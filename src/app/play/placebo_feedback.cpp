#include "app/play/placebo_feedback.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string_view>
#include <utility>
#include <vector>

namespace ajp::app::play {

#if defined(VK_KHR_present_id2) && defined(VK_KHR_present_wait2) && defined(VK_EXT_present_timing)
#define AJP_PRESENT_FEEDBACK 1
#endif

namespace {
constexpr std::uint32_t kQueueSize = 16;     // dates en attente de relecture dans le pilote
constexpr std::int64_t kMaxLeadUs = 25'000;  // avance au plus d'une date visée (6.6)
}  // namespace

struct PresentFeedback::Api {
#ifdef AJP_PRESENT_FEEDBACK
  PFN_vkSetSwapchainPresentTimingQueueSizeEXT queue_size = nullptr;
  PFN_vkGetSwapchainTimingPropertiesEXT properties = nullptr;
  PFN_vkGetSwapchainTimeDomainPropertiesEXT domains = nullptr;
  PFN_vkGetPastPresentationTimingEXT past = nullptr;
  PFN_vkWaitForPresent2KHR wait = nullptr;
  PFN_vkGetCalibratedTimestampsKHR calibrated = nullptr;
#endif
};

struct PresentFeedback::TimingStorage {
#ifdef AJP_PRESENT_FEEDBACK
  VkPresentTimingInfoEXT info{.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT};
  VkPresentTimingsInfoEXT infos{.sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT};
  std::array<VkPastPresentationTimingEXT, kQueueSize> past{};
  std::array<VkPresentStageTimeEXT, kQueueSize> stages{};
#endif
};

PresentFeedback::PresentFeedback(pl_vulkan vk, PresentLevel level, const PresentCaps& caps,
                                 std::shared_ptr<spdlog::logger> channel,
                                 std::unique_ptr<WaylandPresentation> wayland)
    : vk_(vk),
      level_(level),
      caps_(caps),
      channel_(std::move(channel)),
      api_(std::make_unique<Api>()),
      wayland_(std::move(wayland)),
      timing_(std::make_unique<TimingStorage>()) {
#ifdef AJP_PRESENT_FEEDBACK
  const auto load = [&](auto& into, const char* fn) {
    into = reinterpret_cast<std::remove_reference_t<decltype(into)>>(vk_->get_proc_addr(vk_->instance, fn));
  };
  load(api_->queue_size, "vkSetSwapchainPresentTimingQueueSizeEXT");
  load(api_->properties, "vkGetSwapchainTimingPropertiesEXT");
  load(api_->domains, "vkGetSwapchainTimeDomainPropertiesEXT");
  load(api_->past, "vkGetPastPresentationTimingEXT");
  load(api_->wait, "vkWaitForPresent2KHR");
  if (caps_.calibrated) load(api_->calibrated, "vkGetCalibratedTimestampsKHR");
  const bool timing = level_ == PresentLevel::Timing || level_ == PresentLevel::TimingAbsolute;
  if (timing && (api_->queue_size == nullptr || api_->past == nullptr || api_->domains == nullptr))
    fall_back("fonctions de present timing absentes");
  // Le premier pixel visible si la surface le date, sinon le premier pixel sorti.
  stage_ = (caps_.present_stages & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT) != 0
               ? VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT
               : VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT;
  if (timing_level(level_) && (caps_.present_stages & stage_) == 0)
    fall_back("la surface ne date ni le premier pixel visible ni sorti");
#endif
}

PresentFeedback::~PresentFeedback() {
  channel_->info(
      "présentation : niveau {}, {} présentées, {} revenues ({} écartées), {} perdues, {} sans demande de "
      "date, {} file pleine ; {:.3f} Hz mesurés, latence de sortie {} µs",
      name(level_), ledger_.issued(), ledger_.returned(), discarded_, ledger_.lost(), ledger_.untimed(),
      queue_full_, clock_.hz(), clock_.output_latency_us());
}

const pl_vulkan_present* PresentFeedback::prepare(pl_swapchain swapchain, std::int64_t now_us,
                                                  std::int64_t target_us) {
  read_ = FeedbackRead{};
  std::uint64_t recreations = 0;
  const VkSwapchainKHR sw = pl_vulkan_swapchain_current(swapchain, &recreations);
  if (sw == VK_NULL_HANDLE) return nullptr;
  if (recreations != recreations_) {
    recreations_ = recreations;
    new_swapchain(sw, now_us);
  }
  drop_if_silent();
  if (level_ == PresentLevel::Wayland) {
    // wp_presentation : pas d'identifiant Vulkan (la swapchain n'en a pas), la demande part au compositeur.
    read_wayland();
    const core::PresentLedger::Issue issue = ledger_.issue(now_us, 0, wayland_ != nullptr);
    last_id_ = issue.id;
    if (wayland_ && issue.timing) wayland_->request(issue.id);
    return nullptr;
  }
  const bool timing = level_ == PresentLevel::Timing || level_ == PresentLevel::TimingAbsolute;
  if (timing) {
    if (domain_.due(now_us)) calibrate(sw, now_us);
    read_back(sw, now_us);
  } else if (level_ == PresentLevel::Wait) {
    poll_wait(sw, now_us);
  }
  // Une date visée seulement au niveau TimingAbsolute en VRR (6.6) : la latence de sortie mesurée retirée,
  // jamais dans le passé ni plus de kMaxLeadUs devant.
  if (level_ != PresentLevel::TimingAbsolute || fixed_refresh() || target_us == 0) {
    target_us = 0;
  } else {
    target_us -= clock_.output_latency_us();
    target_us = target_us <= now_us ? 0 : std::min(target_us, now_us + kMaxLeadUs);
  }
  // Au niveau Timing, la date n'est demandée qu'avec un domaine de temps calé.
  const core::PresentLedger::Issue issue =
      ledger_.issue(now_us, target_us, !timing || (domain_.known() && level_ != PresentLevel::Wait));
  last_id_ = issue.id;
  // Wait et Timing seulement : la swapchain a VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR (present_flags).
  present_ = pl_vulkan_present{.present_id = caps_.present_id2 ? issue.id : 0, .timing = nullptr};
#ifdef AJP_PRESENT_FEEDBACK
  if (timing && issue.timing) {
    VkPresentTimingInfoEXT& info = timing_->info;
    info = VkPresentTimingInfoEXT{.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT};
    info.timeDomainId = domain_id_;
    info.presentStageQueries = stage_;
    info.targetTimeDomainPresentStage = stage_;
    if (target_us != 0) {
      info.targetTime = domain_.to_domain_ns(target_us);
      // En fréquence fixe, au rafraîchissement le plus proche de la date visée.
      if (refresh_duration_ != 0 && refresh_interval_ == refresh_duration_)
        info.flags |= VK_PRESENT_TIMING_INFO_PRESENT_AT_NEAREST_REFRESH_CYCLE_BIT_EXT;
    }
    timing_->infos = VkPresentTimingsInfoEXT{
        .sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, .swapchainCount = 1, .pTimingInfos = &info};
    present_.timing = &timing_->infos;
  }
#endif
  return present_.present_id != 0 ? &present_ : nullptr;
}

// Present timing qui ne peut pas dater : wp_presentation s'il est ouvert (il date), sinon le sondage.
void PresentFeedback::fall_back(std::string_view why) {
  level_ = fallback_level(wayland_ != nullptr);
  ledger_.forget();
  channel_->warn("présentation : {}, niveau {}", why, name(level_));
}

// Present timing qui ne rend rien (NVIDIA 615 sous KDE Wayland : 0 image revenue) retombe (A10-06).
void PresentFeedback::drop_if_silent() {
  if (timing_level(level_) && feedback_silent(ledger_.issued(), ledger_.returned()))
    fall_back(std::format("aucun retour après {} présentations", ledger_.issued()));
}

void PresentFeedback::submitted(VkResult result) {
#ifdef AJP_PRESENT_FEEDBACK
  if (result == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT) {
    ++queue_full_;
    ledger_.queue_full();
  }
#else
  (void)result;
#endif
}

void PresentFeedback::new_swapchain(VkSwapchainKHR swapchain, std::int64_t now_us) {
  clock_.reset();  // P3.44 : d'un mode à l'autre (60 puis 144 Hz), les dates d'avant fausseraient la grille
  if (level_ == PresentLevel::Wayland) return;  // les retours du compositeur suivent la surface
  ledger_.forget();
  domain_.reset();
  local_domain_ = false;
  properties_counter_ = UINT64_MAX;
  domains_counter_ = UINT64_MAX;
  (void)now_us;  // le calage du domaine local se fait à la première présentation (prepare)
#ifdef AJP_PRESENT_FEEDBACK
  if (level_ != PresentLevel::Timing && level_ != PresentLevel::TimingAbsolute) return;
  if (const VkResult r = api_->queue_size(vk_->device, swapchain, kQueueSize); r != VK_SUCCESS) {
    fall_back(std::format("vkSetSwapchainPresentTimingQueueSizeEXT a échoué ({})", static_cast<int>(r)));
    return;
  }
  read_properties(swapchain);
  choose_domain(swapchain);
  if (!domain_.known() && !local_domain_) fall_back("aucun domaine de temps lisible");
#else
  (void)swapchain;
#endif
}

void PresentFeedback::read_properties(VkSwapchainKHR swapchain) {
#ifdef AJP_PRESENT_FEEDBACK
  if (api_->properties == nullptr) return;
  VkSwapchainTimingPropertiesEXT props{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT};
  std::uint64_t counter = 0;
  const VkResult r = api_->properties(vk_->device, swapchain, &props, &counter);
  if (r != VK_SUCCESS) return;  // VK_NOT_READY : pas encore connues, relues au compteur suivant
  properties_counter_ = counter;
  if (props.refreshDuration != refresh_duration_ || props.refreshInterval != refresh_interval_) {
    channel_->info("présentation : rafraîchissement {} ns, intervalle {}", props.refreshDuration,
                   props.refreshInterval == UINT64_MAX ? std::string("variable")
                                                       : std::to_string(props.refreshInterval) + " ns");
  }
  const RefreshState before = refresh_state();
  refresh_duration_ = props.refreshDuration;
  refresh_interval_ = props.refreshInterval;
  if (refresh_state() != before) refresh_changed_ = true;
#else
  (void)swapchain;
#endif
}

void PresentFeedback::choose_domain(VkSwapchainKHR swapchain) {
#ifdef AJP_PRESENT_FEEDBACK
  VkSwapchainTimeDomainPropertiesEXT props{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT};
  std::uint64_t counter = 0;
  if (api_->domains(vk_->device, swapchain, &props, &counter) != VK_SUCCESS || props.timeDomainCount == 0)
    return;
  std::vector<VkTimeDomainKHR> kinds(props.timeDomainCount);
  std::vector<std::uint64_t> ids(props.timeDomainCount);
  props.pTimeDomains = kinds.data();
  props.pTimeDomainIds = ids.data();
  const VkResult r = api_->domains(vk_->device, swapchain, &props, &counter);
  if (r != VK_SUCCESS && r != VK_INCOMPLETE) return;
  domains_counter_ = counter;
  domain_.reset();
  local_domain_ = false;
  // CLOCK_MONOTONIC, celui de host_now_us, tel quel ; sinon un domaine local recalé sur l'hôte.
  int local = -1;
  for (std::uint32_t i = 0; i < props.timeDomainCount; ++i) {
    if (kinds[i] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR) {
      domain_kind_ = kinds[i];
      domain_id_ = ids[i];
      domain_.set_monotonic();
      return;
    }
    if (local < 0 && (kinds[i] == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT ||
                      kinds[i] == VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT))
      local = static_cast<int>(i);
  }
  if (local < 0 || api_->calibrated == nullptr) return;
  domain_kind_ = kinds[static_cast<std::size_t>(local)];
  domain_id_ = ids[static_cast<std::size_t>(local)];
  local_domain_ = true;  // inconnu jusqu'au premier calage (prepare)
#else
  (void)swapchain;
#endif
}

void PresentFeedback::calibrate(VkSwapchainKHR swapchain, std::int64_t now_us) {
#ifdef AJP_PRESENT_FEEDBACK
  if (!local_domain_ || api_->calibrated == nullptr) return;
  const VkSwapchainCalibratedTimestampInfoEXT local{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT,
      .swapchain = swapchain,
      .presentStage = domain_kind_ == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT ? stage_ : 0,
      .timeDomainId = domain_id_};
  const std::array<VkCalibratedTimestampInfoKHR, 2> infos{{
      {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
       .pNext = &local,
       .timeDomain = static_cast<VkTimeDomainKHR>(domain_kind_)},
      {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
       .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR},
  }};
  std::array<std::uint64_t, 2> stamps{};
  std::uint64_t deviation = 0;
  if (api_->calibrated(vk_->device, 2, infos.data(), stamps.data(), &deviation) == VK_SUCCESS)
    domain_.calibrate(stamps[0], stamps[1], now_us);
#else
  (void)swapchain;
  (void)now_us;
#endif
}

void PresentFeedback::read_back(VkSwapchainKHR swapchain, std::int64_t now_us) {
#ifdef AJP_PRESENT_FEEDBACK
  (void)now_us;
  if (ledger_.in_flight() == 0) return;
  TimingStorage& t = *timing_;
  for (std::size_t i = 0; i < kQueueSize; ++i) {
    t.stages[i] = VkPresentStageTimeEXT{};
    t.past[i] = VkPastPresentationTimingEXT{.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT,
                                            .presentStageCount = 1,
                                            .pPresentStages = &t.stages[i]};
  }
  const VkPastPresentationTimingInfoEXT info{
      .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT,
      .flags = VK_PAST_PRESENTATION_TIMING_ALLOW_PARTIAL_RESULTS_BIT_EXT |
               VK_PAST_PRESENTATION_TIMING_ALLOW_OUT_OF_ORDER_RESULTS_BIT_EXT,
      .swapchain = swapchain};
  VkPastPresentationTimingPropertiesEXT props{
      .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
      .presentationTimingCount = kQueueSize,
      .pPresentationTimings = t.past.data()};
  const VkResult r = api_->past(vk_->device, &info, &props);
  if (r != VK_SUCCESS && r != VK_INCOMPLETE) return;
  if (props.timingPropertiesCounter != properties_counter_) read_properties(swapchain);
  if (props.timeDomainsCounter != domains_counter_) choose_domain(swapchain);
  for (std::uint32_t i = 0; i < props.presentationTimingCount && i < kQueueSize; ++i) {
    const VkPastPresentationTimingEXT& p = t.past[i];
    if (p.presentStageCount == 0 || p.pPresentStages[0].time == 0) continue;  // pas encore daté
    const std::uint64_t ns = p.pPresentStages[0].time;
    const std::int64_t visible = p.timeDomain == VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR
                                     ? static_cast<std::int64_t>(ns / 1000)
                                     : domain_.to_host_us(ns);
    accept(p.presentId, visible, true);
  }
#else
  (void)swapchain;
  (void)now_us;
#endif
}

void PresentFeedback::poll_wait(VkSwapchainKHR swapchain, std::int64_t now_us) {
#ifdef AJP_PRESENT_FEEDBACK
  if (api_->wait == nullptr || !caps_.present_wait2) return;
  // Sondé seulement (délai 0) : l'attente bloquante tourne à vide chez certains pilotes. La date n'est que
  // celle du sondage : elle confirme l'affichage sans nourrir la grille des vblanks.
  for (int i = 0; i < 4; ++i) {
    const std::uint64_t id = ledger_.oldest();
    if (id == 0) return;
    const VkPresentWait2InfoKHR info{
        .sType = VK_STRUCTURE_TYPE_PRESENT_WAIT_2_INFO_KHR, .presentId = id, .timeout = 0};
    if (api_->wait(vk_->device, swapchain, &info) != VK_SUCCESS) return;
    accept(id, now_us, false);
  }
#else
  (void)swapchain;
  (void)now_us;
#endif
}

void PresentFeedback::record_stamps(const std::filesystem::path& file) {
  if (!wayland_ || file.empty()) return;
  stamps_.open(file, std::ios::binary | std::ios::trunc);
  if (stamps_) stamps_ << screen_stamp_header();
}

void PresentFeedback::read_wayland() {
  if (!wayland_) return;
  wayland_done_.clear();
  wayland_->read(wayland_done_);
  for (const WaylandDone& done : wayland_done_) {
    if (stamps_.is_open())
      stamps_ << screen_stamp_line(ScreenStamp{.seq = done.id,
                                               .presented_us = done.visible_us,
                                               .refresh_ns = done.refresh_ns,
                                               .msc = done.msc,
                                               .flags = done.flags});
    if (done.visible_us == 0) {
      if (ledger_.report(done.id, 0)) ++discarded_;
      continue;
    }
    if (done.refresh_ns != 0) refresh_duration_ = done.refresh_ns;
    accept(done.id, done.visible_us, true);
  }
}

void PresentFeedback::accept(std::uint64_t id, std::int64_t visible_us, bool dated) {
  const auto sample = ledger_.report(id, visible_us);
  if (!sample) return;
  if (dated) {
    // La période annoncée (swapchain ou compositeur) numérote les vblanks : sans elle, des présentations
    // à l'image neuve feraient apprendre la cadence du contenu (A12-01).
    clock_.set_nominal_period(static_cast<double>(refresh_duration_) / 1000.0);
    clock_.observe(sample->visible_us, sample->id, sample->target_us);
  }
  read_ = FeedbackRead{.id = sample->id, .target_us = sample->target_us, .visible_us = sample->visible_us};
  if (const auto grid = clock_.grid(); grid && dated)
    read_.vsync_k = std::llround(grid->index_of(static_cast<double>(sample->visible_us)));
}

}  // namespace ajp::app::play
