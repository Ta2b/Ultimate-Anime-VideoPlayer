#include "app/play/grade_planner.h"

#include <algorithm>
#include <cmath>
#include <utility>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {
namespace {

int e_steps_of(double e) {
  return static_cast<int>(std::lround(e / kEStep));
}

// L_max (nits affichés) à la surface `area` (0 à 1), linéaire entre les surfaces de render::kAblAreas.
double abl_at(const render::DisplayContract& c, double area) {
  const auto& areas = render::kAblAreas;
  if (area <= areas.front()) return c.abl.front();
  for (std::size_t i = 1; i < areas.size(); ++i) {
    if (area <= areas.at(i)) {
      const double t = (area - areas.at(i - 1)) / (areas.at(i) - areas.at(i - 1));
      return c.abl.at(i - 1) + t * (c.abl.at(i) - c.abl.at(i - 1));
    }
  }
  return c.abl.back();
}

void lower_priority() {
#ifdef __linux__
  (void)setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
#elif defined(_WIN32)
  (void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

}  // namespace

void GradePlan::set_shots(std::vector<media::Shot> shots) {
  shots_ = std::move(shots);
  std::vector<double> apl;
  apl.reserve(shots_.size());
  for (const media::Shot& s : shots_) apl.push_back(s.meta.apl_mean_nits);
  series_apl_nits_ = 0.0;
  if (!apl.empty()) {
    const auto mid = apl.begin() + static_cast<std::ptrdiff_t>(apl.size() / 2);
    std::nth_element(apl.begin(), mid, apl.end());
    series_apl_nits_ = *mid;
  }
  ++generation_;
  given_.clear();
}

void GradePlan::set_contract(const render::DisplayContract& contract) {
  if (contract == contract_) return;
  contract_ = contract;
  ++generation_;
}

void GradePlan::set_look(std::shared_ptr<const render::CubeLut> look) {
  look_ = std::move(look);
  ++generation_;
}

void GradePlan::set_user_e(double e) {
  target_e_ = std::clamp(e, -render::kCurveMaxE, render::kCurveMaxE);
}

render::CurveInput GradePlan::input_of(const media::Shot& shot, double apl_nits, double user_e) const {
  const double k = contract_.k();
  render::CurveInput in;
  in.shot_apl_nits = apl_nits;
  in.series_apl_nits = series_apl_nits_;
  in.user_e = user_e;
  in.content_max_nits = shot.meta.max_scl_nits > 0.0F ? shot.meta.max_scl_nits : 1000.0;
  in.panel_peak_nits = contract_.content_peak_nits();
  // Le limiteur à la surface claire du plan, et le budget de l'image entière, en unités du contenu.
  in.abl_limit_nits =
      abl_at(contract_, std::max<double>(shot.meta.mean.bright, render::kAblAreas.front())) / k;
  in.apl_budget_nits = kAplBudgetShare * contract_.content_frame_average_nits();
  in.lum_hist = shot.meta.mean.lum_hist;
  in.vibrance = config_.vibrance;
  return in;
}

std::optional<std::size_t> GradePlan::shot_at(std::int64_t pts_us) const {
  if (shots_.empty()) return std::nullopt;
  const auto after = std::ranges::upper_bound(shots_, pts_us, {}, &media::Shot::start_us);
  if (after == shots_.begin()) return 0;
  return static_cast<std::size_t>(after - shots_.begin() - 1);
}

BakeJob GradePlan::job_of(std::size_t index, int part) const {
  const media::Shot& shot = shots_.at(index);
  double apl = shot.meta.apl_mean_nits > 0.0F ? shot.meta.apl_mean_nits : shot.first.apl_nits;
  if (shot.fade) apl = part == 0 ? shot.first.apl_nits : shot.last.apl_nits;
  const double e = std::round(applied_e_ / kEStep) * kEStep;
  return BakeJob{.key = {.start_us = shot.start_us,
                         .part = part,
                         .e_steps = e_steps_of(applied_e_),
                         .generation = generation_},
                 .input = input_of(shot, apl, e)};
}

std::vector<BakeJob> GradePlan::wanted(std::int64_t pts_us) const {
  std::vector<BakeJob> out;
  const std::optional<std::size_t> first = shot_at(pts_us);
  if (!first) return out;
  for (std::size_t i = *first; i < std::min(shots_.size(), *first + 1 + kPlanAhead); ++i) {
    for (int part = 0; part < (shots_.at(i).fade ? 2 : 1); ++part) {
      BakeJob job = job_of(i, part);
      if (std::ranges::none_of(cache_, [&](const Entry& e) { return e.key == job.key; }))
        out.push_back(std::move(job));
    }
  }
  return out;
}

std::shared_ptr<const std::vector<float>> GradePlan::bake(const BakeJob& job, const render::CubeLut* look,
                                                          int size) {
  return std::make_shared<const std::vector<float>>(render::bake(render::plan_curve(job.input), look, size));
}

void GradePlan::store(const BakeJob::Key& key, std::shared_ptr<const std::vector<float>> lut) {
  std::erase_if(cache_, [&](const Entry& e) { return e.key == key; });
  cache_.push_front(Entry{.key = key, .lut = std::move(lut)});
  while (cache_.size() > kPlanCache) cache_.pop_back();
}

std::shared_ptr<const std::vector<float>> GradePlan::find(const BakeJob::Key& key) {
  const auto it = std::ranges::find_if(cache_, [&](const Entry& e) { return e.key == key; });
  if (it == cache_.end()) return nullptr;
  cache_.splice(cache_.begin(), cache_, it);  // le plus récent en tête
  return cache_.front().lut;
}

std::vector<GradePlanOrder> GradePlan::take_orders(std::int64_t pts_us, std::int64_t now_us) {
  // E vers sa cible, à kLiveEPerSecond au plus.
  if (e_moved_at_us_) {
    const double step = kLiveEPerSecond * static_cast<double>(now_us - *e_moved_at_us_) / 1e6;
    applied_e_ += std::clamp(target_e_ - applied_e_, -std::max(step, 0.0), std::max(step, 0.0));
  }
  e_moved_at_us_ = now_us;

  std::vector<GradePlanOrder> out;
  const std::optional<std::size_t> first = shot_at(pts_us);
  if (!first) return out;
  std::size_t end = *first;
  std::vector<std::pair<std::int64_t, BakeJob::Key>> giving;
  for (std::size_t i = *first; i < std::min(shots_.size(), *first + 1 + kPlanAhead); ++i, ++end) {
    const media::Shot& shot = shots_.at(i);
    const BakeJob job = job_of(i, 0);
    const auto given =
        std::ranges::find(given_, shot.start_us, &std::pair<std::int64_t, BakeJob::Key>::first);
    if (given != given_.end() && given->second == job.key) continue;
    std::shared_ptr<const std::vector<float>> lut = find(job.key);
    std::shared_ptr<const std::vector<float>> fade_to;
    if (shot.fade) fade_to = find(job_of(i, 1).key);
    if (!lut || (shot.fade && !fade_to)) break;
    out.push_back(GradePlanOrder{.from_pts_us = shot.start_us,
                                 .until_pts_us = shot.end_us,
                                 .size = config_.lut_size,
                                 .lut = std::move(lut),
                                 .fade_to = std::move(fade_to),
                                 .curve = render::plan_curve(job.input),
                                 .e = job.input.user_e});
    giving.emplace_back(shot.start_us, job.key);
  }
  // Un plan plus loin de la fenêtre déjà donné avec une autre clé (E ou contrat changés) : son ordre d'avant
  // l'emporterait sur l'ordre plus récent d'un plan plus tôt. On attend qu'il soit cuit lui aussi.
  for (std::size_t i = end; i < std::min(shots_.size(), *first + 1 + kPlanAhead); ++i) {
    const auto given =
        std::ranges::find(given_, shots_.at(i).start_us, &std::pair<std::int64_t, BakeJob::Key>::first);
    if (given != given_.end() && given->second != job_of(i, 0).key && !out.empty()) return {};
  }
  for (auto& [start, key] : giving) {
    std::erase_if(given_, [&](const auto& g) { return g.first == start; });
    given_.emplace_back(start, key);
  }
  while (given_.size() > 2 * kPlanCache) given_.erase(given_.begin());
  return out;
}

GradePlanner::GradePlanner(PlannerConfig config) : plan_(config) {
  auto started = start_thread(
      "grade", ThreadReport([](const Refusal& refusal) { log_channel("app")->warn("{}", refusal.detail); }),
      [this](const std::stop_token& stop, const ThreadReport& /*report*/) { run(stop); });
  if (started) {
    thread_ = std::move(*started);
  } else {
    log_channel("app")->warn("{} : sans planificateur du grade", started.error().detail);
  }
}

GradePlanner::~GradePlanner() {
  if (thread_.joinable()) {
    thread_.request_stop();
    wake_.notify_all();
    thread_.join();
  }
}

void GradePlanner::set_shots(std::vector<media::Shot> shots) {
  const std::scoped_lock lock(mutex_);
  plan_.set_shots(std::move(shots));
  wake_.notify_all();
}

void GradePlanner::set_contract(const render::DisplayContract& contract) {
  const std::scoped_lock lock(mutex_);
  plan_.set_contract(contract);
  wake_.notify_all();
}

void GradePlanner::set_look(std::shared_ptr<const render::CubeLut> look) {
  const std::scoped_lock lock(mutex_);
  plan_.set_look(std::move(look));
  wake_.notify_all();
}

void GradePlanner::set_user_e(double e) {
  const std::scoped_lock lock(mutex_);
  plan_.set_user_e(e);
  wake_.notify_all();
}

std::vector<GradePlanOrder> GradePlanner::take_orders(std::int64_t pts_us, std::int64_t now_us) {
  const std::scoped_lock lock(mutex_);
  if (pts_us != position_us_) {
    position_us_ = pts_us;
    wake_.notify_all();
  }
  std::vector<GradePlanOrder> orders = plan_.take_orders(pts_us, now_us);
  wake_.notify_all();  // E a pu bouger : de nouvelles cuissons
  return orders;
}

std::size_t GradePlanner::baked() const {
  const std::scoped_lock lock(mutex_);
  return baked_;
}

void GradePlanner::run(const std::stop_token& stop) {
  lower_priority();
  std::unique_lock lock(mutex_);
  while (!stop.stop_requested()) {
    std::vector<BakeJob> jobs;
    if (!wake_.wait(lock, stop, [&] { return !(jobs = plan_.wanted(position_us_)).empty(); })) break;
    const BakeJob job = jobs.front();
    const std::shared_ptr<const render::CubeLut> look = plan_.look_shared();
    const int size = plan_.lut_size();
    lock.unlock();
    std::shared_ptr<const std::vector<float>> lut = GradePlan::bake(job, look.get(), size);
    lock.lock();
    plan_.store(job.key, std::move(lut));
    ++baked_;
  }
}

}  // namespace ajp::app::play
