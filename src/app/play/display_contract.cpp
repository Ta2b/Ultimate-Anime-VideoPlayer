#include "app/play/display_contract.h"

#include <algorithm>
#include <format>
#include <utility>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

namespace ajp::app::play {
namespace {

// Le filtre de SDL voit chaque événement en file, quel que soit le fil : il ne fait que poser le drapeau.
bool on_event(void* context, SDL_Event* event) {
  if (event->type == SDL_EVENT_WINDOW_HDR_STATE_CHANGED || event->type == SDL_EVENT_WINDOW_DISPLAY_CHANGED)
    static_cast<std::atomic<bool>*>(context)->store(true, std::memory_order_relaxed);
  return true;
}

}  // namespace

render::DisplayContract contract_of(const DisplayFacts& facts, const std::optional<KwinOutput>& kwin,
                                    double headroom, const std::optional<MeasuredAbl>& abl) {
  render::DisplayContract out;
  out.oled = facts.oled;
  out.reference_nits = kwin && kwin->reference ? *kwin->reference : render::kContentReferenceNits;
  // La crête : celle réglée dans KDE, sinon la marge de SDL × la référence, sinon l'EDID.
  if (kwin && kwin->peak_override) {
    out.panel_peak_nits = *kwin->peak_override;
  } else if (headroom > 1.0) {
    out.panel_peak_nits = headroom * out.reference_nits;
  } else if (facts.peak_nits) {
    out.panel_peak_nits = *facts.peak_nits;
  }
  out.abl = abl ? abl->nits : render::default_abl(out.panel_peak_nits, out.oled);
  for (double& nits : out.abl) nits = std::min(nits, out.panel_peak_nits);
  out.primaries = facts.primaries;
  out.min_nits = facts.min_nits.value_or(0.0);
  return out;
}

std::string describe(const render::DisplayContract& c) {
  std::string text = std::format(
      "référence {:.0f} nits (k {:.3f}), pic {:.0f} nits ({:.0f} du contenu), MaxFALL {:.0f} ({:.0f} du "
      "contenu), noir {:.4f}{}",
      c.reference_nits, c.k(), c.panel_peak_nits, c.content_peak_nits(), c.abl.back(),
      c.content_frame_average_nits(), c.min_nits, c.oled ? ", OLED" : "");
  if (c.primaries) {
    const render::Chromaticities& p = *c.primaries;
    text += std::format(", primaires R {:.3f}/{:.3f} G {:.3f}/{:.3f} B {:.3f}/{:.3f}", p.rx, p.ry, p.gx, p.gy,
                        p.bx, p.by);
  }
  return text;
}

ContractWatch::ContractWatch(SDL_Window* window, std::optional<DisplayFacts> facts,
                             std::shared_ptr<spdlog::logger> log)
    : window_(window), facts_(std::move(facts)), log_(std::move(log)), kwin_stamp_(kwin_config_stamp()) {
  (void)SDL_AddEventWatch(&on_event, &hdr_changed_);
}

ContractWatch::~ContractWatch() {
  SDL_RemoveEventWatch(&on_event, &hdr_changed_);
}

render::DisplayContract ContractWatch::read() const {
  const DisplayFacts facts = facts_.value_or(DisplayFacts{});
  float headroom = 1.0F;
  if (window_ != nullptr) {
    const SDL_PropertiesID props = SDL_GetWindowProperties(window_);
    headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.0F);
  }
  return contract_of(facts, read_kwin_output(facts.connector), headroom, std::nullopt);
}

std::optional<render::DisplayContract> ContractWatch::tick() {
  const std::optional<std::int64_t> stamp = kwin_config_stamp();
  const bool rewritten = stamp != kwin_stamp_;
  kwin_stamp_ = stamp;
  if (!hdr_changed_.exchange(false, std::memory_order_relaxed) && !rewritten) return std::nullopt;
  render::DisplayContract contract = read();
  if (contract_ && *contract_ == contract) return std::nullopt;
  log_->info("contrat d'écran : {}", describe(contract));
  contract_ = contract;
  return contract_;
}

}  // namespace ajp::app::play
