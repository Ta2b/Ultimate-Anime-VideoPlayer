#include "app/play/chain_choices.h"

#include <filesystem>
#include <utility>
#include <vector>

#include "app/play/chain.h"
#include "app/play/chain_stats.h"
#include "app/play/display_contract.h"
#include "app/play/display_edid.h"
#include "app/play/drba_model.h"
#include "app/play/engine_screen.h"
#include "app/play/engine_switch.h"
#include "app/play/first_run.h"
#include "app/play/gmfss_model.h"
#include "app/play/host_facts.h"
#include "app/play/interp_retired.h"
#include "app/play/model_choice.h"
#include "app/play/paths.h"
#include "app/play/placebo_screen.h"
#include "app/play/settings.h"
#include "app/play/span_int8.h"
#include "app/play/stage_fallback.h"
#include "core/utf8_path.h"

namespace kirari::app::play {

std::function<void(SettingsChoices&, const PlaySettings&)> motion_models_refresh(
    const PlayOptions& options, const PlayerPaths& paths, const GpuFacts& gpu, bool fp8_engine,
    bool gmfss_supported, GmfssMenuProbe probe) {
  return [options, paths, preference = rife_preference(gpu), fp8_engine, gmfss_supported,
          probe = std::move(probe)](SettingsChoices& choices, const PlaySettings& running) mutable {
    auto found = choose_rife_model(options, paths.home, running, preference, paths.models);
    std::optional<ModelChoice> rife;
    if (found) rife = std::move(*found);
    const auto dirs = motion_model_dirs(paths, rife, options.gmfss_dir);
    const auto drba = find_distil_drba(dirs);
    const GmfssAvailability gmfss = gmfss_supported ? probe.read(dirs) : GmfssAvailability{};
    choices.rife_available = rife || drba || offered(MotionEngine::Apple);
    choices.distil_drba_available = drba.has_value();
    choices.gmfss_available = gmfss.available;
    choices.gmfss_half_flow_available = gmfss.half_flow;
    choices.rife_tiers = rife ? sibling_tiers(rife->path, fp8_engine) : TierSupport{};
    choices.drba_tiers = drba ? sibling_tiers(drba->path, fp8_engine) : TierSupport{};
    choices.default_rife_model = rife ? std::move(rife->default_name) : std::string();
    choices.rife_models = rife ? std::move(rife->available) : std::vector<std::string>{};
    choices.rife_folder = dirs.empty() ? std::string() : core::to_utf8(dirs.front());
  };
}

SettingsChoices chain_choices(Player& p, const ChainEngines& c, const SpanPrepare& span_prepare,
                              int auto_peak, bool source_hdr) {
  const media::TrackInfo& video = p.video;
  const double source_fps = fps_of(video.frame_rate);
  const PlaceboScreen& out = p.screen;
  const bool rife_loaded = c.interpolates();
  const bool fp8_engine = c.backend && c.backend->capabilities().fp8;
  const std::optional<DrbaModel> drba_file = find_distil_drba(c.model_places);
  const bool gmfss_supported = gmfss_kept() && c.backend && c.backend->capabilities().gmfss;
  GmfssMenuProbe probe;
  const GmfssAvailability gmfss = gmfss_supported ? probe.read(c.model_places) : GmfssAvailability{};
  // Les modèles du poste, que SPAN tourne ou non : un agrandissement désactivé ne vide pas le menu.
  const std::optional<ModelChoice>& span = c.model ? c.model : c.span_list;
  return SettingsChoices{
      .models = span ? span->available : std::vector<std::string>{},
      .default_model = span ? span->default_name : std::string(),
      .source_width = video.width,
      .source_height = video.height,
      .source_fps = source_fps,
      .source_hdr = source_hdr,
      .source_rate = video.frame_rate,
      .display_hz = display_period_us(out.window()) > 0
                        ? 1e6 / static_cast<double>(display_period_us(out.window()))
                        : 0.0,
      .display_vrr = p.display && p.display->vrr.value_or(false),
      .display_vrr_armed = p.display && p.display->vrr_armed.value_or(false),
      .display_oled = p.display && p.display->oled,
      .rife_available = c.rife_model.has_value() || c.drba_found || offered(MotionEngine::Apple),
      .span_tiers =
          span ? span_tiers(span->path, paths::derived_models(p.paths), fp8_engine,
                            c.backend && c.backend->capabilities().span_fp8_cutlass,
                            p.options.span_native && c.backend && c.backend->capabilities().span_fp16_cutlass)
               : TierSupport{},
      .rife_tiers = c.rife_choice ? sibling_tiers(c.rife_choice->path, fp8_engine) : TierSupport{},
      .drba_tiers = drba_file ? sibling_tiers(drba_file->path, fp8_engine) : TierSupport{},
      .distil_drba_available = c.drba_found,
      .gmfss_half_flow_available = gmfss.half_flow,
      .gmfss_available = gmfss.available,
      .gmfss_loaded = c.gmfss_use != GmfssUse::Off,
      .gmfss_only_auto_cap = gmfss_auto_cap(c, 0, source_fps),
      .rife_loaded = rife_loaded,
      .opened = c.opened,
      .rife_models = c.rife_choice ? c.rife_choice->available : std::vector<std::string>{},
      .default_rife_model = c.rife_choice ? c.rife_choice->default_name : std::string(),
      .rife_folder = c.model_places.empty() ? std::string() : core::to_utf8(c.model_places.front()),
      .refresh_motion_models =
          motion_models_refresh(p.options, p.paths, gpu_facts(p.screen), fp8_engine, gmfss_supported, std::move(probe)),
      .hdr_output = p.screen.hdr_output(),
      .auto_peak = auto_peak,
      .engines = engine_badges(p.paths.engines, c.loaded_engines),
      .chosen_engines = chosen_line(p.engines),
      .engine_in_use = engine_of_library(p.library->path()),
      .engine_note =
          fallback_note(p.engines, p.library->path(), gpu_facts(p.screen), p.library->passed_over_code()),
      .display_text = p.display ? describe(*p.display) : std::string("EDID illisible : OLED inconnu"),
      .source_file = p.options.file,
      .display_height = display_height(out.window()),
      .span_prepare = span_prepare,
      .span_job = p.span_job.note,
      .stage_note = stage_panel_note(c.stage_notes)};
}

}  // namespace kirari::app::play
