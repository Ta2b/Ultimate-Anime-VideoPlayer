#include "app/play/chain_choices.h"

#include <filesystem>
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
#include "app/play/model_choice.h"
#include "app/play/paths.h"
#include "app/play/placebo_screen.h"
#include "app/play/settings.h"
#include "app/play/span_int8.h"
#include "app/play/stage_fallback.h"
#include "core/utf8_path.h"

namespace kirari::app::play {

SettingsChoices chain_choices(Player& p, const ChainEngines& c, const SpanPrepare& span_prepare,
                              int auto_peak, bool source_hdr) {
  const media::TrackInfo& video = p.video;
  const PlaceboScreen& out = p.screen;
  const bool rife_loaded = c.interpolates();
  const bool fp8_engine = c.backend && c.backend->capabilities().fp8;
  const std::optional<DrbaModel> drba_file = find_distil_drba(c.model_places);
  const std::optional<GmfssModels> gmfss = find_gmfss(c.model_places);
  // Les modèles du poste, que SPAN tourne ou non : un agrandissement désactivé ne vide pas le menu.
  const std::optional<ModelChoice>& span = c.model ? c.model : c.span_list;
  return SettingsChoices{
      .models = span ? span->available : std::vector<std::string>{},
      .default_model = span ? span->default_name : std::string(),
      .source_width = video.width,
      .source_height = video.height,
      .source_fps = fps_of(video.frame_rate),
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
      .gmfss_half_flow_available = gmfss && !gmfss->half_pair.empty(),
      .gmfss_available = c.backend && c.backend->capabilities().gmfss && gmfss.has_value(),
      .rife_loaded = rife_loaded,
      .opened = c.opened,
      .rife_models = c.rife_choice ? c.rife_choice->available : std::vector<std::string>{},
      .default_rife_model = c.rife_choice ? c.rife_choice->default_name : std::string(),
      .rife_folder = core::to_utf8(c.model_places.front()),
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
