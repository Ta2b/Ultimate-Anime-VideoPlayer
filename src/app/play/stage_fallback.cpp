#include "app/play/stage_fallback.h"

#include <algorithm>
#include <format>
#include <utility>

#include "app/play/player_engines.h"
#include "app/play/precision.h"
#include "app/play/visible_error.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

bool watch_now(const Refusal& refusal) {
  return refusal.key == kWatchNowKey || refusal.code == RefusalCode::Cancelled;
}

std::string stage_signature(ChainStage stage, const PlaySettings& s, int source_width, int source_height,
                            std::string_view library) {
  // La taille de travail et la source fixent la forme des moteurs ; le modèle, le cran et le module, le
  // graphe et sa construction.
  const std::string common = std::format("{}|{}x{}|{}|{}|{}", library, source_width, source_height,
                                         s.work_height, name(s.downscale), s.restore_lines);
  if (stage == ChainStage::Upscale)
    return std::format("span|{}|{}|{}", common, s.model, name(s.span_precision));
  // L'interpolation dépend aussi de SPAN (ordre du warp, sortie haute) et du facteur le plus haut.
  return std::format("motion|{}|{}|{}|{}|{}|{}|{}|{}|{}", common, name(s.motion_engine), s.rife_model,
                     name(s.motion_precision), s.rife_fused, s.drba_fused, s.upscale, name(s.span_order),
                     s.rife_factor);
}

const StageMemory::Entry* StageMemory::find(ChainStage stage, std::string_view signature,
                                            const std::filesystem::path& file) const {
  const auto found = std::ranges::find_if(entries_, [&](const Entry& e) {
    return e.stage == stage && e.signature == signature && (!e.cancelled || e.file == file);
  });
  return found == entries_.end() ? nullptr : &*found;
}

void StageMemory::remember(Entry entry) {
  std::erase_if(entries_,
                [&](const Entry& e) { return e.stage == entry.stage && e.signature == entry.signature; });
  entries_.push_back(std::move(entry));
}

const StageMemory::Inner* StageMemory::find_inner(std::string_view key) const {
  const auto found = std::ranges::find(inner_, key, &Inner::key);
  return found == inner_.end() ? nullptr : &*found;
}

void StageMemory::remember_inner(Inner inner) {
  std::erase_if(inner_, [&](const Inner& i) { return i.key == inner.key; });
  inner_.push_back(std::move(inner));
}

std::string inner_key(std::string_view what, std::string_view library, const std::filesystem::path& model,
                      int width, int height) {
  return std::format("{}|{}|{}|{}x{}", what, library, core::to_utf8(model), width, height);
}

void InnerFallbacks::note(ChainStage stage, const Refusal& refusal) {
  // Une note par étage : le panneau dit qu'un repli joue, avec la première raison.
  if (std::ranges::any_of(
          notes_, [&](const StageNote& n) { return n.stage == stage && n.change == StageChange::Fallback; }))
    return;
  notes_.push_back({.stage = stage, .change = StageChange::Fallback, .refusal = refusal});
}

const Refusal* InnerFallbacks::refused(ChainStage stage, const std::string& key) {
  if (memory_ == nullptr) return nullptr;
  const StageMemory::Inner* found = memory_->find_inner(key);
  if (found == nullptr) return nullptr;
  log_channel("app")->info("repli retenu pour la séance ({}) : {}", found->refusal.detail, key);
  note(stage, found->refusal);
  return &found->refusal;
}

void InnerFallbacks::failed(ChainStage stage, std::string key, const Refusal& refusal) {
  // « Annuler et regarder », une vidéo glissée ou un appareil perdu ne sont pas des refus du moteur : rien
  // n'est retenu.
  if (watch_now(refusal) || refusal.key == kDroppedVideoKey || refusal.code == RefusalCode::DeviceLost)
    return;
  fresh_ = true;
  note(stage, refusal);
  if (memory_ != nullptr)
    memory_->remember_inner({.stage = stage, .key = std::move(key), .refusal = refusal});
}

namespace {

bool stage_on(ChainStage stage, const PlaySettings& s) {
  return stage == ChainStage::Upscale ? s.upscale : s.rife_factor >= 2;
}

PrecisionTier& tier_of(ChainStage stage, PlaySettings& s) {
  return stage == ChainStage::Upscale ? s.span_precision : s.motion_precision;
}

// L'étage retiré pour cette lecture. Sans SPAN, la source garde sa taille : la réduire n'avait de sens que
// pour SPAN (A29-01).
void drop(ChainStage stage, PlaySettings& s) {
  if (stage == ChainStage::Upscale) {
    s.upscale = false;
    s.work_height = 0;
  } else {
    s.rife_factor = 0;
    s.rife_auto = false;
  }
}

}  // namespace

std::optional<StagePlan> plan_stages(const PlaySettings& wanted, const StageMemory& memory, int source_width,
                                     int source_height, std::string_view library,
                                     const std::filesystem::path& file, bool allow_drop) {
  StagePlan plan{.settings = wanted, .notes = {}};
  // SPAN d'abord : le retirer change la taille de travail, donc la signature de l'interpolation.
  for (const ChainStage stage : {ChainStage::Upscale, ChainStage::Motion}) {
    // Au plus deux tours par étage : le cran ramené à Équilibre, puis l'étage retiré.
    for (int turn = 0; turn < 2 && stage_on(stage, plan.settings); ++turn) {
      const StageMemory::Entry* refused = memory.find(
          stage, stage_signature(stage, plan.settings, source_width, source_height, library), file);
      if (refused == nullptr) break;
      // SPAN essaie déjà ses variantes puis le FP16 livré (chain_setup.cpp) : son refus vaut pour tous les
      // crans. L'interpolation charge le graphe du cran tel quel : le FP16 reste à essayer.
      PrecisionTier& tier = tier_of(stage, plan.settings);
      if (!refused->cancelled && stage == ChainStage::Motion && tier != PrecisionTier::Balanced) {
        tier = PrecisionTier::Balanced;
        plan.notes.push_back({.stage = stage, .change = StageChange::Balanced, .refusal = refused->refusal});
        continue;
      }
      if (!allow_drop) return std::nullopt;
      drop(stage, plan.settings);
      plan.notes.push_back({.stage = stage,
                            .change = refused->cancelled ? StageChange::Cancelled : StageChange::Dropped,
                            .refusal = refused->refusal});
    }
  }
  return plan;
}

namespace {

i18n::Translated note_text(const StageNote& note) {
  const bool span = note.stage == ChainStage::Upscale;
  switch (note.change) {
    case StageChange::Cancelled:
      return span ? i18n::ui_tr("Engine build cancelled: playing without upscaling")
                  : i18n::ui_tr("Engine build cancelled: playing without smoothing");
    case StageChange::Dropped:
      return span ? i18n::ui_tr("Upscaling unavailable: playing without it")
                  : i18n::ui_tr("Smoothing unavailable: playing without it");
    case StageChange::Balanced:
      return span ? i18n::ui_tr("Upscaling precision unavailable: Balanced instead")
                  : i18n::ui_tr("Smoothing precision unavailable: Balanced instead");
    case StageChange::Fallback:
      return span ? i18n::ui_tr("Upscaling: a fallback engine runs")
                  : i18n::ui_tr("Smoothing: a fallback engine runs");
  }
  return {};
}

}  // namespace

i18n::Translated stage_notice(const std::vector<StageNote>& notes) {
  i18n::Translated text;
  for (const StageNote& note : notes) {
    // Un repli interne s'annonce à son premier refus (chain_setup.cpp) ; au panneau seulement ensuite.
    if (note.change == StageChange::Fallback) continue;
    text = text.empty() ? note_text(note) : text + i18n::Translated::data(" · ") + note_text(note);
  }
  if (!text.empty())
    text = text + i18n::Translated::data(" ") + i18n::ui_tr("(Video > Post-processing: Retry)");
  return text;
}

std::string stage_panel_note(const std::vector<StageNote>& notes) {
  std::string text;
  for (const StageNote& note : notes) {
    if (!text.empty()) text += "\n";
    text += note_text(note).str();
    if (note.change != StageChange::Cancelled) text += std::format(" ({})", visible_error(note.refusal));
  }
  return text;
}

}  // namespace ajp::app::play
