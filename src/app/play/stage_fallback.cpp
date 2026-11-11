#include "app/play/stage_fallback.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <utility>

#include "app/play/options.h"
#include "app/play/player_engines.h"
#include "app/play/precision.h"
#include "app/play/visible_error.h"

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
  // L'interpolation dépend aussi de SPAN : l'ordre du warp, que le facteur le plus haut et la sortie fixent.
  // Pas le facteur brut, que l'Auto récrit selon l'écran : un refus retenu à ×5 vaut à ×4 (même ordre), et
  // ×5 et ×2,5 ne se confondent plus (B09-05).
  const int output_height = chain_work_size(s, source_width, source_height).height * kSpanFactor;
  const bool after =
      warp_after_span(s.span_order, output_height, s.rife_factor / std::max(s.rife_factor_den, 1));
  return std::format("motion|{}|{}|{}|{}|{}|{}|{}|{}|{}", common, name(s.motion_engine), s.rife_model,
                     name(s.motion_precision), s.rife_fused, s.drba_fused, s.upscale, name(s.span_order),
                     after ? "warp-after" : "warp-before");
}

const StageMemory::Entry* StageMemory::find(ChainStage stage, std::string_view signature,
                                            const std::filesystem::path& file) const {
  const auto found = std::ranges::find_if(entries_, [&](const Entry& e) {
    return e.stage == stage && e.signature == signature && (!e.cancelled || e.file == file);
  });
  return found == entries_.end() ? nullptr : &*found;
}

void StageMemory::remember(Entry entry) {
  const auto same = [&](const Entry& e) { return e.stage == entry.stage && e.signature == entry.signature; };
  if (entry.cancelled) {
    // Une annulation ne vaut que pour sa vidéo : elle ne remplace ni le refus réel de la même signature, qui
    // vaut pour toutes les vidéos, ni l'annulation d'une autre vidéo (B29-03).
    if (std::ranges::any_of(entries_, [&](const Entry& e) { return same(e) && !e.cancelled; })) return;
    std::erase_if(entries_, [&](const Entry& e) { return same(e) && e.file == entry.file; });
    // Les annulations ne s'accumulent pas sans fin sur une longue séance : les plus anciennes partent.
    constexpr std::ptrdiff_t kMaxCancelled = 32;
    while (std::ranges::count_if(entries_, [](const Entry& e) { return e.cancelled; }) >= kMaxCancelled)
      entries_.erase(std::ranges::find_if(entries_, [](const Entry& e) { return e.cancelled; }));
  } else {
    // Un refus réel recouvre toute annulation de la même signature.
    std::erase_if(entries_, same);
  }
  entries_.push_back(std::move(entry));
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
    // « Annuler et regarder » vaut pour toute la préparation de cette vidéo (B29-02) : SPAN annulé, la
    // taille de travail change et l'interpolation demanderait une construction de plus, avec un second écran
    // de préparation et un second bouton. Elle part avec lui.
    const auto cancelled = std::ranges::find(plan.notes, StageChange::Cancelled, &StageNote::change);
    if (cancelled != plan.notes.end() && stage_on(stage, plan.settings)) {
      if (!allow_drop) return std::nullopt;
      drop(stage, plan.settings);
      plan.notes.push_back({.stage = stage, .change = StageChange::Cancelled, .refusal = cancelled->refusal});
      continue;
    }
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
  }
  return {};
}

}  // namespace

i18n::Translated stage_notice(const std::vector<StageNote>& notes) {
  i18n::Translated text;
  for (const StageNote& note : notes)
    text = text.empty() ? note_text(note) : text + i18n::Translated::data(" · ") + note_text(note);
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
