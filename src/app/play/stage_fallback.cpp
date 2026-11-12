#include "app/play/stage_fallback.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <utility>

#include "app/play/chain_shape.h"
#include "app/play/links.h"
#include "app/play/options.h"
#include "app/play/player_engines.h"
#include "app/play/precision.h"
#include "app/play/pumped_task.h"
#include "app/play/visible_error.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

bool watch_now(const Refusal& refusal) {
  // La fenêtre fermée pendant l'attente (run_pumped) est aussi un Cancelled, mais pas ce choix de
  // l'utilisateur.
  return refusal.key == kWatchNowKey ||
         (refusal.code == RefusalCode::Cancelled && refusal.key != kWindowClosedKey);
}

std::string stage_signature(ChainStage stage, const PlaySettings& s, int source_width, int source_height,
                            std::string_view library, const StageRates& rates) {
  // La taille de travail et la source fixent la forme des moteurs ; le modèle, le cran et le module, le
  // graphe et sa construction.
  const std::string common = std::format("{}|{}x{}|{}|{}|{}", library, source_width, source_height,
                                         s.work_height, name(s.downscale), s.restore_lines);
  if (stage == ChainStage::Upscale)
    return std::format("span|{}|{}|{}", common, s.model, name(s.span_precision));
  // L'interpolation dépend aussi de SPAN : l'ordre du warp, que le facteur le plus haut et la sortie fixent,
  // calculé comme la chaîne le calcule (order_factor, C09-03). Pas le facteur brut, que l'Auto récrit selon
  // l'écran : un refus retenu à ×5 vaut à ×4 (même ordre), et ×5 et ×2,5 ne se confondent plus (B09-05).
  const int output_height = chain_work_size(s, source_width, source_height).height * kSpanFactor;
  const bool after = s.upscale && warp_after_span(s.span_order, output_height,
                                                  order_factor(s, rates.display_hz, rates.source_fps));
  return std::format("motion|{}|{}|{}|{}|{}|{}|{}|{}|{}", common, name(s.motion_engine), s.rife_model,
                     name(s.motion_precision), s.rife_fused, s.drba_fused, s.upscale, name(s.span_order),
                     after ? "warp-after" : "warp-before");
}

const StageMemory::Entry* StageMemory::find(ChainStage stage, std::string_view signature,
                                            const std::filesystem::path& file) const {
  const auto found = std::ranges::find_if(entries_, [&](const Entry& e) {
    return e.stage == stage && e.signature == signature && (!e.for_file_only() || e.file == file);
  });
  return found == entries_.end() ? nullptr : &*found;
}

void StageMemory::remember(Entry entry) {
  const auto same = [&](const Entry& e) { return e.stage == entry.stage && e.signature == entry.signature; };
  if (entry.for_file_only()) {
    // Une annulation (ou une mémoire pleine passagère, C31-09) ne vaut que pour sa vidéo : elle ne remplace
    // ni le refus réel de la même signature, qui vaut pour toutes les vidéos, ni l'entrée d'une autre vidéo
    // (B29-03).
    if (std::ranges::any_of(entries_, [&](const Entry& e) { return same(e) && !e.for_file_only(); })) return;
    std::erase_if(entries_, [&](const Entry& e) { return same(e) && e.file == entry.file; });
    // Elles ne s'accumulent pas sans fin sur une longue séance : les plus anciennes partent.
    constexpr std::ptrdiff_t kMaxCancelled = 32;
    const auto per_file = [](const Entry& e) { return e.for_file_only(); };
    while (std::ranges::count_if(entries_, per_file) >= kMaxCancelled)
      entries_.erase(std::ranges::find_if(entries_, per_file));
  } else {
    // Un refus réel recouvre toute annulation de la même signature.
    std::erase_if(entries_, same);
  }
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

bool display_retreats(const Refusal& refusal) {
  return refusal.key.empty() &&
         (refusal.code == RefusalCode::GpuMemoryExhausted || refusal.code == RefusalCode::VendorCallFailed ||
          refusal.code == RefusalCode::VulkanCallFailed);
}

std::optional<ChainStage> retreat_stage(StageMemory* memory, const Refusal& refusal,
                                        const PlaySettings& played, bool motion, bool upscale,
                                        int source_width, int source_height, std::string_view library,
                                        const std::filesystem::path& file, const StageRates& rates) {
  if (memory == nullptr || watch_now(refusal) || refusal.code == RefusalCode::DeviceLost ||
      refusal.code == RefusalCode::SourceResized || refusal.key == kDroppedVideoKey)
    return std::nullopt;
  const bool motion_on = motion && played.rife_factor >= 2;
  const bool upscale_on = upscale && played.upscale;
  // Seul l'étage que le refus nomme part (C09-01) ; l'anneau ou l'affichage (C31-05), l'interpolation
  // d'abord.
  const std::string_view blamed = blamed_stage_key(refusal.key);
  std::optional<ChainStage> stage;
  if (blamed == kMotionRefusalKey && motion_on) {
    stage = ChainStage::Motion;
  } else if (blamed == kSpanRefusalKey && upscale_on) {
    stage = ChainStage::Upscale;
  } else if (blamed == kRingRefusalKey || display_retreats(refusal)) {
    if (motion_on)
      stage = ChainStage::Motion;
    else if (upscale_on)
      stage = ChainStage::Upscale;
  }
  if (!stage) return std::nullopt;
  memory->remember({.stage = *stage,
                    .signature = stage_signature(*stage, played, source_width, source_height, library, rates),
                    .refusal = refusal,
                    .file = file});
  log_channel("app")->warn("{} refusé après la préparation ({}) : la chaîne se rouvre sans lui",
                           *stage == ChainStage::Upscale ? "SPAN" : "interpolation", refusal.detail);
  return stage;
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
                                     const std::filesystem::path& file, bool allow_drop,
                                     const StageRates& rates) {
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
          stage, stage_signature(stage, plan.settings, source_width, source_height, library, rates), file);
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
