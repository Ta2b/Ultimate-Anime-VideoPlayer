#include "app/play/settings_panel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <imgui.h>

#include "app/play/auto_config.h"
#include "app/play/first_run.h"
#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "app/play/settings_bench_results.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/settings_color_tabs.h"
#include "app/play/settings_engines.h"
#include "app/play/settings_export.h"
#include "app/play/settings_languages.h"
#include "app/play/settings_pages.h"
#include "app/play/settings_sound.h"
#include "app/play/settings_upscale_level.h"
#include "app/play/text_fit.h"
#include "core/engines.h"

namespace ajp::app::play {
namespace {

using panel::kWarn;

// Les onglets, dans l'ordre de l'écran ; les catégories se déduisent du changement de `category` (ADR 0012,
// à la manière de Dolphin) : Général, Vidéo, Audio, Langues, Export.
constexpr std::array<SettingsPage, 9> kPages{{
    {Page::Application, "General",
     "Application"},  // langue, épisode suivant, mises à jour, lecteur par défaut
    {Page::Statistics, "General", "Statistics"},
    {Page::VideoGeneral, "Video", "General"},  // préréglage, moteur, sortie HDR, couleurs, FPS
    {Page::Upscaling, "Video", "Upscaling"},
    {Page::Interpolation, "Video", "Interpolation"},
    {Page::Hdr, "Video", "HDR"},
    {Page::Audio, "Audio", "Audio"},
    {Page::Languages, "Languages", "Languages"},  // langues préférées, sous-titres
    {Page::Export, "Export", "Export"},
}};
static_assert(kPages[2].id == Page::VideoGeneral);  // l'onglet ouvert d'abord (settings_panel.h : page_)

// La catégorie d'un onglet, comptée de 0 dans l'ordre de la colonne.
int category_of(int page) {
  int category = 0;
  for (int i = 1; i <= page; ++i) {
    if (std::string_view(kPages[static_cast<std::size_t>(i)].category) !=
        kPages[static_cast<std::size_t>(i - 1)].category)
      ++category;
  }
  return category;
}

// Le premier onglet d'une catégorie.
int first_page(int category) {
  for (int i = 0; i < static_cast<int>(kPages.size()); ++i) {
    if (category_of(i) == category) return i;
  }
  return 0;
}

int category_count() {
  return category_of(static_cast<int>(kPages.size()) - 1) + 1;
}

}  // namespace

std::span<const SettingsPage> settings_pages() {
  return kPages;
}

SettingsPanel::SettingsPanel() : export_(std::make_unique<ExportDraft>()) {}
bool SettingsPanel::busy() const {
  // alpha_ : le panneau revient après un curseur relâché, en quelques images, même en pause.
  return open_ && (prepare_.pending != nullptr || span_job_pending_ || (machine_ && machine_->running()) ||
                   alpha_ < 0.99F);
}

SettingsPanel::~SettingsPanel() {
  if (prepare_.stop) prepare_.stop->request_stop();  // la tâche dépend du lecteur qui ouvre ce panneau
}

// La colonne des catégories, comme celle de Dolphin : une ligne par catégorie, la choisie sur un fond
// d'accent. Elle ramène à l'onglet vu en dernier dans la catégorie.
void SettingsPanel::draw_categories(float scale) {
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const float row_height = ImGui::GetFontSize() + 16.0F * scale;
  const int current = category_of(page_);
  const ImU32 accent = ImGui::ColorConvertFloat4ToU32(ImVec4(0.26F, 0.47F, 0.82F, 1.0F));
  const ImU32 hover = ImGui::ColorConvertFloat4ToU32(ImVec4(1.0F, 1.0F, 1.0F, 0.07F));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.0F, 2.0F * scale});
  for (int category = 0; category < category_count(); ++category) {
    const int first = first_page(category);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float row_width = ImGui::GetContentRegionAvail().x;
    ImGui::PushID(category);
    if (ImGui::InvisibleButton("##categorie", {row_width, row_height}) && category != current) {
      const int remembered = category_page_[static_cast<std::size_t>(category)];
      page_ = remembered >= 0 ? remembered : first;
      select_tab_ = page_;
    }
    ImGui::PopID();
    if (category == current) {
      draw->AddRectFilled(at, {at.x + row_width, at.y + row_height}, accent, 4.0F * scale);
    } else if (ImGui::IsItemHovered()) {
      draw->AddRectFilled(at, {at.x + row_width, at.y + row_height}, hover, 4.0F * scale);
    }
    draw->AddText({at.x + 12.0F * scale, at.y + (row_height - ImGui::GetFontSize()) * 0.5F},
                  ImGui::GetColorU32(ImGuiCol_Text),
                  i18n::tr(kPages[static_cast<std::size_t>(first)].category));
  }
  ImGui::PopStyleVar();
}

// Les onglets de la catégorie en cours ; rien pour une catégorie d'un seul onglet.
void SettingsPanel::draw_tabs() {
  const int category = category_of(page_);
  const int first = first_page(category);
  int count = 0;
  for (int i = first; i < static_cast<int>(kPages.size()) && category_of(i) == category; ++i) ++count;
  if (count < 2) return;
  if (!ImGui::BeginTabBar("##onglets")) return;
  for (int i = first; i < first + count; ++i) {
    const ImGuiTabItemFlags flags = select_tab_ == i ? ImGuiTabItemFlags_SetSelected : 0;
    if (ImGui::BeginTabItem(i18n::tr(kPages[static_cast<std::size_t>(i)].name), nullptr, flags)) {
      if (select_tab_ < 0 || select_tab_ == i) page_ = i;
      ImGui::EndTabItem();
    }
  }
  select_tab_ = -1;
  ImGui::EndTabBar();
}

void SettingsPanel::draw_page(const PlaySettings& running, const SettingsChoices& choices,
                              std::vector<Input>& actions, Result& result, bool& delay_held,
                              bool& sub_delay_held) {
  switch (kPages[static_cast<std::size_t>(page_)].id) {
    case Page::Application: {
      application_page(draft_, choices, updates_, system_, actions);
      panel::advanced_header();
      if (reset_section(draft_)) result.reopen = draft_;
      break;
    }
    case Page::Statistics: {
      statistics_page(draft_);
      break;
    }
    case Page::VideoGeneral: {
      video_general_page(draft_, choices, actions, engine_draft_);
      panel::advanced_header();
      if (panel::begin_boxes()) {
        panel::next_box();
        display_section(draft_, choices);
        panel::next_box();
        engines_advanced(choices, draft_.engines_dir, engines_folder_, draft_.fast_precision);
        panel::end_boxes();
      }
      // Une ligne de la calibration appliquée l'est tout de suite, sans « Appliquer » : c'est ce qu'on a
      // demandé.
      if (!machine_) machine_ = std::make_unique<MachineNote>();
      if (bench_results_section(draft_, choices, *machine_, actions) && draft_ != running)
        result.reopen = draft_;
      break;
    }
    case Page::Upscaling: {
      image_tab(draft_, choices);
      panel::advanced_header();
      // La réduction et la préparation du FP8 côte à côte ; la seconde ne sert qu'au FP8 choisi.
      if (panel::begin_boxes()) {
        panel::next_box();
        upscale_advanced(draft_, choices);
        panel::next_box();
        ImGui::BeginDisabled(!fp8_in_use(draft_, choices));
        span_prepare_section(choices.span_prepare, choices.span_job ? *choices.span_job : prepare_);
        panel::end_group();
        ImGui::EndDisabled();
        panel::end_boxes();
      }
      break;
    }
    case Page::Interpolation: {
      fluidity_tab(draft_, choices);
      panel::advanced_header();
      motion_advanced(draft_, choices);
      break;
    }
    case Page::Hdr: {
      hdr_page(draft_, choices);
      break;
    }
    case Page::Audio: {
      delay_held = sound_tab(draft_, choices);
      break;
    }
    case Page::Languages: {
      sub_delay_held = languages_page(draft_);
      break;
    }
    case Page::Export: {
      export_tab(*export_, running, choices, actions);
      break;
    }
  }
}

void SettingsPanel::measure_pending(const PlaySettings& running, const SettingsChoices& choices) {
  pending_text_.clear();
  pending_cost_ = ChangeCost::None;
  const Change pending = change_between(live_part(running, draft_, choices.rife_loaded, choices.chain()),
                                        draft_, choices.rife_loaded, choices.chain());
  std::string reason;
  if (pending.cost == ChangeCost::Reopen || pending.cost == ChangeCost::Restart) {
    pending_cost_ = pending.cost;
    reason = pending.reason;
  }
  // Un autre moteur choisi dans Vidéo > Général : la vidéo se rouvre avec lui, au même endroit.
  if (engine_draft_ && engine_draft_ != choices.engine_in_use) {
    const std::string engine =
        i18n::tr_format("the engine changes to {} (built the first time for each model and size)",
                        engine_label(*engine_draft_));
    reason =
        reason.empty() ? engine + i18n::tr(" ; playback resumes at the same spot") : engine + " ; " + reason;
    if (pending_cost_ == ChangeCost::None) pending_cost_ = ChangeCost::Reopen;
  }
  if (pending_cost_ != ChangeCost::None)
    pending_text_ = i18n::tr_format("To apply: {}. Closing the panel discards these changes.", reason);
}

void SettingsPanel::draw_footer(const PlaySettings& running, const SettingsChoices& choices,
                                const PlaySettings& before, bool delay_held, bool sub_delay_held,
                                std::vector<Input>& actions, Result& result) {
  // Un réglage de la chaîne fait à la main : le préréglage devient Personnalisé (les valeurs restent). Avant
  // l'historique : un réglage et son passage en Personnalisé ne font qu'une entrée d'annulation (C23-05).
  if (draft_.preset != Preset::Custom && draft_.preset == before.preset &&
      preset_chain_differs(before, draft_, choices.calibrated.has_value()))
    draft_.preset = Preset::Custom;
  // Annuler et rétablir : un état n'est retenu que lorsque plus aucun curseur n'est tenu (un glissement
  // compte pour un seul changement).
  if (!ImGui::IsAnyItemActive()) history_.settle(draft_);
  ImGui::BeginDisabled(!history_.can_undo());
  if (ImGui::Button(i18n::tr("Undo"))) draft_ = history_.undo(draft_);
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!history_.can_redo());
  if (ImGui::Button(i18n::tr("Redo"))) draft_ = history_.redo(draft_);
  ImGui::EndDisabled();
  ImGui::PopItemWidth();

  // La part directe s'applique tout de suite ; le reste attend « Appliquer ».
  PlaySettings live = live_part(running, draft_, choices.rife_loaded, choices.chain());
  // Chaque délai recale le son (il reprend les blocs en file) : un glissement hacherait le son à
  // chaque image, le délai s'applique donc au relâchement (saisie au clavier comprise).
  if (delay_held) live.audio_delay_ms = running.audio_delay_ms;
  if (sub_delay_held) live.subtitle_delay_ms = running.subtitle_delay_ms;
  if (live != running) result.live = live;
  // Le bandeau est celui dont la hauteur a été réservée au début de l'image (pending_text_, C23-02). Un même
  // bandeau pour tout ce qui attend : recharger la vidéo ou relancer le lecteur, que le bouton dit.
  if (pending_text_.empty()) return;
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
  panel::wrapped(pending_text_);
  ImGui::PopStyleColor();
  const bool restart = pending_cost_ == ChangeCost::Restart;
  if (ImGui::Button(restart ? i18n::tr("Apply (restarts the player)")
                            : i18n::tr("Apply (reloads the video)"))) {
    if (engine_draft_ && engine_draft_ != choices.engine_in_use)
      actions.emplace_back(UseEngine{.id = *engine_draft_});
    engine_draft_.reset();
    if (restart) {
      result.restart = draft_;
    } else if (draft_ != live) {
      result.reopen = draft_;
    }
  }
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Discard changes"))) {
    draft_ = live;
    engine_draft_.reset();
  }
}

void SettingsPanel::scroll_categories() {
  const ImGuiIO& io = ImGui::GetIO();
  // La molette sur la colonne passe d'une catégorie à la suivante ou à la précédente. Une catégorie par
  // geste, pas par cran : une molette libre ou un pavé tactile envoie des dizaines de crans (B23-10).
  if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0F && !wheel_locked_) {
    if (wheel_ * io.MouseWheel < 0.0F) wheel_ = 0.0F;  // le sens a changé : un nouveau geste
    wheel_ += io.MouseWheel;
    if (std::abs(wheel_) >= 1.0F) {
      const int step = wheel_ < 0.0F ? 1 : -1;
      const int category = std::clamp(category_of(page_) + step, 0, category_count() - 1);
      const int remembered = category_page_[static_cast<std::size_t>(category)];
      page_ = remembered >= 0 ? remembered : first_page(category);
      select_tab_ = page_;
      wheel_ = 0.0F;
      wheel_idle_s_ = 0.0F;
      wheel_locked_ = true;
    }
  }
  if (wheel_locked_) {
    // Le geste dure tant que la molette tourne ; un court calme le termine.
    wheel_idle_s_ += io.DeltaTime;
    if (io.MouseWheel != 0.0F) wheel_idle_s_ = 0.0F;
    if (wheel_idle_s_ > 0.35F) wheel_locked_ = false;
  }
}

SettingsPanel::Result SettingsPanel::draw(const PlaySettings& running, const SettingsChoices& choices,
                                          float scale, std::vector<Input>& actions) {
  Result result;
  if (!open_) return result;
  // Réglages changés ailleurs (Tab, volume) : le panneau les reprend, sans perdre ce qui attend.
  if (running != seen_) {
    const ExternalChange external = external_changes(seen_, running, draft_);
    draft_ = follow_running(draft_, seen_, running, choices.rife_loaded);
    history_.adopt(external, running);  // le volume des touches ne se rejoue pas avec « Annuler » (B23-01)
    seen_ = running;
  }
  // La préparation lancée par la boucle du lecteur, pas par ce panneau : sa barre avance aussi en pause
  // (C23-10).
  span_job_pending_ = choices.span_job && choices.span_job->pending != nullptr;
  const ImGuiIO& io = ImGui::GetIO();
  // Un curseur en direct tenu à l'image précédente : le panneau s'efface pour laisser voir l'image, puis
  // revient au relâchement.
  const bool dim = std::exchange(panel::state().dim, false);
  const float target = dim ? 0.18F : 1.0F;
  alpha_ += (target - alpha_) * std::min(1.0F, io.DeltaTime * 14.0F);
  if (std::abs(target - alpha_) < 0.01F) alpha_ = target;
  // Le panneau se déplace par son titre et se redimensionne par tous ses bords, dans la fenêtre moins une
  // marge. Sa place, en parts de la fenêtre (panel_size.h), revient à chaque ouverture et suit la fenêtre
  // quand elle change de taille. La première fois : à droite, sous le titre, au-dessus de la barre de
  // lecture.
  const float pad = 18.0F * scale;
  const PanelArea area{std::max(320.0F * scale, io.DisplaySize.x - 2.0F * pad),
                       std::max(240.0F * scale, io.DisplaySize.y - 2.0F * pad)};
  const float first_width = std::min(1100.0F * scale, area.width);
  const float first_top = 40.0F * scale;
  const PanelRect first{.x = area.width - first_width,
                        .y = first_top,
                        .width = first_width,
                        .height = std::max(240.0F * scale, area.height - first_top - 102.0F * scale)};
  const PanelRect wanted = size_.target(area, first);
  const ImGuiCond imposed = size_.area_changed() ? ImGuiCond_Always : ImGuiCond_Appearing;
  ImGui::SetNextWindowPos({pad + wanted.x, pad + wanted.y}, imposed);
  ImGui::SetNextWindowSize({wanted.width, wanted.height}, imposed);
  ImGui::SetNextWindowSizeConstraints(
      {std::min(520.0F * scale, area.width), std::min(260.0F * scale, area.height)},
      {area.width, area.height});
  bool keep = true;
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha_);
  if (ImGui::Begin((std::string(i18n::tr("Settings")) + "###settings").c_str(), &keep, kFlags)) {
    span_ = {.left = ImGui::GetWindowPos().x, .right = ImGui::GetWindowPos().x + ImGui::GetWindowWidth()};
    size_.observe({.x = ImGui::GetWindowPos().x - pad,
                   .y = ImGui::GetWindowPos().y - pad,
                   .width = ImGui::GetWindowWidth(),
                   .height = ImGui::GetWindowHeight()});
    ImGui::PushItemWidth(-ImGui::GetWindowWidth() * 0.42F);
    bool delay_held = false;
    bool sub_delay_held = false;
    // L'état d'avant les widgets de l'image : copié seulement quand il a changé, pas à chaque image (B23-09).
    if (before_ != draft_) before_ = draft_;
    const PlaySettings& before = before_;
    // Une calibration refaite ici vaut tout de suite pour « Auto », sans attendre la vidéo suivante (B07-13).
    std::optional<SettingsChoices> recalibrated;
    if (machine_ && machine_->fresh && *machine_->fresh != choices.calibrated) {
      recalibrated.emplace(choices);
      recalibrated->calibrated = *machine_->fresh;
    }
    const SettingsChoices& shown = recalibrated ? *recalibrated : choices;
    // Le bandeau « Appliquer » de cette image, mesuré avant de tracer les colonnes : sa hauteur suit son
    // texte, quelle que soit la langue ou le nombre de raisons (C23-02).
    const ImGuiStyle& style = ImGui::GetStyle();
    measure_pending(running, shown);
    float banner = 0.0F;
    if (!pending_text_.empty()) {
      const std::string measured = french_spaces(pending_text_);
      const float text_height =
          ImGui::CalcTextSize(measured.c_str(), nullptr, false, ImGui::GetContentRegionAvail().x).y;
      // Espace, filet, texte, rangée de boutons : l'espacement entre chacun.
      banner = text_height + ImGui::GetFrameHeight() + 4.0F * style.ItemSpacing.y + 2.0F * scale;
    }
    // Sous les deux colonnes : annuler et rétablir (et le bandeau « Appliquer » quand il est là).
    const float footer = banner + ImGui::GetFrameHeightWithSpacing() + 2.0F * style.ItemSpacing.y;
    const float column = std::min(170.0F * scale, ImGui::GetWindowWidth() * 0.26F);
    // Barre de défilement fine, collée au bord extérieur des deux zones.
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0F * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 4.0F * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0F * scale, 6.0F * scale));
    ImGui::BeginChild("##categories", ImVec2(column, -footer), ImGuiChildFlags_Borders);
    ImGui::PopStyleVar();
    scroll_categories();
    draw_categories(scale);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginGroup();
    draw_tabs();
    category_page_[static_cast<std::size_t>(category_of(page_))] = page_;
    // Une page trop étroite garde sa disposition et gagne une barre de défilement horizontale, comme chez
    // Dolphin, plutôt que de tasser ses colonnes.
    const float page_min = 680.0F * scale;
    if (ImGui::GetContentRegionAvail().x < page_min) ImGui::SetNextWindowContentSize(ImVec2(page_min, 0.0F));
    ImGui::BeginChild("##page", ImVec2(0, -footer), ImGuiChildFlags_None,
                      ImGuiWindowFlags_HorizontalScrollbar);
    panel::begin_page();
    ImGui::PushID(page_);
    draw_page(running, shown, actions, result, delay_held, sub_delay_held);
    panel::end_page();
    ImGui::PopID();
    ImGui::EndChild();
    ImGui::EndGroup();
    ImGui::PopStyleVar(2);
    draw_footer(running, shown, before, delay_held, sub_delay_held, actions, result);
  }
  ImGui::End();
  ImGui::PopStyleVar();
  if (!keep) open_ = false;
  return result;
}

}  // namespace ajp::app::play
