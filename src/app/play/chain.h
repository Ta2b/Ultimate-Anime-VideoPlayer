// La chaîne ouverte par « ajp play » et ce que les actions de l'utilisateur y changent : touches,
// menu des paramètres en direct, pistes, profils, grade HDR et vrai noir OLED, soins OLED. Le
// thread main seul y touche (player.cpp ouvre la chaîne et tient la boucle).
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "app/play/display_edid.h"
#include "app/play/input.h"
#include "app/play/live_settings.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"
#include "app/play/profiles.h"
#include "app/play/session.h"
#include "app/play/settings.h"
#include "app/play/ui.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "media/demuxer.h"
#include "render/grade_params.h"

namespace ajp::app::play {

// Les profils de la lecture : user.toml, le profil actif, la source pour choisir leur règle.
struct ProfileState {
  std::filesystem::path user_path;
  Profiles data;
  int active = 1;
  core::SourceInfo source;
  std::string default_model;  // le modèle SPAN par défaut, pour un réglage « modèle par défaut »
};

// Le mode HDR que le profil demande, quand un repli de l'environnement (sortie SDR, échec de RTX
// HDR) en fait tourner un autre : le repli n'est jamais mémorisé à la place de la préférence.
struct HdrIntent {
  bool true_hdr = false;
  bool hdr_native = false;
};

// Ce qui vit le temps d'une ouverture de la chaîne, et ce que les actions y changent.
struct Chain {
  Session& session;
  PlaceboScreen& screen;
  PlayerUi& ui;
  Pipeline& pipeline;
  LiveSettings& live;
  PlaySettings& running;  // réglages en cours, gardés dans play.toml
  ProfileState& profiles;
  bool rife_loaded = false;
  bool true_hdr_open = false;
  int auto_peak = 0;                               // pic automatique de RTX HDR
  int peak_nits = 0;                               // pic donné à RTX HDR, 0 sans lui
  bool dirty = false;                              // play.toml à réécrire
  std::optional<PlaySettings> reopen;              // demandé par le menu des paramètres
  std::optional<EngineChange> engines;             // onglet Moteurs : à faire avant de rouvrir
  std::optional<ExportJob> export_job;             // onglet Export : à faire avant de rouvrir
  std::optional<std::filesystem::path> next_file;  // glissé sur la fenêtre, ou l'épisode voisin
  std::filesystem::path file;                      // la vidéo en cours
  const std::optional<DisplayFacts>& display;
  // Vrai noir OLED : la rampe passée par TrueHDR (codes PQ, de gpu), les seuils en cours, et le
  // recalage demandé (au plus 5 fois par seconde).
  std::vector<float> truehdr_ramp;
  std::optional<render::BlackThresholds> black;
  bool black_dirty = true;
  std::int64_t black_at_us = 0;
  // L'en-tête « chaîne » de l'OSD pour ces réglages, recalculé quand le facteur, la réduction ou le
  // moteur HDR changent à chaud ; vide : l'en-tête reste celui de l'ouverture.
  std::function<std::string(const PlaySettings&)> header_text;
  std::optional<HdrIntent> hdr_wanted;  // présent seulement pendant un repli HDR
  bool source_hdr = false;              // la vidéo est PQ ou HLG : le grade ne la retouche pas comme du SDR
};

// Les réglages avec, pendant un repli HDR, le mode que le profil demande à la place du mode de
// repli : ce que remember_chain et play.toml retiennent.
PlaySettings with_intended_hdr(const Chain& to, PlaySettings settings);

// Couleur de la sortie de RTX HDR : PQ BT.2020 au pic donné à TrueHDR.
ColorInfo pq_color(int peak_nits);
// Les paramètres de TrueHDR tirés des réglages ; le pic de l'écran si aucun n'est fixé.
vendor_hdr::TrueHdrParams true_hdr_params(const PlaySettings& s, int auto_peak);
// RTX HDR : le SDK Maxine importé une fois dans le magasin du lecteur (environ 135 Mo), revérifié
// à chaque lancement, puis l'effet ouvert pour la sortie de SPAN ; l'écran reçoit alors du PQ
// BT.2020 au pic donné à TrueHDR, que rend la fonction.
std::expected<int, Refusal> open_rtx_hdr(const std::filesystem::path& sdk_source, const PlayerPaths& paths,
                                         const PlaySettings& settings, int auto_peak, PlaceboScreen& screen,
                                         gpu::KernelOps& kernels, int width, int height,
                                         const ColorInfo& source);

// Le grade de la chaîne pour chaque sortie : en PQ, route TrueHDR si RTX HDR est ouvert, native si le
// HDR natif est choisi, aucun sinon ; en SDR, sur toutes les routes. Rien là où il est neutre.
render::GradeChoice grade_of(const PlaySettings& s, bool true_hdr_open, int peak_nits, int auto_peak,
                             const std::optional<DisplayFacts>& display = {},
                             const std::optional<render::BlackThresholds>& black = {},
                             bool source_hdr = false);
// La sortie de l'écran, pour l'OSD et le journal.
std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, int peak_nits);
double fps_of(const media::Rational& rate);

ProfilesView view_of(const ProfileState& state);
// La chaîne en cours devient celle du profil actif, dans user.toml.
void remember_chain(ProfileState& state, const PlaySettings& s);
// La chaîne (modèle, hauteur, RIFE, HDR) diffère : le profil actif la retient.
bool chain_differs(const PlaySettings& a, const PlaySettings& b);

// Soins OLED, grade avec les seuils du vrai noir, et recalage de ces seuils sur la rampe mesurée.
void apply_oled_care(Chain& to, const PlaySettings& s);
void set_grade(Chain& to, const PlaySettings& s);
void refresh_black(Chain& to);
// Une action de l'utilisateur (touche, barre, menu) sur la chaîne.
void apply(const Input& input, Chain& to, std::int64_t now_us);

// --stats : une ligne par présentation ; le bilan de fin dans le journal.
void write_record(std::ofstream& stats, const PresentRecord& record);
void log_summary(const Session& session);

}  // namespace ajp::app::play
