// La chaîne ouverte par « ajp play » et ce que les actions de l'utilisateur y changent : touches,
// menu des paramètres en direct, pistes, profils, grade HDR et vrai noir OLED, soins OLED. Le
// thread main seul y touche (player.cpp ouvre la chaîne et tient la boucle).
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
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
  int auto_peak = 0;                   // pic automatique de RTX HDR
  int peak_nits = 0;                   // pic donné à RTX HDR, 0 sans lui
  bool dirty = false;                  // play.toml à réécrire
  std::optional<PlaySettings> reopen;  // demandé par le menu des paramètres
  const std::optional<DisplayFacts>& display;
  // Vrai noir OLED : la rampe passée par TrueHDR (codes PQ, de gpu), les seuils en cours, et le
  // recalage demandé (au plus 5 fois par seconde).
  std::vector<float> truehdr_ramp;
  std::optional<render::BlackThresholds> black;
  bool black_dirty = true;
  std::int64_t black_at_us = 0;
};

// RTX HDR : le SDK Maxine importé une fois dans le magasin du lecteur (environ 135 Mo), revérifié
// à chaque lancement, puis l'effet ouvert pour la sortie de SPAN ; l'écran reçoit alors du PQ
// BT.2020 au pic donné à TrueHDR, que rend la fonction.
std::expected<int, Refusal> open_rtx_hdr(const std::filesystem::path& sdk_source, const PlayerPaths& paths,
                                         const PlaySettings& settings, int auto_peak, PlaceboScreen& screen,
                                         gpu::KernelOps& kernels, int width, int height);

// Le grade HDR de la chaîne : route TrueHDR si RTX HDR est ouvert, native si le HDR natif est choisi,
// aucun en SDR ; rien si le grade est neutre.
std::optional<render::GradeParams> grade_of(const PlaySettings& s, bool true_hdr_open, int peak_nits,
                                            int auto_peak, const std::optional<DisplayFacts>& display = {},
                                            const std::optional<render::BlackThresholds>& black = {});
// La sortie de l'écran, pour l'OSD et le journal.
std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, int peak_nits);
double fps_of(const media::Rational& rate);

ProfilesView view_of(const ProfileState& state);
// La chaîne en cours devient celle du profil actif, dans user.toml.
void remember_chain(ProfileState& state, const PlaySettings& s);

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
