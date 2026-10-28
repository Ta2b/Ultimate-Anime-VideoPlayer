// Les moteurs d'une lecture : construits (ou lus du cache) puis chargés, la préparation à l'écran,
// et ce qui survit à une réouverture de la chaîne (Player). Interne à player.cpp ; GMFSS y a ses
// deux moteurs, aux tailles fixées pour la lecture (gmfss_model.h).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "app/play/backend_library.h"
#include "app/play/build_progress.h"
#include "app/play/chain.h"
#include "app/play/display_edid.h"
#include "app/play/drba_model.h"
#include "app/play/engine_install.h"
#include "app/play/gmfss_model.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"
#include "app/play/settings.h"
#include "app/play/ui.h"
#include "core/clock.h"
#include "core/settings.h"
#include "gpu/backend.h"
#include "media/demuxer.h"

namespace ajp::app::play {

// Un moteur à construire, tel que la préparation le montre.
struct EngineLabel {
  std::string name;     // « SPAN », « RIFE »
  std::string model;    // nom du modèle
  std::string backend;  // « tensorrt », « vulkan »
  int width = 0;        // entrée du moteur
  int height = 0;
  double expected_s = kDefaultBuildSeconds;
};

// Instantané demandé (--snapshot) : pris pendant la première construction de moteur qui dure jusqu'à
// at_us, sinon pendant la lecture.
struct SnapshotPlan {
  std::optional<std::filesystem::path> path;
  std::int64_t at_us = 0;
  int frames = 1;
  // Demande l'instantané à l'écran si since_us remonte à at_us ; true s'il vient d'être demandé
  // (une présentation doit suivre, même en pause).
  bool maybe(PlaceboScreen& screen, std::int64_t since_us) {
    if (!path || host_now_us() - since_us < at_us) return false;
    screen.snapshot_next(*path, frames);
    path.reset();
    return true;
  }
};

// Ce qui survit à une réouverture de la chaîne : la fenêtre, l'interface, la bibliothèque du
// backend (que l'onglet Moteurs peut changer), les touches et l'instantané d'essai, les pistes
// choisies.
struct Player {
  const PlayOptions& options;
  const PlayerPaths& paths;
  const EngineCatalog& catalog;  // les moteurs téléchargeables (engines.lock)
  PlaceboScreen& screen;
  PlayerUi& ui;
  std::shared_ptr<BackendLibrary> library;
  media::TrackInfo video;
  std::filesystem::path settings_path;
  PlaySettings from_file;  // play.toml au lancement
  PlaySettings started;    // réglages du début de la lecture (options comprises)
  SnapshotPlan snapshot;
  bool details = false;  // Tab pendant une préparation
  std::size_t next_key = 0;
  std::int64_t began_us = 0;
  std::optional<int> audio_track;
  std::optional<std::optional<int>> subtitle_track;
  ProfileState profiles;
  core::Settings engines;               // settings.toml : les moteurs choisis à l'écran de bienvenue
  std::optional<DisplayFacts> display;  // l'écran selon son EDID (OLED ou non)
  std::filesystem::path resume_path;    // mémoire de lecture (resume.h) ; vide : aucune
  bool resumed = false;                 // la lecture reprend là où elle s'était arrêtée : à annoncer
  std::string notice;                   // à annoncer à la prochaine ouverture (moteur changé)
  bool quit = false;                    // la dernière ouverture s'est fermée par q ou la fenêtre
  // Réglages d'une réouverture demandée : écrits dans play.toml une fois la chaîne ouverte, pas avant
  // (une chaîne qui ne s'ouvre pas ne doit pas devenir celle du prochain lancement).
  std::optional<PlaySettings> unsaved;
};

// Les messages des backends, sur le canal « backend » du journal.
gpu::BackendLog backend_log();

// Une image de la superposition avant une étape bloquante (chargement d'un moteur, RTX HDR) : la
// fenêtre dit ce qu'elle attend au lieu de rester noire.
void show_loading(PlaceboScreen& screen, PlayerUi& ui, std::string label);

// Construit (ou lit du cache) puis charge un moteur, la préparation à l'écran. Rien si
// l'utilisateur ferme la fenêtre. lane : la voie où le moteur tournera (ses activations y sont
// posées, gpu/lanes.h) ; le backend revient ensuite sur Upscale.
std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_engine(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, const std::filesystem::path& onnx,
    const gpu::Shape& shape, EngineLabel label, gpu::Lane lane);

// GMFSS Fortuna pour des images de travail work_width × work_height : ses deux graphes, trouvés dans
// le premier de dirs qui les a, fixés à la taille remplie de RIFE (dérivés dans le dossier de
// données), construits puis chargés. Rien si l'utilisateur ferme la fenêtre. Refus : FieldMissing
// (graphes absents, avec leurs noms et les dossiers essayés), et ceux de la dérivation et du backend.
struct GmfssEngines {
  std::unique_ptr<gpu::Engine> pair;
  std::unique_ptr<gpu::Engine> fusion;
  int align = 64;  // multiple des plans remplis des deux graphes (gmfss_model.h)
};
// DistilDRBA pour des images de travail work_width × work_height : son graphe, trouvé dans le
// premier de dirs qui l'a, fixé à la taille remplie de RIFE, construit puis chargé. Rien si
// l'utilisateur ferme la fenêtre. Refus : FieldMissing (graphe absent), et ceux de la dérivation
// et du backend. align : le multiple des plans remplis du graphe trouvé (64, ou 128 au demi-flot).
std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_distil_drba(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend,
    std::span<const std::filesystem::path> dirs, int work_width, int work_height, int& align);

// L'interpolation d'Apple pour des images de travail work_width × work_height, chargée comme RIFE
// image (plans v2, remplis au multiple de 64) : son modèle (apple_model.h), essayé à la
// construction par le runtime Core ML. Rien si l'utilisateur ferme la fenêtre. Refus
// FeatureUnavailable : un autre backend que Core ML ; ceux du modèle et du runtime (macOS avant
// 15.4, puce ou taille refusées).
std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_apple(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, int work_width, int work_height);

std::expected<std::optional<GmfssEngines>, Refusal> prepare_gmfss(Player& p, const gpu::BackendInit& init,
                                                                  gpu::Backend& backend,
                                                                  std::span<const std::filesystem::path> dirs,
                                                                  GmfssVariant variant, int work_width,
                                                                  int work_height, bool quality = false);

}  // namespace ajp::app::play
