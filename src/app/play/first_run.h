// L'écran de bienvenue du premier lancement : le GPU vu par Vulkan, les moteurs que
// core::recommend_engines propose pour lui (Vulkan toujours, les autres au choix, avec badges et
// présélection) et le profil de démarrage. « Commencer » écrit settings.toml (moteurs choisis,
// first_run_done) et user.toml (start_profile) ; --first-run remontre l'écran, comme le bouton de
// l'onglet Moteurs pendant la lecture.
//
// SPAN et RIFE passent par TensorRT sur une carte NVIDIA qui l'a choisi, sinon par le backend
// Vulkan ; RTX HDR passe par Maxine, remplacé sans elle par le HDR natif. Ce qui change se dit au
// journal.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/settings.h"
#include "app/play/settings_panel.h"
#include "core/engines.h"
#include "core/gpu_device.h"
#include "core/settings.h"

namespace ajp::app::play {

// Le GPU de rendu, tel que l'écran l'a ouvert.
struct GpuFacts {
  std::string name;  // « NVIDIA GeForce RTX 4070 Ti »
  GpuVendor vendor = GpuVendor::Other;
  std::uint64_t vram_bytes = 0;
  std::uint64_t host_visible_vram_bytes = 0;  // tas VRAM visible de l'hôte : toute la VRAM avec Resizable BAR
  bool tensor_cores = false;  // NVIDIA avec matrices coopératives Vulkan : une RTX
  bool wmma = false;          // AMD RDNA3 ou RDNA4 avec matrices coopératives Vulkan
};

// La carte AMD a-t-elle des unités matricielles (WMMA : RDNA3, RDNA4) ? Par le nom du pilote :
// la puce chez RADV (« (RADV GFX1201) »), le produit ailleurs (« RX 9070 XT »).
bool amd_matrix_cores(std::string_view gpu_name);

// ROCm (donc MIGraphX) ne sert officiellement que Navi 21, RDNA3, RDNA4 et les Instinct : une
// RX 6500M (Navi 24) ou un iGPU Radeon ne doit pas se voir proposer son téléchargement.
bool rocm_listed(std::string_view gpu_name);
// MIGraphX de Windows ML sert les AMD RDNA 3 et après, cartes et APU (Radeon 740M à 780M, 840M à
// 890M, 8040S à 8060S) : la Radeon 860M y est, pas une RX 6000.
bool winml_listed(std::string_view gpu_name);
core::SystemInfo system_of(const GpuFacts& gpu, core::Os os);
// « NVIDIA GeForce RTX 4070 Ti · 12 Go · cœurs Tensor »
std::string gpu_line(const GpuFacts& gpu);

// Resizable BAR (ou Smart Access Memory) : « actif, 16 Go », « inactif, 256 Mo », ou rien quand la
// carte n'est pas une carte à mémoire dédiée ou que le pilote n'expose pas de tas visible.
std::string rebar_text(const GpuFacts& gpu);

// Profil de démarrage conseillé : 1 (1440p, RIFE ×3) pour une RTX ou une RDNA3/RDNA4 de 11 Go et
// plus, 7 (1080p, ×2)
// pour une RTX plus petite et pour les autres cartes de 4 Go et plus hors Intel, 9 (source) sinon.
int start_profile_for(const GpuFacts& gpu);

// Un moteur que ce lecteur sait télécharger (engines.lock pour cet OS).
struct EngineDownload {
  core::EngineId id = core::EngineId::TensorRt;
  bool installed = false;  // déjà extrait sous <données>/backends
  std::uint64_t bytes = 0;
  std::string license;  // à accepter avant le téléchargement, vide si aucune
  std::string missing;  // prérequis du système absent (missing_prerequisite), vide sinon
};

// Ce qui est déjà sur ce poste.
struct EnginesHere {
  // Le système dont l'écran propose les moteurs : celui de l'hôte, ou celui qu'un test vérifie.
  core::Os os = core::kHostOs;
  bool tensorrt = false;       // backend ajp_backend_tensorrt trouvé
  bool migraphx = false;       // ajp_backend_migraphx
  bool openvino = false;       // ajp_backend_openvino
  bool coreml = false;         // ajp_backend_coreml (Mac : le moteur Metal de l'écran)
  bool maxine = false;         // SDK Maxine importé, ou importable depuis --truehdr-sdk
  std::string maxine_status;   // « importé », « à importer depuis /usr/local/VideoFX »…
  std::string maxine_license;  // non vide : installable depuis NGC (maxine_install.h), sa licence
  std::vector<EngineDownload> downloads;
};

struct EngineRow {
  core::EngineOffer offer;
  bool usable = false;  // installé, ou installable par ce lecteur
  bool chosen = false;
  std::string status;     // « livré », « installé », « pas encore installable dans cette version »
  bool download = false;  // à télécharger si coché
  std::string license;    // licence à accepter (download seulement)
  bool license_accepted = false;
  bool needs_key = false;       // téléchargement par la clé d'API NGC de l'utilisateur (Maxine)
  std::array<char, 128> key{};  // saisie, jamais écrite sur disque
};

struct FirstRunView {
  std::string gpu;
  std::vector<EngineRow> engines;
  ProfilesView profiles;   // .start : le profil choisi
  bool done = false;       // « Commencer »
  bool revisit = false;    // rouvert depuis l'onglet Moteurs : la lecture reprend ensuite
  bool cancelled = false;  // « Annuler » (revisit seulement) : rien ne change
};

// Les lignes de recommend_engines pour ce GPU, cochées selon la présélection au premier lancement,
// selon le choix de current ensuite (Vulkan toujours) ; profiles.start garde le profil donné.
FirstRunView first_run_view(const GpuFacts& gpu, const EnginesHere& here, ProfilesView profiles,
                            const core::Settings& current = {});

// settings avec les moteurs choisis, ceux présents, les préférés qui en découlent, et
// first_run_done.
core::Settings settings_after(const FirstRunView& view, core::Settings settings);

// settings sans id (moteur qui n'a pas pu s'installer), ses préférés recalculés.
core::Settings without_engine(core::Settings settings, core::EngineId id);

// id fait partie des moteurs choisis à l'écran de bienvenue.
bool chosen(const core::Settings& settings, core::EngineId id);

// Les bibliothèques de backend à essayer, dans l'ordre : celles de TensorRT, MIGraphX et OpenVINO
// seulement si leur moteur a été choisi, Vulkan restant le recours.
std::vector<std::filesystem::path> backends_within(std::span<const std::filesystem::path> candidates,
                                                   const core::Settings& engines);

// La chaîne que les moteurs choisis permettent : sans TensorRT ni Vulkan, ni SPAN ni RIFE ; sans
// Maxine, le HDR natif au lieu de RTX HDR. notes dit ce qui a changé.
PlaySettings within_engines(PlaySettings settings, const core::Settings& engines,
                            std::vector<std::string>& notes);
// « Vulkan, TensorRT, NVIDIA Maxine », pour l'onglet Moteurs.
std::string chosen_line(const core::Settings& engines);

std::string_view engine_label(core::EngineId id);
std::string_view badge_label(core::Badge badge);
std::string_view reason_label(core::Reason reason);

// Dessine l'écran dans le cadre ImGui en cours ; view.done passe à vrai sur « Commencer » (« Continuer
// la lecture » quand il est rouvert), view.cancelled sur « Annuler ».
void draw_first_run(FirstRunView& view, float scale);

}  // namespace ajp::app::play
