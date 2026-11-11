// L'assistant du premier lancement (wizard.h : Bienvenue, Langues, Intégration, Moteur, Prêt) : le GPU vu
// par Vulkan, les moteurs que core::recommend_engines propose pour lui (Vulkan toujours, les autres au
// choix, avec badges et présélection). Sa fin écrit settings.toml (moteurs choisis, first_run_done) ;
// --first-run le remontre, comme le bouton de l'onglet Moteurs pendant la lecture.
//
// SPAN et RIFE passent par TensorRT sur une carte NVIDIA qui l'a choisi, sinon par le backend
// Vulkan. Ce qui change se dit au journal.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "app/play/settings.h"
#include "app/play/settings_panel.h"
#include "app/play/wizard.h"
#include "core/engines.h"
#include "core/gpu_device.h"
#include "core/settings.h"

namespace ajp::app::play {

// Le GPU de rendu, tel que l'écran l'a ouvert.
struct GpuFacts {
  std::string name;  // « NVIDIA GeForce RTX 4070 Ti »
  std::string
      driver;  // version du pilote (VkPhysicalDeviceProperties::driverVersion, en décimal) ; vide : inconnue
  GpuVendor vendor = GpuVendor::Other;
  std::uint64_t vram_bytes = 0;
  std::uint64_t host_visible_vram_bytes = 0;  // tas VRAM visible de l'hôte : toute la VRAM avec Resizable BAR
  bool tensor_cores = false;                  // NVIDIA avec matrices coopératives Vulkan : une RTX
  bool wmma = false;                          // AMD RDNA3 ou RDNA4 avec matrices coopératives Vulkan
  bool mesa_nvk = false;                      // A37-11 : carte NVIDIA sous NVK, sans le pilote de NVIDIA
};

// La carte AMD a-t-elle des unités matricielles (WMMA : RDNA3, RDNA4) ? Par le nom du pilote :
// la puce chez RADV (« (RADV GFX1201) »), le produit ailleurs (« RX 9070 XT »).
bool amd_matrix_cores(std::string_view gpu_name);

core::SystemInfo system_of(const GpuFacts& gpu, core::Os os);
// P3.6 : le pilote NVIDIA que demande le TensorRT livré (CUDA 13), et la version lue du pilote (majeure,
// mineure) ; rien hors NVIDIA, sous NVK (version de Mesa, A37-11) ou sans version lisible.
inline constexpr int kTensorRtDriverMajor = 580;
std::optional<std::pair<int, int>> nvidia_driver(const GpuFacts& gpu);
// A37-08, B34-07 : le minimum de CUDA 13 avec sa mineure, d'après le tableau « minimum driver version » des
// notes de version de CUDA 13.0 : 580.65 sous Linux, 580.88 sous Windows (à revérifier sur la machine de
// Tatoub avec un pilote 580.65 à 580.87 sous Windows : TensorRT doit être refusé avant le téléchargement).
#ifdef __linux__
inline constexpr std::pair<int, int> kTensorRtDriverMinimum{kTensorRtDriverMajor, 65};
#else
inline constexpr std::pair<int, int> kTensorRtDriverMinimum{kTensorRtDriverMajor, 88};
#endif
// A37-08, A37-11 : pourquoi le pilote empêche TensorRT (pilote NVIDIA trop ancien, ou NVK) ; vide sinon,
// et hors NVIDIA.
std::string tensorrt_driver_problem(const GpuFacts& gpu);
// « NVIDIA GeForce RTX 4070 Ti · 12 Go · cœurs Tensor »
std::string gpu_line(const GpuFacts& gpu);

// Resizable BAR (ou Smart Access Memory) : « actif, 16 Go », « inactif, 256 Mo », ou rien quand la
// carte n'est pas une carte à mémoire dédiée ou que le pilote n'expose pas de tas visible.
std::string rebar_text(const GpuFacts& gpu);

// Un moteur que ce lecteur sait télécharger (engines.lock pour cet OS).
struct EngineDownload {
  core::EngineId id = core::EngineId::TensorRt;
  bool installed = false;  // déjà extrait sous <données>/backends
  std::uint64_t bytes = 0;
  std::uint64_t needed = 0;     // la place que l'installation demande au disque (engine_install.h :
                                // InstallSpace::peak) ; 0 : inconnue
  std::uint64_t available = 0;  // la place libre là où le moteur s'installe
  std::string license;          // à accepter avant le téléchargement, vide si aucune
};

// Ce qui est déjà sur ce poste.
struct EnginesHere {
  // Le système dont l'écran propose les moteurs : celui de l'hôte, ou celui qu'un test vérifie.
  core::Os os = core::kHostOs;
  bool tensorrt = false;  // backend ajp_backend_tensorrt trouvé
  bool coreml = false;    // ajp_backend_coreml (Mac : le moteur Metal de l'écran)
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
  std::uint64_t bytes = 0;  // la taille à télécharger (download seulement)
};

// P3.4 : les lignes Moteurs du résumé de l'assistant : les moteurs retenus, avec la taille à télécharger, et
// chaque moteur coché dont la licence n'est pas acceptée, qui ne s'installera pas, et le moteur recommandé
// pour cette carte s'il est resté décoché (A25-02).
std::vector<std::string> engines_summary(const std::vector<EngineRow>& rows);
// Ce moteur coché s'installera-t-il ? Non s'il se télécharge sous une licence qui n'est pas acceptée
// (B25-05).
bool engine_installs(const EngineRow& row);

struct FirstRunView {
  std::string gpu;
  std::vector<EngineRow> engines;
  bool done = false;       // « Commencer »
  bool revisit = false;    // rouvert depuis l'onglet Moteurs : la lecture reprend ensuite
  bool cancelled = false;  // « Annuler » (pendant une lecture seulement) : rien ne change
  // Ouvert pendant une lecture, l'assistant entier compris (A25-14) : « Annuler » rend la lecture, fermer
  // la fenêtre n'est plus la seule sortie.
  bool during_playback = false;
  // L'assistant du premier lancement (wizard.h) ; vide au rouvrir depuis l'onglet Moteurs (revisit).
  Wizard wizard;
  WizardFacts facts;
  std::string tier_line;     // « Niveau estimé : Rapide »
  std::string display_line;  // « Écran 3840×2160 · 165 Hz »
  bool laptop = false;       // un portable : la construction des moteurs y dure le double (A25-03)
  // Refait les textes traduits à la construction (ligne GPU, niveau, écran, état des moteurs) quand la
  // langue de l'interface change à l'écran Bienvenue (B25-02) ; les cases et les accords ne bougent pas.
  std::function<void(FirstRunView&)> retranslate;
};

// Les lignes de recommend_engines pour ce GPU, cochées selon la présélection au premier lancement,
// selon le choix de current ensuite (Vulkan toujours).
FirstRunView first_run_view(const GpuFacts& gpu, const EnginesHere& here, const core::Settings& current = {});

// settings avec les moteurs choisis, ceux présents, les préférés qui en découlent, et
// first_run_done.
core::Settings settings_after(const FirstRunView& view, core::Settings settings);

// settings sans id (moteur qui n'a pas pu s'installer), ses préférés recalculés.
core::Settings without_engine(core::Settings settings, core::EngineId id);

// id fait partie des moteurs choisis à l'écran de bienvenue.
bool chosen(const core::Settings& settings, core::EngineId id);

// Les bibliothèques de backend à essayer, dans l'ordre : celles de TensorRT et de Core ML seulement si
// leur moteur a été choisi, Vulkan restant le recours.
std::vector<std::filesystem::path> backends_within(std::span<const std::filesystem::path> candidates,
                                                   const core::Settings& engines);

// La chaîne que les moteurs choisis permettent : sans TensorRT ni Vulkan, ni SPAN ni RIFE. notes dit ce
// qui a changé.
PlaySettings within_engines(PlaySettings settings, const core::Settings& engines,
                            std::vector<std::string>& notes);
// « Vulkan, TensorRT », pour l'onglet Moteurs.
std::string chosen_line(const core::Settings& engines);

std::string_view engine_label(core::EngineId id);
std::string_view badge_label(core::Badge badge);
std::string_view reason_label(core::Reason reason);

// Dessine l'écran dans le cadre ImGui en cours ; view.done passe à vrai sur « Commencer » (« Continuer
// la lecture » quand il est rouvert), view.cancelled sur « Annuler ».
void draw_first_run(FirstRunView& view, float scale);

}  // namespace ajp::app::play
