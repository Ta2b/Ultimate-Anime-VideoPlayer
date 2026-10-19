// L'écran de bienvenue du premier lancement : le GPU vu par Vulkan, les moteurs que
// core::recommend_engines propose pour lui (Vulkan toujours, les autres au choix, avec badges et
// présélection) et le profil de démarrage. « Commencer » écrit settings.toml (moteurs choisis,
// first_run_done) et user.toml (start_profile) ; --first-run remontre l'écran.
//
// Dans la tranche verticale, SPAN et RIFE passent par TensorRT et RTX HDR par Maxine : sans eux,
// la lecture garde l'image source (pas encore d'inférence Vulkan ici) et le dit au journal.
#pragma once

#include <cstdint>
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
  bool tensor_cores = false;  // NVIDIA avec matrices coopératives Vulkan : une RTX
};

core::SystemInfo system_of(const GpuFacts& gpu);
// « NVIDIA GeForce RTX 4070 Ti · 12 Go · cœurs Tensor »
std::string gpu_line(const GpuFacts& gpu);

// Profil de démarrage conseillé : 1 (1440p, RIFE ×3) pour une RTX de 11 Go et plus, 7 (1080p, ×2)
// pour une RTX plus petite, 9 (source) sans cœurs Tensor.
int start_profile_for(const GpuFacts& gpu);

// Ce qui est déjà sur ce poste.
struct EnginesHere {
  bool tensorrt = false;      // backend ajp_backend_tensorrt trouvé
  bool maxine = false;        // SDK Maxine importé, ou importable depuis --truehdr-sdk
  std::string maxine_status;  // « importé », « à importer depuis /usr/local/VideoFX »…
};

struct EngineRow {
  core::EngineOffer offer;
  bool usable = false;  // installé, ou installable par ce lecteur
  bool chosen = false;
  std::string status;  // « livré », « installé », « pas encore installable dans cette version »
};

struct FirstRunView {
  std::string gpu;
  std::vector<EngineRow> engines;
  ProfilesView profiles;  // .start : le profil choisi
  bool done = false;      // « Commencer »
};

// Les lignes de recommend_engines pour ce GPU, cochées selon la présélection au premier lancement,
// selon le choix de current ensuite (Vulkan toujours) ; profiles.start garde le profil donné.
FirstRunView first_run_view(const GpuFacts& gpu, const EnginesHere& here, ProfilesView profiles,
                            const core::Settings& current = {});

// settings avec les moteurs choisis, ceux présents, les préférés qui en découlent, et
// first_run_done.
core::Settings settings_after(const FirstRunView& view, core::Settings settings);

// La chaîne que les moteurs choisis permettent : sans TensorRT, ni SPAN ni RIFE ; sans Maxine, pas
// de RTX HDR. notes dit ce qui a été retiré.
PlaySettings within_engines(PlaySettings settings, const core::Settings& engines,
                            std::vector<std::string>& notes);
// « Vulkan, TensorRT, NVIDIA Maxine », pour l'onglet Moteurs.
std::string chosen_line(const core::Settings& engines);

std::string_view engine_label(core::EngineId id);
std::string_view badge_label(core::Badge badge);
std::string_view reason_label(core::Reason reason);

// Dessine l'écran dans le cadre ImGui en cours ; view.done passe à vrai sur « Commencer ».
void draw_first_run(FirstRunView& view, float scale);

}  // namespace ajp::app::play
