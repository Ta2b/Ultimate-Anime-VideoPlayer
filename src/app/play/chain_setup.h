// Les moteurs d'une ouverture de la chaîne, pour ces réglages : le backend, SPAN, et l'estimateur du
// mouvement (RIFE ou sa variante mouvement, DistilDRBA), construits ou lus du cache
// puis chargés, la préparation à l'écran. La lecture (player.cpp) et l'export (export_run.h) les
// ouvrent de la même façon, et l'étage gpu se décrit d'eux (upscale()).
#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/backend_library.h"
#include "app/play/chain_shape.h"
#include "app/play/live_settings.h"
#include "app/play/model_choice.h"
#include "app/play/options.h"
#include "app/play/player_engines.h"
#include "app/play/ring_stage.h"
#include "app/play/settings.h"
#include "app/play/span_int8.h"
#include "app/play/stage_fallback.h"
#include "app/play/upscale_stage.h"
#include "core/clock.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/rife.h"
#include "media/hw_api.h"
#include "media/source.h"

namespace ajp::app::play {

// Les décodeurs à essayer pour un backend : NVDEC en mémoire CUDA pour TensorRT, puis le décodage
// logiciel pour ce que NVDEC refuse (H.264 10 bits, VP9 profils 1 et 3…), envoyé par le backend
// (host_upload.h) ; ailleurs, Vulkan Video sur l'appareil de l'écran quand FFmpeg le partage
// (shared_vulkan : les images restent sur le GPU, le backend Vulkan les lit en place), puis, sous
// Linux, VA-API (le bloc vidéo d'AMD ou d'Intel) et, sous Windows, D3D11VA, copiés en mémoire hôte,
// puis le décodage logiciel.
// Sur un Mac, VideoToolbox (le bloc média de la puce) d'abord, copié en mémoire hôte.
// Sur une carte NVIDIA, VA-API n'existe que par un pilote tiers au-dessus de NVDEC : l'essai est
// sauté (nvidia_gpu).
std::vector<media::HwApi> decoder_attempts(std::string_view backend, bool shared_vulkan,
                                           bool nvidia_gpu = false);

// Où chercher le graphe de DistilDRBA : à côté du modèle RIFE choisi, puis dans les dossiers RIFE.
std::vector<std::filesystem::path> motion_model_dirs(const PlayerPaths& paths,
                                                     const std::optional<ModelChoice>& rife);

// Images décodées tenues au plus en même temps : par le thread source, par l'étage gpu
// (RingStage::kMaxFrames en vol) et par le backend après ingest. Autant de surfaces NVDEC rendues
// sans copie (DecoderOptions::direct_surfaces).
inline constexpr int kHeldDecodedFrames =
    media::kSourceHeldFrames + static_cast<int>(RingStage::kMaxFrames) + gpu::kMaxIngestHeldFrames;
// B15-03 : plus celle que timed() retient jusqu'à l'image suivante.
static_assert(kHeldDecodedFrames + 1 <= media::kMaxExtraHwFrames, "NVDEC : 32 surfaces en plus au plus");

// Les options du décodeur pour ce backend et cet écran : les essais de decoder_attempts ; sous
// TensorRT, qui lit les images NVDEC en place, les surfaces directes ; ailleurs, la copie en
// mémoire hôte des images matérielles (DecoderOptions::download).
media::DecoderOptions decoder_options(std::string_view backend, const GpuDevice& render,
                                      const media::HwDeviceSource* shared);

// Ordre de déclaration = ordre inverse de destruction : les moteurs partent avant leur backend.
struct ChainEngines {
  ChainEngines() = default;
  ChainEngines(ChainEngines&&) noexcept = default;
  ChainEngines& operator=(ChainEngines&&) noexcept = default;
  ChainEngines(const ChainEngines&) = delete;
  ChainEngines& operator=(const ChainEngines&) = delete;
  // Le travail encore enfilé (une dernière passe du thread gpu arrêté) finit avant que les moteurs, leurs
  // graphes et leurs poids partent, sur tous les chemins : lecture, export, sortie anticipée (A30-01).
  ~ChainEngines() {
    if (backend) backend->drain();
  }

  BackendLibrary::Instance backend;
  std::string backend_name;           // « tensorrt », « vulkan »
  std::optional<ModelChoice> model;   // SPAN ; rien : sans agrandissement
  std::optional<SpanInt8> span_int8;  // sa variante INT8, construite à sa place (span_int8.h)
  int factor = 1;                     // kSpanFactor avec SPAN
  int top_factor = 1;                 // le plus haut facteur d'interpolation que la lecture peut prendre
  // La forme de cette ouverture (anneau, ordre du warp) et ses mesures : ce qu'un réglage changé en direct
  // doit retrouver, sinon la chaîne se rouvre (chain_shape.h).
  OpenedChain opened;
  WorkSize work;       // entrée des moteurs pour la vidéo de la lecture
  std::string reduce;  // « 1920x1080 → Area 1280x720 », ou « 1920x1080 »
  // RIFE cherché même sans interpolation : le menu peut l'allumer.
  std::optional<ModelChoice> rife_choice;
  std::optional<std::filesystem::path> rife_model;
  std::vector<std::filesystem::path> model_places;  // dossiers du graphe de DistilDRBA
  bool drba_found = false;
  int drba_align = 64;                   // multiple des plans remplis de DistilDRBA : 128 au demi-flot
  WorkSize rife_size;                    // où RIFE mouvement estime (720p pour une sortie 4K), sinon work
  kernels::RifeGeometry geometry;        // de rife_size
  kernels::RifeGeometry image_geometry;  // de work
  int rife_align = kernels::kRifeAlign;
  int rife_planes = kernels::kRifeInputChannels;
  bool skip_held = false;               // --skip-held (UpscaleSetup::skip_held)
  bool one_lane = false;                // --one-lane (UpscaleSetup::one_lane)
  bool rife_warm = false;               // --rife-warm (UpscaleSetup::rife_warm)
  bool smooth_drawings = false;         // --smooth-drawings (UpscaleSetup::smooth_drawings)
  std::unique_ptr<gpu::Engine> engine;  // SPAN
  std::unique_ptr<gpu::Engine> drba;
  std::unique_ptr<gpu::Engine> rife;  // RIFE image, ou l'interpolation d'Apple (apple)
  std::unique_ptr<gpu::Engine> rife_motion;
  std::unique_ptr<gpu::Engine> rife_motion_calm;  // --rife-calm-model (P6.17), à la forme de rife_motion
  bool rife_fused = false;  // rife ou rife_motion est RIFE fusionné (six moteurs et noyaux de colle)
  bool drba_fused = false;  // drba est DistilDRBA fusionné (quatre moteurs et noyaux de colle)
  // Le fusionné essayé puis refusé, le réseau entier à sa place (A03-21) : voir OpenedChain.
  std::optional<RefusalCode> rife_fused_refused;
  std::optional<RefusalCode> drba_fused_refused;
  bool drba_motion = false;  // drba rend flot et masque (variante mouvement, warp après SPAN)
  WorkSize drba_size;        // où DistilDRBA estime : rife_size en variante mouvement, sinon work
  bool apple = false;
  // Les réglages que la chaîne joue vraiment : ceux demandés, sans les étages refusés et avec les crans
  // ramenés à Équilibre (stage_fallback.h) ; stage_notes dit ce qui a changé.
  PlaySettings played;
  std::vector<StageNote> stage_notes;

  bool use_drba() const { return drba != nullptr; }
  bool use_apple() const { return apple && rife != nullptr; }
  bool interpolates() const { return rife || rife_motion || use_drba(); }
  // « DistilDRBA », « Apple » ou « RIFE », pour les compteurs.
  std::string_view motion_name() const;
  // « RIFE ×2 · rife_v4.25 · », ou rien sans interpolation.
  std::string motion_label(const PlaySettings& settings) const;
  // « SPAN ×2 · modèle · vulkan · 1920x1080 → 3840x2160 », ou « sans agrandissement · … ».
  std::string span_label() const;

  // L'étage gpu pour des images source width × height écrites dans slots. clock nullptr : aucune
  // paire sautée pour retard (export).
  UpscaleSetup upscale(const PlaySettings& settings, std::shared_ptr<LiveSettings> live,
                       const MasterClock* clock, int width, int height,
                       std::vector<RingSlotDesc> slots) const;
};

// Le Rapide est demandé : choisi dans les réglages et pas replié pour cette séance.
constexpr bool free_requested(const PlaySettings& settings, bool fallback) {
  return settings.auto_free && !fallback;
}

// Le backend de p.library, puis les moteurs que settings demandent pour p.video. Rien : la fenêtre
// fermée pendant une préparation. Un étage refusé ou annulé (SPAN, interpolation) est retenu dans memory et
// la chaîne se prépare sans lui (ChainEngines::played, stage_notes) ; un cran FP8 ou FP32 refusé ne ramène
// que son étage à Équilibre. memory nullptr (export, banc) : les crans retombent encore, mais un étage refusé
// reste un refus. Refus : backend refusé, ou étage refusé sans mémoire.
std::expected<std::optional<ChainEngines>, Refusal> prepare_chain(Player& p, const PlaySettings& settings,
                                                                  StageMemory* memory);

}  // namespace ajp::app::play
