// Les moteurs d'une ouverture de la chaîne, pour ces réglages : le backend, SPAN, et l'estimateur du
// mouvement (RIFE ou sa variante mouvement, GMFSS Fortuna, DistilDRBA), construits ou lus du cache
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
#include "app/play/gmfss_model.h"
#include "app/play/live_settings.h"
#include "app/play/model_choice.h"
#include "app/play/options.h"
#include "app/play/player_engines.h"
#include "app/play/ring_stage.h"
#include "app/play/settings.h"
#include "app/play/span_int8.h"
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
// Linux, VA-API (le bloc vidéo d'AMD ou d'Intel) copié en mémoire hôte, puis le décodage logiciel.
std::vector<media::HwApi> decoder_attempts(std::string_view backend, bool shared_vulkan);

// Images décodées tenues au plus en même temps : par le thread source, par l'étage gpu
// (RingStage::kMaxFrames en vol) et par le backend après ingest. Autant de surfaces NVDEC rendues
// sans copie (DecoderOptions::direct_surfaces).
inline constexpr int kHeldDecodedFrames =
    media::kSourceHeldFrames + static_cast<int>(RingStage::kMaxFrames) + gpu::kMaxIngestHeldFrames;
static_assert(kHeldDecodedFrames <= media::kMaxExtraHwFrames, "NVDEC : 32 surfaces en plus au plus");

// Les options du décodeur pour ce backend et cet écran : les essais de decoder_attempts ; sous
// TensorRT, qui lit les images NVDEC en place, les surfaces directes ; ailleurs, la copie en
// mémoire hôte des images matérielles (DecoderOptions::download).
media::DecoderOptions decoder_options(std::string_view backend, const GpuDevice& render,
                                      const media::HwDeviceSource* shared);

// Ordre de déclaration = ordre inverse de destruction : les moteurs partent avant leur backend.
struct ChainEngines {
  BackendLibrary::Instance backend;
  std::string backend_name;           // « tensorrt », « vulkan »
  std::optional<ModelChoice> model;   // SPAN ; rien : sans agrandissement
  std::optional<SpanInt8> span_int8;  // sa variante INT8, construite à sa place (span_int8.h)
  int factor = 1;                     // kSpanFactor avec SPAN
  WorkSize work;                      // entrée des moteurs pour la vidéo de la lecture
  std::string reduce;                 // « 1920x1080 → Area 1280x720 », ou « 1920x1080 »
  // RIFE cherché même sans interpolation : le menu peut l'allumer.
  std::optional<ModelChoice> rife_choice;
  std::optional<std::filesystem::path> rife_model;
  std::vector<std::filesystem::path> gmfss_places;  // dossiers des graphes de GMFSS et DistilDRBA
  GmfssVariant gmfss_variant = GmfssVariant::Base;
  bool gmfss_found = false;
  bool drba_found = false;
  WorkSize rife_size;                    // où RIFE mouvement estime (720p pour une sortie 4K), sinon work
  kernels::RifeGeometry geometry;        // de rife_size
  kernels::RifeGeometry image_geometry;  // de work
  int rife_align = kernels::kRifeAlign;
  int rife_planes = kernels::kRifeInputChannels;
  bool one_lane = false;                // --one-lane (UpscaleSetup::one_lane)
  std::unique_ptr<gpu::Engine> engine;  // SPAN
  GmfssEngines gmfss;
  std::unique_ptr<gpu::Engine> drba;
  std::unique_ptr<gpu::Engine> rife;
  std::unique_ptr<gpu::Engine> rife_motion;

  bool use_gmfss() const { return gmfss.pair != nullptr; }
  bool use_drba() const { return drba != nullptr; }
  bool interpolates() const { return rife || rife_motion || use_gmfss() || use_drba(); }
  // « GMFSS », « DistilDRBA » ou « RIFE », pour les compteurs.
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

// Le backend de p.library, puis les moteurs que settings demandent pour p.video. Rien : la fenêtre
// fermée pendant une préparation. Refus : modèle introuvable, backend ou moteur refusés.
std::expected<std::optional<ChainEngines>, Refusal> prepare_chain(Player& p, const PlaySettings& settings);

}  // namespace ajp::app::play
