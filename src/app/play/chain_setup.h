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
#include <utility>
#include <vector>

#include "app/play/anime4k_choice.h"
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
#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/rife.h"
#include "media/hw_api.h"
#include "media/source.h"

namespace kirari::app::play {

// Ce que le rapport « Signaler un problème » dit de la lecture en cours (B39-02) : la vidéo (nom, codec,
// taille, cadence, transfert de la source), l'écran, la sortie HDR10 ou SDR, la chaîne jouée. Les faits sont
// remplacés à chaque nouvelle vidéo, et placés en tête du rapport : la fin du journal les perd après une
// longue séance.
struct PlaybackFacts {
  std::string file;   // le nom du fichier seul
  std::string codec;  // « hevc », « h264 »
  int width = 0;
  int height = 0;
  double frame_rate = 0;  // 0 : inconnue
  std::string transfer;   // « SDR », « PQ », « HLG »
  std::string screen;     // describe_hdr : pic, HDR du système, fréquence
  bool hdr_output = false;
  std::string chain;  // l'étiquette de la chaîne : SPAN, RIFE, facteur
};
std::vector<std::pair<std::string, std::string>> playback_facts(const PlaybackFacts& facts);
// Les faits de la lecture qui s'ouvre, retenus pour le rapport.
void note_playback_facts(const media::TrackInfo& video, const ColorInfo& source, const std::string& screen,
                         bool hdr_output, const std::string& chain, const std::filesystem::path& file);
// La chaîne qui joue : au journal avec le nom du fichier, et dans les faits du rapport.
void announce_playback(const Player& p, const ColorInfo& source, bool hdr_output, const std::string& chain);

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
// sans copie (DecoderOptions::direct_surfaces). Le Smooth en tient plus en vol (smooth_extra_frames), la
// stabilisation temporelle aussi (temporal_extra_frames).
inline constexpr int kHeldDecodedFrames =
    media::kSourceHeldFrames + static_cast<int>(RingStage::kMaxFrames) + gpu::kMaxIngestHeldFrames;
// Ce que l'étage gpu tient en plus : les images du Smooth, la fenêtre de la stabilisation temporelle.
struct HeldExtras {
  bool smooth_drawings = false;
  bool temporal_fix = false;
};
constexpr int held_decoded_frames(HeldExtras extras) {
  return kHeldDecodedFrames + static_cast<int>(smooth_extra_frames(extras.smooth_drawings)) +
         static_cast<int>(temporal_extra_frames(extras.temporal_fix));
}
static_assert(held_decoded_frames({.smooth_drawings = true, .temporal_fix = true}) + 1 <=
                  media::kMaxExtraHwFrames,
              "NVDEC: 32 extra surfaces at most, the one frame the decoder still holds included (B15-03)");

// Les options du décodeur pour ce backend et cet écran : les essais de decoder_attempts ; sous
// TensorRT, qui lit les images NVDEC en place, les surfaces directes ; ailleurs, la copie en
// mémoire hôte des images matérielles (DecoderOptions::download).
// extras : ce que l'étage gpu tient en plus (held_decoded_frames).
media::DecoderOptions decoder_options(std::string_view backend, const GpuDevice& render,
                                      const media::HwDeviceSource* shared, HeldExtras extras = {});

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
  std::string backend_name;          // « tensorrt », « vulkan »
  std::optional<ModelChoice> model;  // SPAN ; rien : sans agrandissement, ou Anime4K
  // Anime4K à la place de SPAN (anime4k_choice.h) : la chaîne rend l'image de travail telle quelle (factor
  // 1), l'écran et l'export y posent les shaders. span_list : la liste du menu, que model ne porte pas alors.
  std::optional<render::Anime4kChoice> anime4k;
  std::optional<ModelChoice> span_list;
  std::optional<SpanInt8> span_int8;  // sa variante INT8, construite à sa place (span_int8.h)
  int factor = 1;                     // kSpanFactor avec SPAN
  int order_factor = 0;  // le facteur qui décide l'ordre du warp et de SPAN (0 : inconnu, en Auto)
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
  bool span_reuse = true;               // --no-span-reuse (UpscaleSetup::span_reuse)
  bool span_tiles = true;               // --no-span-tiles (UpscaleSetup::span_reuse)
  bool span_tiles_check = false;        // --span-tiles-check (UpscaleSetup::span_reuse)
  bool one_lane = false;                // --one-lane (UpscaleSetup::one_lane)
  bool rife_warm = false;               // --rife-warm (UpscaleSetup::rife_warm)
  bool rife_encode_all = false;         // --rife-encode-all (UpscaleSetup::rife_encode_all)
  bool rife_at_slot = false;            // --rife-at-slot (UpscaleSetup::rife_at_slot)
  bool scene_map = false;               // --scene-map (UpscaleSetup::scene_map)
  bool smooth_drawings = false;         // --smooth-drawings (UpscaleSetup::smooth_drawings)
  bool temporal_fix_whole = false;      // --no-tfix-cache (chain_temporal.cpp)
  bool pair_triage = false;             // --pair-triage (UpscaleSetup::pair_triage)
  std::unique_ptr<gpu::Engine> engine;  // SPAN
  // La stabilisation temporelle avant SPAN (chain_temporal.cpp), à la taille de travail : la fenêtre et
  // l'encodeur de ses images (prepare/onnx_cache.h) ; nuls : éteinte ou refusée.
  std::unique_ptr<gpu::Engine> temporal_fix;
  std::unique_ptr<gpu::Engine> temporal_encoder;
  int temporal_features = 0;  // canaux d'une entrée de traits (prepare::feature_channels)
  std::unique_ptr<gpu::Engine> drba;
  std::unique_ptr<gpu::Engine> rife;  // RIFE image, ou l'interpolation d'Apple (apple)
  std::unique_ptr<gpu::Engine> rife_motion;
  std::unique_ptr<gpu::Engine> rife_motion_calm;  // --rife-calm-model (P6.17), à la forme de rife_motion
  // Échelle par paire (PlaySettings::pair_scale) : rife_motion à rife_half_size, pour les paires à mouvement
  // fort ; nul sans l'option ou si sa préparation est refusée.
  std::unique_ptr<gpu::Engine> rife_motion_half;
  WorkSize rife_half_size;
  // GMFSS Fortuna Union (chain_gmfss.cpp) : ses deux graphes, le mélange ou seul ; gmfss_px, le seuil du
  // mélange en pixels à 720 lignes.
  std::unique_ptr<gpu::Engine> gmfss_pair;
  std::unique_ptr<gpu::Engine> gmfss_fusion;
  GmfssUse gmfss_use = GmfssUse::Off;
  float gmfss_px = 15.0F;
  std::int64_t gmfss_cost_us = 0;  // GmfssSetup::cost_us et pair_us
  std::int64_t gmfss_pair_us = 0;
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
  bool fallback_fresh = false;  // B08-02 : un repli interne refusé à cette ouverture, pas retenu d'avant
  std::vector<LoadedEngine> loaded_engines;  // B08-12 : pour les badges de l'onglet Moteurs

  // --shot-plan (shot_plan.h) : l'Auto plan par plan ; drba_by_shot, DistilDRBA chargé avec RIFE mouvement,
  // l'un ou l'autre selon le plan (chain_shot_plan.cpp). DistilDRBA n'est alors pas le moteur de la chaîne.
  bool shot_plan = false;
  bool drba_by_shot = false;
  bool use_drba() const { return drba != nullptr && !drba_by_shot; }
  bool use_apple() const { return apple && rife != nullptr; }
  bool interpolates() const { return rife || rife_motion || use_drba() || gmfss_use == GmfssUse::Only; }
  // « DistilDRBA », « Apple » ou « RIFE », pour les compteurs.
  std::string_view motion_name() const;
  HeldExtras held_extras() const {
    return {.smooth_drawings = smooth_drawings, .temporal_fix = temporal_fix != nullptr};
  }
  // C08-08 : le moteur dans la clé de la mémoire de l'Auto (core/auto_memory.h) : l'interpolation, son
  // modèle, fusionné ou entier, image ou mouvement, et le SPAN chargé (span_variant : FP16, FP8, INT8). Un
  // palier appris sur RIFE 4.25 lite ne vaut pas pour RIFE 4.26.
  std::string auto_memory_engine(std::string_view span_variant) const;
  // « RIFE ×2 · rife_v4.25 · », ou rien sans interpolation.
  std::string motion_label(const PlaySettings& settings) const;
  // « SPAN ×2 · modèle · vulkan · 1920x1080 → 3840x2160 », « Anime4K ×2 · M · mode A · … », ou « sans
  // agrandissement · … ».
  std::string span_label() const;

  // L'étage gpu pour des images source width × height écrites dans slots. clock nullptr : aucune
  // paire sautée pour retard (export).
  UpscaleSetup upscale(const PlaySettings& settings, std::shared_ptr<LiveSettings> live,
                       const MasterClock* clock, int width, int height,
                       std::vector<RingSlotDesc> slots) const;
};

// Le module du moteur (TensorRT, Vulkan) : il entre dans les signatures des refus retenus.
std::string library_name(const Player& p);

// SPAN ou Anime4K, quand l'agrandissement tourne (span_runs ; c.work déjà posé) : Anime4K choisi, la liste du
// menu et ses shaders lus sur l'écran, pour la lecture comme pour l'export ; refusés, la chaîne reste sans
// agrandissement (l'image de travail, agrandie par le filtre de la fenêtre). Vrai : SPAN est à charger
// (chain_upscaler.cpp).
bool take_upscaler(Player& p, ChainEngines& c, const PlaySettings& settings);

// Le backend de p.library, puis les moteurs que settings demandent pour p.video. Rien : la fenêtre
// fermée pendant une préparation. Un étage refusé ou annulé (SPAN, interpolation) est retenu dans memory et
// la chaîne se prépare sans lui (ChainEngines::played, stage_notes) ; un cran FP8 ou FP32 refusé ne ramène
// que son étage à Équilibre. memory nullptr (export, banc) : les crans retombent encore, mais un étage refusé
// reste un refus. Refus : backend refusé, ou étage refusé sans mémoire.
std::expected<std::optional<ChainEngines>, Refusal> prepare_chain(Player& p, const PlaySettings& settings,
                                                                  StageMemory* memory);

// Échelle par paire (chain_half.cpp) : le second RIFE mouvement, à demi-taille, dans c.rife_motion_half quand
// settings.pair_scale le demande et que c a un RIFE mouvement. Rien si l'utilisateur ferme la fenêtre ; un
// refus laisse la lecture à pleine taille (journal), retenu pour la séance dans inner (C08-12).
std::optional<bool> load_half_rife(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                   const PlaySettings& settings, InnerFallbacks& inner,
                                   const std::string& library, const std::filesystem::path& models_dir);

// GMFSS (chain_gmfss.cpp) : ce que settings demande (le mélange avec RIFE, ou GMFSS seul), puis ses deux
// moteurs dans c, à c.rife_size pour le mélange (après load_rife), à c.work seul (avant). Rien si
// l'utilisateur ferme la fenêtre ; un refus (moteur autre que TensorRT, graphes absents, construction) laisse
// RIFE, la raison au journal, retenu pour la séance dans inner.
GmfssUse gmfss_wanted(const PlaySettings& settings);
// Le coût d'une paire GMFSS du mélange au-delà de RIFE, à la taille size (GmfssSetup::cost_us, pair_us).
struct GmfssCost {
  std::int64_t total_us = 0;
  std::int64_t pair_us = 0;  // le graphe « paire » seul
};
GmfssCost gmfss_cost(WorkSize size, bool half);
// Le plafond de la cadence Auto (Hz, 0 : l'écran) quand GMFSS tourne, ce qu'une RTX 4070 Ti tient en travail
// 720p (une fusion de 11 à 13 ms par instant, une paire de 35 ms, 12 au demi-flot) au lieu de monter à ×8 :
// GMFSS seul, ×2 de la source (×2,5 au demi-flot) ; le mélange, budgété comme si chaque paire était forte,
// ×3 (×4 au demi-flot, à mesurer), là où l'Auto Stable se posait sur un anime nerveux au lieu d'osciller de
// ×1 à ×7 entre scènes calmes et fortes. cap, le plafond choisi.
int gmfss_auto_cap(const ChainEngines& c, int cap, double source_fps);
std::optional<bool> load_gmfss(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                               const PlaySettings& settings, InnerFallbacks& inner,
                               const std::string& library);
// L'interpolation entière : GMFSS seul d'abord (RIFE ne se charge que s'il est refusé), puis RIFE, puis le
// mélange sur le RIFE chargé. Faux : la fenêtre fermée.
std::expected<bool, StageFailure> load_motion(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                              const PlaySettings& settings, InnerFallbacks& inner,
                                              const std::string& library);

// L'Auto plan par plan (chain_shot_plan.cpp, --shot-plan). drba_for_shots : DistilDRBA est-il à charger en
// plus de RIFE (RIFE choisi, DistilDRBA trouvé, warp après SPAN pour que les deux soient en variante
// mouvement) ? settle_shot_plan, après le chargement : les deux moteurs vont ensemble, ou DistilDRBA est
// rendu et RIFE joue seul (la raison au journal).
bool drba_for_shots(const ChainEngines& c, const PlaySettings& settings);
void settle_shot_plan(ChainEngines& c);

// La stabilisation temporelle (chain_temporal.cpp) : le modèle de settings.temporal_fix à c.work dans
// c.temporal_fix, sous TensorRT. Rien si l'utilisateur ferme la fenêtre ; un refus (autre moteur, modèle
// absent, construction) laisse la lecture sans elle, la raison au journal, retenu pour la séance dans inner.
std::vector<std::filesystem::path> temporal_fix_dirs(const PlayerPaths& paths);
std::optional<bool> load_temporal_fix(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                      const PlaySettings& settings, InnerFallbacks& inner,
                                      const std::string& library);

// RIFE (chain_rife.cpp) : c.rife_size, la géométrie, puis le fusionné, la variante mouvement ou RIFE image,
// au cran de précision de settings. Faux : la fenêtre fermée pendant une construction ; un refus de l'étage
// le rend.
std::expected<bool, StageFailure> load_rife(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                            const PlaySettings& settings, InnerFallbacks& inner,
                                            const std::string& library);

}  // namespace kirari::app::play
