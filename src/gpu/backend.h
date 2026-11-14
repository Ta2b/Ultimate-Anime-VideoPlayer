// Interface d'un backend d'inférence. Chaque backend est une bibliothèque
// chargée à l'exécution (ajp_backend_<nom>.so / .dll), installée seulement si
// l'utilisateur l'a choisie : le lecteur démarre sans CUDA ni TensorRT.
//
// La frontière passe des types C++ (std::string, std::function, std::expected…) : un
// backend se construit avec la même bibliothèque standard que le lecteur, ce que
// l'empreinte kBackendLayout vérifie au chargement. Aucune exception ne la traverse : chaque
// erreur revient en Refusal, et les points d'entrée extern "C" sont noexcept.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>
#include <version>

#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "gpu/lanes.h"
#include "gpu/tensor.h"

namespace ajp::gpu {

// Version de cette interface. Un backend d'une autre version est refusé au chargement.
// 2 : BackendInit à la création, noyaux de la chaîne (kernels()), alloc_tensor().
// 3 : enqueue(), signal() et wait() rendent leur refus ; ajp_backend_layout().
// 4 : KernelOps sans le flux optique matériel (open_optical_flow, optical_flow).
// 5 : KernelOps::soft_splat_all (les splats d'un instant en un passage).
// 6 : voies de calcul (gpu/lanes.h) : Backend::use_lane, BackendCaps::lanes.
// 7 : KernelOps::rife_pair (une paire sans réécrire le maillage de RIFE).
// 8 : KernelOps::soft_splat_fill (les splats d'un instant et le remplissage de l'instant ensemble).
// 9 : KernelOps::warp_blend_ring, warp_blend_poses_ring, gmfss_compose_ring ; BackendCaps::ring_targets.
// 10 : KernelOps::write_yuv (export par puits CUDA) ; BackendCaps::yuv_targets.
// 11 : BackendCaps::span_fp8_cutlass (le SPAN entier en FP8 par noyaux CUTLASS, docs/span-fp8-cutlass.md).
// 12 : BackendCaps::rife_fused (RIFE fusionné : six moteurs TensorRT et noyaux de colle CUDA, manifeste «
// .rifefused »).
// 13 : BackendCaps::drba_fused (DistilDRBA fusionné : quatre moteurs TensorRT et noyaux CUDA, manifeste «
// .drbafused »).
// 14 : Backend::derive_span_fp8 (dérivation native du paquet « .span8 », docs/span-fp8-cutlass.md).
// 15 : BackendCaps::gmfss_fused (GMFSS fusionné : la paire en deux moteurs, les caractéristiques d'une image
// et le flot, avec des tenseurs d'entrée et de sortie décalés dans un tampon ; docs/gmfss-fusionne.md).
// 16 : KernelOps::scene_stats et ses voisines, ajoutées au milieu de la table virtuelle par le grade
// adaptatif (rc.6) sans changer de version : un module TensorRT d'avant appelait write_ring_true_hdr à leur
// place et plantait au lancement d'une vidéo en RTX HDR. Toute virtuelle ajoutée ou déplacée dans KernelOps
// ou Backend demande une nouvelle version.
// 17 : KernelOps::deband (débandage dans la chaîne CUDA) ; RTX HDR retiré (open_true_hdr, set_true_hdr,
// scene_stats, write_ring_true_hdr, true_hdr_ramp).
// 18 : DistilDRBA fusionné à deux sorties (flot et masque, variante mouvement du warp après SPAN).
// 19 : KernelOps::frame_stats_start et frame_stats_take (analyse v2 pour le grade,
// kernels/spec/frame_stats.h).
// 20 : KernelOps::health, map_fences et unmap_fences (étape 11.5 : signal d'export écrit par le GPU,
// contexte perdu vu sans attendre le délai).
// 21 : Backend::enqueue_ring (étape 3.4b : le SPAN FP8 écrit l'emplacement de l'anneau depuis son épilogue).
// 22 : SpanFp8Derivation::scores_acc16 et kernels_ms_acc16 (étape 3.5a).
// 23 : SpanFp8Request::hard, SpanFp8ImageScore au 99,9e centile, SpanFp8Derivation::hard_scores (étape 3.7a).
// 24 : BackendCaps::gpu_arch et span_fp8_kernels, SpanFp8Request::validate_only, SpanFp8Derivation::gpu_arch
// et kernels_version (étape 3.7b).
// 25 : SpanFp8Request::min_free_ratio (étape 3.6).
// 26 : BackendCaps::ring_rgb10a2, RingFormat::Rgb10a2Unorm importé (étape 5.5a).
// 27 : RingWrite::grade, KernelOps::grades_ring et set_grade (grade exact dans l'anneau, chantier 2.9).
// 28 : KernelOps::gpu_profile (chantier 0.2) ; GradeOrder et GradeVersion gagnent la LUT de la courbe par
// plan (8.6b).
// 29 : le grade de l'anneau retiré (RingWrite::grade, GradeSlot, GradeOrder, grades_ring, set_grade).
// 30 : GMFSS retiré (gmfss_frames, soft_splat, soft_splat_all, soft_splat_fill, gmfss_compose,
// gmfss_compose_ring, fill ; BackendCaps::gmfss_fused).
// 31 : Engine::next_warm (RIFE à chaud, P6.15) ; PairScores::moving (la zone qui bouge, P6.6 et P6.18).
// 32 : Backend::drain (les files attendues avant la destruction des moteurs, A30-01).
// 33 : BackendInit::profile_dispatches (--profile-dispatches atteint le backend, A45-08).
// 34 : retirés, sans usage dans le lecteur (A21-10, A21-11) : Backend::import, ExternalImage,
// BackendCaps::import_formats, KernelOps::rife_pad, measure, previous_scores, IngestSpec::rife_conversion ;
// KernelOps::SceneStats passe au lecteur (chain_shots.h).
// 35 : huit RefusalCode jamais produits retirés (B20-04) : les codes suivants changent de numéro, un refus
// d'un backend d'avant se lirait sous un autre nom.
// 36 : Engine::next_pair_follows (RIFE fusionné : l'encodage de B repris comme celui de A à la paire
// suivante).
// 37 : Motion::warp (le warp final en Catmull-Rom, en option).
// 38 : Artifact::reduced (l'avis de précision réduite du repli Vulkan, montré à l'écran : C45-01).
// 39 : Artifact::reused (le moteur vient du cache : un refus au chargement le met de côté, C05-01).
// 40 : BackendInit::filtered_warp (le warp par texture, à l'essai) ; SpanFp8Request::fp16, timing_width et
// timing_height ; SpanFp8Derivation::trt_fp16_ms, timed_size et validation_peaks (E01).
// 41 : BackendInit::trials (les options d'essai des chantiers, un bit chacune ; kTrialL2Keep).
// 42 : BackendCaps::span_fp16_cutlass (le SPAN FP16 natif par noyaux CUTLASS, RTX 40 et RTX 50).
inline constexpr int kBackendAbi = 42;

using DeviceUuid = ajp::DeviceUuid;

enum class DeviceMatch : std::uint8_t { Same, Different, Unknown };

// Un backend journalise par BackendInit::log, que app relie à log_channel("backend")
// (core/log.h) ; ses canaux y vont par install_module_log_sinks. Le rappel ne lève jamais.
enum class BackendLogLevel : std::uint8_t { Debug, Info, Warn, Error };
using BackendLog = std::function<void(BackendLogLevel level, std::string_view message)>;

// Les bits de BackendInit::trials. Un bit retiré n'est pas repris par un autre essai.
// --l2-keep : l'encodage gardé de RIFE fusionné tenu en L2 (fenêtre d'accès persistante, au plus 16 Mo
// réservés). Gain à mesurer face au L2 qu'il prend à SPAN. Lu par le backend TensorRT ; sans effet ailleurs.
inline constexpr std::uint32_t kTrialL2Keep = 1U << 0U;
// « Vitesse avant précision » décochée (Général > Moteurs, PlaySettings::fast_precision) : la porte du
// moteur Vulkan ne garde une précision réduite qu'à 0,5 dB des noyaux de référence, même quand la perte ne se
// voit pas (net_gate.h). Un réglage plutôt qu'un essai, porté par ces bits pour ne pas monter kBackendAbi (un
// paquet TensorRT déjà installé reste valable). Lu par le backend Vulkan ; sans effet ailleurs.
inline constexpr std::uint32_t kStrictPrecision = 1U << 1U;
// 1U << 2U : l'ancien --tiled-warp (mouvement du milieu par point du flot de la tuile), retiré faute de gain
// (docs/interpolation.md) ; un bit libre, que plus rien ne lit.
// --warp-l2 : le warp TensorRT tient le flot du mouvement en L2 persistant (réserve de --l2-keep). Même
// image. Lu par le backend TensorRT ; sans effet ailleurs.
inline constexpr std::uint32_t kTrialWarpL2 = 1U << 3U;
// --rife-hwc8 : les blocs du RIFE fusionné construits en E/S HWC8 (--inputIOFormats=hwc8), leurs noyaux de
// colle écrivant et lisant ce format : plus de copies de reformatage (41 % du dernier bloc). Même calcul, au
// rangement près. Lu par le backend TensorRT ; sans effet ailleurs.
inline constexpr std::uint32_t kTrialRifeHwc8 = 1U << 4U;

// Ce que le lecteur passe à ajp_backend_create. Le backend garde ce qu'il lui faut de render,
// cache_dir et log ; render reste valide jusqu'à ajp_backend_destroy. refusal ne vaut que le
// temps de l'appel : le backend n'en garde pas l'adresse.
struct BackendInit {
  const GpuDevice* render = nullptr;  // appareil de rendu : le backend s'y lie ou refuse
  std::filesystem::path cache_dir;    // artefacts construits (moteurs, SPIR-V réglé)
  BackendLog log;  // appelé depuis n'importe quel thread du backend ; vide : rien n'est écrit
  // --profile-dispatches (option d'essai) : temps GPU et nombre de dispatchs par shader des réseaux, au
  // journal quand un réseau se décharge. Lu par le backend Vulkan ; sans effet ailleurs.
  bool profile_dispatches = false;
  // --filtered-warp (option d'essai, éteinte par défaut) : le warp des images intermédiaires lit images, flot
  // et masque par le filtre bilinéaire matériel (fraction sur 8 bits, écart au warp exact jusqu'au gradient
  // local / 512). Perte de qualité dont le gain n'est pas mesuré : elle ne sert qu'à la mesure de
  // docs/interpolation.md. Lu par le backend TensorRT ; sans effet ailleurs.
  bool filtered_warp = false;
  // Les options d'essai des chantiers, éteintes par défaut, pour la mesure : un bit par essai (kTrial…). Un
  // essai de plus prend un bit, sans monter kBackendAbi.
  std::uint32_t trials = 0;
  // Rempli quand ajp_backend_create rend nullptr : AbiMismatch, DeviceMismatch,
  // LibraryUnavailable… ; nullptr si l'appelant n'en veut pas.
  Refusal* refusal = nullptr;
};

// Les architectures CUDA où le FP8 (graphes Q/DQ) est permis : Ada (sm 8.9) et Hopper (sm 9.0). Blackwell
// (sm 10.x, sm 12.x : RTX 50) attend une mesure de qualité et de vitesse qui le justifie (intention du
// produit) ; avant Ada, pas d'unités FP8.
constexpr bool fp8_allowed(int sm_major, int sm_minor) {
  return (sm_major == 8 && sm_minor >= 9) || sm_major == 9;
}

// Ce que sait faire le backend sur l'appareil choisi.
struct BackendCaps {
  std::string name;  // "tensorrt", "vulkan", "metal"…
  std::string device_name;
  std::string driver_version;
  bool needs_host_sync = false;  // wait() attend côté hôte ; compté dans host_waits
  bool fp16 = false;
  bool int8 = false;
  bool fp8 =
      false;  // graphes Q/DQ FP8 (E4M3) par les unités matricielles : fp8_allowed (Ada, Hopper ; pas RTX 50)
  // Le SPAN entier en FP8 par noyaux CUTLASS (paquet « .span8 » à la place de l'ONNX) : sm 8.9 seulement,
  // dans un backend construit avec CUTLASS.
  bool span_fp8_cutlass = false;
  // Le SPAN FP16 natif par les mêmes noyaux (paquet « .span8 » au drapeau fp16, --span-native) : sm 8.6, 8.9
  // et 12.0 (SpanFp8Net::supported_fp16_device), jugé sur la carte au chargement.
  bool span_fp16_cutlass = false;
  // L'architecture de l'appareil (« sm_89 ») et la version des noyaux du SPAN FP8
  // (SpanFp8Net::kKernelsVersion) : un reçu v2 n'est pris que sur les mêmes (étape 3.7b).
  std::string gpu_arch;
  int span_fp8_kernels = 0;
  // RIFE coupé en six moteurs TensorRT et noyaux de colle (manifeste « .rifefused » à la place de l'ONNX,
  // kernels/spec/rife_fused.h).
  bool rife_fused = false;
  // DistilDRBA coupé en quatre moteurs TensorRT et noyaux de colle (manifeste « .drbafused » à la place de
  // l'ONNX, kernels/spec/drba_fused.h).
  bool drba_fused = false;
  bool matrix_units = false;  // unités matricielles utilisées
  // Voies qui se recouvrent vraiment sur l'appareil (gpu/lanes.h) ; 1 : use_lane ne change rien.
  int lanes = 1;
  // Images intermédiaires écrites directement dans le slot (KernelOps::warp_blend_ring…).
  bool ring_targets = false;
  // Emplacements RGB10A2 UNORM importés et écrits (store_ring, étape 5.5) ; sinon RGBA16 seul.
  bool ring_rgb10a2 = false;
  // KernelOps::write_yuv : une image écrite en YUV dans la mémoire CUDA de l'encodeur, sur le
  // contexte primaire de l'appareil de rendu (celui de NVENC).
  bool yuv_targets = false;
};

// Un modèle ONNX tel que livré, identifié par son hachage.
struct ModelSpec {
  std::string id;  // "2x_AnimeJaNai_HD_V3_Compact", "rife-4.26"…
  std::filesystem::path onnx_path;
  std::string sha256;  // du fichier ONNX
  TensorType precision = TensorType::F16;
};

// Formes d'entrée pour lesquelles construire le moteur (min, opt, max), NCHW.
struct ShapeSet {
  Shape min;
  Shape opt;
  Shape max;
};

// Rapport de progression d'une construction, appelé depuis le thread preparer.
// stop demande l'abandon : build() rend alors Refusal Cancelled au plus tôt.
struct Progress {
  std::function<void(float fraction, const std::string& stage)> report;
  std::stop_token stop;
};

// Une perte de précision consentie à la vitesse (repli Vulkan : sommes en float16 gardées par l'autotuner,
// dans la limite de la porte de fidélité). Le lecteur la montre à l'écran ; le journal ne suffit pas
// (C45-01). layers = 0 : rien à dire. Les valeurs non mesurées le sont dites : une perte non jugée contre le
// calcul en float, un gain repris du cache de l'autotuner.
struct PrecisionNotice {
  std::uint32_t layers = 0;
  bool loss_measured = false;
  float loss_db = 0.0F;  // PSNR des choix gardés contre le calcul en float
  bool gain_measured = false;
  float gain_percent = 0.0F;  // vitesse gagnée sur les couches mesurées, face aux sommes en float
  friend bool operator==(const PrecisionNotice&, const PrecisionNotice&) = default;
};

// Résultat sérialisé d'une construction (moteur TensorRT, SPIR-V réglé…), mis en cache.
struct Artifact {
  std::string cache_key;
  std::filesystem::path path;
  // Vrai quand le moteur vient du cache, pas de cette construction : s'il est refusé au chargement, il peut
  // être périmé (pilote changé) ou abîmé, et le reconstruire a un sens (C05-01).
  bool reused = false;
  PrecisionNotice reduced;  // vide hors du repli Vulkan
};

// La dérivation native du paquet « .span8 » (Backend::derive_span_fp8, docs/span-fp8-cutlass.md).
// Une image 8 bits : trois plans R, G, B de height × width octets, l'un après l'autre.
struct SpanStill {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;
};

struct SpanFp8Request {
  // Un paquet « .span8 » (gpu/span_fp8/package.h) dont seuls les poids comptent : ses échelles sont celles
  // que la calibration remplace.
  std::vector<std::byte> package;
  std::vector<SpanStill> calibration;  // les maxima des tenseurs se relèvent sur elles
  std::vector<SpanStill> validation;   // mises de côté : jugent la sortie FP8 contre la référence FP16
  std::vector<SpanStill> hard;         // cas durs (gpu/span_fp8/hard_cases.h) : jugés aussi, sans calibrer
  // Validation seule (étape 3.7b : une taille nouvelle) : les échelles et le drapeau acc16 du paquet tels
  // quels, sans calibration (calibration peut être vide), une seule variante jugée.
  bool validate_only = false;
  // Étape 3.6 : refus GpuMemoryExhausted si la VRAM libre est sous min_free_ratio × le besoin estimé de la
  // dérivation (0 : pas de contrôle ; la préparation lancée par un préréglage met 1,5).
  double min_free_ratio = 0.0;
  // C02-01 : le moteur TensorRT du SPAN FP16 que la lecture garde sans le paquet, construit pour
  // timing_width × timing_height (Backend::build) ; le backend le chronomètre comme le FP8, sur le même flux
  // et par tours intercalés (trt_fp16_ms). Absent : rien n'est mesuré contre lui (essais, outils).
  std::optional<Artifact> fp16;
  // C02-05 : la taille de travail de la lecture, où les temps se mesurent ; la qualité se juge sur la découpe
  // des images (au plus span_fp8::kCropMaxWidth × kCropMaxHeight). 0 : la taille des images de validation.
  int timing_width = 0;
  int timing_height = 0;
};

struct SpanFp8ImageScore {
  double psnr_db = 0.0;
  double ssim = 0.0;
  double error_p999 = 0.0;       // 99,9e centile de |FP8 − FP16| (étape 3.7a)
  double dark_error_p999 = 0.0;  // le même sur les pixels sombres de la référence
};

struct SpanFp8Derivation {
  std::array<float, 25> peaks{};  // maximum de |valeur| de chacun des 25 tenseurs quantifiés
  std::array<float, 25>
      scales{};  // les échelles E4M3 qu'on en tire (maximum / 448), celles qui ont été jugées
  std::vector<SpanFp8ImageScore> scores;  // une par image de validation
  double kernels_ms = 0.0;                // médiane du SPAN FP8 sur la taille des images
  // La même chose, conv_cat et to_pixel accumulant en f16 (étape 3.5a) ; vide si le backend ne l'essaie pas.
  std::vector<SpanFp8ImageScore> scores_acc16;
  double kernels_ms_acc16 = 0.0;
  // Les cas durs de la requête, pour chaque variante (étape 3.7a).
  std::vector<SpanFp8ImageScore> hard_scores;
  std::vector<SpanFp8ImageScore> hard_scores_acc16;
  std::string gpu_arch;     // « sm_89 » (étape 3.7b)
  int kernels_version = 0;  // SpanFp8Net::kKernelsVersion
  std::string reference;    // d'où vient la sortie de référence, « cutlass-fp16 »
  // C02-01 : une passe du SPAN FP16 de TensorRT (SpanFp8Request::fp16), mesurée avec les noyaux FP8 ; 0 :
  // non mesuré.
  double trt_fp16_ms = 0.0;
  std::string timed_size;  // « 2560x1440 » : la taille où les trois temps ont été pris (C02-05)
  // C02-04 : le maximum de |valeur| de chaque tenseur quantifié, relevé par la référence FP16 sur les images
  // de calibration et de validation (pas les cas durs) : span_fp8::judge_peaks le compare à 448 × scales.
  std::array<float, 25> validation_peaks{};
};

// Moteur chargé et prêt à recevoir enqueue(). Possédé par un Program. Son code vit dans la
// bibliothèque du backend : il est détruit avant le Backend qui l'a chargé, donc avant
// ajp_backend_destroy et le déchargement de la bibliothèque.
class Engine {
 public:
  virtual ~Engine() = default;
  virtual std::span<const std::string> input_names() const = 0;
  virtual std::span<const std::string> output_names() const = 0;
  // L'appel suivant reprend les mêmes tenseurs d'entrée que le précédent, dont seuls les plans du temps ont
  // changé (rife_time) : un moteur qui garde le travail des images de l'appel précédent le saute (D6, une
  // fenêtre préparée une fois, un appel par instant). Sans effet sur les autres.
  virtual void next_times_only() {}
  // L'appel suivant prend pour images A celles qui étaient les images B de l'appel précédent sur ce moteur,
  // inchangées (la paire suivante d'un même plan) : un moteur qui garde l'encodage de B le reprend pour A et
  // n'encode que la nouvelle B. Sans effet sur les autres.
  virtual void next_pair_follows() {}
  // P6.15 : l'appel suivant peut reprendre les blocs grossiers de RIFE de l'appel précédent (même moteur,
  // paire voisine, même instant) au lieu de les recalculer. Vrai si le moteur le fera ; faux : calcul entier.
  virtual bool next_warm() { return false; }
};

// Point de synchronisation sur la file du backend.
struct Fence {
  std::uintptr_t handle = 0;  // CUevent, VkSemaphore timeline…
  std::uint64_t value = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendCaps capabilities() const = 0;
  virtual DeviceMatch match(const DeviceUuid& render_device) const = 0;  // UUID du rendu
  virtual std::expected<Artifact, Refusal> build(const ModelSpec& model, const ShapeSet& shapes,
                                                 Progress& progress) = 0;
  virtual std::expected<std::unique_ptr<Engine>, Refusal> load(const Artifact& artifact) = 0;
  // Tenseur neuf sur l'appareil ; une dimension nulle, négative ou un total qui déborde est
  // refusé (ShapeInvalid), un type que le backend ne sait pas tenir aussi (FormatUnsupported).
  virtual std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_tensor(TensorType type,
                                                                             const Shape& shape) = 0;
  // Noyaux de la chaîne sur les tenseurs de ce backend, vivants autant que lui ; nullptr pour
  // un backend qui ne fait que construire et exécuter des moteurs (prepare).
  virtual KernelOps* kernels() = 0;
  // Les trois appels de la file rendent l'erreur de l'API (enqueueV3, cuEventRecord,
  // vkQueueSubmit, perte de l'appareil…) en refus nommé qui cite l'appel, jamais en silence.
  virtual std::expected<void, Refusal> enqueue(Engine& engine, std::span<const Tensor> inputs,
                                               std::span<Tensor> outputs) = 0;
  virtual std::expected<Fence, Refusal> signal() = 0;
  virtual std::expected<void, Refusal> wait(const Fence& fence) = 0;  // côté GPU, sauf si needs_host_sync
  // La voie des appels suivants de ce thread (gpu/lanes.h) : load() y pose les activations d'un
  // moteur, enqueue() et les noyaux s'enfilent sur sa file. Les dépendances entre voies sont
  // l'affaire du backend ; signal() suit le travail de toutes. Un backend à une voie l'ignore.
  virtual void use_lane(Lane /*lane*/) {}
  // Attend côté hôte la fin de tout le travail enfilé sur chaque voie : à appeler avant de détruire les
  // moteurs et les tampons qu'il lit encore (A30-01). Un backend dont les destructions attendent déjà
  // leur file l'ignore.
  virtual void drain() {}
  // Dérive le paquet « .span8 » du SPAN entier en FP8 (docs/span-fp8-cutlass.md) : relève sur les images de
  // calibration le maximum de chaque tenseur quantifié par une référence FP16 (mêmes convolutions CUTLASS,
  // sans quantification), en tire les échelles, puis juge le réseau FP8 contre cette référence sur les images
  // de validation (PSNR, SSIM). Synchrone, sur un thread à lui ; progress.report rend la fraction faite, et
  // progress.stop l'abandonne (Cancelled). Ne touche ni aux moteurs chargés ni aux fichiers. Refus : un
  // backend qui ne le sait pas (FormatUnsupported), GPU autre que sm 8.9, taille d'image que les noyaux ne
  // prennent pas, paquet illisible, mémoire (GpuMemoryExhausted), échec CUDA, maximum nul ou non fini.
  virtual std::expected<SpanFp8Derivation, Refusal> derive_span_fp8(const SpanFp8Request& /*request*/,
                                                                    Progress& /*progress*/) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "ce moteur ne dérive pas le SPAN FP8"});
  }
  // enqueue, puis l'image de outputs[0] dans l'emplacement de target, au protocole de write_ring (attente de
  // wait_value, signal de signal_value). Un backend qui sait écrire l'emplacement depuis le moteur même (le
  // SPAN FP8 du backend TensorRT, étape 3.4b) le fait sans écriture à part ; outputs[0] est écrit aussi.
  virtual std::expected<void, Refusal> enqueue_ring(Engine& engine, std::span<const Tensor> inputs,
                                                    std::span<Tensor> outputs, const RingTarget& target) {
    if (outputs.empty() || target.slot == nullptr)
      return std::unexpected(
          Refusal{RefusalCode::InputMismatch, "enqueue_ring : sortie ou emplacement absents"});
    KernelOps* ops = kernels();
    if (ops == nullptr)
      return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "enqueue_ring : backend sans noyaux"});
    if (auto ok = enqueue(engine, inputs, outputs); !ok) return ok;
    return ops->write_ring(outputs[0], *target.slot, target.sync);
  }
};

// Empreinte binaire de la frontière : FNV-1a 64 bits sur une suite de nombres.
constexpr std::uint64_t layout_fingerprint(std::initializer_list<std::uint64_t> values) {
  std::uint64_t hash = 0xCBF2'9CE4'8422'2325ULL;
  for (const std::uint64_t value : values) {
    for (unsigned byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (8U * byte)) & 0xFFU;
      hash *= 0x0000'0100'0000'01B3ULL;
    }
  }
  return hash;
}

// Taille et alignement de chaque type, dans l'ordre : ce qui fixe leur disposition en mémoire.
template <class... T>
constexpr std::uint64_t shape_fingerprint() {
  return layout_fingerprint({((std::uint64_t{sizeof(T)} << 16U) | std::uint64_t{alignof(T)})...});
}

// La bibliothèque standard et ce qui change sa disposition : libstdc++ et son ABI de
// std::string, libc++ et son numéro d'ABI, la STL de MSVC, ses itérateurs de débogage et son
// CRT (/MD ou /MT : deux tas différents).
#if defined(_LIBCPP_VERSION)
inline constexpr std::uint64_t kStdLibrary = 0x200'0000 + _LIBCPP_ABI_VERSION;
#elif defined(__GLIBCXX__)
inline constexpr std::uint64_t kStdLibrary = 0x100'0000 + _GLIBCXX_USE_CXX11_ABI;
#elif defined(_MSVC_STL_VERSION)
#if defined(_DLL)
inline constexpr std::uint64_t kStdLibrary = 0x300'0100 + _ITERATOR_DEBUG_LEVEL;
#else
inline constexpr std::uint64_t kStdLibrary = 0x300'0000 + _ITERATOR_DEBUG_LEVEL;
#endif
#else
inline constexpr std::uint64_t kStdLibrary = 0;
#endif

// Empreinte de tout ce qui traverse la frontière. kBackendAbi date le contrat ; l'empreinte
// trahit ce que kBackendAbi laisse passer : une bibliothèque construite contre d'autres
// en-têtes sans changement de version, ou avec une autre bibliothèque standard.
// AJP_BOUNDARY_HASH : le texte des en-têtes de la frontière (cmake/BoundaryHash.cmake), posé par ajp::gpu. Il
// voit ce que les tailles ne voient pas : l'ordre des virtuelles, une signature changée.
#ifndef AJP_BOUNDARY_HASH
#error "AJP_BOUNDARY_HASH manque : lier ajp::gpu"
#endif
inline constexpr std::uint64_t kBoundaryHash = AJP_BOUNDARY_HASH;

inline constexpr std::uint64_t kBackendLayout = layout_fingerprint({
    static_cast<std::uint64_t>(kBackendAbi),
    kBoundaryHash,
    kStdLibrary,
    sizeof(void*),
    shape_fingerprint<std::string, std::vector<std::string>, std::filesystem::path, std::function<void()>,
                      std::shared_ptr<const FrameData>, std::unique_ptr<Engine>, std::stop_token>(),
    shape_fingerprint<Refusal, std::expected<void, Refusal>, std::expected<Tensor, Refusal>,
                      std::expected<Artifact, Refusal>, std::expected<std::unique_ptr<Engine>, Refusal>,
                      std::expected<std::unique_ptr<TensorBuffer>, Refusal>, std::expected<Fence, Refusal>>(),
    shape_fingerprint<GpuDevice, VulkanDevice, QueueFamily, BackendInit, BackendCaps, ModelSpec, ShapeSet,
                      Progress, Artifact, Fence, Tensor, Shape>(),
    shape_fingerprint<DecodedFrame, ColorInfo, HdrMetadata, VulkanImageAccess,
                      std::expected<VulkanImageAccess, Refusal>>(),
    shape_fingerprint<IngestSpec, Motion, RingWrite, PairScores, RingSlotDesc, ExportedHandle, RingTarget,
                      YuvTarget>(),
    shape_fingerprint<SpanStill, SpanFp8Request, SpanFp8ImageScore, SpanFp8Derivation,
                      std::expected<SpanFp8Derivation, Refusal>>(),
    // Ce que KernelOps rend au lecteur (A21-01) : les statistiques du grade et la mémoire de l'appareil.
    shape_fingerprint<kernels::FrameStats, std::optional<kernels::FrameStats>,
                      std::expected<std::optional<kernels::FrameStats>, Refusal>, KernelOps::DeviceMemory,
                      GpuStageSummary, std::vector<GpuStageSummary>>(),
});

// Ce que le chargeur vérifie avant ajp_backend_create, avec ajp_backend_abi() et
// ajp_backend_layout() lus dans la bibliothèque. Refus : AbiMismatch, qui cite les valeurs.
inline std::expected<void, Refusal> check_backend_abi(int abi, std::uint64_t layout) {
  if (abi != kBackendAbi) {
    return std::unexpected(
        Refusal{RefusalCode::AbiMismatch,
                std::format("backend d'interface {}, le lecteur attend {}", abi, kBackendAbi)});
  }
  if (layout != kBackendLayout) {
    return std::unexpected(
        Refusal{RefusalCode::AbiMismatch,
                std::format("empreinte binaire {:016x}, le lecteur attend {:016x} : en-têtes ou bibliothèque "
                            "standard différents",
                            layout, kBackendLayout)});
  }
  return {};
}

}  // namespace ajp::gpu

// Points d'entrée exportés par chaque bibliothèque de backend, tous noexcept : sous MSVC, une
// exception sortie d'une fonction extern "C" est un comportement indéfini chez l'appelant.
// Le chargeur lit d'abord ajp_backend_abi() et ajp_backend_layout() (check_backend_abi) :
// une autre version ou une autre empreinte est refusée (AbiMismatch) avant de passer un
// BackendInit dont la forme pourrait différer. ajp_backend_create rend nullptr si abi
// diffère de kBackendAbi, si init est nul ou si l'appareil ne convient pas, et dit pourquoi
// dans init->refusal. ajp_backend_destroy vient après la destruction de chaque Engine et
// TensorBuffer du backend.
//
// Les déclarations portent l'export de leurs définitions : MSVC refuse une définition
// __declspec(dllexport) d'une fonction déclarée sans (C2375). Dans le lecteur, qui ne les définit
// ni ne les appelle (dlsym), elles ne produisent rien.
#if defined(_WIN32)
#define AJP_BACKEND_EXPORT __declspec(dllexport)
#else
#define AJP_BACKEND_EXPORT __attribute__((visibility("default")))
#endif
extern "C" AJP_BACKEND_EXPORT int ajp_backend_abi() noexcept;               // kBackendAbi
extern "C" AJP_BACKEND_EXPORT std::uint64_t ajp_backend_layout() noexcept;  // kBackendLayout
extern "C" AJP_BACKEND_EXPORT ajp::gpu::Backend* ajp_backend_create(
    int abi, const ajp::gpu::BackendInit* init) noexcept;
extern "C" AJP_BACKEND_EXPORT void ajp_backend_destroy(ajp::gpu::Backend* backend) noexcept;

namespace ajp::gpu {

// Types des points d'entrée, pour les adresses lues par dlsym ou GetProcAddress.
using BackendAbiFn = decltype(&ajp_backend_abi);
using BackendLayoutFn = decltype(&ajp_backend_layout);
using BackendCreateFn = decltype(&ajp_backend_create);
using BackendDestroyFn = decltype(&ajp_backend_destroy);

}  // namespace ajp::gpu
