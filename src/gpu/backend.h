// Interface d'un backend d'inférence. Chaque backend est une bibliothèque
// chargée à l'exécution (ajp_backend_<nom>.so / .dll), installée seulement si
// l'utilisateur l'a choisie : le lecteur démarre sans CUDA, ROCm ni TensorRT.
//
// La frontière passe des types C++ (std::string, std::function, std::expected…) : un
// backend se construit avec la même bibliothèque standard que le lecteur, ce que
// l'empreinte kBackendLayout vérifie au chargement. Aucune exception ne la traverse : chaque
// erreur revient en Refusal, et les points d'entrée extern "C" sont noexcept.
#pragma once

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
inline constexpr int kBackendAbi = 11;

using DeviceUuid = ajp::DeviceUuid;

enum class DeviceMatch : std::uint8_t { Same, Different, Unknown };

// Un backend journalise par BackendInit::log, que app relie à log_channel("backend")
// (core/log.h) ; ses canaux y vont par install_module_log_sinks. Le rappel ne lève jamais.
enum class BackendLogLevel : std::uint8_t { Debug, Info, Warn, Error };
using BackendLog = std::function<void(BackendLogLevel level, std::string_view message)>;

// Ce que le lecteur passe à ajp_backend_create. Le backend garde ce qu'il lui faut de render,
// cache_dir et log ; render reste valide jusqu'à ajp_backend_destroy. refusal ne vaut que le
// temps de l'appel : le backend n'en garde pas l'adresse.
struct BackendInit {
  const GpuDevice* render = nullptr;  // appareil de rendu : le backend s'y lie ou refuse
  std::filesystem::path cache_dir;    // artefacts construits (moteurs, SPIR-V réglé)
  BackendLog log;  // appelé depuis n'importe quel thread du backend ; vide : rien n'est écrit
  // Rempli quand ajp_backend_create rend nullptr : AbiMismatch, DeviceMismatch,
  // LibraryUnavailable… ; nullptr si l'appelant n'en veut pas.
  Refusal* refusal = nullptr;
};

// Ce que sait faire le backend sur l'appareil choisi.
struct BackendCaps {
  std::string name;  // "tensorrt", "vulkan", "metal"…
  std::string device_name;
  std::string driver_version;
  bool needs_host_sync = false;  // wait() attend côté hôte ; compté dans host_waits
  bool fp16 = false;
  bool int8 = false;
  bool fp8 = false;  // graphes Q/DQ FP8 (E4M3) calculés par les unités matricielles : Ada (sm 8.9) et après
  // Le SPAN entier en FP8 par noyaux CUTLASS (paquet « .span8 » à la place de l'ONNX) : sm 8.9 seulement,
  // dans un backend construit avec CUTLASS.
  bool span_fp8_cutlass = false;
  bool matrix_units = false;                // unités matricielles utilisées
  std::vector<PixelFormat> import_formats;  // formats acceptés par import()
  // Voies qui se recouvrent vraiment sur l'appareil (gpu/lanes.h) ; 1 : use_lane ne change rien.
  int lanes = 1;
  // Images intermédiaires écrites directement dans le slot (KernelOps::warp_blend_ring…).
  bool ring_targets = false;
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

// Résultat sérialisé d'une construction (moteur TensorRT, SPIR-V réglé…), mis en cache.
struct Artifact {
  std::string cache_key;
  std::filesystem::path path;
};

// Moteur chargé et prêt à recevoir enqueue(). Possédé par un Program. Son code vit dans la
// bibliothèque du backend : il est détruit avant le Backend qui l'a chargé, donc avant
// ajp_backend_destroy et le déchargement de la bibliothèque.
class Engine {
 public:
  virtual ~Engine() = default;
  virtual std::span<const std::string> input_names() const = 0;
  virtual std::span<const std::string> output_names() const = 0;
};

// Image d'un autre module (surface NVDEC copiée, VkImage du ring) à importer sans copie.
struct ExternalImage {
  FrameMemory memory = FrameMemory::Host;
  PixelFormat format = PixelFormat::Rgba16Unorm;
  int width = 0;
  int height = 0;
  std::uintptr_t handle = 0;  // CUdeviceptr, VkImage, mémoire exportée…
  std::uintptr_t ready = 0;   // événement ou sémaphore à attendre
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
  // Refuse un format inattendu au lieu de le convertir en silence.
  virtual std::expected<Tensor, Refusal> import(const ExternalImage& image) = 0;
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
inline constexpr std::uint64_t kBackendLayout = layout_fingerprint({
    static_cast<std::uint64_t>(kBackendAbi),
    kStdLibrary,
    sizeof(void*),
    shape_fingerprint<std::string, std::vector<std::string>, std::filesystem::path, std::function<void()>,
                      std::shared_ptr<const FrameData>, std::unique_ptr<Engine>, std::stop_token>(),
    shape_fingerprint<Refusal, std::expected<void, Refusal>, std::expected<Tensor, Refusal>,
                      std::expected<Artifact, Refusal>, std::expected<std::unique_ptr<Engine>, Refusal>,
                      std::expected<std::unique_ptr<TensorBuffer>, Refusal>, std::expected<Fence, Refusal>>(),
    shape_fingerprint<GpuDevice, VulkanDevice, QueueFamily, BackendInit, BackendCaps, ModelSpec, ShapeSet,
                      Progress, Artifact, ExternalImage, Fence, Tensor, Shape>(),
    shape_fingerprint<DecodedFrame, ColorInfo, HdrMetadata, VulkanImageAccess,
                      std::expected<VulkanImageAccess, Refusal>>(),
    shape_fingerprint<IngestSpec, Motion, RingWrite, PairScores, std::optional<PairScores>, RingSlotDesc,
                      ExportedHandle, RingTarget, YuvTarget>(),
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
