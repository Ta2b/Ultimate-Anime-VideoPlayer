// Interface d'un backend d'inférence. Chaque backend est une bibliothèque
// chargée à l'exécution (ajp_backend_<nom>.so / .dll), installée seulement si
// l'utilisateur l'a choisie : le lecteur démarre sans CUDA, ROCm ni TensorRT.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "gpu/tensor.h"

namespace ajp::gpu {

// Version de cette interface. Un backend d'une autre version est refusé au chargement.
// 2 : BackendInit à la création, noyaux de la chaîne (kernels()), alloc_tensor().
inline constexpr int kBackendAbi = 2;

using DeviceUuid = ajp::DeviceUuid;

enum class DeviceMatch : std::uint8_t { Same, Different, Unknown };

// Un backend a sa propre copie de spdlog : il journalise par BackendInit::log, que app relie
// à log_channel("backend") (core/log.h).
enum class BackendLogLevel : std::uint8_t { Debug, Info, Warn, Error };
using BackendLog = std::function<void(BackendLogLevel level, std::string_view message)>;

// Ce que le lecteur passe à ajp_backend_create, qui en garde une copie. render reste valide
// jusqu'à ajp_backend_destroy ; refusal, le temps de l'appel.
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
  bool matrix_units = false;                // unités matricielles utilisées
  std::vector<PixelFormat> import_formats;  // formats acceptés par import()
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

// Moteur chargé et prêt à recevoir enqueue(). Possédé par un Program.
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
  virtual DeviceMatch match(const DeviceUuid& render_device) const = 0;  // UUID ou LUID du rendu
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
  virtual void enqueue(Engine& engine, std::span<const Tensor> inputs, std::span<Tensor> outputs) = 0;
  virtual Fence signal() = 0;
  virtual void wait(const Fence& fence) = 0;  // côté GPU, sauf si needs_host_sync
};

}  // namespace ajp::gpu

// Points d'entrée exportés par chaque bibliothèque de backend. Le chargeur lit d'abord
// ajp_backend_abi() : une autre version est refusée (AbiMismatch) avant de passer un
// BackendInit dont la forme pourrait différer. ajp_backend_create rend nullptr si abi
// diffère de kBackendAbi, si init est nul ou si l'appareil ne convient pas, et dit pourquoi
// dans init->refusal.
extern "C" int ajp_backend_abi();
extern "C" ajp::gpu::Backend* ajp_backend_create(int abi, const ajp::gpu::BackendInit* init);
extern "C" void ajp_backend_destroy(ajp::gpu::Backend* backend);
