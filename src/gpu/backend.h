// Interface d'un backend d'inférence. Chaque backend est une bibliothèque
// chargée à l'exécution (ajp_backend_<nom>.so / .dll), installée seulement si
// l'utilisateur l'a choisie : le lecteur démarre sans CUDA, ROCm ni TensorRT.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/frame.h"
#include "core/types.h"

namespace ajp::gpu {

// Version de cette interface. Un backend d'une autre version est refusé au chargement.
inline constexpr int kBackendAbi = 1;

using DeviceUuid = std::array<std::uint8_t, 16>;

enum class DeviceMatch : std::uint8_t { Same, Different, Unknown };

enum class TensorType : std::uint8_t { F32, F16, U8, I8 };

// Ce que sait faire le backend sur l'appareil choisi.
struct BackendCaps {
  std::string name;                       // "tensorrt", "vulkan", "metal"…
  std::string device_name;
  std::string driver_version;
  bool needs_host_sync = false;           // wait() attend côté hôte ; compté dans host_waits
  bool fp16 = false;
  bool int8 = false;
  bool matrix_units = false;              // unités matricielles utilisées
  std::vector<PixelFormat> import_formats;  // formats acceptés par import()
};

// Un modèle ONNX tel que livré, identifié par son hachage.
struct ModelSpec {
  std::string id;                    // "2x_AnimeJaNai_HD_V3_Compact", "rife-4.26"…
  std::filesystem::path onnx_path;
  std::string sha256;                // du fichier ONNX
  TensorType precision = TensorType::F16;
};

// Formes d'entrée pour lesquelles construire le moteur (min, opt, max), NCHW.
struct Shape {
  std::array<std::int64_t, 4> dims{};
};
struct ShapeSet {
  Shape min;
  Shape opt;
  Shape max;
};

// Rapport de progression d'une construction, appelé depuis le thread preparer.
struct Progress {
  std::function<void(float fraction, const std::string& stage)> report;
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

// Tenseur résident sur l'appareil du backend. Ne possède pas forcément sa mémoire.
struct Tensor {
  TensorType type = TensorType::F16;
  Shape shape;
  std::uintptr_t handle = 0;  // CUdeviceptr, VkBuffer…
  std::uint64_t offset = 0;
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
  virtual void enqueue(Engine& engine, std::span<const Tensor> inputs, std::span<Tensor> outputs) = 0;
  virtual Fence signal() = 0;
  virtual void wait(const Fence& fence) = 0;  // côté GPU, sauf si needs_host_sync
};

}  // namespace ajp::gpu

// Point d'entrée exporté par chaque bibliothèque de backend. Renvoie nullptr si
// abi diffère de kBackendAbi ou si l'appareil ne convient pas.
extern "C" ajp::gpu::Backend* ajp_backend_create(int abi);
extern "C" void ajp_backend_destroy(ajp::gpu::Backend* backend);
