// Réseaux confiés à un autre runtime (MIGraphX sur AMD, OpenVINO sur Intel) derrière le backend
// Vulkan : la chaîne (ingestion, rife_input, warp, ring…) garde les noyaux Vulkan sur l'appareil
// de rendu, seul le réseau passe par le runtime du fabricant.
//
// Les tenseurs des réseaux vivent alors en mémoire hôte visible du GPU (MemoryUse::Readback,
// en cache côté hôte) : le runtime les lit et les écrit par leur projection, après l'attente sur
// l'hôte de la file Vulkan (BackendCaps::needs_host_sync), et rend la main une fois ses sorties
// écrites. Une copie de plus qu'une interop de mémoire, mais rien n'est partagé entre les deux
// pilotes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/vulkan/compute/caps.h"

namespace ajp::gpu::vk::net {

// Un tenseur d'entrée ou de sortie, vu par l'hôte : NCHW dense, dans le type du réseau.
struct HostBinding {
  std::byte* data = nullptr;
  std::uint64_t bytes = 0;
  TensorType type = TensorType::F16;
  Shape shape;
};

// La forme fixée d'une entrée, par son nom.
struct InputShape {
  std::string name;
  std::vector<std::int64_t> dims;
};

// Un réseau chargé par le runtime.
class RuntimeNet {
 public:
  virtual ~RuntimeNet() = default;
  virtual std::span<const std::string> input_names() const = 0;
  virtual std::span<const std::string> output_names() const = 0;
  // Calcule les sorties, écrites avant le retour. Refus : celui du runtime, nommé.
  virtual std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                           std::span<const HostBinding> outputs) = 0;
};

class Runtime {
 public:
  virtual ~Runtime() = default;
  virtual std::string_view name() const = 0;  // BackendCaps::name : « migraphx », « openvino »
  virtual std::string device_name() const = 0;
  virtual std::string version() const = 0;  // entre dans la clé des artefacts
  virtual std::string_view extension() const = 0;
  // Compile le modèle, ses entrées fixées à ces formes (vide : celles du fichier, toutes
  // statiques), et l'écrit à path. Refus : ceux du runtime, Cancelled.
  virtual std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                             const std::filesystem::path& path, Progress& progress) = 0;
  virtual std::expected<std::unique_ptr<RuntimeNet>, Refusal> load(const std::filesystem::path& path) = 0;
};

// Crée le runtime sur l'appareil de cette description Vulkan (même UUID, sinon le premier du
// fabricant). Refus DeviceMismatch : appareil d'un autre fabricant, que le backend Vulkan seul
// sert ; LibraryUnavailable, DeviceUnavailable : runtime absent ou sans appareil.
using RuntimeFactory = std::function<std::expected<std::shared_ptr<Runtime>, Refusal>(const DeviceCaps&)>;

// Noms des entrées et des sorties d'un réseau, rangés à côté de son artefact (un par ligne,
// « in » ou « out » puis le nom), pour les runtimes dont l'artefact ne les garde pas.
std::expected<void, Refusal> write_io_names(const std::filesystem::path& artifact,
                                            std::span<const std::string> inputs,
                                            std::span<const std::string> outputs);
struct IoNames {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};
std::expected<IoNames, Refusal> read_io_names(const std::filesystem::path& artifact);

// Copie count éléments de from vers to en convertissant entre F32 et F16 (même type : une copie).
// Refus FormatUnsupported : un type entier différent de l'autre.
std::expected<void, Refusal> copy_converted(const std::byte* from, TensorType from_type, std::byte* to,
                                            TensorType to_type, std::size_t count);

// Les noms de l'interface d'un fichier ONNX, lus par gpu_ir.
std::expected<IoNames, Refusal> onnx_io_names(const std::filesystem::path& onnx);

// Les dimensions symboliques des entrées d'un fichier ONNX fixées par ces formes : (symbole,
// valeur), une fois chacun. Refus InputMismatch : un symbole qui recevrait deux valeurs.
using DimOverride = std::pair<std::string, std::int64_t>;
std::expected<std::vector<DimOverride>, Refusal> symbolic_overrides(const std::filesystem::path& onnx,
                                                                    std::span<const InputShape> inputs);

}  // namespace ajp::gpu::vk::net
