// Partagé par les fichiers du runtime MIGraphX et VitisAI de Windows ML (winml_runtime.cpp,
// winml_manifest.cpp) et leurs tests : le manifeste de l'artefact et le choix de l'appareil. Rien d'exposé
// hors du module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::winml_rt {

// Le manifeste de l'artefact : le fichier ONNX, une ligne par dimension symbolique fixée, si le
// réseau ouvert est la copie <artefact>.onnx (interface dans le type des tenseurs de la chaîne,
// entrées à la forme de la construction), si MIGraphX calcule en half (quantize_fp16), et si le
// réseau tourne sur le NPU (VitisAI, sa copie <artefact>.npu.onnx en FLOAT), mesuré contre le GPU.
struct Manifest {
  std::filesystem::path onnx;
  std::vector<vk::net::DimOverride> dims;
  bool io = false;
  bool fp16 = false;
  bool npu = false;
};

std::filesystem::path cache_of(const std::filesystem::path& artifact);      // <artefact>.mxcache
std::filesystem::path onnx_of(const std::filesystem::path& artifact);       // <artefact>.onnx
std::filesystem::path npu_onnx_of(const std::filesystem::path& artifact);   // <artefact>.npu.onnx
std::filesystem::path npu_cache_of(const std::filesystem::path& artifact);  // <artefact>.vaicache

// Refus CacheIo.
std::expected<void, Refusal> write_manifest(const std::filesystem::path& path, const Manifest& manifest);
std::expected<Manifest, Refusal> read_manifest(const std::filesystem::path& path);

// Un appareil que MIGraphX propose, par ses identifiants PCI.
struct DeviceId {
  std::uint32_t vendor = 0;
  std::uint32_t device = 0;
};

// Celui de la chaîne Vulkan (mêmes identifiants), sinon le premier GPU AMD proposé ; aucun : rien.
std::optional<std::size_t> pick_device(std::span<const DeviceId> offered, DeviceId chain);

}  // namespace ajp::gpu::winml_rt
