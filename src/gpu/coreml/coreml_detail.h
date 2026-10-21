// Partagé par les fichiers du runtime Core ML (coreml_runtime.cpp, coreml_measure.cpp) : les
// unités de calcul, le manifeste de l'artefact, les types de tenseurs d'ONNX Runtime, la mesure
// des répartitions. Rien d'exposé hors du module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/coreml/coreml_units.h"
#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::coreml_rt {

std::unexpected<Refusal> refuse(RefusalCode code, std::string text);

// Le cadre des images de l'interpolation d'Apple (coreml_vt.h), à la place d'un réseau.
struct VtFrames {
  int width = 0;
  int height = 0;
};

// Le manifeste de l'artefact : le fichier ONNX, une ligne par dimension symbolique fixée, les
// unités retenues par la mesure (absentes d'un artefact plus ancien), et si le réseau ouvert est
// la copie <artefact>.onnx, réécrite à l'interface : dans le type des tenseurs de la chaîne, aux
// formes de la construction (onnx_of). Pour le modèle d'un interpolateur du système
// (gpu/system_interpolator.h), le cadre de l'interpolation d'Apple, sans session ONNX Runtime.
struct Manifest {
  std::filesystem::path onnx;
  std::vector<vk::net::DimOverride> dims;
  std::optional<Units> units;
  bool io = false;
  std::optional<VtFrames> videotoolbox;
};

std::filesystem::path cache_of(const std::filesystem::path& artifact);  // <artefact>.mlcache
std::filesystem::path onnx_of(const std::filesystem::path& artifact);   // <artefact>.onnx

ONNXTensorElementDataType element_of(TensorType type);
std::optional<TensorType> type_of(ONNXTensorElementDataType type);
std::size_t element_bytes(TensorType type);

// Ouvre une session du manifeste sur ces unités (CoremlRuntime::open).
using OpenSession = std::function<std::expected<Ort::Session, Refusal>(Units)>;

// Chaque répartition sur les mêmes entrées, le GPU d'abord, qui sert de référence : une
// répartition dont la sortie s'en écarte est écartée. Le Neural Engine s'il reste dans la
// tolérance du meilleur temps, sinon la plus rapide. Un réseau qui déforme reste au GPU ;
// rien de mesurable : les unités par défaut (units_of).
std::expected<Units, Refusal> choose_units(const Manifest& manifest,
                                           std::span<const vk::net::InputShape> inputs, Progress& progress,
                                           const OpenSession& open);

// Les unités d'un artefact : celles de la mesure ; sans elle (un artefact plus ancien, une
// mesure impossible), le GPU pour un réseau qui déforme (GridSample, que le Neural Engine
// n'exécute pas), sinon toutes.
std::expected<Units, Refusal> units_of(const Manifest& manifest);

}  // namespace ajp::gpu::coreml_rt
