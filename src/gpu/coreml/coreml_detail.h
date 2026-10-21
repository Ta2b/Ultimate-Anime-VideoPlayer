// Partagé par les fichiers du runtime Core ML (coreml_runtime.cpp, coreml_measure.cpp) : les
// unités de calcul, le manifeste de l'artefact, les types de tenseurs d'ONNX Runtime, la mesure
// des répartitions. Rien d'exposé hors du module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/coreml/coreml_direct.h"
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
// unités retenues par la mesure (absentes d'un artefact plus ancien), si le réseau ouvert est
// la copie <artefact>.onnx, réécrite à l'interface : dans le type des tenseurs de la chaîne, aux
// formes de la construction (onnx_of), et si Core ML l'exécute directement, sans ONNX Runtime
// (coreml_direct.h, direct_of). Pour le modèle d'un interpolateur du système
// (gpu/system_interpolator.h), le cadre de l'interpolation d'Apple, sans session ONNX Runtime.
struct Manifest {
  std::filesystem::path onnx;
  std::vector<vk::net::DimOverride> dims;
  std::optional<Units> units;
  bool io = false;
  std::optional<VtFrames> videotoolbox;
  bool direct = false;
};

std::filesystem::path cache_of(const std::filesystem::path& artifact);   // <artefact>.mlcache
std::filesystem::path onnx_of(const std::filesystem::path& artifact);    // <artefact>.onnx
std::filesystem::path direct_of(const std::filesystem::path& artifact);  // <artefact>.mlmodelc

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

// Ouvre le modèle compilé en direct (open_direct) pour ces noms d'entrées et de sorties.
using OpenDirect = std::function<std::expected<std::unique_ptr<DirectModel>, Refusal>(
    std::span<const std::string> inputs, std::span<const std::string> outputs)>;

// Écart permis entre Core ML direct et ONNX Runtime, qui exécutent le même modèle compilé sur les
// mêmes unités : la moyenne, et le pire pixel (une ligne mal recopiée passerait la moyenne).
inline constexpr double kDirectMaxMeanError = 1e-3;
inline constexpr double kDirectMaxError = 2e-2;

// L'écart entre la session et Core ML direct sur l'image de test de choose_units, deux passages
// de suite (le second sur les enveloppes gardées) : la pire sortie, en moyenne et au pire pixel.
// Refus : ceux d'open, de Core ML, d'ONNX Runtime ; FormatUnsupported : d'autres types ou formes.
struct DirectError {
  double mean = 0.0;
  double max = 0.0;
};
std::expected<DirectError, Refusal> direct_error(Ort::Session& session,
                                                 std::span<const vk::net::InputShape> inputs,
                                                 const OpenDirect& open);

// Les unités d'un artefact : celles de la mesure ; sans elle (un artefact plus ancien, une
// mesure impossible), le GPU pour un réseau qui déforme (GridSample, que le Neural Engine
// n'exécute pas), sinon toutes.
std::expected<Units, Refusal> units_of(const Manifest& manifest);

}  // namespace ajp::gpu::coreml_rt
