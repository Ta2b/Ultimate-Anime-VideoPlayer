// La mesure des répartitions Core ML (coreml_detail.h) : des entrées reproductibles, quelques
// inférences par répartition, la médiane de leurs temps et la première sortie pour comparer.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <format>
#include <limits>

#include "gpu/coreml/coreml_detail.h"

namespace ajp::gpu::coreml_rt {
namespace {

using vk::net::InputShape;

// Mesure d'une session : inférences de chauffe, puis médiane des suivantes.
constexpr int kWarmupRuns = 2;
constexpr std::size_t kTimedRuns = 5;
// Le Neural Engine garde un réseau s'il ne coûte pas plus que ce facteur du meilleur temps : il
// laisse le GPU, partagé avec la chaîne MoltenVK, RIFE et le rendu, à tout le reste.
constexpr double kAneTolerance = 1.25;
// Écart moyen permis entre deux répartitions sur la même entrée (images entre 0 et 1).
constexpr double kMaxMeanError = 0.01;

// Des entrées fixes et reproductibles entre 0 et 1, aux formes fixées par build (sinon celles du
// modèle), les mêmes pour chaque répartition mesurée.
struct Probe {
  std::vector<std::string> inputs;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<TensorType> types;
  std::vector<std::vector<std::byte>> data;
  std::vector<std::string> outputs;
};

std::expected<Probe, Refusal> make_probe(Ort::Session& session, std::span<const InputShape> fixed) {
  Probe probe;
  Ort::AllocatorWithDefaultOptions allocator;
  for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
    std::string name = session.GetInputNameAllocated(i, allocator).get();
    const Ort::TypeInfo info = session.GetInputTypeInfo(i);
    const auto tensor = info.GetTensorTypeAndShapeInfo();
    const std::optional<TensorType> type = type_of(tensor.GetElementType());
    if (!type)
      return refuse(RefusalCode::FormatUnsupported, std::format("Core ML : {} n'est pas flottant", name));
    std::vector<std::int64_t> shape = tensor.GetShape();
    if (const auto at = std::ranges::find(fixed, name, &InputShape::name); at != fixed.end())
      shape = at->dims;
    std::size_t count = 1;
    for (const std::int64_t d : shape) {
      if (d <= 0)
        return refuse(RefusalCode::ShapeInvalid, std::format("Core ML : {}, dimension libre", name));
      count *= static_cast<std::size_t>(d);
    }
    std::vector<float> values(count);
    for (std::size_t k = 0; k < count; ++k)
      values[k] = static_cast<float>((k * 2654435761U) % 1024U) / 1023.0F;
    std::vector<std::byte> bytes(count * element_bytes(*type));
    if (auto ok = vk::net::copy_converted(reinterpret_cast<const std::byte*>(values.data()), TensorType::F32,
                                          bytes.data(), *type, count);
        !ok)
      return std::unexpected(ok.error());
    probe.inputs.push_back(std::move(name));
    probe.shapes.push_back(std::move(shape));
    probe.types.push_back(*type);
    probe.data.push_back(std::move(bytes));
  }
  for (std::size_t i = 0; i < session.GetOutputCount(); ++i)
    probe.outputs.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
  return probe;
}

struct Timing {
  double ms = 0.0;               // médiane d'une inférence
  std::vector<float> first_out;  // première sortie, pour comparer deux répartitions
};

// Lève les exceptions d'ONNX Runtime.
std::expected<Timing, Refusal> measure(Ort::Session& session, Probe& probe) {
  const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> values;
  std::vector<const char*> in_names;
  for (std::size_t i = 0; i < probe.inputs.size(); ++i) {
    values.push_back(Ort::Value::CreateTensor(memory, probe.data[i].data(), probe.data[i].size(),
                                              probe.shapes[i].data(), probe.shapes[i].size(),
                                              element_of(probe.types[i])));
    in_names.push_back(probe.inputs[i].c_str());
  }
  std::vector<const char*> out_names;
  for (const std::string& name : probe.outputs) out_names.push_back(name.c_str());
  const auto run = [&] {
    return session.Run(Ort::RunOptions{nullptr}, in_names.data(), values.data(), values.size(),
                       out_names.data(), out_names.size());
  };
  for (int i = 0; i < kWarmupRuns; ++i) static_cast<void>(run());
  std::array<double, kTimedRuns> times{};
  std::vector<Ort::Value> outputs;
  for (double& t : times) {
    const auto start = std::chrono::steady_clock::now();
    outputs = run();
    t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  std::ranges::nth_element(times, times.begin() + static_cast<std::ptrdiff_t>(kTimedRuns / 2));
  Timing timing{.ms = times[kTimedRuns / 2], .first_out = {}};
  if (outputs.empty()) return timing;
  const auto info = outputs.front().GetTensorTypeAndShapeInfo();
  const std::optional<TensorType> type = type_of(info.GetElementType());
  if (!type) return timing;
  timing.first_out.resize(info.GetElementCount());
  if (auto ok = vk::net::copy_converted(static_cast<const std::byte*>(outputs.front().GetTensorRawData()),
                                        *type, reinterpret_cast<std::byte*>(timing.first_out.data()),
                                        TensorType::F32, timing.first_out.size());
      !ok)
    return std::unexpected(ok.error());
  return timing;
}

// Sans sortie flottante à comparer : nul.
double mean_error(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  if (a.empty()) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
  return sum / static_cast<double>(a.size());
}

}  // namespace

std::expected<Units, Refusal> units_of(const Manifest& manifest) {
  if (manifest.units) return *manifest.units;
  auto warps = vk::net::onnx_uses_op(manifest.onnx, "GridSample");
  if (!warps) return std::unexpected(warps.error());
  return *warps ? Units::CpuAndGpu : Units::All;
}

std::expected<Units, Refusal> choose_units(const Manifest& manifest, std::span<const InputShape> inputs,
                                           Progress& progress, const OpenSession& open) {
  auto fallback = units_of(manifest);
  if (!fallback || *fallback == Units::CpuAndGpu) return fallback;
  struct Tried {
    Units units;
    double ms;
  };
  std::vector<Tried> kept;
  std::optional<std::vector<float>> reference;
  constexpr std::array kOrder{Units::CpuAndGpu, Units::All, Units::CpuAndNeuralEngine};
  for (std::size_t i = 0; i < kOrder.size(); ++i) {
    if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction annulée");
    if (progress.report)
      progress.report(0.1F + 0.25F * static_cast<float>(i),
                      std::format("mesure Core ML ({})", units_name(kOrder[i])));
    auto session = open(kOrder[i]);
    if (!session) continue;
    try {
      auto probe = make_probe(*session, inputs);
      if (!probe) return fallback;
      auto timing = measure(*session, *probe);
      if (!timing) continue;
      if (!reference) {
        reference = std::move(timing->first_out);
      } else if (mean_error(*reference, timing->first_out) > kMaxMeanError) {
        continue;
      }
      kept.push_back({.units = kOrder[i], .ms = timing->ms});
    } catch (const std::exception&) {
      continue;  // une répartition que Core ML refuse à l'exécution
    }
  }
  if (kept.empty()) return fallback;
  const auto fastest = std::ranges::min_element(kept, {}, &Tried::ms);
  const auto ane = std::ranges::find(kept, Units::CpuAndNeuralEngine, &Tried::units);
  if (ane != kept.end() && ane->ms <= kAneTolerance * fastest->ms) return ane->units;
  return fastest->units;
}

}  // namespace ajp::gpu::coreml_rt
