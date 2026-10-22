// La mesure des répartitions Core ML (coreml_detail.h) : l'image de test de vk::net::probe_image
// à chaque entrée, la médiane des temps de kUnitTimedRuns inférences par répartition et la
// première sortie pour comparer, puis vk::net::choose_unit.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <optional>

#include "gpu/coreml/coreml_detail.h"
#include "gpu/vulkan/backend/net_measure.h"

namespace ajp::gpu::coreml_rt {
namespace {

using vk::net::InputShape;

// Des entrées fixes et reproductibles entre 0 et 1 (l'image de test, décalée d'une entrée à la
// suivante), aux formes fixées par build (sinon celles du modèle), les mêmes pour chaque
// répartition mesurée.
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
    const std::vector<float> values = vk::net::probe_image(shape, static_cast<int>(i));
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
  out_names.reserve(probe.outputs.size());
  for (const std::string& name : probe.outputs) out_names.push_back(name.c_str());
  const auto run = [&] {
    return session.Run(Ort::RunOptions{nullptr}, in_names.data(), values.data(), values.size(),
                       out_names.data(), out_names.size());
  };
  for (int i = 0; i < vk::net::kUnitWarmupRuns; ++i) static_cast<void>(run());
  std::array<double, vk::net::kUnitTimedRuns> times{};
  std::vector<Ort::Value> outputs;
  for (double& t : times) {
    const auto start = std::chrono::steady_clock::now();
    outputs = run();
    t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  Timing timing{.ms = vk::net::median(times), .first_out = {}};
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

// Une sortie en F32, pour comparer.
std::expected<std::vector<float>, Refusal> as_floats(const std::byte* data, TensorType type,
                                                     std::size_t count) {
  std::vector<float> out(count);
  if (auto ok = vk::net::copy_converted(data, type, reinterpret_cast<std::byte*>(out.data()), TensorType::F32,
                                        count);
      !ok)
    return std::unexpected(ok.error());
  return out;
}

// Une forme NCHW de la chaîne : rang 4, dimensions positives.
std::optional<Shape> shape_of(std::span<const std::int64_t> dims) {
  Shape shape;
  if (dims.size() != shape.dims.size()) return std::nullopt;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) return std::nullopt;
    shape.dims[i] = dims[i];
  }
  return shape;
}

double max_error(std::span<const float> a, std::span<const float> b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    worst = std::max(worst, static_cast<double>(std::abs(a[i] - b[i])));
  return a.size() == b.size() ? worst : std::numeric_limits<double>::infinity();
}

}  // namespace

std::expected<DirectError, Refusal> direct_error(Ort::Session& session, std::span<const InputShape> inputs,
                                                 const OpenDirect& open,
                                                 const std::shared_ptr<Surfaces>& surfaces) {
  const auto unsupported = [](std::string text) {
    return refuse(RefusalCode::FormatUnsupported, std::format("Core ML direct : {}", text));
  };
  try {
    auto probe = make_probe(session, inputs);
    if (!probe) return std::unexpected(probe.error());
    auto direct = open(probe->inputs, probe->outputs);
    if (!direct) return std::unexpected(direct.error());
    const DirectModel& model = **direct;
    std::vector<vk::net::HostBinding> in;
    for (std::size_t i = 0; i < probe->inputs.size(); ++i) {
      if (model.input_types()[i] != probe->types[i])
        return unsupported(std::format("{} : type", probe->inputs[i]));
      const std::optional<Shape> shape = shape_of(probe->shapes[i]);
      if (!shape) return unsupported(std::format("{} : forme hors NCHW", probe->inputs[i]));
      in.push_back({.data = probe->data[i].data(),
                    .bytes = probe->data[i].size(),
                    .type = probe->types[i],
                    .shape = *shape});
    }
    // La référence : la session, toutes les sorties.
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<Ort::Value> values;
    std::vector<const char*> in_names;
    for (std::size_t i = 0; i < probe->inputs.size(); ++i) {
      values.push_back(Ort::Value::CreateTensor(memory, probe->data[i].data(), probe->data[i].size(),
                                                probe->shapes[i].data(), probe->shapes[i].size(),
                                                element_of(probe->types[i])));
      in_names.push_back(probe->inputs[i].c_str());
    }
    std::vector<const char*> out_names;
    for (const std::string& name : probe->outputs) out_names.push_back(name.c_str());
    const std::vector<Ort::Value> reference =
        session.Run(Ort::RunOptions{nullptr}, in_names.data(), values.data(), values.size(), out_names.data(),
                    out_names.size());
    std::vector<std::vector<float>> expected;
    std::vector<std::vector<std::byte>> out_bytes;
    std::vector<vk::net::HostBinding> out;
    for (std::size_t i = 0; i < reference.size(); ++i) {
      const auto info = reference[i].GetTensorTypeAndShapeInfo();
      const std::optional<TensorType> type = type_of(info.GetElementType());
      if (!type || *type != model.output_types()[i])
        return unsupported(std::format("{} : type", probe->outputs[i]));
      const std::vector<std::int64_t> dims = info.GetShape();
      const std::optional<Shape> shape = shape_of(dims);
      if (!shape) return unsupported(std::format("{} : forme hors NCHW", probe->outputs[i]));
      auto floats = as_floats(static_cast<const std::byte*>(reference[i].GetTensorRawData()), *type,
                              info.GetElementCount());
      if (!floats) return std::unexpected(floats.error());
      expected.push_back(std::move(*floats));
      out_bytes.emplace_back(info.GetElementCount() * element_bytes(*type));
      out.push_back({.data = out_bytes.back().data(),
                     .bytes = out_bytes.back().size(),
                     .type = *type,
                     .shape = *shape});
    }
    // L'écart des sorties de out à la référence, versé dans mean et max.
    const auto compare = [&](std::span<const vk::net::HostBinding> got_at, double& mean,
                             double& max) -> std::expected<void, Refusal> {
      for (std::size_t i = 0; i < got_at.size(); ++i) {
        auto got = as_floats(got_at[i].data, got_at[i].type, expected[i].size());
        if (!got) return std::unexpected(got.error());
        mean = std::max(mean, vk::net::mean_error(expected[i], *got));
        max = std::max(max, max_error(expected[i], *got));
      }
      return {};
    };
    DirectError worst;
    for (int pass = 0; pass < 2; ++pass) {
      for (auto& bytes : out_bytes) std::ranges::fill(bytes, std::byte{0});
      if (auto ok = (*direct)->predict(in, out); !ok) return std::unexpected(ok.error());
      if (auto ok = compare(out, worst.mean, worst.max); !ok) return std::unexpected(ok.error());
    }
    if (surfaces == nullptr) {
      worst.surfaces_skipped = "pas d'IOSurface ici";
      return worst;
    }
    // Les mêmes liaisons sur des IOSurface, l'entrée recopiée, la sortie remise à zéro.
    std::vector<std::unique_ptr<vk::net::HostMemory>> held;
    std::vector<vk::net::HostBinding> in_surface = in;
    std::vector<vk::net::HostBinding> out_surface = out;
    for (std::vector<vk::net::HostBinding>* list : {&in_surface, &out_surface}) {
      for (vk::net::HostBinding& b : *list) {
        auto surface = allocate_surface(surfaces, b.bytes, 0, b.type, b.shape);
        if (!surface) {
          worst.surfaces_skipped = surface.error().detail;
          return worst;
        }
        b.data = (*surface)->data();
        held.push_back(std::move(*surface));
      }
    }
    for (std::size_t i = 0; i < in.size(); ++i) std::memcpy(in_surface[i].data, in[i].data, in[i].bytes);
    for (const vk::net::HostBinding& b : out_surface) std::memset(b.data, 0, b.bytes);
    if (auto ok = (*direct)->predict(in_surface, out_surface); !ok) {
      worst.surfaces_skipped = ok.error().detail;
      return worst;
    }
    if (!model.on_surfaces()) {
      worst.surfaces_skipped = "Core ML ne prend pas les IOSurface de ses tenseurs";
      return worst;
    }
    if (auto ok = compare(out_surface, worst.surface_mean, worst.surface_max); !ok)
      return std::unexpected(ok.error());
    worst.on_surfaces = true;
    return worst;
  } catch (const std::exception& e) {
    return refuse(RefusalCode::DeviceUnavailable, std::format("Core ML direct, comparaison : {}", e.what()));
  }
}

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
  // La première répartition mesurée sert de référence ; le Neural Engine seul est préféré (il
  // laisse le GPU, partagé avec la chaîne MoltenVK, RIFE et le rendu, à tout le reste).
  constexpr std::array kOrder{Units::CpuAndGpu, Units::All, Units::CpuAndNeuralEngine};
  constexpr std::size_t kPreferred = 2;
  static_assert(kOrder[kPreferred] == Units::CpuAndNeuralEngine);
  std::array<std::optional<vk::net::UnitMeasure>, kOrder.size()> measured{};
  std::optional<std::vector<float>> reference;
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
      const double error = reference ? vk::net::mean_error(*reference, timing->first_out) : 0.0;
      if (!reference) reference = std::move(timing->first_out);
      measured[i] = vk::net::UnitMeasure{.ms = timing->ms, .error = error};
    } catch (const std::exception&) {
      continue;  // une répartition que Core ML refuse à l'exécution
    }
  }
  const std::optional<std::size_t> chosen = vk::net::choose_unit(measured, kPreferred);
  if (!chosen) return fallback;
  return kOrder[*chosen];
}

}  // namespace ajp::gpu::coreml_rt
