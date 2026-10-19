#include "gpu/cpu/cpu_backend.h"

#include <format>
#include <string>
#include <utility>
#include <vector>

#include "core/thread.h"
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::cpu {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

class CpuEngine final : public Engine {
 public:
  explicit CpuEngine(ir::Graph graph) : graph_(std::move(graph)) {
    for (const ir::TensorId id : graph_.inputs) inputs_.push_back(graph_.tensors[id].name);
    for (const ir::TensorId id : graph_.outputs) outputs_.push_back(graph_.tensors[id].name);
  }
  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  const ir::Graph& graph() const { return graph_; }

 private:
  ir::Graph graph_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

std::expected<void, Refusal> run(const HostMemory& memory, const CpuEngine& engine,
                                 std::span<const Tensor> inputs, std::span<Tensor> outputs) {
  const ir::Graph& graph = engine.graph();
  if (inputs.size() != graph.inputs.size() || outputs.size() != graph.outputs.size()) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("{} entrée(s) et {} sortie(s) pour un moteur qui en a {} et {}", inputs.size(),
                              outputs.size(), graph.inputs.size(), graph.outputs.size()));
  }
  std::vector<ir::NamedValue> values;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const ir::Tensor& declared = graph.tensors[graph.inputs[i]];
    if (declared.type == ir::DataType::I64) {
      return refuse(RefusalCode::FormatUnsupported, std::format("entrée {} : I64", declared.name));
    }
    auto flat = read_values(memory, inputs[i]);
    if (!flat) return std::unexpected(std::move(flat.error()));
    const auto& dims = inputs[i].shape.dims;
    values.push_back(
        {.name = declared.name,
         .value = {.type = declared.type, .dims = {dims.begin(), dims.end()}, .f = std::move(*flat)}});
  }
  auto results = ir::run(graph, values);
  if (!results) return std::unexpected(std::move(results.error()));
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (auto ok = write_values(memory, outputs[i], (*results)[i].value.f); !ok) return ok;
  }
  return {};
}

}  // namespace

BackendCaps CpuBackend::capabilities() const {
  return {.name = "cpu", .device_name = "CPU (référence, tests et débogage)", .fp16 = true};
}

std::expected<Artifact, Refusal> CpuBackend::build(const ModelSpec& model, const ShapeSet&,
                                                   Progress& progress) {
  if (progress.stop.stop_requested())
    return refuse(RefusalCode::Cancelled, std::format("construction de {}", model.id));
  try {
    auto graph = ir::load_onnx_file(model.onnx_path);
    if (!graph) return std::unexpected(std::move(graph.error()));
    if (progress.report) progress.report(1.0F, "lecture ONNX");
    return Artifact{.cache_key = std::format("cpu-{}-{}", model.id, model.sha256), .path = model.onnx_path};
  } catch (...) {
    return std::unexpected(refusal_from_exception(RefusalCode::ModelUnreadable, model.id));
  }
}

std::expected<std::unique_ptr<Engine>, Refusal> CpuBackend::load(const Artifact& artifact) {
  try {
    auto graph = ir::load_onnx_file(artifact.path);
    if (!graph) return std::unexpected(std::move(graph.error()));
    if (auto folded = ir::fold_constants(*graph); !folded) return std::unexpected(std::move(folded.error()));
    ir::eliminate_dead_code(*graph);
    return std::make_unique<CpuEngine>(std::move(*graph));
  } catch (...) {
    return std::unexpected(refusal_from_exception(RefusalCode::ModelUnreadable, artifact.cache_key));
  }
}

std::expected<Tensor, Refusal> CpuBackend::import(const ExternalImage&) {
  return refuse(RefusalCode::FormatUnsupported,
                "le backend CPU n'importe rien : une image hôte passe par ingest");
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> CpuBackend::alloc_tensor(TensorType type,
                                                                               const Shape& shape) {
  auto count = element_count(shape);
  if (!count) return std::unexpected(std::move(count.error()));
  try {
    return std::make_unique<HostBuffer>(memory_, type, shape, *count * element_size(type));
  } catch (...) {
    return std::unexpected(refusal_from_exception(RefusalCode::ShapeInvalid, "alloc_tensor"));
  }
}

std::expected<void, Refusal> CpuBackend::enqueue(Engine& engine, std::span<const Tensor> inputs,
                                                 std::span<Tensor> outputs) {
  const auto* cpu = dynamic_cast<const CpuEngine*>(&engine);
  if (cpu == nullptr) return refuse(RefusalCode::InputMismatch, "enqueue : moteur d'un autre backend");
  try {
    return run(memory_, *cpu, inputs, outputs);
  } catch (...) {
    return std::unexpected(refusal_from_exception(RefusalCode::ShapeInvalid, "enqueue"));
  }
}

}  // namespace ajp::gpu::cpu
