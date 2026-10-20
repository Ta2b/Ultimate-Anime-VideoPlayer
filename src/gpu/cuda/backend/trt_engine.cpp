#include "gpu/cuda/backend/trt_engine.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <format>
#include <fstream>
#include <iterator>

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Forme NCHW fixe d'un tenseur ; rien si le rang n'est pas 4 ou si une dimension est libre.
std::optional<Shape> fixed_shape(const nvinfer1::Dims& dims) {
  if (dims.nbDims != 4) return std::nullopt;
  Shape shape;
  for (int i = 0; i < 4; ++i) {
    if (dims.d[i] <= 0) return std::nullopt;
    shape.dims[static_cast<std::size_t>(i)] = dims.d[i];
  }
  return shape;
}

}  // namespace

TrtEngine::~TrtEngine() {
  for (const Graph& graph : graphs_) {
    if (graph.exec != nullptr) cudaGraphExecDestroy(graph.exec);
  }
}

std::expected<std::unique_ptr<TrtEngine>, Refusal> TrtEngine::load(nvinfer1::IRuntime& runtime,
                                                                   const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return refuse(RefusalCode::IoFailed, std::format("moteur {} illisible", path.string()));
  const std::vector<char> blob{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (blob.empty()) return refuse(RefusalCode::IoFailed, std::format("moteur {} vide", path.string()));

  std::unique_ptr<TrtEngine> engine(new TrtEngine());
  engine->engine_.reset(runtime.deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine->engine_) {
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("IRuntime::deserializeCudaEngine({}) a échoué", path.filename().string()));
  }
  engine->context_.reset(engine->engine_->createExecutionContext());
  if (!engine->context_) {
    return refuse(RefusalCode::VendorCallFailed, "ICudaEngine::createExecutionContext a échoué");
  }
  // Entrées d'abord : une forme libre prend celle du profil (OPT), que le contexte fixe ; les
  // sorties se lisent ensuite sur le contexte, formes résolues.
  for (const bool inputs : {true, false}) {
    for (int i = 0; i < engine->engine_->getNbIOTensors(); ++i) {
      const char* name = engine->engine_->getIOTensorName(i);
      const bool is_input = engine->engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
      if (is_input != inputs) continue;
      if (engine->engine_->getTensorDataType(name) != nvinfer1::DataType::kHALF) {
        return refuse(RefusalCode::FormatUnsupported, std::format("tenseur {} : FP16 attendu", name));
      }
      nvinfer1::Dims dims = engine->context_->getTensorShape(name);
      if (is_input && !fixed_shape(dims)) {
        dims = engine->engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
        if (!engine->context_->setInputShape(name, dims)) {
          return refuse(RefusalCode::ShapeInvalid, std::format("tenseur {} : forme du profil refusée", name));
        }
      }
      const auto shape = fixed_shape(dims);
      if (!shape) {
        return refuse(RefusalCode::ShapeInvalid, std::format("tenseur {} : forme NCHW fixe attendue", name));
      }
      (is_input ? engine->inputs_ : engine->outputs_).emplace_back(name);
      (is_input ? engine->input_shapes_ : engine->output_shapes_).push_back(*shape);
    }
  }
  // Une entrée et une sortie (l'image), ou plusieurs (la variante mouvement de RIFE : flot, masque ;
  // les deux graphes de GMFSS), dans l'ordre du graphe.
  if (engine->inputs_.empty() || engine->inputs_.size() > kMaxInputs || engine->outputs_.empty() ||
      engine->outputs_.size() > kMaxOutputs) {
    return refuse(RefusalCode::GraphInvalid,
                  std::format("moteur à 1 à {} entrées et 1 à {} sorties attendu", kMaxInputs, kMaxOutputs));
  }
  return engine;
}

std::expected<void, Refusal> TrtEngine::enqueue(cudaStream_t stream, std::span<const Tensor> inputs,
                                                std::span<const Tensor> outputs) {
  if (inputs.size() != inputs_.size() || outputs.size() != outputs_.size()) {
    return refuse(RefusalCode::ShapeInvalid, "nombre d'entrées ou de sorties différent du moteur");
  }
  const auto bind = [this](const std::string& name, const Tensor& tensor, const Shape& expected) -> bool {
    if (tensor.type != TensorType::F16 || !(tensor.shape == expected)) return false;
    // NOLINTNEXTLINE(performance-no-int-to-ptr) : un tenseur porte son adresse CUDA en entier
    void* address = reinterpret_cast<void*>(tensor.handle + tensor.offset);
    return context_->setTensorAddress(name.c_str(), address);
  };
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!bind(inputs_[i], inputs[i], input_shapes_[i])) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("entrée {} : forme ou type différents du moteur", inputs_[i]));
    }
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (!bind(outputs_[i], outputs[i], output_shapes_[i])) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("sortie {} : forme ou type différents du moteur", outputs_[i]));
    }
  }
  if (graphs_off_) return launch(stream, nullptr);
  std::vector<std::uintptr_t> addresses;
  addresses.reserve(inputs.size() + outputs.size());
  for (const Tensor& t : inputs) addresses.push_back(t.handle + t.offset);
  for (const Tensor& t : outputs) addresses.push_back(t.handle + t.offset);
  auto found = std::ranges::find(graphs_, addresses, &Graph::addresses);
  if (found == graphs_.end()) {
    if (graphs_.size() >= kMaxGraphs) return launch(stream, nullptr);
    graphs_.push_back(Graph{.addresses = std::move(addresses)});
    found = graphs_.end() - 1;
  }
  return launch(stream, &*found);
}

std::expected<void, Refusal> TrtEngine::launch(cudaStream_t stream, Graph* graph) {
  const auto direct = [&]() -> std::expected<void, Refusal> {
    if (!context_->enqueueV3(stream))
      return refuse(RefusalCode::VendorCallFailed, "IExecutionContext::enqueueV3 a échoué");
    return {};
  };
  if (graph == nullptr) return direct();
  if (graph->exec != nullptr) {
    if (cudaGraphLaunch(graph->exec, stream) != cudaSuccess)
      return refuse(RefusalCode::VendorCallFailed, "cudaGraphLaunch a échoué");
    return {};
  }
  if (!graph->warmed) {  // le premier lancement fait ses allocations hors capture
    graph->warmed = true;
    return direct();
  }
  cudaGraph_t captured = nullptr;
  bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
  if (ok) {
    const bool queued = context_->enqueueV3(stream);
    ok = cudaStreamEndCapture(stream, &captured) == cudaSuccess && queued && captured != nullptr;
  }
  if (ok) ok = cudaGraphInstantiate(&graph->exec, captured, 0) == cudaSuccess;
  if (captured != nullptr) cudaGraphDestroy(captured);
  if (!ok) {
    // Moteur que la capture refuse : lancements directs, l'erreur de capture effacée.
    graph->exec = nullptr;
    graphs_off_ = true;
    static_cast<void>(cudaGetLastError());
    return direct();
  }
  if (cudaGraphLaunch(graph->exec, stream) != cudaSuccess)
    return refuse(RefusalCode::VendorCallFailed, "cudaGraphLaunch a échoué");
  return {};
}

}  // namespace ajp::gpu::cuda
