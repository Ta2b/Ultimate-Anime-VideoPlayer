#include "gpu/cuda/backend/trt_engine.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"

namespace ajp::gpu::cuda {
namespace {

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

// Le pas par pixel d'un tenseur d'E/S (TrtEngine::input_pitch) : 0 en plans, C arrondi au multiple de 8 en
// HWC8 ; tout autre rangement est refusé.
std::expected<int, Refusal> tensor_pitch(const nvinfer1::ICudaEngine& engine, const char* name,
                                         const Shape& shape) {
  const nvinfer1::TensorFormat format = engine.getTensorFormat(name);
  if (format == nvinfer1::TensorFormat::kLINEAR) return 0;
  if (format == nvinfer1::TensorFormat::kHWC8) return (static_cast<int>(shape.dims[1]) + 7) / 8 * 8;
  return refuse(RefusalCode::FormatUnsupported,
                std::format("tenseur {} : rangé en plans ou en HWC8 attendu", name));
}

// Un plan au-delà ne vient pas de nos modèles (le plus gros pèse quelques centaines de Mo).
constexpr std::uintmax_t kMaxPlanBytes = std::uintmax_t{8} << 30U;

struct Plan {
  std::unique_ptr<char[]> bytes;
  std::size_t size = 0;
};

// Le plan entier d'un seul read, sa taille donnée par le système de fichiers (tellg() sur un dossier
// rend n'importe quoi) : l'istreambuf_iterator d'avant lisait octet par octet, 0,5 à 1 s par moteur
// de 150 Mo au démarrage (constat T3). IStreamReaderV2 ne ferait pas mieux sans GPUDirect Storage :
// le lecteur d'exemple de NVIDIA le dit (samples/common/streamReader.h), et ses lectures vers la
// mémoire d'appareil passent de toute façon par un tampon hôte.
std::expected<Plan, Refusal> read_plan(const std::filesystem::path& path) {
  std::error_code error;
  const bool regular = std::filesystem::is_regular_file(path, error);
  const std::uintmax_t size = regular ? std::filesystem::file_size(path, error) : 0;
  if (!regular || error)
    return refuse(RefusalCode::IoFailed, std::format("moteur {} illisible", core::to_utf8(path)));
  if (size == 0) return refuse(RefusalCode::IoFailed, std::format("moteur {} vide", core::to_utf8(path)));
  if (size > kMaxPlanBytes) {
    return refuse(RefusalCode::IoFailed, std::format("moteur {} de {} octets, au-delà de {}",
                                                     core::to_utf8(path), size, kMaxPlanBytes));
  }
  Plan plan{.bytes = std::make_unique_for_overwrite<char[]>(static_cast<std::size_t>(size)),
            .size = static_cast<std::size_t>(size)};
  std::ifstream file(path, std::ios::binary);
  file.read(plan.bytes.get(), static_cast<std::streamsize>(size));
  if (!file || file.gcount() != static_cast<std::streamsize>(size)) {
    return refuse(RefusalCode::IoFailed,
                  std::format("moteur {} tronqué pendant sa lecture", core::to_utf8(path)));
  }
  return plan;
}

// IRuntime::deserializeCudaEngine a échoué. Un manque de mémoire vidéo n'accuse pas le plan : le lecteur ne
// le met de côté que pour VendorCallFailed (C05-01). La mémoire libre se compare à la taille du plan (ses
// poids y vont) ; l'erreur CUDA en attente, elle, dit aussi un appareil perdu.
std::unexpected<Refusal> deserialize_failed(const std::filesystem::path& path, std::size_t plan_size) {
  const cudaError_t pending = clear_error();
  if (pending != cudaSuccess)
    return std::unexpected(cuda_refusal(pending, "IRuntime::deserializeCudaEngine"));
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes < plan_size) {
    return refuse(RefusalCode::GpuMemoryExhausted,
                  std::format("IRuntime::deserializeCudaEngine({}) : mémoire vidéo pleine ({} Mo libres pour "
                              "un plan de {} Mo)",
                              core::to_utf8(path.filename()), free_bytes >> 20U, plan_size >> 20U));
  }
  static_cast<void>(clear_error());
  return refuse(RefusalCode::VendorCallFailed,
                std::format("IRuntime::deserializeCudaEngine({}) a échoué", core::to_utf8(path.filename())));
}

// Les files auxiliaires que le moteur demande, rangées dans streams au fur et à mesure : le destructeur du
// moteur les détruit, même après un refus.
std::expected<void, Refusal> make_aux_streams(int count, std::vector<cudaStream_t>& streams) {
  for (int i = 0; i < count; ++i) {
    cudaStream_t aux = nullptr;
    if (const cudaError_t made = cudaStreamCreateWithFlags(&aux, cudaStreamNonBlocking);
        made != cudaSuccess) {
      if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
        return std::unexpected(lost_refusal(lost, "cudaStreamCreateWithFlags (file auxiliaire)"));
      return refuse(RefusalCode::VendorCallFailed, "cudaStreamCreateWithFlags (file auxiliaire)");
    }
    streams.push_back(aux);
  }
  return {};
}

}  // namespace

TrtEngine::~TrtEngine() {
  drop_graphs();
  context_.reset();
  for (cudaStream_t aux : aux_streams_) cudaStreamDestroy(aux);
}

std::int64_t TrtEngine::activation_bytes() const {
  return static_cast<std::int64_t>(engine_->getDeviceMemorySizeV2());
}

void TrtEngine::drop_graphs() {
  for (cudaGraphExec_t exec : execs_) {
    if (exec != nullptr) cudaGraphExecDestroy(exec);
  }
  execs_.clear();
  book_.clear();
}

std::expected<std::unique_ptr<TrtEngine>, Refusal> TrtEngine::load(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& path,
    std::shared_ptr<SharedActivations> shared) {
  auto plan = read_plan(path);
  if (!plan) return std::unexpected(std::move(plan.error()));

  std::unique_ptr<TrtEngine> engine(new TrtEngine());
  engine->engine_.reset(runtime.deserializeCudaEngine(plan->bytes.get(), plan->size));
  plan->bytes.reset();  // les poids sont sur le GPU : le plan hôte (des centaines de Mo) part tout de suite
  if (!engine->engine_) return deserialize_failed(path, plan->size);
  engine->context_.reset(shared ? engine->engine_->createExecutionContext(
                                      nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED)
                                : engine->engine_->createExecutionContext());
  if (!engine->context_) {
    return refuse(RefusalCode::VendorCallFailed, "ICudaEngine::createExecutionContext a échoué");
  }
  if (auto made = make_aux_streams(engine->engine_->getNbAuxStreams(), engine->aux_streams_); !made)
    return std::unexpected(std::move(made.error()));
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
      const auto pitch = tensor_pitch(*engine->engine_, name, *shape);
      if (!pitch) return std::unexpected(pitch.error());
      (is_input ? engine->inputs_ : engine->outputs_).emplace_back(name);
      (is_input ? engine->input_shapes_ : engine->output_shapes_).push_back(*shape);
      (is_input ? engine->input_pitches_ : engine->output_pitches_).push_back(*pitch);
    }
  }
  // Une entrée et une sortie (l'image), ou plusieurs (la variante mouvement de RIFE : flot, masque ;
  // DistilDRBA fusionné), dans l'ordre du graphe.
  if (engine->inputs_.empty() || engine->inputs_.size() > kMaxInputs || engine->outputs_.empty() ||
      engine->outputs_.size() > kMaxOutputs) {
    return refuse(RefusalCode::GraphInvalid,
                  std::format("moteur à 1 à {} entrées et 1 à {} sorties attendu", kMaxInputs, kMaxOutputs));
  }
  // La zone commune ne grandit que pour un moteur accepté : elle ne rapetisse jamais, un moteur refusé
  // ensuite lui laisserait sa taille pour toute la séance.
  if (shared) {
    if (auto ok = shared->reserve(engine->activation_bytes()); !ok) return std::unexpected(ok.error());
    engine->shared_ = std::move(shared);
  }
  return engine;
}

std::expected<void, Refusal> TrtEngine::place(const std::shared_ptr<SharedActivations>& arena,
                                              std::span<const Tensor> inputs,
                                              std::span<const Tensor> outputs) {
  if (inputs.size() != inputs_.size() || outputs.size() != outputs_.size()) {
    return refuse(RefusalCode::ShapeInvalid, "nombre d'entrées ou de sorties différent du moteur");
  }
  // Lancé sur une autre voie que celle de son chargement : ses activations passent dans la zone
  // de cette voie (posée juste après).
  if (shared_ && arena && arena != shared_) {
    if (auto ok = arena->reserve(activation_bytes()); !ok) return ok;
    shared_ = arena;
    shared_generation_ = 0;
  }
  // Une zone rendue plus petite que ce moteur (refus d'agrandir pour un autre, puis ancienne taille
  // refusée aussi) : redemandée ici, et seul ce moteur échoue si elle ne vient pas.
  if (shared_ && shared_->bytes() < activation_bytes()) {
    if (auto ok = shared_->reserve(activation_bytes()); !ok) return ok;
  }
  // La zone commune a grandi depuis le dernier lancement : posée à nouveau, les graphes capturés
  // sur l'ancienne adresse jetés.
  if (shared_ && shared_generation_ != shared_->generation()) {
    context_->setDeviceMemoryV2(shared_->data(), shared_->bytes());
    drop_graphs();
    shared_generation_ = shared_->generation();
  }
  // Ses activations écrivent le bas de la zone : un moteur fusionné qui y gardait des tampons plus bas que
  // leur fin les refait (C03-01). Un moteur à mémoire propre n'écrit pas la zone.
  if (shared_) shared_->used(this, activation_bytes());
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
  // Pour le prochain enqueueV3 seulement, d'après TensorRT : reposées à chaque lancement.
  if (!aux_streams_.empty())
    context_->setAuxStreams(aux_streams_.data(), static_cast<std::int32_t>(aux_streams_.size()));
  return {};
}

std::expected<void, Refusal> TrtEngine::match_aux_priority(cudaStream_t stream) {
  if (aux_streams_.empty()) return {};
  int priority = 0;
  if (cudaStreamGetPriority(stream, &priority) != cudaSuccess) {
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "cudaStreamGetPriority"));
    return {};  // priorité inconnue : les files gardent la leur
  }
  if (priority == aux_priority_) return {};
  // Une file détruite pendant un travail en vol attend sa fin (cudaStreamDestroy) : sans risque ici.
  for (cudaStream_t& aux : aux_streams_) {
    cudaStream_t fresh = nullptr;
    if (cudaStreamCreateWithPriority(&fresh, cudaStreamNonBlocking, priority) != cudaSuccess) {
      if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
        return std::unexpected(lost_refusal(lost, "cudaStreamCreateWithPriority (file auxiliaire)"));
      return {};  // gardée telle quelle
    }
    cuda_warn(cudaStreamDestroy(aux), "cudaStreamDestroy");
    aux = fresh;
  }
  aux_priority_ = priority;
  drop_graphs();  // les graphes capturés nommaient les anciennes files
  return {};
}

std::expected<void, Refusal> TrtEngine::enqueue_plain(cudaStream_t stream,
                                                      const std::shared_ptr<SharedActivations>& arena,
                                                      std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
  if (auto ok = match_aux_priority(stream); !ok) return ok;
  if (auto ok = place(arena, inputs, outputs); !ok) return ok;
  if (!context_->enqueueV3(stream))
    return refuse(RefusalCode::VendorCallFailed, "IExecutionContext::enqueueV3 a échoué");
  return {};
}

std::expected<void, Refusal> TrtEngine::enqueue(cudaStream_t stream,
                                                const std::shared_ptr<SharedActivations>& arena,
                                                std::span<const Tensor> inputs,
                                                std::span<const Tensor> outputs) {
  if (auto ok = match_aux_priority(stream); !ok) return ok;
  if (auto ok = place(arena, inputs, outputs); !ok) return ok;
  if (book_.skip()) return launch(stream, {});
  std::vector<std::uintptr_t> addresses;
  addresses.reserve(inputs.size() + outputs.size());
  for (const Tensor& t : inputs) addresses.push_back(t.handle + t.offset);
  for (const Tensor& t : outputs) addresses.push_back(t.handle + t.offset);
  return launch(stream, book_.next(addresses));
}

std::expected<void, Refusal> TrtEngine::launch(cudaStream_t stream,
                                               const GraphBook<std::vector<std::uintptr_t>>::Pick& pick) {
  const auto direct = [&]() -> std::expected<void, Refusal> {
    if (!context_->enqueueV3(stream))
      return refuse(RefusalCode::VendorCallFailed, "IExecutionContext::enqueueV3 a échoué");
    return {};
  };
  if (pick.step == GraphStep::Direct || pick.step == GraphStep::Overflow || pick.step == GraphStep::Warm)
    return direct();  // Warm : le premier lancement fait ses allocations hors capture
  if (execs_.size() <= pick.index) execs_.resize(pick.index + 1, nullptr);
  cudaGraphExec_t& exec = execs_[pick.index];
  // Un rejeu refusé sans erreur collante : graphe détruit, lancement direct, recapture au passage suivant
  // (GraphBook::launch_failed) ; une erreur collante : contexte perdu.
  const auto replay = [&]() -> std::expected<void, Refusal> {
    const cudaError_t launched = cudaGraphLaunch(exec, stream);
    if (launched == cudaSuccess) return {};
    if (sticky(launched)) return std::unexpected(cuda_refusal(launched, "cudaGraphLaunch"));
    cuda_warn(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    exec = nullptr;
    book_.launch_failed(pick.index);
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "cudaGraphLaunch"));
    return direct();
  };
  if (pick.step == GraphStep::Replay) return replay();
  // Un emplacement repris par un autre jeu d'adresses (registre plein, GraphBook) garde l'ancien graphe.
  if (exec != nullptr) {
    cuda_warn(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    exec = nullptr;
  }
  cudaGraph_t captured = nullptr;
  bool queued = false;
  cudaError_t error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error == cudaSuccess) {
    queued = context_->enqueueV3(stream);
    error = cudaStreamEndCapture(stream, &captured);  // termine aussi une capture invalidée
  }
  if (error == cudaSuccess && queued && captured != nullptr) error = cudaGraphInstantiate(&exec, captured, 0);
  if (captured != nullptr) cudaGraphDestroy(captured);
  if (error != cudaSuccess || !queued || captured == nullptr) {
    // Capture refusée : rien n'a tourné (la capture enregistre sans exécuter), le lancement direct fait le
    // travail et l'erreur est effacée. Le lancement suivant réessaie ; après kCaptureAttempts refus, le
    // registre coupe les graphes et le backend le dit dans le journal.
    exec = nullptr;
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "capture du graphe"));
    book_.captured(false, !queued                ? "enqueueV3 a échoué pendant la capture"
                          : error != cudaSuccess ? cudaGetErrorString(error)
                                                 : "graphe vide");
    return direct();
  }
  book_.captured(true);
  return replay();
}

}  // namespace ajp::gpu::cuda
