#include "gpu/cuda/trt/engine.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <utility>

#include "gpu/cuda/trt/cuda_objects.h"
#include "gpu/cuda/trt/engine_impl.h"

namespace ajp::gpu::trt {
namespace {

std::expected<std::size_t, Refusal> index_of(std::span<const TensorDesc> tensors, std::string_view name) {
  for (std::size_t k = 0; k < tensors.size(); ++k) {
    if (tensors[k].name == name) return k;
  }
  return std::unexpected(
      Refusal{RefusalCode::TensorUnknown, std::format("tenseur « {} » absent du moteur", name)});
}

Refusal trt_refusal(TrtLog& log, std::string_view call) {
  return Refusal{RefusalCode::TensorRtFailed, std::string(call) + " a échoué" + error_suffix(log)};
}

}  // namespace

TrtEngine::TrtEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TrtEngine::TrtEngine(TrtEngine&& other) noexcept = default;
TrtEngine& TrtEngine::operator=(TrtEngine&& other) noexcept = default;
TrtEngine::~TrtEngine() = default;

std::span<const TensorDesc> TrtEngine::tensors() const {
  return impl_->tensors;
}

const TensorDesc* TrtEngine::find(std::string_view name) const {
  const auto found = std::ranges::find(impl_->tensors, name, &TensorDesc::name);
  return found == impl_->tensors.end() ? nullptr : &*found;
}

int TrtEngine::profile_count() const {
  return impl_->engine->getNbOptimizationProfiles();
}

int TrtEngine::profile() const {
  return impl_->profile;
}

std::int64_t TrtEngine::device_memory_bytes() const {
  return impl_->engine->getDeviceMemorySizeV2();
}

std::expected<void, Refusal> TrtEngine::select_profile(int index, cudaStream_t stream) {
  if (index < 0 || index >= profile_count()) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("profil {} hors de 0..{}", index, profile_count() - 1)});
  }
  impl_->state->log.take_last_error();
  if (!impl_->context->setOptimizationProfileAsync(index, stream)) {
    return std::unexpected(
        trt_refusal(impl_->state->log, std::format("setOptimizationProfileAsync({})", index)));
  }
  impl_->profile = index;
  std::ranges::fill(impl_->addresses, nullptr);
  return {};
}

std::expected<void, Refusal> TrtEngine::set_input_shape(std::string_view name,
                                                        std::span<const std::int64_t> dims) {
  auto index = index_of(impl_->tensors, name);
  if (!index) return std::unexpected(index.error());
  const TensorDesc& desc = impl_->tensors[*index];
  if (desc.io != TensorIo::Input) {
    return std::unexpected(
        Refusal{RefusalCode::TensorUnknown, std::format("« {} » est une sortie, sa forme se déduit", name)});
  }
  const ShapeRange& range = desc.profiles[static_cast<std::size_t>(impl_->profile)];
  // Le rang du profil vient du moteur : il tient dans nvinfer1::Dims.
  bool inside = dims.size() == range.min.size();
  for (std::size_t k = 0; inside && k < dims.size(); ++k) {
    inside = range.min[k] <= dims[k] && dims[k] <= range.max[k];
  }
  if (!inside) {
    return std::unexpected(
        Refusal{RefusalCode::ShapeOutOfProfile,
                std::format("entrée « {} » : forme {} hors du profil {} (min {}, max {})", name,
                            dims_text(dims), impl_->profile, dims_text(range.min), dims_text(range.max))});
  }
  nvinfer1::Dims shape{};
  shape.nbDims = static_cast<std::int32_t>(dims.size());
  std::ranges::copy(dims, shape.d);
  impl_->state->log.take_last_error();
  if (!impl_->context->setInputShape(desc.name.c_str(), shape)) {
    return std::unexpected(
        trt_refusal(impl_->state->log, std::format("setInputShape({}, {})", name, dims_text(dims))));
  }
  return {};
}

std::expected<Dims, Refusal> TrtEngine::shape(std::string_view name) const {
  auto index = index_of(impl_->tensors, name);
  if (!index) return std::unexpected(index.error());
  const nvinfer1::Dims dims = impl_->context->getTensorShape(impl_->tensors[*index].name.c_str());
  Dims out(dims.d, dims.d + std::max(dims.nbDims, 0));
  if (dims.nbDims < 0 || std::ranges::any_of(out, [](std::int64_t d) { return d < 0; })) {
    return std::unexpected(
        Refusal{RefusalCode::ShapeOutOfProfile,
                std::format("forme de « {} » encore inconnue : une entrée dynamique attend set_input_shape()",
                            name)});
  }
  return out;
}

std::expected<void, Refusal> TrtEngine::bind(std::string_view name, void* device) {
  auto index = index_of(impl_->tensors, name);
  if (!index) return std::unexpected(index.error());
  if (device == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, std::format("adresse nulle pour le tenseur « {} »", name)});
  }
  impl_->state->log.take_last_error();
  if (!impl_->context->setTensorAddress(impl_->tensors[*index].name.c_str(), device)) {
    return std::unexpected(trt_refusal(impl_->state->log, std::format("setTensorAddress({})", name)));
  }
  impl_->addresses[*index] = device;
  return {};
}

std::expected<void, Refusal> TrtEngine::enqueue(cudaStream_t stream, const EnqueueEvents& events) {
  for (std::size_t k = 0; k < impl_->tensors.size(); ++k) {
    if (impl_->addresses[k] == nullptr) {
      return std::unexpected(Refusal{
          RefusalCode::TensorUnknown,
          std::format("tenseur « {} » sans adresse : bind() avant enqueue()", impl_->tensors[k].name)});
    }
  }
  if (!impl_->context->allInputDimensionsSpecified()) {
    return std::unexpected(Refusal{RefusalCode::ShapeOutOfProfile,
                                   "forme d'entrée dynamique non posée : set_input_shape() avant enqueue()"});
  }
  TrtLog& log = impl_->state->log;
  log.take_last_error();
  // nullptr retire l'événement d'une exécution précédente.
  if (!impl_->context->setInputConsumedEvent(events.input_consumed)) {
    return std::unexpected(trt_refusal(log, "setInputConsumedEvent"));
  }
  if (!impl_->context->enqueueV3(stream)) return std::unexpected(trt_refusal(log, "enqueueV3"));
  if (events.done != nullptr) return cuda_check(cudaEventRecord(events.done, stream), "cudaEventRecord(fin)");
  return {};
}

}  // namespace ajp::gpu::trt
