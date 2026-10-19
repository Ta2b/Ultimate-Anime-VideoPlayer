#include "gpu/cuda/trt/runtime.h"

#include <NvInferRuntime.h>
#include <format>
#include <fstream>
#include <system_error>
#include <utility>
#include <vector>

#include "gpu/cuda/trt/engine_impl.h"
#include "gpu/cuda/trt/runtime_state.h"

namespace ajp::gpu::trt {
namespace {

static_assert(NV_TENSORRT_MAJOR >= 11, "TensorRT 11 ou plus : le runtime suit trtexec 11.3");

namespace fs = std::filesystem;

std::string path_text(const fs::path& path) {
  const std::u8string text = path.u8string();
  return {text.begin(), text.end()};
}

std::expected<std::vector<std::byte>, Refusal> read_engine(const fs::path& path) {
  std::error_code error;
  const std::uintmax_t size = fs::file_size(path, error);
  if (error) {
    return std::unexpected(Refusal{
        RefusalCode::CacheIo, std::format("moteur {} illisible : {}", path_text(path), error.message())});
  }
  if (size == 0 || size > kMaxEngineBytes) {
    return std::unexpected(Refusal{
        RefusalCode::EngineInvalid,
        std::format("moteur {} de {} octets, hors de 1..{}", path_text(path), size, kMaxEngineBytes)});
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file || file.gcount() != static_cast<std::streamsize>(bytes.size())) {
    return std::unexpected(
        Refusal{RefusalCode::CacheIo, std::format("lecture incomplète du moteur {}", path_text(path))});
  }
  return bytes;
}

}  // namespace

std::string LibraryVersion::text() const {
  return std::format("{}.{}.{}.{}", major, minor, patch, build);
}

TrtRuntime::TrtRuntime(std::shared_ptr<RuntimeState> state) : state_(std::move(state)) {}

std::expected<TrtRuntime, Refusal> TrtRuntime::create(std::shared_ptr<spdlog::logger> log) {
  const std::int32_t loaded = getInferLibVersion();
  if (loaded / 10000 != NV_TENSORRT_MAJOR) {
    return std::unexpected(
        Refusal{RefusalCode::LibraryUnavailable,
                std::format("libnvinfer {} chargée, TensorRT {} attendu (en-têtes {}.{}.{})", loaded,
                            NV_TENSORRT_MAJOR, NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH)});
  }
  auto state = std::make_shared<RuntimeState>(std::move(log));
  state->runtime.reset(nvinfer1::createInferRuntime(state->log));
  if (!state->runtime) {
    return std::unexpected(
        Refusal{RefusalCode::TensorRtFailed, "createInferRuntime a rendu nul" + error_suffix(state->log)});
  }
  // Sinon les messages et les erreurs de ce runtime iraient au journal d'un autre.
  if (getLogger() != &state->log) {
    return std::unexpected(
        Refusal{RefusalCode::TensorRtFailed,
                "un autre runtime ou constructeur TensorRT vit encore avec son propre journal ; "
                "TensorRT n'en garde qu'un par processus"});
  }
  return TrtRuntime(std::move(state));
}

LibraryVersion TrtRuntime::version() const {
  const std::int32_t v = getInferLibVersion();
  return {.major = v / 10000,
          .minor = (v / 100) % 100,
          .patch = v % 100,
          .build = getInferLibBuildVersion(),
          .headers_build = NV_TENSORRT_BUILD};
}

TrtLog& TrtRuntime::log() const {
  return state_->log;
}

std::expected<TrtEngine, Refusal> TrtRuntime::load_file(const fs::path& engine) const {
  auto bytes = read_engine(engine);
  if (!bytes) return std::unexpected(bytes.error());
  auto loaded = load(*bytes);
  if (!loaded) {
    return std::unexpected(
        Refusal{loaded.error().code, std::format("{} : {}", path_text(engine), loaded.error().detail)});
  }
  return loaded;
}

std::expected<TrtEngine, Refusal> TrtRuntime::load(std::span<const std::byte> plan) const {
  if (plan.empty()) return std::unexpected(Refusal{RefusalCode::EngineInvalid, "moteur vide : 0 octets"});
  auto impl = std::make_unique<TrtEngine::Impl>();
  impl->state = state_;
  state_->log.take_last_error();
  impl->engine.reset(state_->runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!impl->engine) {
    return std::unexpected(Refusal{
        RefusalCode::EngineInvalid,
        std::format("TensorRT refuse le moteur de {} octets (abîmé, ou construit par une autre version ou "
                    "pour un autre GPU){}",
                    plan.size(), error_suffix(state_->log))});
  }
  auto tensors = describe_io(*impl->engine);
  if (!tensors) return std::unexpected(tensors.error());
  impl->tensors = std::move(*tensors);
  impl->addresses.assign(impl->tensors.size(), nullptr);
  impl->context.reset(impl->engine->createExecutionContext());
  if (!impl->context) {
    return std::unexpected(
        Refusal{RefusalCode::TensorRtFailed,
                std::format("createExecutionContext a rendu nul ({} octets d'activations){}",
                            impl->engine->getDeviceMemorySizeV2(), error_suffix(state_->log))});
  }
  return TrtEngine(std::move(impl));
}

}  // namespace ajp::gpu::trt
