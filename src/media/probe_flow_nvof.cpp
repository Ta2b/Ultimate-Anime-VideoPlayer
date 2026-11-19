// La session NVOF de la pré-analyse (ProbeFlow, media/probe_flow.h), d'après les en-têtes de l'API 2.0
// (third_party/nvidia-optical-flow, BSD). La bibliothèque vient du pilote et se charge à l'exécution
// (libnvidia-opticalflow.so.1, nvofapi64.dll) ; le pilote CUDA passe par media/cuda_driver.h.
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "media/cuda_driver.h"
#include "media/probe_flow.h"

// <cuda.h>, que l'en-tête inclut, est media/nvof_cuda/cuda.h : quatre types, sans le toolkit.
#include "nvOpticalFlowCuda.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace kirari::media {
namespace {

#if defined(_WIN32)
constexpr const char* kLibrary = "nvofapi64.dll";
void* open_library() {
  // Le pilote est dans System32 : jamais une copie posée à côté de la vidéo (comme nvcuda.dll, P1.22).
  return reinterpret_cast<void*>(LoadLibraryExA(kLibrary, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
}
void* find_symbol(void* library, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
}
#else
constexpr const char* kLibrary = "libnvidia-opticalflow.so.1";
void* open_library() {
  return dlopen(kLibrary, RTLD_NOW | RTLD_LOCAL);
}
void* find_symbol(void* library, const char* name) {
  return dlsym(library, name);
}
#endif

using CreateInstance = NV_OF_STATUS NVOFAPI(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);

// NV_OF_API_VERSION, sans la conversion à l'ancienne de sa macro (-Wold-style-cast).
constexpr auto kApiVersion = static_cast<uint32_t>((NV_OF_API_MAJOR_VERSION << 4U) | NV_OF_API_MINOR_VERSION);
constexpr auto kLumaBytes = static_cast<std::size_t>(kProbeFlowWidth);
constexpr auto kVectorBytes = static_cast<std::size_t>(kProbeFlowCellsX) * sizeof(FlowVector);

}  // namespace

// La bibliothèque n'est jamais déchargée : le pilote garde ses fils (comme libcuda, media/cuda_driver.h).
// Chaque appel de NVOF qui touche l'appareil (l'initialisation, les tampons, l'exécution, la destruction) se
// fait le contexte rendu courant, comme dans les exemples du SDK : le fil de la pré-analyse n'en a aucun.
struct ProbeFlow::Session {
  // Un tampon de NVOF, son adresse sur l'appareil et l'écart de ses lignes, lus à sa création.
  struct Buffer {
    NvOFGPUBufferHandle handle = nullptr;
    std::uintptr_t address = 0;
    std::size_t pitch = 0;
  };

  std::shared_ptr<CudaDriver> driver;
  PrimaryContext context;
  std::uintptr_t stream = 0;
  NV_OF_CUDA_API_FUNCTION_LIST calls{};
  NvOFHandle handle = nullptr;
  Buffer input;
  Buffer reference;
  Buffer flow;

  Session() = default;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;
  ~Session() {
    if (driver == nullptr) return;
    // Une exécution refusée a pu laisser une copie en route sur le flux : elle finit avant que ses tampons
    // soient rendus.
    if (stream != 0) (void)driver->synchronize(context.context, stream);
    (void)driver->in_context(context.context, [this] {
      for (const Buffer* buffer : {&input, &reference, &flow})
        if (buffer->handle != nullptr) calls.nvOFDestroyGPUBufferCuda(buffer->handle);
      if (handle != nullptr) calls.nvOFDestroy(handle);
    });
    if (stream != 0) driver->destroy_stream(context.context, stream);
  }

  std::string failed(NV_OF_STATUS status, const char* call) const {
    std::string detail = std::format("{} : erreur {}", call, static_cast<int>(status));
    if (handle != nullptr) {
      std::array<char, 256> text{};
      auto size = static_cast<uint32_t>(text.size());
      if (calls.nvOFGetLastError(handle, text.data(), &size) == NV_OF_SUCCESS && text[0] != '\0')
        detail += std::format(" ({})", std::string_view(text.data()));
    }
    return detail;
  }

  // La session et ses trois tampons, le contexte courant ; un message d'erreur, vide si tout va.
  std::string start() {
    std::string error;
    if (auto entered = driver->in_context(context.context, [&] { error = create(); }); !entered)
      return entered.error().detail;
    return error;
  }

  std::string create() {
    if (const auto status = calls.nvCreateOpticalFlowCuda(std::bit_cast<CUcontext>(context.context), &handle);
        status != NV_OF_SUCCESS)
      return failed(status, "nvCreateOpticalFlowCuda");
    const auto io = std::bit_cast<CUstream>(stream);
    if (const auto status = calls.nvOFSetIOCudaStreams(handle, io, io); status != NV_OF_SUCCESS)
      return failed(status, "nvOFSetIOCudaStreams");
    NV_OF_INIT_PARAMS init{};
    init.width = kProbeFlowWidth;
    init.height = kProbeFlowHeight;
    init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_SLOW;  // 256 × 144 : la meilleure qualité ne coûte presque rien
    init.enableExternalHints = NV_OF_FALSE;
    init.enableOutputCost = NV_OF_FALSE;  // Turing n'en a pas ; motion_of juge l'appariement lui-même
    init.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    init.enableRoi = NV_OF_FALSE;
    if (const auto status = calls.nvOFInit(handle, &init); status != NV_OF_SUCCESS)
      return failed(status, "nvOFInit");
    const NV_OF_BUFFER_DESCRIPTOR luma{.width = kProbeFlowWidth,
                                       .height = kProbeFlowHeight,
                                       .bufferUsage = NV_OF_BUFFER_USAGE_INPUT,
                                       .bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8};
    const NV_OF_BUFFER_DESCRIPTOR vectors{.width = kProbeFlowCellsX,
                                          .height = kProbeFlowCellsY,
                                          .bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT,
                                          .bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2};
    for (auto [desc, row, into] :
         {std::tuple{&luma, kLumaBytes, &input}, std::tuple{&luma, kLumaBytes, &reference},
          std::tuple{&vectors, kVectorBytes, &flow}}) {
      if (std::string error = create_buffer(*desc, row, *into); !error.empty()) return error;
    }
    return {};
  }

  // Un tampon de desc, ses lignes d'au moins row octets ; un message d'erreur, vide si tout va.
  std::string create_buffer(const NV_OF_BUFFER_DESCRIPTOR& desc, std::size_t row, Buffer& into) {
    if (const auto status =
            calls.nvOFCreateGPUBufferCuda(handle, &desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &into.handle);
        status != NV_OF_SUCCESS)
      return failed(status, "nvOFCreateGPUBufferCuda");
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    if (const auto status = calls.nvOFGPUBufferGetStrideInfo(into.handle, &stride); status != NV_OF_SUCCESS)
      return failed(status, "nvOFGPUBufferGetStrideInfo");
    into.address = static_cast<std::uintptr_t>(calls.nvOFGPUBufferGetCUdeviceptr(into.handle));
    into.pitch = stride.strideInfo[0].strideXInBytes;
    if (into.address == 0 || into.pitch < row)
      return std::format("tampon de NVOF inattendu (lignes de {} octets pour {})", into.pitch, row);
    return {};
  }

  // Les vecteurs de current (l'entrée) face à previous (la référence) ; un message d'erreur si refus.
  std::expected<std::vector<FlowVector>, std::string> run(const ProbeLuma& previous,
                                                          const ProbeLuma& current) {
    const std::uintptr_t ctx = context.context;
    for (auto [luma, buffer] : {std::pair{&current, &input}, std::pair{&previous, &reference}}) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) : copie vers l'appareil, la source est lue
      void* host = const_cast<std::uint8_t*>(luma->data());
      if (auto ok = driver->copy_host_2d(ctx, stream, true, buffer->address, buffer->pitch, host, kLumaBytes,
                                         kLumaBytes, kProbeFlowHeight);
          !ok)
        return std::unexpected(ok.error().detail);
    }
    NV_OF_STATUS status = NV_OF_SUCCESS;
    auto ran = driver->in_context(ctx, [&] {
      NV_OF_EXECUTE_INPUT_PARAMS in{};
      in.inputFrame = input.handle;
      in.referenceFrame = reference.handle;
      in.disableTemporalHints = NV_OF_TRUE;  // chaque paire seule : sauts, cache et export rejouables
      NV_OF_EXECUTE_OUTPUT_PARAMS out{};
      out.outputBuffer = flow.handle;
      status = calls.nvOFExecute(handle, &in, &out);
    });
    if (!ran) return std::unexpected(ran.error().detail);
    if (status != NV_OF_SUCCESS) return std::unexpected(failed(status, "nvOFExecute"));
    std::vector<FlowVector> vectors(static_cast<std::size_t>(kProbeFlowCellsX) * kProbeFlowCellsY);
    if (auto ok = driver->copy_host_2d(ctx, stream, false, flow.address, flow.pitch, vectors.data(),
                                       kVectorBytes, kVectorBytes, kProbeFlowCellsY);
        !ok)
      return std::unexpected(ok.error().detail);
    if (auto ok = driver->synchronize(ctx, stream); !ok) return std::unexpected(ok.error().detail);
    return vectors;
  }
};

ProbeFlow::ProbeFlow(std::unique_ptr<Session> session) : session_(std::move(session)) {}
ProbeFlow::~ProbeFlow() = default;

std::expected<std::unique_ptr<ProbeFlow>, std::string> ProbeFlow::open(int device) {
  const auto off = [](std::string why) { return std::unexpected(std::move(why)); };
  auto driver = CudaDriver::load();
  if (!driver) return off(driver.error().detail);
  void* library = open_library();
  if (library == nullptr) return off(std::format("{} absente", kLibrary));
  auto* create = reinterpret_cast<CreateInstance*>(find_symbol(library, "NvOFAPICreateInstanceCuda"));
  if (create == nullptr) return off("NvOFAPICreateInstanceCuda absent");
  auto session = std::make_unique<Session>();
  if (const NV_OF_STATUS status = create(kApiVersion, &session->calls); status != NV_OF_SUCCESS)
    return off(std::format("API {}.{} refusée (erreur {})", NV_OF_API_MAJOR_VERSION, NV_OF_API_MINOR_VERSION,
                           static_cast<int>(status)));
  auto context = (*driver)->retain_primary(device);
  if (!context) return off(context.error().detail);
  auto stream = (*driver)->create_stream(context->context);
  if (!stream) return off(stream.error().detail);
  session->driver = std::move(*driver);
  session->context = std::move(*context);
  session->stream = *stream;
  if (std::string error = session->start(); !error.empty()) return off(std::move(error));
  return std::unique_ptr<ProbeFlow>(new ProbeFlow(std::move(session)));
}

std::optional<ProbeMotion> ProbeFlow::push(const DecodedFrame& frame) {
  if (session_ == nullptr) return std::nullopt;
  std::optional<ProbeLuma> current = probe_luma_of(frame);
  if (!current) {
    previous_.reset();
    return std::nullopt;
  }
  return push(std::move(*current));
}

std::optional<ProbeMotion> ProbeFlow::push(ProbeLuma current) {
  if (session_ == nullptr || current.size() != static_cast<std::size_t>(kProbeFlowWidth) * kProbeFlowHeight)
    return std::nullopt;
  std::optional<ProbeMotion> out;
  if (previous_) {
    auto vectors = session_->run(*previous_, current);
    if (!vectors) {
      failure_ = std::move(vectors.error());
      session_.reset();
      previous_.reset();
      return std::nullopt;
    }
    out = motion_of(*previous_, current, *vectors);
  }
  previous_ = std::move(current);
  return out;
}

std::optional<std::string> ProbeFlow::take_failure() {
  return std::exchange(failure_, std::nullopt);
}

void ProbeFlow::forget() {
  previous_.reset();
}

}  // namespace kirari::media
