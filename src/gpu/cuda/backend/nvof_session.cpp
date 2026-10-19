// Session du flux optique matériel (nvof_session.h), d'après la documentation des en-têtes de
// l'API 2.0 (third_party/nvidia-optical-flow).
#include "gpu/cuda/backend/nvof_session.h"

#include <algorithm>
#include <array>
#include <cuda.h>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "kernels/spec/optical_flow.h"
#include "nvOpticalFlowCuda.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

#if defined(_WIN32)
constexpr const char* kLibrary = "nvofapi64.dll";
void* open_library() {
  return reinterpret_cast<void*>(LoadLibraryA(kLibrary));
}
void* find_symbol(void* library, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
}
void close_library(void* library) {
  FreeLibrary(static_cast<HMODULE>(library));
}
#else
constexpr const char* kLibrary = "libnvidia-opticalflow.so.1";
void* open_library() {
  return dlopen(kLibrary, RTLD_NOW | RTLD_LOCAL);
}
void* find_symbol(void* library, const char* name) {
  return dlsym(library, name);
}
void close_library(void* library) {
  dlclose(library);
}
#endif

using CreateInstance = NV_OF_STATUS NVOFAPI(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);

}  // namespace

struct NvofSession::Api {
  void* library = nullptr;
  NV_OF_CUDA_API_FUNCTION_LIST calls{};
  ~Api() {
    if (library != nullptr) close_library(library);
  }
};

// Un sens : sa session, ses deux lumas et son flot.
struct NvofSession::Side {
  const NV_OF_CUDA_API_FUNCTION_LIST* calls = nullptr;
  NvOFHandle handle = nullptr;
  NvOFGPUBufferHandle input = nullptr;
  NvOFGPUBufferHandle reference = nullptr;
  NvOFGPUBufferHandle flow = nullptr;
  ~Side() {
    for (NvOFGPUBufferHandle buffer : {input, reference, flow}) {
      if (buffer != nullptr) calls->nvOFDestroyGPUBufferCuda(buffer);
    }
    if (handle != nullptr) calls->nvOFDestroy(handle);
  }

  // Le refus d'un appel, avec le dernier message de la session s'il y en a un.
  std::unexpected<Refusal> failed(NV_OF_STATUS status, std::string_view call) const {
    std::string detail = std::format("flux optique : {} : erreur {}", call, static_cast<int>(status));
    if (handle != nullptr) {
      std::array<char, 256> text{};
      auto size = static_cast<uint32_t>(text.size());
      if (calls->nvOFGetLastError(handle, text.data(), &size) == NV_OF_SUCCESS && text[0] != '\0')
        detail += std::format(" ({})", std::string_view(text.data()));
    }
    return refuse(RefusalCode::VendorCallFailed, std::move(detail));
  }

  Pitched pitched(NvOFGPUBufferHandle buffer) const {
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    if (calls->nvOFGPUBufferGetStrideInfo(buffer, &stride) != NV_OF_SUCCESS) return {};
    // NOLINTNEXTLINE(performance-no-int-to-ptr) : CUdeviceptr est une adresse de l'appareil.
    return {.data = reinterpret_cast<void*>(calls->nvOFGPUBufferGetCUdeviceptr(buffer)),
            .pitch = stride.strideInfo[0].strideXInBytes};
  }
};

std::expected<std::unique_ptr<NvofSession>, Refusal> NvofSession::open(cudaStream_t stream, int width,
                                                                       int height) {
  if (width < 32 || height < 32) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("flux optique : luma {}x{} trop petite", width, height));
  }
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr)
    return refuse(RefusalCode::VendorCallFailed, "flux optique : pas de contexte CUDA courant");
  std::unique_ptr<NvofSession> session(new NvofSession());
  session->api_ = std::make_unique<Api>();
  Api& api = *session->api_;
  api.library = open_library();
  if (api.library == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("flux optique : {} absente (pilote NVIDIA sans flux optique)", kLibrary));
  }
  auto* create = reinterpret_cast<CreateInstance*>(find_symbol(api.library, "NvOFAPICreateInstanceCuda"));
  if (create == nullptr)
    return refuse(RefusalCode::LibraryUnavailable, "flux optique : NvOFAPICreateInstanceCuda absent");
  if (const NV_OF_STATUS status = create(NV_OF_API_VERSION, &api.calls); status != NV_OF_SUCCESS) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("flux optique : API {}.{} refusée par le pilote (erreur {})",
                              NV_OF_API_MAJOR_VERSION, NV_OF_API_MINOR_VERSION, static_cast<int>(status)));
  }
  session->width_ = width;
  session->height_ = height;
  for (auto& owned : session->sides_) {
    owned = std::make_unique<Side>();
    Side& side = *owned;
    side.calls = &api.calls;
    if (const auto status = api.calls.nvCreateOpticalFlowCuda(context, &side.handle); status != NV_OF_SUCCESS)
      return side.failed(status, "nvCreateOpticalFlowCuda");
    // La grille : la préférée si l'appareil la propose, sinon 4 (seule grille des Turing).
    if (session->grid_ == 0) {
      uint32_t count = 0;
      std::vector<uint32_t> grids;
      if (api.calls.nvOFGetCaps(side.handle, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &count) ==
          NV_OF_SUCCESS) {
        grids.resize(count);
        if (api.calls.nvOFGetCaps(side.handle, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, grids.data(),
                                  &count) != NV_OF_SUCCESS)
          grids.clear();
      }
      const bool preferred =
          std::ranges::find(grids, static_cast<uint32_t>(kernels::kPreferredFlowGrid)) != grids.end();
      session->grid_ = preferred ? kernels::kPreferredFlowGrid : 4;
    }
    if (const auto status = api.calls.nvOFSetIOCudaStreams(side.handle, stream, stream);
        status != NV_OF_SUCCESS)
      return side.failed(status, "nvOFSetIOCudaStreams");
    NV_OF_INIT_PARAMS init{};
    init.width = static_cast<uint32_t>(width);
    init.height = static_cast<uint32_t>(height);
    init.outGridSize = static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(session->grid_);
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_SLOW;  // la meilleure qualité ; son coût se mesure sur la RTX
    init.enableExternalHints = NV_OF_FALSE;
    init.enableOutputCost = NV_OF_FALSE;
    init.hPrivData = nullptr;
    init.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    init.enableRoi = NV_OF_FALSE;
    if (const auto status = api.calls.nvOFInit(side.handle, &init); status != NV_OF_SUCCESS)
      return side.failed(status, "nvOFInit");
    const NV_OF_BUFFER_DESCRIPTOR luma{.width = init.width,
                                       .height = init.height,
                                       .bufferUsage = NV_OF_BUFFER_USAGE_INPUT,
                                       .bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8};
    const NV_OF_BUFFER_DESCRIPTOR vectors{.width = static_cast<uint32_t>(session->cells_x()),
                                          .height = static_cast<uint32_t>(session->cells_y()),
                                          .bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT,
                                          .bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2};
    for (auto [desc, into] : {std::pair{&luma, &side.input}, std::pair{&luma, &side.reference},
                              std::pair{&vectors, &side.flow}}) {
      if (const auto status =
              api.calls.nvOFCreateGPUBufferCuda(side.handle, desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, into);
          status != NV_OF_SUCCESS)
        return side.failed(status, "nvOFCreateGPUBufferCuda");
    }
  }
  return session;
}

NvofSession::~NvofSession() {
  for (auto& side : sides_) side.reset();  // tampons et sessions avant la bibliothèque
}

int NvofSession::cells_x() const {
  return kernels::flow_cells(width_, grid_);
}
int NvofSession::cells_y() const {
  return kernels::flow_cells(height_, grid_);
}

NvofSession::Pitched NvofSession::input(Direction direction) const {
  const Side& side = *sides_[static_cast<std::size_t>(direction)];
  return side.pitched(side.input);
}
NvofSession::Pitched NvofSession::reference(Direction direction) const {
  const Side& side = *sides_[static_cast<std::size_t>(direction)];
  return side.pitched(side.reference);
}
NvofSession::Pitched NvofSession::flow(Direction direction) const {
  const Side& side = *sides_[static_cast<std::size_t>(direction)];
  return side.pitched(side.flow);
}

std::expected<void, Refusal> NvofSession::execute() {
  for (const auto& owned : sides_) {
    const Side& side = *owned;
    NV_OF_EXECUTE_INPUT_PARAMS in{};
    in.inputFrame = side.input;
    in.referenceFrame = side.reference;
    in.externalHints = nullptr;
    in.disableTemporalHints = NV_OF_FALSE;  // images successives d'une vidéo
    in.numRois = 0;
    in.roiData = nullptr;
    NV_OF_EXECUTE_OUTPUT_PARAMS out{};
    out.outputBuffer = side.flow;
    out.outputCostBuffer = nullptr;
    if (const auto status = side.calls->nvOFExecute(side.handle, &in, &out); status != NV_OF_SUCCESS)
      return side.failed(status, "nvOFExecute");
  }
  return {};
}

}  // namespace ajp::gpu::cuda
