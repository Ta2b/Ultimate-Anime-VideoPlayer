// Session du flux optique matériel (nvof_session.h), d'après la documentation des en-têtes de
// l'API 2.0 (third_party/nvidia-optical-flow).
#include "gpu/cuda/backend/nvof_session.h"

#include <algorithm>
#include <array>
#include <cuda.h>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "kernels/spec/scene_flow.h"
#include "nvOpticalFlowCuda.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace kirari::gpu::cuda {
namespace {

// La bibliothèque n'est jamais déchargée : le pilote garde ses fils après nvOFDestroy (comme la pré-analyse,
// media/probe_flow_nvof.cpp) ; la décharger à la fin d'une lecture ou à un changement de taille ôterait leur
// code sous eux.
#if defined(_WIN32)
constexpr const char* kLibrary = "nvofapi64.dll";
void* open_library() {
  // Le pilote est dans System32 : jamais une copie posée à côté du lecteur ou de la vidéo (P1.22).
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

// NV_OF_API_VERSION, sans la conversion à l'ancienne de sa macro (-Wold-style-cast, clang-tidy).
constexpr auto kApiVersion = static_cast<uint32_t>((NV_OF_API_MAJOR_VERSION << 4U) | NV_OF_API_MINOR_VERSION);

// Une capacité à plusieurs valeurs ; vide si le pilote ne la donne pas.
std::vector<uint32_t> caps(const NV_OF_CUDA_API_FUNCTION_LIST& calls, NvOFHandle handle, NV_OF_CAPS which) {
  uint32_t count = 0;
  if (calls.nvOFGetCaps(handle, which, nullptr, &count) != NV_OF_SUCCESS || count == 0) return {};
  std::vector<uint32_t> values(count);
  if (calls.nvOFGetCaps(handle, which, values.data(), &count) != NV_OF_SUCCESS) return {};
  values.resize(count);
  return values;
}

// Une file à la priorité la plus haute (le plus petit nombre) : la réduction et la copie de la carte passent
// devant les blocs en attente des voies, la carte arrive tôt.
std::expected<void, Refusal> high_priority_stream(cudaStream_t& stream) {
  int least = 0;
  int greatest = 0;
  if (auto ok =
          cuda_check(cudaDeviceGetStreamPriorityRange(&least, &greatest), "cudaDeviceGetStreamPriorityRange");
      !ok)
    return ok;
  return cuda_check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest),
                    "cudaStreamCreateWithPriority");
}

}  // namespace

struct NvofSession::Api {
  NV_OF_CUDA_API_FUNCTION_LIST calls{};
  NvOFHandle handle = nullptr;
  ~Api() {
    if (handle != nullptr) calls.nvOFDestroy(handle);
  }

  // Le refus d'un appel, avec le dernier message de la session s'il y en a un.
  std::unexpected<Refusal> failed(NV_OF_STATUS status, std::string_view call) const {
    std::string detail = std::format("flux optique : {} : erreur {}", call, static_cast<int>(status));
    if (handle != nullptr) {
      std::array<char, 256> text{};
      auto size = static_cast<uint32_t>(text.size());
      if (calls.nvOFGetLastError(handle, text.data(), &size) == NV_OF_SUCCESS && text[0] != '\0')
        detail += std::format(" ({})", std::string_view(text.data()));
    }
    return refuse(RefusalCode::VendorCallFailed, std::move(detail));
  }
};

struct NvofSession::Buffers {
  const NV_OF_CUDA_API_FUNCTION_LIST* calls = nullptr;
  std::array<NvOFGPUBufferHandle, kLumas> lumas{};
  NvOFGPUBufferHandle flow = nullptr;
  NvOFGPUBufferHandle cost = nullptr;
  // Adresses et pas, lus une fois à la création : jamais nuls ensuite (un noyau ne lit pas une adresse
  // nulle).
  std::array<Pitched, kLumas> lumas_at{};
  Pitched flow_at;
  Pitched cost_at;
  ~Buffers() {
    for (NvOFGPUBufferHandle buffer : lumas) {
      if (buffer != nullptr) calls->nvOFDestroyGPUBufferCuda(buffer);
    }
    for (NvOFGPUBufferHandle buffer : {flow, cost}) {
      if (buffer != nullptr) calls->nvOFDestroyGPUBufferCuda(buffer);
    }
  }
  // Les adresses et pas des tampons, lus une fois ; faux si l'un est illisible.
  bool read_addresses() {
    for (std::size_t i = 0; i < lumas.size(); ++i) lumas_at[i] = pitched(lumas[i]);
    flow_at = pitched(flow);
    cost_at = pitched(cost);
    return std::ranges::all_of(lumas_at, readable) && readable(flow_at) && readable(cost_at);
  }
  static bool readable(const Pitched& at) { return at.data != nullptr && at.pitch != 0; }
  Pitched pitched(NvOFGPUBufferHandle buffer) const {
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    if (calls->nvOFGPUBufferGetStrideInfo(buffer, &stride) != NV_OF_SUCCESS) return {};
    // NOLINTNEXTLINE(performance-no-int-to-ptr) : CUdeviceptr est une adresse de l'appareil.
    return {.data = reinterpret_cast<void*>(calls->nvOFGPUBufferGetCUdeviceptr(buffer)),
            .pitch = stride.strideInfo[0].strideXInBytes};
  }
};

std::expected<std::unique_ptr<NvofSession>, Refusal> NvofSession::open(int width, int height) {
  if (width < kernels::kSceneFlowMinSide || height < kernels::kSceneFlowMinSide) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("flux optique : luma {}x{} trop petite", width, height));
  }
  // La carte de coût n'existe qu'à partir d'Ampere (8.x) ; sans elle, pas de confiance, donc pas de carte.
  int device = 0;
  int major = 0;
  if (auto ok = cuda_check(cudaGetDevice(&device), "cudaGetDevice"); !ok) return std::unexpected(ok.error());
  if (auto ok = cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
                           "cudaDeviceGetAttribute");
      !ok)
    return std::unexpected(ok.error());
  if (major < 8)
    return refuse(RefusalCode::FormatUnsupported,
                  "flux optique : pas de carte de coût avant Ampere (RTX 30)");
  // Le contexte primaire du runtime, courant sur ce thread après le premier appel.
  if (auto ok = cuda_check(cudaFree(nullptr), "cudaFree"); !ok) return std::unexpected(ok.error());
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr)
    return refuse(RefusalCode::VendorCallFailed, "flux optique : pas de contexte CUDA courant");

  std::unique_ptr<NvofSession> session(new NvofSession());
  session->width_ = width;
  session->height_ = height;
  session->api_ = std::make_unique<Api>();
  Api& api = *session->api_;
  void* library = open_library();
  if (library == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("flux optique : {} absente (pilote NVIDIA sans flux optique)", kLibrary));
  }
  auto* create = reinterpret_cast<CreateInstance*>(find_symbol(library, "NvOFAPICreateInstanceCuda"));
  if (create == nullptr)
    return refuse(RefusalCode::LibraryUnavailable, "flux optique : NvOFAPICreateInstanceCuda absent");
  if (const NV_OF_STATUS status = create(kApiVersion, &api.calls); status != NV_OF_SUCCESS) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("flux optique : API {}.{} refusée par le pilote (erreur {})",
                              NV_OF_API_MAJOR_VERSION, NV_OF_API_MINOR_VERSION, static_cast<int>(status)));
  }
  if (const auto status = api.calls.nvCreateOpticalFlowCuda(context, &api.handle); status != NV_OF_SUCCESS)
    return api.failed(status, "nvCreateOpticalFlowCuda");

  // Les capacités : la grille 4 et la taille, sans rien supposer.
  const std::vector<uint32_t> grids = caps(api.calls, api.handle, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES);
  if (std::ranges::find(grids, static_cast<uint32_t>(kernels::kSceneFlowGrid)) == grids.end()) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("flux optique : grille {} non proposée", kernels::kSceneFlowGrid));
  }
  const auto bound = [&](NV_OF_CAPS which, uint32_t fallback) {
    const std::vector<uint32_t> values = caps(api.calls, api.handle, which);
    return values.empty() ? fallback : values.front();
  };
  const auto w = static_cast<uint32_t>(width);
  const auto h = static_cast<uint32_t>(height);
  if (w < bound(NV_OF_CAPS_WIDTH_MIN, 0) || h < bound(NV_OF_CAPS_HEIGHT_MIN, 0) ||
      w > bound(NV_OF_CAPS_WIDTH_MAX, w) || h > bound(NV_OF_CAPS_HEIGHT_MAX, h)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("flux optique : luma {}x{} hors des tailles du pilote", width, height));
  }

  if (auto ok = high_priority_stream(session->stream_); !ok) return std::unexpected(ok.error());
  if (const auto status = api.calls.nvOFSetIOCudaStreams(api.handle, session->stream_, session->stream_);
      status != NV_OF_SUCCESS)
    return api.failed(status, "nvOFSetIOCudaStreams");

  NV_OF_INIT_PARAMS init{};
  init.width = w;
  init.height = h;
  init.outGridSize = static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(kernels::kSceneFlowGrid);
  init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
  init.mode = NV_OF_MODE_OPTICALFLOW;
  // Moyen : la carte ne décide que « bouge ou pas », « correspond ou pas » ; le banc 2.2 en mesure le coût.
  init.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
  init.enableExternalHints = NV_OF_FALSE;
  init.enableOutputCost = NV_OF_TRUE;
  init.hPrivData = nullptr;
  init.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
  init.enableRoi = NV_OF_FALSE;
  if (const auto status = api.calls.nvOFInit(api.handle, &init); status != NV_OF_SUCCESS)
    return api.failed(status, "nvOFInit");

  session->buffers_ = std::make_unique<Buffers>();
  Buffers& buffers = *session->buffers_;
  buffers.calls = &api.calls;
  const NV_OF_BUFFER_DESCRIPTOR luma{.width = w,
                                     .height = h,
                                     .bufferUsage = NV_OF_BUFFER_USAGE_INPUT,
                                     .bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8};
  const auto cells_x = static_cast<uint32_t>(session->cells_x());
  const auto cells_y = static_cast<uint32_t>(session->cells_y());
  const NV_OF_BUFFER_DESCRIPTOR vectors{.width = cells_x,
                                        .height = cells_y,
                                        .bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT,
                                        .bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2};
  const NV_OF_BUFFER_DESCRIPTOR costs{.width = cells_x,
                                      .height = cells_y,
                                      .bufferUsage = NV_OF_BUFFER_USAGE_COST,
                                      .bufferFormat = NV_OF_BUFFER_FORMAT_UINT8};
  for (auto [desc, into] : {std::pair{&luma, &buffers.lumas[0]}, std::pair{&luma, &buffers.lumas[1]},
                            std::pair{&luma, &buffers.lumas[2]}, std::pair{&vectors, &buffers.flow},
                            std::pair{&costs, &buffers.cost}}) {
    if (const auto status =
            api.calls.nvOFCreateGPUBufferCuda(api.handle, desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, into);
        status != NV_OF_SUCCESS)
      return api.failed(status, "nvOFCreateGPUBufferCuda");
  }
  if (!buffers.read_addresses())
    return refuse(RefusalCode::VendorCallFailed, "flux optique : adresse ou pas d'un tampon illisible");
  return session;
}

NvofSession::~NvofSession() {
  if (stream_ != nullptr) static_cast<void>(cudaStreamSynchronize(stream_));
  buffers_.reset();  // tampons avant la session
  api_.reset();
  if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(stream_));
}

int NvofSession::cells_x() const {
  return kernels::scene_flow_cells(width_);
}
int NvofSession::cells_y() const {
  return kernels::scene_flow_cells(height_);
}

NvofSession::Pitched NvofSession::luma(int slot) const {
  return buffers_->lumas_at[static_cast<std::size_t>(slot)];
}
NvofSession::Pitched NvofSession::flow() const {
  return buffers_->flow_at;
}
NvofSession::Pitched NvofSession::cost() const {
  return buffers_->cost_at;
}

std::expected<void, Refusal> NvofSession::execute(int a, int b, bool restart) {
  NV_OF_EXECUTE_INPUT_PARAMS in{};
  in.inputFrame = buffers_->lumas[static_cast<std::size_t>(a)];
  in.referenceFrame = buffers_->lumas[static_cast<std::size_t>(b)];
  in.externalHints = nullptr;
  // Images successives d'un même plan : les vecteurs précédents guident. Coupés à la demande, sinon le
  // résultat dépendrait de l'historique (et l'export ne rejouerait plus la lecture).
  in.disableTemporalHints = restart ? NV_OF_TRUE : NV_OF_FALSE;
  in.numRois = 0;
  in.roiData = nullptr;
  NV_OF_EXECUTE_OUTPUT_PARAMS out{};
  out.outputBuffer = buffers_->flow;
  out.outputCostBuffer = buffers_->cost;
  if (const auto status = api_->calls.nvOFExecute(api_->handle, &in, &out); status != NV_OF_SUCCESS)
    return api_->failed(status, "nvOFExecute");
  return {};
}

}  // namespace kirari::gpu::cuda
