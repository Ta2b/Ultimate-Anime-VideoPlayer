// nvof_dump : lit des tâches sur l'entrée standard, une par ligne,
//   <largeur> <hauteur> <luma a> <luma b> <sortie>
// (lumas 8 bits brutes, largeur × hauteur octets), et écrit dans <sortie> les deux flux NVOF de la
// paire, avant (a → b) puis arrière (b → a) : chacun (largeur/4) × (hauteur/4) vecteurs int16
// (x, y) en S10.5, comme le matériel les rend. Une ligne « ok <sortie> » par tâche réussie.
//
// Les réglages sont ceux que le lecteur prendra : grille 4, qualité la plus haute (SLOW), sans
// indices temporels (les paires d'entraînement ne se suivent pas). Une session par taille.
#include <cuda.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nvOpticalFlowCuda.h"

namespace {

constexpr std::uint32_t kGrid = 4;

[[noreturn]] void die(const std::string& what) {
  std::cerr << "nvof_dump : " << what << '\n';
  std::exit(1);
}

void check(CUresult r, const char* call) {
  if (r != CUDA_SUCCESS) die(std::string(call) + " : erreur CUDA " + std::to_string(static_cast<int>(r)));
}

using CreateInstance = NV_OF_STATUS NVOFAPI(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);

struct Api {
  NV_OF_CUDA_API_FUNCTION_LIST calls{};
  Api() {
    void* library = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) die("libnvidia-opticalflow.so.1 absente (pilote NVIDIA)");
    auto* create = reinterpret_cast<CreateInstance*>(dlsym(library, "NvOFAPICreateInstanceCuda"));
    if (create == nullptr || create(NV_OF_API_VERSION, &calls) != NV_OF_SUCCESS) die("API NVOF refusée");
  }
};

void ok(const Api& api, NvOFHandle handle, NV_OF_STATUS status, const char* call) {
  if (status == NV_OF_SUCCESS) return;
  std::string text = std::string(call) + " : erreur " + std::to_string(static_cast<int>(status));
  char message[256] = {};
  uint32_t size = sizeof(message);
  if (handle != nullptr && api.calls.nvOFGetLastError(handle, message, &size) == NV_OF_SUCCESS)
    text += std::string(" (") + message + ")";
  die(text);
}

// Une session d'une taille : ses deux lumas et son flux.
struct Session {
  const Api& api;
  std::uint32_t width, height;
  NvOFHandle handle = nullptr;
  NvOFGPUBufferHandle input = nullptr, reference = nullptr, flow = nullptr;

  Session(const Api& a, CUcontext context, std::uint32_t w, std::uint32_t h) : api(a), width(w), height(h) {
    ok(api, nullptr, api.calls.nvCreateOpticalFlowCuda(context, &handle), "nvCreateOpticalFlowCuda");
    NV_OF_INIT_PARAMS init{};
    init.width = w;
    init.height = h;
    init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_SLOW;
    ok(api, handle, api.calls.nvOFInit(handle, &init), "nvOFInit");
    const NV_OF_BUFFER_DESCRIPTOR luma{.width = w, .height = h, .bufferUsage = NV_OF_BUFFER_USAGE_INPUT,
                                       .bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8};
    const NV_OF_BUFFER_DESCRIPTOR vectors{.width = cells(w), .height = cells(h),
                                          .bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT,
                                          .bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2};
    for (auto [desc, into] : {std::pair{&luma, &input}, std::pair{&luma, &reference}, std::pair{&vectors, &flow}})
      ok(api, handle, api.calls.nvOFCreateGPUBufferCuda(handle, desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, into),
         "nvOFCreateGPUBufferCuda");
  }
  ~Session() {
    for (NvOFGPUBufferHandle b : {input, reference, flow}) {
      if (b != nullptr) api.calls.nvOFDestroyGPUBufferCuda(b);
    }
    if (handle != nullptr) api.calls.nvOFDestroy(handle);
  }
  static std::uint32_t cells(std::uint32_t n) { return (n + kGrid - 1) / kGrid; }

  void upload(NvOFGPUBufferHandle buffer, const std::vector<std::uint8_t>& host, std::uint32_t w,
              std::uint32_t h, std::uint32_t bytes_per_row) {
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    ok(api, handle, api.calls.nvOFGPUBufferGetStrideInfo(buffer, &stride), "nvOFGPUBufferGetStrideInfo");
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = host.data();
    copy.srcPitch = bytes_per_row;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = api.calls.nvOFGPUBufferGetCUdeviceptr(buffer);
    copy.dstPitch = stride.strideInfo[0].strideXInBytes;
    copy.WidthInBytes = bytes_per_row;
    copy.Height = h;
    static_cast<void>(w);
    check(cuMemcpy2D(&copy), "cuMemcpy2D");
  }

  std::vector<std::uint8_t> run(const std::vector<std::uint8_t>& from, const std::vector<std::uint8_t>& to) {
    upload(input, from, width, height, width);
    upload(reference, to, width, height, width);
    NV_OF_EXECUTE_INPUT_PARAMS in{};
    in.inputFrame = input;
    in.referenceFrame = reference;
    in.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS out{};
    out.outputBuffer = flow;
    ok(api, handle, api.calls.nvOFExecute(handle, &in, &out), "nvOFExecute");
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    ok(api, handle, api.calls.nvOFGPUBufferGetStrideInfo(flow, &stride), "nvOFGPUBufferGetStrideInfo");
    const std::uint32_t row = cells(width) * 4U;  // deux int16
    std::vector<std::uint8_t> host(static_cast<std::size_t>(row) * cells(height));
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = api.calls.nvOFGPUBufferGetCUdeviceptr(flow);
    copy.srcPitch = stride.strideInfo[0].strideXInBytes;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = host.data();
    copy.dstPitch = row;
    copy.WidthInBytes = row;
    copy.Height = cells(height);
    check(cuMemcpy2D(&copy), "cuMemcpy2D");
    return host;
  }
};

std::vector<std::uint8_t> read_luma(const std::string& path, std::size_t bytes) {
  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint8_t> data(bytes);
  if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes)))
    die(path + " : luma trop courte");
  return data;
}

}  // namespace

int main() {
  check(cuInit(0), "cuInit");
  CUdevice device{};
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context = nullptr;
  check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
  const Api api;
  std::unique_ptr<Session> session;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream fields(line);
    std::uint32_t w = 0, h = 0;
    std::string a, b, out;
    if (!(fields >> w >> h >> a >> b >> out)) die("tâche illisible : " + line);
    if (!session || session->width != w || session->height != h) {
      session.reset();
      session = std::make_unique<Session>(api, context, w, h);
    }
    const auto la = read_luma(a, static_cast<std::size_t>(w) * h);
    const auto lb = read_luma(b, static_cast<std::size_t>(w) * h);
    const auto forward = session->run(la, lb);
    const auto backward = session->run(lb, la);
    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(forward.data()), static_cast<std::streamsize>(forward.size()));
    file.write(reinterpret_cast<const char*>(backward.data()), static_cast<std::streamsize>(backward.size()));
    if (!file) die(out + " : écriture");
    std::cout << "ok " << out << std::endl;
  }
  session.reset();
  cuDevicePrimaryCtxRelease(device);
  return 0;
}
