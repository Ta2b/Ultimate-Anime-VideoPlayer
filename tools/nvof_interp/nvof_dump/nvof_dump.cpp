// nvof_dump : lit des tâches sur l'entrée standard, une par ligne, champs séparés par des TABULATIONS
// (un chemin peut contenir des espaces, par exemple « C:\Users\Prénom Nom\AppData\Local\Temp », C46-11) :
//   <largeur> <hauteur> <luma a> <luma b> <sortie>
// (lumas 8 bits brutes, largeur × hauteur octets), et écrit dans <sortie> les deux flux NVOF de la
// paire, avant (a → b) puis arrière (b → a) : chacun (largeur/4) × (hauteur/4) vecteurs int16
// (x, y) en S10.5, comme le matériel les rend. Une ligne « ok <sortie> » par tâche réussie.
//
// Par défaut, grille 4 et qualité la plus haute (SLOW), sans indices temporels (les paires
// d'entraînement ne se suivent pas). Une session par taille. Options (banc nvof_warp.py) :
//   --grid 1|2|4        taille des blocs (1 et 2 : Ampere et après seulement) ;
//   --preset slow|medium|fast.
// Chaque appel au matériel écrit sur la sortie d'erreur « temps_ms <l>x<h> <ms> » (exécution et
// synchronisation, sans les copies), pour le banc.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nvOpticalFlowCuda.h"

// Le pilote NVIDIA : libnvidia-opticalflow.so.1 sous Linux, nvofapi64.dll sous Windows (B46-19).
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

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
#ifdef _WIN32
    HMODULE library = LoadLibraryA("nvofapi64.dll");
    if (library == nullptr) die("nvofapi64.dll absente (pilote NVIDIA)");
    auto* create = reinterpret_cast<CreateInstance*>(
        reinterpret_cast<void*>(GetProcAddress(library, "NvOFAPICreateInstanceCuda")));
#else
    void* library = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) die("libnvidia-opticalflow.so.1 absente (pilote NVIDIA)");
    auto* create = reinterpret_cast<CreateInstance*>(dlsym(library, "NvOFAPICreateInstanceCuda"));
#endif
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

  std::uint32_t grid;

  Session(const Api& a, CUcontext context, std::uint32_t w, std::uint32_t h, std::uint32_t g,
          NV_OF_PERF_LEVEL preset)
      : api(a), width(w), height(h), grid(g) {
    ok(api, nullptr, api.calls.nvCreateOpticalFlowCuda(context, &handle), "nvCreateOpticalFlowCuda");
    NV_OF_INIT_PARAMS init{};
    init.width = w;
    init.height = h;
    init.outGridSize = g == 1   ? NV_OF_OUTPUT_VECTOR_GRID_SIZE_1
                       : g == 2 ? NV_OF_OUTPUT_VECTOR_GRID_SIZE_2
                                : NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = preset;
    ok(api, handle, api.calls.nvOFInit(handle, &init), "nvOFInit");
    const NV_OF_BUFFER_DESCRIPTOR luma{.width = w,
                                       .height = h,
                                       .bufferUsage = NV_OF_BUFFER_USAGE_INPUT,
                                       .bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8};
    const NV_OF_BUFFER_DESCRIPTOR vectors{.width = cells(w),
                                          .height = cells(h),
                                          .bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT,
                                          .bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2};
    for (auto [desc, into] :
         {std::pair{&luma, &input}, std::pair{&luma, &reference}, std::pair{&vectors, &flow}})
      ok(api, handle,
         api.calls.nvOFCreateGPUBufferCuda(handle, desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, into),
         "nvOFCreateGPUBufferCuda");
  }
  ~Session() {
    for (NvOFGPUBufferHandle b : {input, reference, flow}) {
      if (b != nullptr) api.calls.nvOFDestroyGPUBufferCuda(b);
    }
    if (handle != nullptr) api.calls.nvOFDestroy(handle);
  }
  std::uint32_t cells(std::uint32_t n) const { return (n + grid - 1) / grid; }

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
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    const auto start = std::chrono::steady_clock::now();
    ok(api, handle, api.calls.nvOFExecute(handle, &in, &out), "nvOFExecute");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    const std::chrono::duration<double, std::milli> spent = std::chrono::steady_clock::now() - start;
    std::cerr << "temps_ms " << width << 'x' << height << ' ' << spent.count() << '\n';
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

int main(int argc, char** argv) {
  std::uint32_t grid = 4;
  NV_OF_PERF_LEVEL preset = NV_OF_PERF_LEVEL_SLOW;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const std::string value = i + 1 < argc ? argv[i + 1] : "";
    if (arg == "--grid" && (value == "1" || value == "2" || value == "4")) {
      grid = static_cast<std::uint32_t>(std::stoul(value));
    } else if (arg == "--preset" && value == "slow") {
      preset = NV_OF_PERF_LEVEL_SLOW;
    } else if (arg == "--preset" && value == "medium") {
      preset = NV_OF_PERF_LEVEL_MEDIUM;
    } else if (arg == "--preset" && value == "fast") {
      preset = NV_OF_PERF_LEVEL_FAST;
    } else {
      die("option inconnue : " + arg + " (--grid 1|2|4, --preset slow|medium|fast)");
    }
    ++i;
  }
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
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> fields;
    {
      std::istringstream split(line);
      for (std::string field; std::getline(split, field, '\t');) fields.push_back(field);
    }
    if (fields.size() != 5) die("tâche illisible (5 champs séparés par des tabulations attendus) : " + line);
    std::uint32_t w = 0, h = 0;
    try {
      w = static_cast<std::uint32_t>(std::stoul(fields[0]));
      h = static_cast<std::uint32_t>(std::stoul(fields[1]));
    } catch (const std::exception&) {
      die("tâche illisible (largeur ou hauteur) : " + line);
    }
    const std::string &a = fields[2], &b = fields[3], &out = fields[4];
    if (!session || session->width != w || session->height != h) {
      session.reset();
      session = std::make_unique<Session>(api, context, w, h, grid, preset);
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
