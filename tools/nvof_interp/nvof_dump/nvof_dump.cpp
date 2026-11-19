// nvof_dump : lit des tâches sur l'entrée standard, une par ligne, champs séparés par des TABULATIONS
// (un chemin peut contenir des espaces, par exemple « C:\Users\Prénom Nom\AppData\Local\Temp », C46-11) :
//   <largeur> <hauteur> <luma a> <luma b> <sortie>
// (lumas 8 bits brutes, largeur × hauteur octets), et écrit dans <sortie> les flux NVOF de la paire :
// avant (a → b) puis arrière (b → a), chacun (largeur/G) × (hauteur/G) vecteurs int16 (x, y) en S10.5,
// comme le matériel les rend. Une ligne « ok <sortie> » par tâche réussie.
//
// Par défaut, grille 4 et qualité la plus haute (SLOW), sans indices temporels (les paires
// d'entraînement ne se suivent pas) : le format qu'attend precompute_nvof.py. Une session par taille.
// Options (banc nvof_scene.py, porte 2.1) :
//   --grid 1|2|4          taille des blocs (1 et 2 : Ampere et après ; lue dans les capacités) ;
//   --preset slow|medium|fast ;
//   --one-way             le sens avant seulement ;
//   --cost                ajoute après chaque flux sa carte de coût, un octet par bloc (0 : sûr) ;
//   --hints               indices temporels, avec --one-way : les tâches doivent se suivre dans la vidéo.
//
// La session a ses propres flux CUDA (nvOFSetIOCudaStreams), jamais le flux par défaut : c'est ce
// que fera le lecteur pour tourner à côté de TensorRT (porte 2.2). Chaque appel écrit sur la sortie
// d'erreur « temps_ms <l>x<h> <ms> », le temps GPU entre deux événements posés autour de
// nvOFExecute, sans les copies. Au début : « gpu <nom> », « cuda <version du pilote> » et
// « grilles <liste> » (les grilles que la carte accepte).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "nvOpticalFlowCuda.h"

// Le pilote NVIDIA : libnvidia-opticalflow.so.1 sous Linux, nvofapi64.dll sous Windows (B46-19).
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

struct Options {
  std::uint32_t grid = 4;
  NV_OF_PERF_LEVEL preset = NV_OF_PERF_LEVEL_SLOW;
  bool one_way = false;
  bool cost = false;
  bool hints = false;
};

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

// Les grilles que la carte accepte, lues sur une session ouverte (Turing : 4 seulement).
std::vector<std::uint32_t> grids_of(const Api& api, NvOFHandle handle) {
  uint32_t count = 0;
  ok(api, handle, api.calls.nvOFGetCaps(handle, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &count),
     "nvOFGetCaps");
  std::vector<std::uint32_t> values(count);
  ok(api, handle,
     api.calls.nvOFGetCaps(handle, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, values.data(), &count),
     "nvOFGetCaps");
  values.resize(count);
  return values;
}

// Une session d'une taille : ses deux lumas, son flux, sa carte de coût, ses flux CUDA.
struct Session {
  const Api& api;
  const Options& options;
  std::uint32_t width, height;
  NvOFHandle handle = nullptr;
  NvOFGPUBufferHandle input = nullptr, reference = nullptr, flow = nullptr, cost = nullptr;
  CUstream in_stream = nullptr, out_stream = nullptr;
  CUevent start = nullptr, stop = nullptr;

  Session(const Api& a, const Options& o, CUcontext context, std::uint32_t w, std::uint32_t h)
      : api(a), options(o), width(w), height(h) {
    ok(api, nullptr, api.calls.nvCreateOpticalFlowCuda(context, &handle), "nvCreateOpticalFlowCuda");
    static bool told = false;
    if (!told) {
      told = true;
      std::string list;
      bool found = false;
      for (std::uint32_t g : grids_of(api, handle)) {
        list += (list.empty() ? "" : ",") + std::to_string(g);
        found = found || g == options.grid;
      }
      std::cerr << "grilles " << list << std::endl;
      if (!found)
        die("grille " + std::to_string(options.grid) + " refusée par la carte (accepte " + list + ")");
    }
    NV_OF_INIT_PARAMS init{};
    init.width = w;
    init.height = h;
    init.outGridSize = static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(options.grid);
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = options.preset;
    init.enableOutputCost = options.cost ? NV_OF_TRUE : NV_OF_FALSE;
    ok(api, handle, api.calls.nvOFInit(handle, &init), "nvOFInit");
    check(cuStreamCreate(&in_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    check(cuStreamCreate(&out_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    ok(api, handle, api.calls.nvOFSetIOCudaStreams(handle, in_stream, out_stream), "nvOFSetIOCudaStreams");
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate");
    const NV_OF_BUFFER_DESCRIPTOR luma{.width = w,
                                       .height = h,
                                       .bufferUsage = NV_OF_BUFFER_USAGE_INPUT,
                                       .bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8};
    const NV_OF_BUFFER_DESCRIPTOR vectors{.width = cells(w),
                                          .height = cells(h),
                                          .bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT,
                                          .bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2};
    const NV_OF_BUFFER_DESCRIPTOR costs{.width = cells(w),
                                        .height = cells(h),
                                        .bufferUsage = NV_OF_BUFFER_USAGE_COST,
                                        .bufferFormat = NV_OF_BUFFER_FORMAT_UINT8};
    std::vector<std::pair<const NV_OF_BUFFER_DESCRIPTOR*, NvOFGPUBufferHandle*>> wanted{
        {&luma, &input}, {&luma, &reference}, {&vectors, &flow}};
    if (options.cost) wanted.emplace_back(&costs, &cost);
    for (auto [desc, into] : wanted)
      ok(api, handle,
         api.calls.nvOFCreateGPUBufferCuda(handle, desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, into),
         "nvOFCreateGPUBufferCuda");
  }
  ~Session() {
    for (NvOFGPUBufferHandle b : {input, reference, flow, cost}) {
      if (b != nullptr) api.calls.nvOFDestroyGPUBufferCuda(b);
    }
    if (handle != nullptr) api.calls.nvOFDestroy(handle);
    for (CUevent e : {start, stop}) {
      if (e != nullptr) cuEventDestroy(e);
    }
    for (CUstream s : {in_stream, out_stream}) {
      if (s != nullptr) cuStreamDestroy(s);
    }
  }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  [[nodiscard]] std::uint32_t cells(std::uint32_t n) const { return (n + options.grid - 1) / options.grid; }

  // Copie hôte ↔ tampon NVOF (pas de ligne du tampon), sur le flux d'entrée ou de sortie de la session.
  void copy(NvOFGPUBufferHandle buffer, void* host, std::uint32_t row, std::uint32_t rows, bool up) {
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    ok(api, handle, api.calls.nvOFGPUBufferGetStrideInfo(buffer, &stride), "nvOFGPUBufferGetStrideInfo");
    CUDA_MEMCPY2D c{};
    const CUdeviceptr device = api.calls.nvOFGPUBufferGetCUdeviceptr(buffer);
    if (up) {
      c.srcMemoryType = CU_MEMORYTYPE_HOST;
      c.srcHost = host;
      c.srcPitch = row;
      c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      c.dstDevice = device;
      c.dstPitch = stride.strideInfo[0].strideXInBytes;
    } else {
      c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      c.srcDevice = device;
      c.srcPitch = stride.strideInfo[0].strideXInBytes;
      c.dstMemoryType = CU_MEMORYTYPE_HOST;
      c.dstHost = host;
      c.dstPitch = row;
    }
    c.WidthInBytes = row;
    c.Height = rows;
    CUstream s = up ? in_stream : out_stream;
    check(cuMemcpy2DAsync(&c, s), "cuMemcpy2DAsync");
    check(cuStreamSynchronize(s), "cuStreamSynchronize");
  }

  // Un sens : le flux puis, avec --cost, sa carte de coût, ajoutés à out.
  void run(std::vector<std::uint8_t>& from, std::vector<std::uint8_t>& to, std::vector<std::uint8_t>& out,
           bool first) {
    copy(input, from.data(), width, height, true);
    copy(reference, to.data(), width, height, true);
    NV_OF_EXECUTE_INPUT_PARAMS in{};
    in.inputFrame = input;
    in.referenceFrame = reference;
    in.disableTemporalHints = (options.hints && !first) ? NV_OF_FALSE : NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS result{};
    result.outputBuffer = flow;
    result.outputCostBuffer = cost;
    check(cuEventRecord(start, in_stream), "cuEventRecord");
    ok(api, handle, api.calls.nvOFExecute(handle, &in, &result), "nvOFExecute");
    check(cuEventRecord(stop, out_stream), "cuEventRecord");
    check(cuEventSynchronize(stop), "cuEventSynchronize");
    float ms = 0.0F;
    check(cuEventElapsedTime(&ms, start, stop), "cuEventElapsedTime");
    std::cerr << "temps_ms " << width << 'x' << height << ' ' << ms << std::endl;
    const std::size_t at = out.size();
    const std::uint32_t row = cells(width) * 4U;  // deux int16
    out.resize(at + static_cast<std::size_t>(row) * cells(height));
    copy(flow, out.data() + at, row, cells(height), false);
    if (options.cost) {
      const std::size_t c = out.size();
      out.resize(c + static_cast<std::size_t>(cells(width)) * cells(height));
      copy(cost, out.data() + c, cells(width), cells(height), false);
    }
  }
};

std::vector<std::uint8_t> read_luma(const std::string& path, std::size_t bytes) {
  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint8_t> data(bytes);
  if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes)))
    die(path + " : luma trop courte");
  return data;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) die(arg + " : valeur manquante");
      return argv[++i];
    };
    if (arg == "--grid") {
      const std::string v = value();
      if (v != "1" && v != "2" && v != "4") die("--grid : 1, 2 ou 4");
      o.grid = static_cast<std::uint32_t>(std::stoul(v));
    } else if (arg == "--preset") {
      const std::string v = value();
      if (v == "slow")
        o.preset = NV_OF_PERF_LEVEL_SLOW;
      else if (v == "medium")
        o.preset = NV_OF_PERF_LEVEL_MEDIUM;
      else if (v == "fast")
        o.preset = NV_OF_PERF_LEVEL_FAST;
      else
        die("--preset : slow, medium ou fast");
    } else if (arg == "--one-way") {
      o.one_way = true;
    } else if (arg == "--cost") {
      o.cost = true;
    } else if (arg == "--hints") {
      o.hints = true;
    } else {
      die("option inconnue : " + arg);
    }
  }
  if (o.hints && !o.one_way) die("--hints demande --one-way (le sens arrière casserait la suite)");
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse(argc, argv);
  check(cuInit(0), "cuInit");
  CUdevice device{};
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  char gpu[256] = {};
  check(cuDeviceGetName(gpu, sizeof(gpu), device), "cuDeviceGetName");
  int driver = 0;
  check(cuDriverGetVersion(&driver), "cuDriverGetVersion");
  std::cerr << "gpu " << gpu << "\ncuda " << driver << std::endl;
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
    bool first = false;
    if (!session || session->width != w || session->height != h) {
      session.reset();
      session = std::make_unique<Session>(api, options, context, w, h);
      first = true;
    }
    auto la = read_luma(a, static_cast<std::size_t>(w) * h);
    auto lb = read_luma(b, static_cast<std::size_t>(w) * h);
    std::vector<std::uint8_t> result;
    session->run(la, lb, result, first);
    if (!options.one_way) session->run(lb, la, result, true);
    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size()));
    if (!file) die(out + " : écriture");
    std::cout << "ok " << out << std::endl;
  }
  session.reset();
  cuDevicePrimaryCtxRelease(device);
  return 0;
}
