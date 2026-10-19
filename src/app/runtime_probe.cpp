#include "app/runtime_probe.h"

#include <format>
#include <span>

namespace ajp::app {
namespace {

// Premier nom qui se charge, ou std::nullopt.
std::optional<std::string> first_present(const LibraryLoader& loader, std::span<const std::string> names) {
  for (const std::string& name : names) {
    if (loader.open(name)) return name;
  }
  return std::nullopt;
}

// Signature de cuDriverGetVersion (cuda.h) : utilisable sans cuInit.
using CuDriverGetVersion = int(int* version);
constexpr int kCudaSuccess = 0;

std::optional<CudaDriver> probe_cuda(const LibraryLoader& loader, std::span<const std::string> names,
                                     std::vector<std::string>& notes) {
  for (const std::string& name : names) {
    auto library = loader.open(name);
    if (!library) continue;
    auto* get_version = (*library)->symbol<CuDriverGetVersion>("cuDriverGetVersion");
    if (get_version == nullptr) {
      notes.push_back(std::format("{} : cuDriverGetVersion absent", name));
      continue;
    }
    int version = 0;
    if (const int result = get_version(&version); result != kCudaSuccess || version <= 0) {
      notes.push_back(
          std::format("{} : cuDriverGetVersion rend l'erreur {} (version {})", name, result, version));
      continue;
    }
    return CudaDriver{.library = name, .version = version};
  }
  return std::nullopt;
}

}  // namespace

RuntimeNames runtime_names(core::Os os) {
  switch (os) {
    case core::Os::Linux:
      return {.cuda = {"libcuda.so.1"},
              .hip = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"},
              .migraphx = {"libmigraphx_c.so", "libmigraphx_c.so.3"},
              .openvino = {"libopenvino.so", "libopenvino_c.so"}};
    case core::Os::Windows:
      return {.cuda = {"nvcuda.dll"},
              .hip = {"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"},
              .migraphx = {"migraphx_c.dll"},
              .openvino = {"openvino.dll", "openvino_c.dll"}};
    case core::Os::MacOs:
      return {};  // ni CUDA, ni ROCm ; OpenVINO n'y sert pas de GPU
  }
  return {};
}

RuntimeReport probe_runtimes(const LibraryLoader& loader, core::Os os) {
  const RuntimeNames names = runtime_names(os);
  RuntimeReport report;
  report.cuda = probe_cuda(loader, names.cuda, report.notes);
  report.hip = first_present(loader, names.hip);
  report.migraphx = first_present(loader, names.migraphx);
  report.openvino = first_present(loader, names.openvino);
  return report;
}

}  // namespace ajp::app
