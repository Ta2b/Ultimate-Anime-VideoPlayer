#include "vendor_hdr/sdk_layout.h"

#include <algorithm>
#include <array>
#include <string>
#include <system_error>
#include <vector>

namespace ajp::vendor_hdr {
namespace {

constexpr std::string_view kMaxineCore =
    "le dossier Core Linux x64 de Maxine VFX 1.3 est incomplet (CUDA 12, NPP et TensorRT 10 compris)";
constexpr std::string_view kMaxineFeature =
    "la fonction TrueHDR manque : l'installer avec install_feature.sh --feature nvvfxtruehdr";
constexpr std::string_view kMaxineHeaders =
    "les en-têtes du SDK manquent : choisir le dossier complet du SDK, include/ compris";

// Repris de manager-v2 Services/TrueHdrSdkStore.cs (CoreRequired, ValidateRequired) : le jeu
// minimal d'exécution de Maxine, plus les en-têtes que vérifie maxine_abi_check.
constexpr std::array<RequiredFile, 16> kMaxineRequired{{
    {.path = "lib/libNVCVImage.so", .missing = kMaxineCore},
    {.path = "lib/libVideoFX.so", .missing = kMaxineCore},
    {.path = "lib/libVideoFXLocal.so", .missing = kMaxineCore},
    {.path = "lib/libnvngxruntime.so", .missing = kMaxineCore},
    {.path = "external/cuda/lib/libcudart.so.12", .missing = kMaxineCore},
    {.path = "external/cuda/lib/libnppc.so.12", .missing = kMaxineCore},
    {.path = "external/cuda/lib/libnppial.so.12", .missing = kMaxineCore},
    {.path = "external/cuda/lib/libnppicc.so.12", .missing = kMaxineCore},
    {.path = "external/cuda/lib/libnppidei.so.12", .missing = kMaxineCore},
    {.path = "external/cuda/lib/libnppig.so.12", .missing = kMaxineCore},
    {.path = "external/tensorrt/lib/libnvinfer.so.10", .missing = kMaxineCore},
    {.path = "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so", .missing = kMaxineFeature},
    {.path = "features/nvvfxtruehdr/lib/libnvidia-ngx-truehdr.so", .prefix = true, .missing = kMaxineFeature},
    {.path = "include/nvCVImage.h", .format = BinaryFormat::Text, .missing = kMaxineHeaders},
    {.path = "include/nvVideoEffects.h", .format = BinaryFormat::Text, .missing = kMaxineHeaders},
    {.path = "features/nvvfxtruehdr/include/nvVFXTrueHDR.h",
     .format = BinaryFormat::Text,
     .missing = kMaxineHeaders},
}};

constexpr std::array<std::string_view, 6> kMaxineCopied{
    "lib", "external", "include", "share", "features/nvvfxtruehdr", "version.h"};

// Chemin CUDA du RTX Video SDK 1.1 : la DLL d'entrée et la fonction NGX TrueHDR qu'elle
// charge. Noms à confirmer sur rtx_video_sdk_v1.1.0.zip (voir rtx_video_abi.h).
constexpr std::string_view kRtxDll = "rtx_video_api.dll";
constexpr std::array<RequiredFile, 2> kRtxRequired{{
    {.path = kRtxDll,
     .format = BinaryFormat::PeX64,
     .missing = "la DLL du RTX Video SDK 1.1 manque : choisir le dossier du SDK qui contient ses DLL"},
    {.path = "nvngx_truehdr.dll",
     .format = BinaryFormat::PeX64,
     .missing = "nvngx_truehdr.dll manque : choisir le dossier du SDK qui contient ses DLL"},
}};

constexpr std::array<std::string_view, 4> kRtxCopied{kRtxDll, "nvngx_truehdr.dll", "nvngx_vsr.dll",
                                                     "cudart64_12.dll"};

constexpr SdkLayout kRtxLayout{.kind = SdkKind::RtxVideo,
                               .display_name = "RTX Video SDK 1.1",
                               .directory = "rtx-video",
                               .main_library = kRtxDll,
                               .required = kRtxRequired,
                               .copied = kRtxCopied,
                               .major = 1,
                               .minor = 1};

constexpr SdkLayout kMaxineLayout{.kind = SdkKind::MaxineVfx,
                                  .display_name = "Maxine VFX 1.3 (TrueHDR)",
                                  .directory = "maxine-vfx",
                                  .main_library = "lib/libVideoFX.so",
                                  .required = kMaxineRequired,
                                  .copied = kMaxineCopied,
                                  .major = 1,
                                  .minor = 3};

}  // namespace

const SdkLayout& layout(SdkKind kind) {
  return kind == SdkKind::RtxVideo ? kRtxLayout : kMaxineLayout;
}

std::string_view display_name(SdkKind kind) {
  return layout(kind).display_name;
}

std::optional<std::filesystem::path> locate(const std::filesystem::path& root, const RequiredFile& file) {
  std::filesystem::path relative(file.path);
  std::error_code error;
  if (!file.prefix) {
    if (std::filesystem::exists(root / relative, error)) return relative;
    return std::nullopt;
  }
  const std::string stem = relative.filename().string();
  std::vector<std::string> names;
  for (std::filesystem::directory_iterator it(root / relative.parent_path(), error), end; !error && it != end;
       it.increment(error)) {
    std::string name = it->path().filename().string();
    if (name.starts_with(stem)) names.push_back(std::move(name));
  }
  if (names.empty()) return std::nullopt;
  return relative.parent_path() / *std::ranges::min_element(names);
}

}  // namespace ajp::vendor_hdr
