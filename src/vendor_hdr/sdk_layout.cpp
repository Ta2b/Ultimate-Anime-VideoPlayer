#include "vendor_hdr/sdk_layout.h"

#include <algorithm>
#include <array>
#include <string>
#include <system_error>
#include <vector>

#include "vendor_hdr/path_text.h"

namespace ajp::vendor_hdr {
namespace {

constexpr std::string_view kMaxineCore =
    "le dossier Core Linux x64 de Maxine VFX 1.3 est incomplet (CUDA 12, NPP et cuDNN compris)";
constexpr std::string_view kMaxineFeature =
    "la fonction TrueHDR manque : l'installer avec install_feature.sh --feature nvvfxtruehdr";
constexpr std::string_view kMaxineHeaders =
    "les en-têtes du SDK manquent : choisir le dossier complet du SDK, include/ compris";

// Ce que TrueHDR charge vraiment, relevé dans /proc/self/maps d'un processus qui l'a créé, chargé
// et exécuté sur la RTX (VFXSDK 1.3.0.0, 19 octobre 2026), dans l'ordre des dépendances lues
// par readelf -d : libcudart et NPP (libnppc d'abord), libcudnn.so.9 (exigée par
// libVideoFXLocal, ses sous-bibliothèques ne se chargent jamais), puis le cœur et la fonction.
// TensorRT, cuBLAS et cuFFT ne sont jamais chargés : ni exigés ni copiés. Le modèle
// libnvidia-ngx-truehdr.so.* est chargé par le runtime NGX lui-même (memfd), pas par nous.
constexpr std::array<std::string_view, 12> kMaxineRuntime{
    "external/cuda/lib/libcudart.so.12",
    "external/cuda/lib/libnppc.so.12",
    "external/cuda/lib/libnppial.so.12",
    "external/cuda/lib/libnppicc.so.12",
    "external/cuda/lib/libnppidei.so.12",
    "external/cuda/lib/libnppig.so.12",
    "external/cuda/lib/libcudnn.so.9",
    "lib/libNVCVImage.so",
    "lib/libVideoFX.so",
    "lib/libVideoFXLocal.so",
    "lib/libnvngxruntime.so",
    "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so",
};

// Le jeu d'exécution, plus le modèle NGX de la fonction et les en-têtes que vérifie
// maxine_abi_check (manager-v2 Services/TrueHdrSdkStore.cs, CoreRequired et ValidateRequired).
constexpr auto kMaxineRequired = [] {
  std::array<RequiredFile, kMaxineRuntime.size() + 4> files{};
  std::size_t at = 0;
  for (const std::string_view path : kMaxineRuntime) {
    files[at++] = {.path = path, .missing = path.starts_with("features/") ? kMaxineFeature : kMaxineCore};
  }
  files[at++] = {.path = "features/nvvfxtruehdr/lib/libnvidia-ngx-truehdr.so",
                 .prefix = true,
                 .missing = kMaxineFeature};
  files[at++] = {.path = "include/nvCVImage.h", .format = BinaryFormat::Text, .missing = kMaxineHeaders};
  files[at++] = {.path = "include/nvVideoEffects.h", .format = BinaryFormat::Text, .missing = kMaxineHeaders};
  files[at++] = {.path = kMaxineFeatureHeader, .format = BinaryFormat::Text, .missing = kMaxineHeaders};
  return files;
}();

// Copiés : le cœur, les en-têtes, share/ (version et licences), la fonction, et d'external/
// les seules bibliothèques chargées (environ 100 Mo au lieu de 4,3 Go : cuBLAS, cuFFT, cuDNN
// complet et TensorRT restent dans le SDK de l'utilisateur).
constexpr auto kMaxineCopied = [] {
  std::array<std::string_view, 11> entries{"lib", "include", "share", "features/nvvfxtruehdr"};
  std::size_t at = 4;
  for (const std::string_view path : kMaxineRuntime) {
    if (path.starts_with("external/")) entries[at++] = path;
  }
  return entries;
}();

// Chemin CUDA du RTX Video SDK 1.1 : le pont aji_rtx.dll (voir rtx_video_abi.h) et la fonction
// NGX TrueHDR qu'il charge à côté de lui.
constexpr std::string_view kRtxDll = "aji_rtx.dll";
constexpr std::array<RequiredFile, 2> kRtxRequired{{
    {.path = kRtxDll,
     .format = BinaryFormat::PeX64,
     .missing =
         "aji_rtx.dll manque : choisir le dossier qui contient le pont aji_rtx et les DLL du RTX Video SDK"},
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
                               .libraries = BinaryFormat::PeX64,
                               .major = 1,
                               .minor = 1};

constexpr SdkLayout kMaxineLayout{.kind = SdkKind::MaxineVfx,
                                  .display_name = "Maxine VFX 1.3 (TrueHDR)",
                                  .directory = "maxine-vfx",
                                  .main_library = "lib/libVideoFX.so",
                                  .required = kMaxineRequired,
                                  .copied = kMaxineCopied,
                                  .libraries = BinaryFormat::ElfX86_64,
                                  .major = 1,
                                  .minor = 3};

// Dossiers dont chaque bibliothèque se vérifie à l'import, en plus des externes chargées.
constexpr std::array<std::string_view, 2> kMaxineLibraryDirectories{"lib", kMaxineFeatureLibraries};

bool is_name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

// Un fichier, lien suivi : un lien pendant ou un dossier n'est pas une bibliothèque.
bool is_file(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

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
  const std::string stem = utf8(relative.filename());
  std::vector<std::string> names;
  for (std::filesystem::directory_iterator it(root / relative.parent_path(), error), end; !error && it != end;
       it.increment(error)) {
    std::string name = utf8(it->path().filename());
    if (name.starts_with(stem)) names.push_back(std::move(name));
  }
  if (names.empty()) return std::nullopt;
  return relative.parent_path() / *std::ranges::min_element(names);
}

std::span<const std::string_view> maxine_runtime() {
  return kMaxineRuntime;
}

bool is_library_name(std::string_view name) {
  const std::size_t suffix = name.find(".so");
  if (!name.starts_with("lib") || suffix == std::string_view::npos || suffix == 3) return false;
  if (!std::ranges::all_of(name.substr(3, suffix - 3), is_name_char)) return false;
  for (std::string_view rest = name.substr(suffix + 3); !rest.empty();) {
    if (rest.front() != '.') return false;
    rest.remove_prefix(1);
    const std::size_t digits = std::min(rest.find_first_not_of("0123456789"), rest.size());
    if (digits == 0) return false;
    rest.remove_prefix(digits);
  }
  return true;
}

std::vector<std::filesystem::path> loadable_libraries(SdkKind kind, const std::filesystem::path& root) {
  std::vector<std::filesystem::path> libraries;
  if (kind == SdkKind::RtxVideo) {
    for (const std::string_view copied : kRtxCopied) {
      if (is_file(root / copied)) libraries.emplace_back(copied);
    }
    return libraries;
  }
  for (const std::string_view runtime : kMaxineRuntime) {
    if (runtime.starts_with("external/") && is_file(root / runtime)) libraries.emplace_back(runtime);
  }
  for (const std::string_view directory : kMaxineLibraryDirectories) {
    std::vector<std::filesystem::path> names;
    std::error_code error;
    for (std::filesystem::directory_iterator it(root / directory, error), end; !error && it != end;
         it.increment(error)) {
      if (is_library_name(utf8(it->path().filename())) && is_file(it->path())) {
        names.push_back(std::filesystem::path(directory) / it->path().filename());
      }
    }
    std::ranges::sort(names);
    libraries.insert(libraries.end(), names.begin(), names.end());
  }
  return libraries;
}

}  // namespace ajp::vendor_hdr
