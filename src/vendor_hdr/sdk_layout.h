// Contenu attendu de chaque SDK TrueHDR : fichiers exigés, parties copiées dans le dossier de
// données, version prise en charge. Une table par SDK : une nouvelle disposition d'archive se
// suit ici, et nulle part ailleurs.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "vendor_hdr/binary_probe.h"
#include "vendor_hdr/sdk_kind.h"

namespace ajp::vendor_hdr {

// Dossier des bibliothèques de la fonction TrueHDR de Maxine.
inline constexpr std::string_view kMaxineFeatureLibraries = "features/nvvfxtruehdr/lib";

// Marqueur de version du SDK Maxine (VFXSDK_linux_1.3.0.0) et préfixe de ses macros :
// NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR, _MINOR, _RELEASE, _BUILD.
inline constexpr std::string_view kMaxineVersionHeader = "share/videofx_version.h";
inline constexpr std::string_view kMaxineVersionPrefix = "NVIDIA_VIDEOEFFECTS_SDK_VERSION";
// En-tête de la fonction TrueHDR, qui porte sa propre version : #define NVVFXTRUEHDR_VERSION "1.3.0.0".
inline constexpr std::string_view kMaxineFeatureHeader = "features/nvvfxtruehdr/include/nvVFXTrueHDR.h";
inline constexpr std::string_view kMaxineFeatureVersion = "NVVFXTRUEHDR_VERSION";

struct RequiredFile {
  std::string_view path;  // relatif à la racine du SDK, séparateurs « / »
  BinaryFormat format = BinaryFormat::ElfX86_64;
  bool prefix = false;       // le nom du fichier commence seulement par celui de path (« .so.1.3 »)
  std::string_view missing;  // ce que l'utilisateur doit faire s'il manque
};

struct SdkLayout {
  SdkKind kind = SdkKind::RtxVideo;
  std::string_view display_name;
  std::string_view directory;     // sous-dossier du magasin : « rtx-video », « maxine-vfx »
  std::string_view main_library;  // bibliothèque d'entrée du SDK, chargée par son chargeur
  std::span<const RequiredFile> required;
  std::span<const std::string_view> copied;      // fichiers ou dossiers copiés quand ils existent
  BinaryFormat libraries = BinaryFormat::PeX64;  // format de toutes ses bibliothèques
  std::uint32_t major = 0;                       // version prise en charge : majeur et mineur
  std::uint32_t minor = 0;
};

const SdkLayout& layout(SdkKind kind);

// Chemin relatif réel d'un fichier exigé sous root : le chemin exact, ou pour un préfixe le
// premier nom du dossier, par ordre alphabétique, qui commence par lui.
std::optional<std::filesystem::path> locate(const std::filesystem::path& root, const RequiredFile& file);

// Bibliothèques Maxine que le chargeur ouvre, toutes exigées, dans l'ordre de leurs
// dépendances : CUDA et NPP, puis le cœur, puis la fonction (maxine_runtime.h).
std::span<const std::string_view> maxine_runtime();

// Toutes les bibliothèques d'un SDK que le chargeur ou le SDK peuvent ouvrir sous root, en
// chemins relatifs. Un lien pendant n'en est pas une.
//   Maxine : celles d'external/ que maxine_runtime() nomme, dans son ordre, puis celles de lib/
//            et de la fonction, triées par dossier puis par nom ;
//   RTX Video : les DLL copiées présentes à la racine.
// Le magasin vérifie leur architecture à l'import, comme TrueHdrSdkStore.cs:134-135.
std::vector<std::filesystem::path> loadable_libraries(SdkKind kind, const std::filesystem::path& root);

// « libnvinfer.so.10 » : nom de bibliothèque partagée Linux, comme Library du Manager
// (manager-v2 Services/TrueHdrSdkStore.cs:28, ^lib[A-Za-z0-9_-]+\.so(\.[0-9]+)*$).
bool is_library_name(std::string_view name);

}  // namespace ajp::vendor_hdr
