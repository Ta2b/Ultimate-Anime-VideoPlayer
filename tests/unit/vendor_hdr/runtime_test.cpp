// Bibliothèques du SDK Maxine ouvertes par chemin à la place de LD_LIBRARY_PATH. Fige ce que
// le lecteur actuel obtenait par ta2b/mpv-animejanai-hdr launch-linux.sh:29-33 (lib/, fonction,
// external/cuda/lib et external/tensorrt/lib) et animejanai-inference-rtx@118c272
// src/pipeline_hdr.c:55-61 (toute la fonction s'ouvre, sinon refus).
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "support.h"
#include "vendor_hdr/maxine_runtime.h"
#include "vendor_hdr/sdk_layout.h"

using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;
namespace fs = std::filesystem;

namespace {

// Bibliothèques exigées par sdk_layout pour Maxine : 11 du cœur, 2 de la fonction.
constexpr std::size_t kRequiredLibraries = 13;

std::vector<std::string> relatives(const std::vector<LoadedLibrary>& loaded) {
  std::vector<std::string> names;
  for (const LoadedLibrary& library : loaded) names.push_back(library.relative.generic_string());
  return names;
}

std::ptrdiff_t position(const std::vector<std::string>& names, std::string_view name) {
  return std::ranges::find(names, name) - names.begin();
}

bool contains(const std::vector<std::string>& names, std::string_view name) {
  return std::ranges::find(names, name) != names.end();
}

std::string refusal_of(const fs::path& root) {
  auto loaded = load_maxine_runtime(root);
  REQUIRE_FALSE(loaded);
  CHECK(loaded.error().code == RefusalCode::LibraryUnavailable);
  return loaded.error().detail;
}

}  // namespace

TEST_CASE("runtime : noms de bibliothèques partagées", "[runtime]") {
  for (const char* name :
       {"libVideoFX.so", "libnvinfer.so.10", "libnvidia-ngx-truehdr.so.1.3.0", "libcudnn_ops.so.9"}) {
    CHECK(is_library_name(name));
  }
  for (const char* name : {"lib.so", "VideoFX.so", "libVideoFX.so.", "libVideoFX.so.1a", "libVideoFX.sox",
                           "libfoo.so..1", "libfoo.a", "libfoo.so.1.debug", "notes.txt"}) {
    CHECK_FALSE(is_library_name(name));
  }
}

TEST_CASE("runtime : tout ce que LD_LIBRARY_PATH aurait trouvé est ouvert", "[runtime]") {
  TempDir dir;
  make_maxine_sdk(dir.path(), {.libraries = LibraryContent::Fakes});
  // Dépendances que le SDK ouvrirait par leur nom, hors du jeu minimal exigé.
  fs::copy_file(AJP_FAKE_DEP_PATH, dir.path() / "external/cuda/lib/libcublas.so.12");
  fs::copy_file(AJP_FAKE_DEP_PATH, dir.path() / "external/tensorrt/lib/libnvinfer_plugin.so.10");
  // Une bibliothèque facultative qui ne se charge pas ne bloque rien.
  write_file(dir.path() / "external/tensorrt/lib/libnvinfer_builder_resource.so.10", elf_header());
  // Ni l'alias d'une bibliothèque déjà retenue, ni ce qui n'est pas une bibliothèque.
  fs::create_symlink("libVideoFX.so", dir.path() / "lib/libVideoFX.so.1");
  write_file(dir.path() / "lib/notes.txt", "pas une bibliothèque");

  auto loaded = load_maxine_runtime(dir.path());
  REQUIRE(loaded);
  const std::vector<std::string> names = relatives(*loaded);
  CHECK(names.size() == kRequiredLibraries + 2);
  CHECK(contains(names, "external/cuda/lib/libcublas.so.12"));
  CHECK(contains(names, "external/tensorrt/lib/libnvinfer_plugin.so.10"));
  CHECK_FALSE(contains(names, "external/tensorrt/lib/libnvinfer_builder_resource.so.10"));
  CHECK_FALSE(contains(names, "lib/libVideoFX.so.1"));
  CHECK_FALSE(contains(names, "lib/notes.txt"));

  // Dépendances d'abord, puis le cœur, puis la fonction.
  CHECK(position(names, "external/tensorrt/lib/libnvinfer.so.10") < position(names, "lib/libVideoFX.so"));
  CHECK(position(names, "lib/libVideoFX.so") <
        position(names, "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so"));

  CHECK(find_library(*loaded, "lib/libVideoFX.so") != nullptr);
  CHECK(find_library(*loaded, "lib/absent.so") == nullptr);
}

TEST_CASE("runtime : une bibliothèque exigée qui ne s'ouvre pas refuse", "[runtime]") {
  TempDir dir;
  make_maxine_sdk(dir.path(), {.libraries = LibraryContent::Fakes});
  SECTION("illisible par le chargeur") {
    write_file(dir.path() / "lib/libVideoFXLocal.so", elf_header());
    CHECK(refusal_of(dir.path()).starts_with("libVideoFXLocal.so ne se charge pas : "));
  }
  SECTION("lien pendant") {
    fs::remove(dir.path() / "external/cuda/lib/libnppc.so.12");
    fs::create_symlink("libnppc.so.12.4", dir.path() / "external/cuda/lib/libnppc.so.12");
    CHECK(refusal_of(dir.path()) == "external/cuda/lib/libnppc.so.12 absent du SDK");
  }
}

TEST_CASE("runtime : toute la fonction TrueHDR doit s'ouvrir", "[runtime]") {
  // pipeline_hdr.c:59 refuse une bibliothèque de la fonction qui ne se charge pas.
  TempDir dir;
  make_maxine_sdk(dir.path(), {.libraries = LibraryContent::Fakes});
  write_file(dir.path() / "features/nvvfxtruehdr/lib/libnvVFXTrueHDRUtils.so.1", elf_header());
  CHECK(refusal_of(dir.path()).starts_with("libnvVFXTrueHDRUtils.so.1 ne se charge pas : "));

  fs::copy_file(AJP_FAKE_DEP_PATH, dir.path() / "features/nvvfxtruehdr/lib/libnvVFXTrueHDRUtils.so.1",
                fs::copy_options::overwrite_existing);
  auto loaded = load_maxine_runtime(dir.path());
  REQUIRE(loaded);
  CHECK(contains(relatives(*loaded), "features/nvvfxtruehdr/lib/libnvVFXTrueHDRUtils.so.1"));
}
