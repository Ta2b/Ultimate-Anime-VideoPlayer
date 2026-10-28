// Bibliothèques du SDK Maxine ouvertes par chemin à la place de LD_LIBRARY_PATH. Fige le jeu
// relevé sur le vrai SDK 1.3.0.0 (sdk_layout.cpp, kMaxineRuntime) : exactement celles que
// TrueHDR charge, dans l'ordre de leurs dépendances, toutes exigées.
#include <cstddef>
#include <filesystem>
#include <string>
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

std::vector<std::string> relatives(const std::vector<LoadedLibrary>& loaded) {
  std::vector<std::string> names;
  for (const LoadedLibrary& library : loaded) names.push_back(library.relative.generic_string());
  return names;
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

TEST_CASE("runtime : le jeu relevé sur le SDK 1.3.0.0", "[runtime]") {
  const std::vector<std::string> expected{"external/cuda/lib/libcudart.so.12",
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
                                          "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so"};
  CHECK(std::vector<std::string>(maxine_runtime().begin(), maxine_runtime().end()) == expected);
  // Chacune est exigée par le magasin.
  for (const std::string& runtime : expected) {
    bool required = false;
    for (const RequiredFile& file : layout(SdkKind::MaxineVfx).required)
      required = required || file.path == runtime;
    CHECK(required);
  }
}

TEST_CASE("runtime : seules les bibliothèques du jeu sont ouvertes, dans son ordre", "[runtime]") {
  TempDir dir;
  make_maxine_sdk(dir.path(), {.libraries = LibraryContent::Fakes});
  // Ce que LD_LIBRARY_PATH aurait pu trouver, mais que TrueHDR ne charge jamais.
  fs::copy_file(AJP_FAKE_DEP_PATH, dir.path() / "external/cuda/lib/libcublas.so.12");
  fs::create_directories(dir.path() / "external/tensorrt/lib");
  fs::copy_file(AJP_FAKE_DEP_PATH, dir.path() / "external/tensorrt/lib/libnvinfer.so.10");
  write_file(dir.path() / "lib/libVideoFXTriton.so", elf_header());  // ne se charge pas : jamais tenté

  auto loaded = load_maxine_runtime(dir.path());
  REQUIRE(loaded);
  CHECK(relatives(*loaded) == std::vector<std::string>(maxine_runtime().begin(), maxine_runtime().end()));
  CHECK(find_library(*loaded, "lib/libVideoFX.so") != nullptr);
  CHECK(find_library(*loaded, "external/cuda/lib/libcublas.so.12") == nullptr);
}

TEST_CASE("runtime : une bibliothèque du jeu qui manque ou ne s'ouvre pas refuse", "[runtime]") {
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
  SECTION("fonction TrueHDR absente") {
    fs::remove(dir.path() / "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so");
    CHECK(refusal_of(dir.path()) == "features/nvvfxtruehdr/lib/libnvVFXTrueHDR.so absent du SDK");
  }
}
