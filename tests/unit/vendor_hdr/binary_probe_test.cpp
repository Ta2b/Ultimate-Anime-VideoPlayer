// Formats et versions lus dans les fichiers d'un SDK, sans rien charger.
// check_binary reprend ValidateElf de manager-v2 Services/TrueHdrSdkStore.cs.
#include "vendor_hdr/binary_probe.h"

#include <catch2/catch_test_macros.hpp>

#include "support.h"
#include "vendor_hdr/sdk_version.h"

using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;

TEST_CASE("binaire : bibliothèque Linux x64", "[binary]") {
  TempDir dir;
  write_file(dir.path() / "good.so", elf_header());
  write_file(dir.path() / "arm.so", elf_header(183));
  write_file(dir.path() / "empty.so", "");
  write_file(dir.path() / "windows.so", pe_dll(std::nullopt));

  CHECK(check_binary(dir.path(), "good.so", BinaryFormat::ElfX86_64));
  CHECK(check_binary(dir.path(), "empty.so", BinaryFormat::ElfX86_64).error().detail == "empty.so est vide");
  CHECK(check_binary(dir.path(), "missing.so", BinaryFormat::ElfX86_64).error().detail ==
        "missing.so est absent");
  for (const char* wrong : {"arm.so", "windows.so"}) {
    auto refused = check_binary(dir.path(), wrong, BinaryFormat::ElfX86_64);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::SdkRejected);
    CHECK(refused.error().detail.starts_with(std::string(wrong) +
                                             " n'est pas une bibliothèque Linux x64 (ELF)"));
  }
}

TEST_CASE("binaire : DLL Windows x64", "[binary]") {
  TempDir dir;
  write_file(dir.path() / "good.dll", pe_dll(std::nullopt));
  write_file(dir.path() / "arm64.dll", pe_dll(std::nullopt, 0xAA64));
  write_file(dir.path() / "linux.dll", elf_header());
  std::string exe = pe_dll(std::nullopt);
  exe[0x96] = 0x22;  // caractéristiques sans IMAGE_FILE_DLL
  exe[0x97] = 0x00;
  write_file(dir.path() / "program.dll", exe);

  CHECK(check_binary(dir.path(), "good.dll", BinaryFormat::PeX64));
  for (const char* wrong : {"arm64.dll", "linux.dll", "program.dll"}) {
    CHECK(check_binary(dir.path(), wrong, BinaryFormat::PeX64).error().detail ==
          std::string(wrong) + " n'est pas une DLL Windows x64");
  }
}

TEST_CASE("binaire : un en-tête n'a qu'à exister et ne pas être vide", "[binary]") {
  TempDir dir;
  write_file(dir.path() / "include/a.h", "#pragma once\n");
  write_file(dir.path() / "include/empty.h", "");
  CHECK(check_binary(dir.path(), "include/a.h", BinaryFormat::Text));
  CHECK(check_binary(dir.path(), "include/empty.h", BinaryFormat::Text).error().detail ==
        "include/empty.h est vide");
}

TEST_CASE("binaire : FileVersion d'une DLL", "[binary]") {
  TempDir dir;
  write_file(dir.path() / "versioned.dll", pe_dll(SdkVersion{.parts = {1, 1, 0, 42}}));
  write_file(dir.path() / "bare.dll", pe_dll(std::nullopt));
  write_file(dir.path() / "linux.so", elf_header());

  const auto version = pe_file_version(dir.path() / "versioned.dll");
  REQUIRE(version);
  CHECK(version->text() == "1.1.0.42");
  CHECK_FALSE(pe_file_version(dir.path() / "bare.dll"));
  CHECK_FALSE(pe_file_version(dir.path() / "linux.so"));
  CHECK_FALSE(pe_file_version(dir.path() / "missing.dll"));
}

TEST_CASE("version : texte", "[version]") {
  CHECK(parse_version("1.3")->text() == "1.3.0.0");
  CHECK(parse_version("1.3.0.21")->parts == std::array<std::uint32_t, 4>{1, 3, 0, 21});
  for (const char* wrong : {"", "1", "1.", ".3", "1.3.0.0.0", "1.a", "v1.3", "1..3"}) {
    CHECK_FALSE(parse_version(wrong));
  }
}

TEST_CASE("version : share/videofx_version.h de Maxine", "[version]") {
  constexpr std::string_view kPrefix = "NVIDIA_VIDEOEFFECTS_SDK_VERSION";
  // Le texte réel de VFXSDK_linux 1.3.0.0, licence omise.
  const auto version = parse_version_header(version_header_text(SdkVersion{.parts = {1, 3, 0, 0}}), kPrefix);
  REQUIRE(version);
  CHECK(version->text() == "1.3.0.0");

  CHECK(parse_version_header("#  define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MINOR   3 // commentaire\n"
                             "#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR 1\n"
                             "#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_BUILD 7\n",
                             kPrefix)
            ->text() == "1.3.0.7");
  // _MAJOR_MINOR (« 1,3 ») n'est pas _MAJOR ; release et build sont facultatifs.
  CHECK_FALSE(parse_version_header("#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR_MINOR 1,3\n", kPrefix));
  CHECK_FALSE(parse_version_header("#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR 1\n", kPrefix));
  CHECK_FALSE(
      parse_version_header("#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR one\n"
                           "#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MINOR 3\n",
                           kPrefix));
  CHECK_FALSE(parse_version_header("#define NVVFX_MAJOR 1\n#define NVVFX_MINOR 3\n", kPrefix));
}

TEST_CASE("version : nom réel versionné", "[version]") {
  CHECK(parse_versioned_name("libVideoFX.so.1.3.0", "libVideoFX.so")->text() == "1.3.0.0");
  CHECK_FALSE(parse_versioned_name("libVideoFX.so", "libVideoFX.so"));
  CHECK_FALSE(parse_versioned_name("libVideoFX.so.", "libVideoFX.so"));
  CHECK_FALSE(parse_versioned_name("libVideoFXLocal.so.1.3", "libVideoFX.so"));
}

TEST_CASE("version : SDK Maxine", "[version]") {
  TempDir dir;
  // La fonction TrueHDR porte sa version, qui doit suivre celle du SDK.
  write_file(dir.path() / "features/nvvfxtruehdr/include/nvVFXTrueHDR.h",
             "#define NVVFXTRUEHDR_VERSION \"1.3.0.0\"\n#define NVVFX_FX_TRUE_HDR \"TrueHDR\"\n");
  SECTION("share/videofx_version.h") {
    write_file(dir.path() / "share/videofx_version.h",
               version_header_text(SdkVersion{.parts = {1, 3, 0, 0}}));
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path())->text() == "1.3.0.0");
  }
  SECTION("nom réel derrière le lien lib/libVideoFX.so") {
    write_file(dir.path() / "lib/libVideoFX.so.1.3.0", elf_header());
    std::filesystem::create_symlink("libVideoFX.so.1.3.0", dir.path() / "lib/libVideoFX.so");
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path())->text() == "1.3.0.0");
  }
  SECTION("les deux s'accordent") {
    write_file(dir.path() / "share/videofx_version.h",
               version_header_text(SdkVersion{.parts = {1, 3, 0, 0}}));
    write_file(dir.path() / "lib/libVideoFX.so.1.3.0", elf_header());
    std::filesystem::create_symlink("libVideoFX.so.1.3.0", dir.path() / "lib/libVideoFX.so");
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path())->text() == "1.3.0.0");
  }
  SECTION("le nom réel ne porte pas le build de l'en-tête") {
    // libVideoFX.so.1.3.0 pour un SDK 1.3.0.21 : seuls les nombres du nom se comparent.
    write_file(dir.path() / "share/videofx_version.h",
               version_header_text(SdkVersion{.parts = {1, 3, 0, 21}}));
    write_file(dir.path() / "lib/libVideoFX.so.1.3.0", elf_header());
    std::filesystem::create_symlink("libVideoFX.so.1.3.0", dir.path() / "lib/libVideoFX.so");
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path())->text() == "1.3.0.21");
    // Un correctif différent reste un SDK mélangé.
    write_file(dir.path() / "share/videofx_version.h",
               version_header_text(SdkVersion{.parts = {1, 3, 1, 0}}));
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path()).error().detail ==
          "share/videofx_version.h annonce 1.3.1.0, mais lib/libVideoFX.so est en 1.3.0.0 : SDK mélangé");
  }
  SECTION("sans marqueur, refusé") {
    write_file(dir.path() / "lib/libVideoFX.so", elf_header());
    auto refused = read_sdk_version(SdkKind::MaxineVfx, dir.path());
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::SdkRejected);
    CHECK(refused.error().detail.starts_with("version du SDK introuvable"));
  }
}

TEST_CASE("version : RTX Video SDK", "[version]") {
  TempDir dir;
  make_rtx_sdk(dir.path(), SdkVersion{.parts = {1, 1, 0, 0}});
  CHECK(read_sdk_version(SdkKind::RtxVideo, dir.path())->text() == "1.1.0.0");
  make_rtx_sdk(dir.path(), std::nullopt);
  CHECK(read_sdk_version(SdkKind::RtxVideo, dir.path()).error().detail ==
        "version du SDK introuvable : nvngx_truehdr.dll n'a pas de FileVersion");
}
