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

TEST_CASE("version : version.h à la manière de Maxine", "[version]") {
  const auto version = parse_version_header(
      "#ifndef VERSION_H\n#define NVVFX_MAJOR_VERSION 1\n#  define NVVFX_MINOR_VERSION   3 // commentaire\n"
      "#define NVVFX_PATCH_VERSION 0\n#define NVVFX_BUILD_VERSION 7\n");
  REQUIRE(version);
  CHECK(version->text() == "1.3.0.7");

  // Le premier préfixe complet l'emporte ; correctif et build sont facultatifs.
  CHECK(parse_version_header("#define OTHER_MAJOR_VERSION 9\n#define NVVFX_MAJOR_VERSION 1\n"
                             "#define NVVFX_MINOR_VERSION 3\n")
            ->text() == "1.3.0.0");
  CHECK_FALSE(parse_version_header("#define NVVFX_MAJOR_VERSION 1\n"));
  CHECK_FALSE(parse_version_header("#define NVVFX_MAJOR_VERSION one\n#define NVVFX_MINOR_VERSION 3\n"));
}

TEST_CASE("version : nom réel versionné", "[version]") {
  CHECK(parse_versioned_name("libVideoFX.so.1.3.0", "libVideoFX.so")->text() == "1.3.0.0");
  CHECK_FALSE(parse_versioned_name("libVideoFX.so", "libVideoFX.so"));
  CHECK_FALSE(parse_versioned_name("libVideoFX.so.", "libVideoFX.so"));
  CHECK_FALSE(parse_versioned_name("libVideoFXLocal.so.1.3", "libVideoFX.so"));
}

TEST_CASE("version : SDK Maxine", "[version]") {
  TempDir dir;
  SECTION("version.h à la racine") {
    write_file(dir.path() / "version.h", "#define NVVFX_MAJOR_VERSION 1\n#define NVVFX_MINOR_VERSION 3\n");
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path())->text() == "1.3.0.0");
  }
  SECTION("version.h dans include/") {
    write_file(dir.path() / "include/version.h",
               "#define NVVFX_MAJOR_VERSION 1\n#define NVVFX_MINOR_VERSION 2\n");
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path())->text() == "1.2.0.0");
  }
  SECTION("nom réel derrière le lien lib/libVideoFX.so") {
    write_file(dir.path() / "lib/libVideoFX.so.1.3.0.0", elf_header());
    std::filesystem::create_symlink("libVideoFX.so.1.3.0.0", dir.path() / "lib/libVideoFX.so");
    CHECK(read_sdk_version(SdkKind::MaxineVfx, dir.path())->text() == "1.3.0.0");
  }
  SECTION("sans marqueur, refusé") {
    write_file(dir.path() / "lib/libVideoFX.so", elf_header());
    auto refused = read_sdk_version(SdkKind::MaxineVfx, dir.path());
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::SdkRejected);
  }
}

TEST_CASE("version : RTX Video SDK", "[version]") {
  TempDir dir;
  make_rtx_sdk(dir.path(), SdkVersion{.parts = {1, 1, 0, 0}});
  CHECK(read_sdk_version(SdkKind::RtxVideo, dir.path())->text() == "1.1.0.0");
  make_rtx_sdk(dir.path(), std::nullopt);
  CHECK(read_sdk_version(SdkKind::RtxVideo, dir.path()).error().detail ==
        "version du SDK introuvable : rtx_video_api.dll n'a pas de FileVersion");
}
