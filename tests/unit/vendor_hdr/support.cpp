#include "support.h"

#include <format>
#include <fstream>
#include <random>
#include <variant>

#include "vendor_hdr/maxine_abi.h"
#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr::test {
namespace {

namespace fs = std::filesystem;

void put16(std::string& bytes, std::size_t at, std::uint32_t value) {
  bytes[at] = static_cast<char>(value & 0xFFU);
  bytes[at + 1] = static_cast<char>((value >> 8U) & 0xFFU);
}

void put32(std::string& bytes, std::size_t at, std::uint32_t value) {
  put16(bytes, at, value & 0xFFFFU);
  put16(bytes, at + 2, value >> 16U);
}

std::string read_all(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string library_bytes(LibraryContent content, std::string_view path) {
  switch (content) {
    case LibraryContent::ElfStub:
      return elf_header();
    case LibraryContent::Empty:
      return {};
    case LibraryContent::Fakes:
      // libVideoFX et libNVCVImage portent les fonctions du SDK ; les autres ne font que se charger.
      return read_all(path == "lib/libVideoFX.so" || path == "lib/libNVCVImage.so" ? AJP_FAKE_MAXINE_PATH
                                                                                   : AJP_FAKE_DEP_PATH);
  }
  return {};
}

}  // namespace

TempDir::TempDir() {
  std::random_device entropy;
  path_ = fs::temp_directory_path() / std::format("ajp-vendor-hdr-{:08x}{:08x}", entropy(), entropy());
  fs::create_directories(path_);
}

TempDir::~TempDir() {
  std::error_code ignored;
  fs::remove_all(path_, ignored);
}

void write_file(const fs::path& file, std::string_view bytes) {
  fs::create_directories(file.parent_path());
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out << bytes;
}

std::string elf_header(std::uint16_t machine) {
  std::string bytes(64, '\0');
  bytes[0] = 0x7F;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;         // ELFCLASS64
  bytes[5] = 1;         // petit-boutiste
  bytes[6] = 1;         // EV_CURRENT
  put16(bytes, 16, 3);  // ET_DYN
  put16(bytes, 18, machine);
  put32(bytes, 20, 1);
  return bytes;
}

std::string pe_dll(std::optional<SdkVersion> version, std::uint16_t machine) {
  std::string bytes(0x300, '\0');
  bytes[0] = 'M';
  bytes[1] = 'Z';
  put32(bytes, 0x3C, 0x80);
  bytes[0x80] = 'P';
  bytes[0x81] = 'E';
  put16(bytes, 0x84, machine);
  put16(bytes, 0x86, 1);       // une section
  put16(bytes, 0x94, 0xF0);    // taille de l'en-tête optionnel PE32+
  put16(bytes, 0x96, 0x2022);  // DLL, exécutable, grandes adresses
  put16(bytes, 0x98, 0x20B);   // PE32+
  constexpr std::size_t kSection = 0x80 + 24 + 0xF0;
  bytes.replace(kSection, 5, ".rsrc");
  put32(bytes, kSection + 16, 0x100);  // SizeOfRawData
  put32(bytes, kSection + 20, 0x200);  // PointerToRawData
  if (version) {
    constexpr std::size_t kFixed = 0x200 + 40;  // après l'en-tête de VS_VERSIONINFO
    put32(bytes, kFixed, 0xFEEF04BD);
    put32(bytes, kFixed + 4, 0x00010000);
    put32(bytes, kFixed + 8, (version->parts[0] << 16U) | version->parts[1]);
    put32(bytes, kFixed + 12, (version->parts[2] << 16U) | version->parts[3]);
  }
  return bytes;
}

std::string abi_header_text() {
  std::string text = "#pragma once\n";
  for (const maxine::AbiConstant& constant : maxine::kAbi) {
    if (const auto* number = std::get_if<long long>(&constant.value)) {
      text += std::format("#define {} {}\n", constant.name, *number);
    } else {
      text += std::format("#define {} \"{}\"\n", constant.name, std::get<std::string_view>(constant.value));
    }
  }
  return text;
}

std::string version_header_text(const SdkVersion& version) {
  const auto [major, minor, release, build] = version.parts;
  return std::format(
      "/* licence MIT */\n\n#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR {}\n"
      "#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MINOR {}\n#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_RELEASE {}\n"
      "#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_BUILD {}\n\n#define NVIDIA_VIDEOEFFECTS_SDK_VERSION "
      "{},{},{},{}\n"
      "#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR_MINOR {},{}\n",
      major, minor, release, build, major, minor, release, build, major, minor);
}

void make_maxine_sdk(const fs::path& root, const MaxineSdkOptions& options) {
  for (const RequiredFile& required : layout(SdkKind::MaxineVfx).required) {
    std::string path(required.path);
    if (required.prefix) path += ".1.3.0";
    const bool feature = path.starts_with("features/nvvfxtruehdr/lib");
    if (feature && !options.feature) continue;
    if (required.format == BinaryFormat::Text) {
      if (!options.headers) continue;
      std::string text = "#pragma once\n";
      if (path == "include/nvCVImage.h") text = abi_header_text();
      if (path == kMaxineFeatureHeader) {
        text += std::format("#define NVVFXTRUEHDR_VERSION \"{}\"\n", options.feature_version);
      }
      write_file(root / path, text);
      continue;
    }
    write_file(root / path, library_bytes(options.libraries, path));
  }
  if (options.version_header) {
    write_file(root / kMaxineVersionHeader,
               version_header_text(parse_version(*options.version_header).value()));
  }
  fs::create_directories(root / "lib" / "models");
  write_file(root / "share" / "license.txt", "Licence du SDK (factice)\n");
}

void make_rtx_sdk(const fs::path& root, std::optional<SdkVersion> version) {
  write_file(root / layout(SdkKind::RtxVideo).main_library, pe_dll(std::nullopt));
  write_file(root / "nvngx_truehdr.dll", pe_dll(version));
}

Done RecordingStream::hand_to_legacy() {
  events.emplace_back("hand");
  if (fail_hand) return refuse(RefusalCode::VendorCallFailed, "cuEventRecord a échoué");
  return {};
}

Done RecordingStream::take_from_legacy() {
  events.emplace_back("take");
  return {};
}

HostImage::HostImage(int image_width, int image_height, TrueHdrPixel image_pixel)
    : pixels(static_cast<std::size_t>(image_width) * static_cast<std::size_t>(image_height)),
      width(image_width),
      height(image_height),
      pixel(image_pixel) {}

TrueHdrImage HostImage::view() {
  return TrueHdrImage{.device = reinterpret_cast<std::uintptr_t>(pixels.data()),
                      .pitch = static_cast<std::size_t>(width) * 4,
                      .width = width,
                      .height = height,
                      .pixel = pixel};
}

}  // namespace ajp::vendor_hdr::test
