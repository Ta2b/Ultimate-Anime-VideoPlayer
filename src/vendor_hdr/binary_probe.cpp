#include "vendor_hdr/binary_probe.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "vendor_hdr/path_text.h"

namespace ajp::vendor_hdr {
namespace {

using Bytes = std::vector<char>;

// En-têtes ELF et PE, table des sections comprise, tiennent dans les premiers 64 Kio.
constexpr std::size_t kHeadBytes = std::size_t{64} * 1024;
// Une section .rsrc plus grande que 64 Mio n'est pas celle d'une DLL de SDK.
constexpr std::uint32_t kMaxResourceBytes = 64U * 1024U * 1024U;
constexpr std::uint32_t kFixedFileInfoSignature = 0xFEEF04BD;

// Au plus size octets à partir de offset ; moins si le fichier est plus court.
std::optional<Bytes> read_bytes(const std::filesystem::path& file, std::uint64_t offset, std::size_t size) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return std::nullopt;
  in.seekg(static_cast<std::streamoff>(offset));
  if (!in) return std::nullopt;
  Bytes bytes(size);
  in.read(bytes.data(), static_cast<std::streamsize>(size));
  bytes.resize(static_cast<std::size_t>(in.gcount()));
  return bytes;
}

// Entier petit-boutiste de width octets à la position at, s'il tient dans bytes.
std::optional<std::uint32_t> little_endian(const Bytes& bytes, std::size_t at, std::size_t width) {
  if (at > bytes.size() || bytes.size() - at < width) return std::nullopt;
  std::uint32_t value = 0;
  for (std::size_t i = width; i-- > 0;) value = (value << 8U) | static_cast<std::uint8_t>(bytes[at + i]);
  return value;
}

std::optional<std::uint32_t> u16(const Bytes& bytes, std::size_t at) {
  return little_endian(bytes, at, 2);
}
std::optional<std::uint32_t> u32(const Bytes& bytes, std::size_t at) {
  return little_endian(bytes, at, 4);
}

// Repris de manager-v2 Services/TrueHdrSdkStore.cs (ValidateElf) : ELF64, petit-boutiste,
// ET_DYN, EM_X86_64.
bool is_elf_x86_64(const Bytes& head) {
  constexpr std::array<std::uint8_t, 6> kIdent{0x7F, 'E', 'L', 'F', 2, 1};
  for (std::size_t i = 0; i < kIdent.size(); ++i) {
    if (head.size() <= i || static_cast<std::uint8_t>(head[i]) != kIdent[i]) return false;
  }
  return u16(head, 16) == 3U && u16(head, 18) == 62U;
}

struct PeLayout {
  std::size_t section_table = 0;
  std::uint32_t section_count = 0;
};

// DOS « MZ », signature « PE\0\0 », machine AMD64, DLL, en-tête optionnel PE32+.
std::optional<PeLayout> pe_layout(const Bytes& head) {
  if (head.size() < 0x40 || head[0] != 'M' || head[1] != 'Z') return std::nullopt;
  const std::optional<std::uint32_t> pe = u32(head, 0x3C);
  if (!pe || u32(head, *pe) != 0x00004550U) return std::nullopt;
  const std::optional<std::uint32_t> sections = u16(head, *pe + 6);
  const std::optional<std::uint32_t> optional_size = u16(head, *pe + 20);
  const std::optional<std::uint32_t> characteristics = u16(head, *pe + 22);
  if (u16(head, *pe + 4) != 0x8664U || !sections || !optional_size || !characteristics) return std::nullopt;
  if ((*characteristics & 0x2000U) == 0 || u16(head, *pe + 24) != 0x020BU) return std::nullopt;
  const std::size_t table = std::size_t{*pe} + 24 + *optional_size;
  if (table + std::size_t{*sections} * 40 > head.size()) return std::nullopt;
  return PeLayout{.section_table = table, .section_count = *sections};
}

Done check_head(const Bytes& head, BinaryFormat format, const std::string& shown) {
  switch (format) {
    case BinaryFormat::ElfX86_64:
      if (is_elf_x86_64(head)) return {};
      return refuse(RefusalCode::SdkRejected,
                    std::format("{} n'est pas une bibliothèque Linux x64 (ELF) : une archive Windows ou "
                                "ARM ne convient pas",
                                shown));
    case BinaryFormat::PeX64:
      if (pe_layout(head)) return {};
      return refuse(RefusalCode::SdkRejected, std::format("{} n'est pas une DLL Windows x64", shown));
    case BinaryFormat::Text:
      return {};
  }
  return {};
}

}  // namespace

Done check_binary(const std::filesystem::path& root, const std::filesystem::path& relative,
                  BinaryFormat format) {
  const std::string shown = utf8(relative);
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(root / relative, error);
  if (error) return refuse(RefusalCode::SdkRejected, std::format("{} est absent", shown));
  if (size == 0) return refuse(RefusalCode::SdkRejected, std::format("{} est vide", shown));
  const std::optional<Bytes> head = read_bytes(root / relative, 0, kHeadBytes);
  if (!head) return refuse(RefusalCode::IoFailed, std::format("{} est illisible", shown));
  return check_head(*head, format, shown);
}

std::optional<SdkVersion> pe_file_version(const std::filesystem::path& file) {
  const std::optional<Bytes> head = read_bytes(file, 0, kHeadBytes);
  if (!head) return std::nullopt;
  const std::optional<PeLayout> layout = pe_layout(*head);
  if (!layout) return std::nullopt;
  for (std::uint32_t i = 0; i < layout->section_count; ++i) {
    const std::size_t entry = layout->section_table + std::size_t{i} * 40;
    if (std::string_view(head->data() + entry, 8) != std::string_view(".rsrc\0\0\0", 8)) continue;
    const std::optional<std::uint32_t> size = u32(*head, entry + 16);
    const std::optional<std::uint32_t> offset = u32(*head, entry + 20);
    if (!size || !offset) return std::nullopt;
    const std::optional<Bytes> resources = read_bytes(file, *offset, std::min(*size, kMaxResourceBytes));
    if (!resources) return std::nullopt;
    // VS_FIXEDFILEINFO est aligné sur 4 octets : signature, version de structure, puis
    // dwFileVersionMS (majeur, mineur) et dwFileVersionLS (correctif, build).
    for (std::size_t at = 0; at + 16 <= resources->size(); at += 4) {
      if (u32(*resources, at) != kFixedFileInfoSignature) continue;
      const std::uint32_t high = u32(*resources, at + 8).value_or(0);
      const std::uint32_t low = u32(*resources, at + 12).value_or(0);
      return SdkVersion{.parts = {high >> 16U, high & 0xFFFFU, low >> 16U, low & 0xFFFFU}};
    }
    return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace ajp::vendor_hdr
