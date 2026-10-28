#include "prepare/onnx_metadata.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <system_error>
#include <vector>

#include "prepare/protobuf_fields.h"

namespace ajp::prepare {
namespace {

// Une entrée de métadonnée (StringStringEntryProto) plus grande n'en est pas une.
constexpr std::uint64_t kMaxEntryBytes = std::uint64_t{1} << 20U;
constexpr std::uint32_t kMetadataPropsField = 14;  // ModelProto.metadata_props

std::unexpected<Refusal> unreadable(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
}

// Un entier varint lu dans le flux ; false à la fin du fichier ou s'il dépasse 64 bits.
bool read_varint(std::istream& in, std::uint64_t& value) {
  value = 0;
  for (unsigned shift = 0; shift < 64; shift += 7) {
    const int read = in.get();
    if (read == std::istream::traits_type::eof()) return false;
    const auto byte = static_cast<std::uint64_t>(read);
    value |= (byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0) return true;
  }
  return false;
}

}  // namespace

std::expected<std::optional<std::string>, Refusal> onnx_metadata(const std::filesystem::path& file,
                                                                 std::string_view key) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(file, error);
  if (error || size == 0) return unreadable("ONNX illisible ou vide : " + file.string());
  std::ifstream in(file, std::ios::binary);
  if (!in) return unreadable("ONNX illisible : " + file.string());
  std::optional<std::string> found;
  while (static_cast<std::uintmax_t>(in.tellg()) < size) {
    std::uint64_t tag = 0;
    if (!read_varint(in, tag) || (tag >> 3U) == 0) return unreadable("ONNX malformé : " + file.string());
    switch (tag & 7U) {
      case 0: {
        std::uint64_t ignored = 0;
        if (!read_varint(in, ignored)) return unreadable("ONNX tronqué : " + file.string());
        break;
      }
      case 1:
      case 5:
        in.seekg((tag & 7U) == 1U ? 8 : 4, std::ios::cur);
        break;
      case 2: {
        std::uint64_t length = 0;
        if (!read_varint(in, length)) return unreadable("ONNX tronqué : " + file.string());
        const auto here = static_cast<std::uintmax_t>(in.tellg());
        if (length > size - here) return unreadable("ONNX tronqué : " + file.string());
        if ((tag >> 3U) != kMetadataPropsField || length > kMaxEntryBytes) {
          in.seekg(static_cast<std::streamoff>(length), std::ios::cur);  // le graphe, les poids : sautés
          break;
        }
        std::vector<std::byte> entry(static_cast<std::size_t>(length));
        in.read(reinterpret_cast<char*>(entry.data()), static_cast<std::streamsize>(length));
        if (!in) return unreadable("ONNX tronqué : " + file.string());
        protobuf::Fields fields(entry);
        protobuf::RawField field;
        std::string name;
        std::string text;
        while (fields.next(field)) {
          if (!field.delimited) continue;
          const std::string_view bytes(reinterpret_cast<const char*>(field.payload.data()),
                                       field.payload.size());
          if (field.number == 1) name = bytes;
          if (field.number == 2) text = bytes;
        }
        if (!fields.ok()) return unreadable("métadonnée malformée : " + file.string());
        if (name == key) found = std::move(text);
        break;
      }
      default:
        return unreadable("ONNX malformé : " + file.string());
    }
    if (!in || static_cast<std::uintmax_t>(in.tellg()) > size)
      return unreadable("ONNX tronqué : " + file.string());
  }
  return found;
}

}  // namespace ajp::prepare
