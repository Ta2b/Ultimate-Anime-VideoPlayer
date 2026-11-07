#include "prepare/onnx_metadata.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <system_error>
#include <vector>

#include "gpu/ir/protobuf.h"
#include "prepare/protobuf_fields.h"

namespace ajp::prepare {
namespace {

// Une entrée de métadonnée (StringStringEntryProto) plus grande n'en est pas une.
constexpr std::uint64_t kMaxEntryBytes = std::uint64_t{1} << 20U;
constexpr std::uint32_t kMetadataPropsField = 14;  // ModelProto.metadata_props

std::unexpected<Refusal> unreadable(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
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
    // L'en-tête du champ (au plus 15 octets : étiquette et longueur), décodé par gpu/ir/protobuf.h ;
    // le graphe et les poids sont sautés sans être lus.
    const auto here = static_cast<std::uintmax_t>(in.tellg());
    std::array<std::byte, 15> head{};
    in.read(reinterpret_cast<char*>(head.data()),
            static_cast<std::streamsize>(std::min<std::uintmax_t>(head.size(), size - here)));
    in.clear();
    auto header = gpu::ir::proto::field_header(std::span(head).first(static_cast<std::size_t>(in.gcount())));
    if (!header) return unreadable("ONNX malformé : " + file.string());
    const std::uintmax_t body = here + header->size;
    if (header->payload > size - body) return unreadable("ONNX tronqué : " + file.string());
    in.seekg(static_cast<std::streamoff>(body));
    if (header->wire == gpu::ir::proto::WireType::Length && header->number == kMetadataPropsField &&
        header->payload <= kMaxEntryBytes) {
      std::vector<std::byte> entry(static_cast<std::size_t>(header->payload));
      in.read(reinterpret_cast<char*>(entry.data()), static_cast<std::streamsize>(entry.size()));
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
    } else {
      in.seekg(static_cast<std::streamoff>(body + header->payload));
    }
    if (!in || static_cast<std::uintmax_t>(in.tellg()) > size)
      return unreadable("ONNX tronqué : " + file.string());
  }
  return found;
}

}  // namespace ajp::prepare
