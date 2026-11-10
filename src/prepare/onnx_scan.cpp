// Repris de animejanai-inference-rtx@118c272 src/onnx_flow.cpp:14-58 et 104-176,
// et src/model_identity.cpp:52-60 pour is_safe_location (sans le refus de
// « model.onnx », propre à l'instantané du fork que ajp ne fait pas). Les numéros de champ
// sont ceux d'onnx.proto (ONNX 1.19.1) ; on ne descend que dans les messages
// déclarés : les octets bruts d'un tenseur ne sont jamais lus comme du protobuf.
#include "prepare/onnx_scan.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>
#include <string_view>
#include <utility>

#include "prepare/protobuf_fields.h"
#include "prepare/utf8.h"

namespace ajp::prepare {
namespace {

using Bytes = std::span<const std::byte>;

std::unexpected<Refusal> malformed(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
}

// Le lecteur du format filaire est celui de gpu/ir (protobuf_fields.h, P1.31) : un second décodeur,
// ici, tronquait le numéro de champ en 32 bits et acceptait un dixième octet de varint quelconque, et
// deux lectures du même fichier ne s'accordaient pas (A04-10).
using Field = protobuf::RawField;
using Reader = protobuf::Fields;

std::string as_text(Bytes bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

enum class Kind : std::uint8_t { Model, Graph, Node, Attribute, Tensor, Sparse, Function, Training };

// Le type du message contenu dans le champ number d'un message kind, s'il en contient un.
std::optional<Kind> child_kind(Kind kind, std::uint32_t number) {
  switch (kind) {
    case Kind::Model:
      if (number == 7) return Kind::Graph;
      if (number == 20) return Kind::Training;
      if (number == 25) return Kind::Function;
      return std::nullopt;
    case Kind::Graph:
      if (number == 1) return Kind::Node;
      if (number == 5) return Kind::Tensor;
      if (number == 15) return Kind::Sparse;
      return std::nullopt;
    case Kind::Node:
      return number == 5 ? std::optional(Kind::Attribute) : std::nullopt;
    case Kind::Attribute:
      if (number == 5 || number == 10) return Kind::Tensor;
      if (number == 6 || number == 11) return Kind::Graph;
      if (number == 22 || number == 23) return Kind::Sparse;
      return std::nullopt;
    case Kind::Sparse:
      return number == 1 || number == 2 ? std::optional(Kind::Tensor) : std::nullopt;
    case Kind::Function:
      if (number == 7) return Kind::Node;
      if (number == 11) return Kind::Attribute;
      return std::nullopt;
    case Kind::Training:
      return number == 1 || number == 2 ? std::optional(Kind::Graph) : std::nullopt;
    case Kind::Tensor:
      return std::nullopt;
  }
  return std::nullopt;
}

class LocationWalker {
 public:
  std::expected<void, Refusal> walk(Bytes data, Kind kind, unsigned depth) {
    if (depth > 64) return malformed("métadonnées ONNX imbriquées au-delà de 64 niveaux");
    Reader reader(data);
    Field field;
    bool external = false;
    bool metadata = false;
    bool graph = false;
    std::string location;
    while (reader.next(field)) {
      if (kind == Kind::Tensor && field.number == 14) external = field.scalar == 1;  // data_location
      if (!field.delimited) continue;
      if (kind == Kind::Tensor && field.number == 13) {  // external_data
        metadata = true;
        auto entry = read_entry(field.payload);
        if (!entry) return std::unexpected(entry.error());
        if (entry->first == "location") {
          if (!location.empty() && location != entry->second)
            return malformed("emplacements de poids contradictoires");
          location = entry->second;
        }
        continue;
      }
      const auto child = child_kind(kind, field.number);
      if (!child) continue;
      if (kind == Kind::Model && *child == Kind::Graph) graph = true;
      if (auto ok = walk(field.payload, *child, depth + 1); !ok) return ok;
    }
    if (!reader.ok()) return malformed("métadonnées de poids ONNX malformées");
    if (kind == Kind::Model && !graph) return malformed("graphe ONNX absent");
    if (kind == Kind::Tensor && (external || metadata)) {
      // Un emplacement qui n'est pas de l'UTF-8 ferait lever path_from_utf8 sous Windows (A04-01).
      if (location.empty() || location.find('\0') != std::string::npos || !is_utf8(location)) {
        return malformed("emplacement de poids absent ou invalide");
      }
      found_.insert(location);
    }
    return {};
  }

  std::vector<std::string> locations() const { return {found_.begin(), found_.end()}; }

 private:
  // StringStringEntryProto : key = 1, value = 2.
  static std::expected<std::pair<std::string, std::string>, Refusal> read_entry(Bytes data) {
    Reader reader(data);
    Field field;
    std::pair<std::string, std::string> entry;
    while (reader.next(field)) {
      if (field.delimited && field.number == 1) entry.first = as_text(field.payload);
      if (field.delimited && field.number == 2) entry.second = as_text(field.payload);
    }
    if (!reader.ok()) return malformed("external_data ONNX malformé");
    return entry;
  }

  std::set<std::string> found_;
};

// TensorShapeProto.Dimension : dim_value = 1, dim_param = 2.
std::int64_t read_dimension(Bytes data) {
  Reader reader(data);
  Field field;
  std::int64_t value = -1;
  while (reader.next(field)) {
    if (!field.delimited && field.number == 1) value = static_cast<std::int64_t>(field.scalar);
  }
  return reader.ok() && value > 0 ? value : -1;
}

// ValueInfoProto : name = 1, type = 2 ; TypeProto.tensor_type = 1 ;
// TypeProto.Tensor : elem_type = 1, shape = 2 ; TensorShapeProto.dim = 1.
bool read_input(Bytes data, OnnxInput& input) {
  Reader info(data);
  Field field;
  while (info.next(field)) {
    if (field.delimited && field.number == 1) input.name = as_text(field.payload);
    if (!field.delimited || field.number != 2) continue;
    Reader type(field.payload);
    Field tensor;
    while (type.next(tensor)) {
      if (!tensor.delimited || tensor.number != 1) continue;
      Reader parts(tensor.payload);
      Field part;
      while (parts.next(part)) {
        if (!part.delimited && part.number == 1) input.elem_type = static_cast<std::int32_t>(part.scalar);
        if (!part.delimited || part.number != 2) continue;
        Reader shape(part.payload);
        Field dim;
        while (shape.next(dim)) {
          if (dim.delimited && dim.number == 1) input.dims.push_back(read_dimension(dim.payload));
        }
        if (!shape.ok()) return false;
      }
      if (!parts.ok()) return false;
    }
    if (!type.ok()) return false;
  }
  return info.ok();
}

// GraphProto : node = 1 (NodeProto.op_type = 4), initializer = 5 (TensorProto.name = 8), input = 11.
std::expected<void, Refusal> read_graph(Bytes data, OnnxSummary& summary) {
  Reader graph(data);
  Field field;
  std::set<std::string> initializers;
  std::vector<OnnxInput> inputs;
  while (graph.next(field)) {
    if (!field.delimited) continue;
    if (field.number == 1 || field.number == 5) {
      Reader inner(field.payload);
      Field item;
      const std::uint32_t wanted = field.number == 1 ? 4 : 8;
      while (inner.next(item)) {
        if (!item.delimited || item.number != wanted) continue;
        if (field.number == 1 && as_text(item.payload) == "GridSample") ++summary.grid_samples;
        if (field.number == 5) initializers.insert(as_text(item.payload));
      }
      if (!inner.ok()) return malformed("nœud ou initialiseur ONNX malformé");
      if (field.number == 1) ++summary.nodes;
    } else if (field.number == 11) {
      OnnxInput input;
      if (!read_input(field.payload, input)) return malformed("entrée ONNX malformée");
      inputs.push_back(std::move(input));
    }
  }
  if (!graph.ok()) return malformed("graphe ONNX malformé");
  for (OnnxInput& input : inputs) {
    if (!initializers.contains(input.name)) summary.inputs.push_back(std::move(input));
  }
  return {};
}

}  // namespace

bool OnnxInput::is_static() const {
  return !dims.empty() && std::ranges::all_of(dims, [](std::int64_t d) { return d > 0; });
}

std::expected<std::vector<std::string>, Refusal> onnx_external_locations(Bytes model) {
  if (model.empty()) return malformed("ONNX vide");
  LocationWalker walker;
  if (auto ok = walker.walk(model, Kind::Model, 0); !ok) return std::unexpected(ok.error());
  return walker.locations();
}

std::expected<OnnxSummary, Refusal> scan_onnx(Bytes model) {
  auto locations = onnx_external_locations(model);
  if (!locations) return std::unexpected(locations.error());
  OnnxSummary summary{.external_locations = std::move(*locations)};
  Reader top(model);
  Field field;
  while (top.next(field)) {
    if (!field.delimited || field.number != 7) continue;
    if (auto ok = read_graph(field.payload, summary); !ok) return std::unexpected(ok.error());
  }
  if (!top.ok()) return malformed("ONNX malformé");
  return summary;
}

bool is_safe_location(const std::string& location) {
  if (!is_utf8(location)) return false;  // path_from_utf8 lèverait sous Windows (A04-01)
  const std::filesystem::path relative = path_from_utf8(location);
  // has_root_path : sous Windows, « /etc/passwd » n'est pas absolu (pas de lecteur) mais part de
  // la racine du lecteur courant ; « C:poids.bin » a un lecteur sans racine.
  if (location.empty() || relative.has_root_path() || location.find('\\') != std::string::npos ||
      location.find(':') != std::string::npos) {
    return false;
  }
  return std::ranges::none_of(relative, [](const std::filesystem::path& part) { return part == ".."; });
}

}  // namespace ajp::prepare
