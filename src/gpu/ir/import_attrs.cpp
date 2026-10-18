// Attributs ONNX vers attributs typés. Un attribut du mauvais type, ou une
// valeur hors du sous-ensemble pris en charge (Resize cubique, Pad wrap…),
// est refusé avec le nom de l'opérateur ; rien n'est corrigé en silence.
#include "gpu/ir/import_attrs.h"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"

namespace ajp::gpu::ir::detail {
namespace {

using onnx::AttributeType;

class AttrReader {
 public:
  explicit AttrReader(const onnx::Node& node) : node_(node) {}

  bool has(std::string_view name) const { return find(name) != nullptr; }

  std::expected<std::int64_t, Refusal> integer(std::string_view name, std::int64_t fallback) const {
    const onnx::Attribute* attr = find(name);
    if (attr == nullptr) return fallback;
    if (!typed(*attr, AttributeType::Int)) return wrong_type(name);
    return attr->i;
  }

  std::expected<float, Refusal> real(std::string_view name, float fallback) const {
    const onnx::Attribute* attr = find(name);
    if (attr == nullptr) return fallback;
    if (!typed(*attr, AttributeType::Float)) return wrong_type(name);
    return attr->f;
  }

  std::expected<std::string, Refusal> text(std::string_view name, std::string fallback) const {
    const onnx::Attribute* attr = find(name);
    if (attr == nullptr) return fallback;
    if (!typed(*attr, AttributeType::String)) return wrong_type(name);
    return attr->s;
  }

  std::expected<Dims, Refusal> integers(std::string_view name) const {
    const onnx::Attribute* attr = find(name);
    if (attr == nullptr) return Dims{};
    if (!typed(*attr, AttributeType::Ints)) return wrong_type(name);
    return attr->ints;
  }

  std::unexpected<Refusal> unsupported(std::string_view detail) const {
    return std::unexpected(
        Refusal{RefusalCode::OnnxUnsupported, std::format("{} {} : {}", node_.op_type, node_.name, detail)});
  }

  std::unexpected<Refusal> invalid(std::string_view detail) const {
    return std::unexpected(
        Refusal{RefusalCode::GraphInvalid, std::format("{} {} : {}", node_.op_type, node_.name, detail)});
  }

 private:
  const onnx::Attribute* find(std::string_view name) const {
    for (const onnx::Attribute& attr : node_.attributes) {
      if (attr.name == name) return &attr;
    }
    return nullptr;
  }

  // Les très anciens exports omettent le type : on accepte alors l'attribut tel quel.
  static bool typed(const onnx::Attribute& attr, AttributeType type) {
    return attr.type == type || attr.type == AttributeType::Undefined;
  }

  std::unexpected<Refusal> wrong_type(std::string_view name) const {
    return invalid(std::format("attribut {} du mauvais type", name));
  }

  const onnx::Node& node_;
};

// Choix d'une valeur textuelle parmi une liste fermée.
template <class E>
std::expected<E, Refusal> choose(const AttrReader& reader, std::string_view name, std::string fallback,
                                 std::initializer_list<std::pair<std::string_view, E>> choices) {
  auto value = reader.text(name, std::move(fallback));
  if (!value) return std::unexpected(value.error());
  for (const auto& [text, choice] : choices) {
    if (*value == text) return choice;
  }
  return reader.unsupported(std::format("{} = {} non pris en charge", name, *value));
}

template <std::size_t N>
std::expected<std::array<std::int64_t, N>, Refusal> fixed(const AttrReader& reader, std::string_view name,
                                                          std::array<std::int64_t, N> fallback) {
  auto values = reader.integers(name);
  if (!values) return std::unexpected(values.error());
  if (values->empty()) return fallback;
  if (values->size() != N)
    return reader.unsupported(std::format("{} : {} valeurs, convolution 2D seulement", name, values->size()));
  std::array<std::int64_t, N> out{};
  std::ranges::copy(*values, out.begin());
  return out;
}

std::expected<AutoPad, Refusal> auto_pad(const AttrReader& reader) {
  return choose<AutoPad>(reader, "auto_pad", "NOTSET",
                         {{"NOTSET", AutoPad::NotSet},
                          {"SAME_UPPER", AutoPad::SameUpper},
                          {"SAME_LOWER", AutoPad::SameLower},
                          {"VALID", AutoPad::Valid}});
}

// Champs communs aux fenêtres 2D : noyau, pas, dilatation, marges, auto_pad.
struct WindowAttrs {
  Pair kernel;
  Pair strides;
  Pair dilations;
  Quad pads;
  AutoPad auto_pad;
};

std::expected<WindowAttrs, Refusal> window_attrs(const AttrReader& r) {
  auto kernel = fixed<2>(r, "kernel_shape", {0, 0});
  auto strides = fixed<2>(r, "strides", {1, 1});
  auto dilations = fixed<2>(r, "dilations", {1, 1});
  auto pads = fixed<4>(r, "pads", {0, 0, 0, 0});
  auto pad_mode = auto_pad(r);
  if (!kernel) return std::unexpected(kernel.error());
  if (!strides) return std::unexpected(strides.error());
  if (!dilations) return std::unexpected(dilations.error());
  if (!pads) return std::unexpected(pads.error());
  if (!pad_mode) return std::unexpected(pad_mode.error());
  return WindowAttrs{
      .kernel = *kernel, .strides = *strides, .dilations = *dilations, .pads = *pads, .auto_pad = *pad_mode};
}

std::expected<Attrs, Refusal> conv_attrs(const AttrReader& r, OpKind op) {
  auto w = window_attrs(r);
  auto group = r.integer("group", 1);
  if (!w) return std::unexpected(w.error());
  if (!group) return std::unexpected(group.error());
  if (op == OpKind::Conv) {
    return ConvAttrs{.kernel = w->kernel,
                     .strides = w->strides,
                     .dilations = w->dilations,
                     .pads = w->pads,
                     .auto_pad = w->auto_pad,
                     .group = *group};
  }
  auto output_padding = fixed<2>(r, "output_padding", {0, 0});
  auto output_shape = r.integers("output_shape");
  if (!output_padding) return std::unexpected(output_padding.error());
  if (!output_shape) return std::unexpected(output_shape.error());
  return ConvTransposeAttrs{.kernel = w->kernel,
                            .strides = w->strides,
                            .dilations = w->dilations,
                            .pads = w->pads,
                            .output_padding = *output_padding,
                            .output_shape = std::move(*output_shape),
                            .auto_pad = w->auto_pad,
                            .group = *group};
}

std::expected<Attrs, Refusal> pool_attrs(const AttrReader& r) {
  if (!r.has("kernel_shape")) return r.invalid("kernel_shape obligatoire");
  auto w = window_attrs(r);
  auto ceil_mode = r.integer("ceil_mode", 0);
  auto count_include_pad = r.integer("count_include_pad", 0);
  if (!w) return std::unexpected(w.error());
  if (!ceil_mode) return std::unexpected(ceil_mode.error());
  if (!count_include_pad) return std::unexpected(count_include_pad.error());
  if (*ceil_mode != 0) return r.unsupported("ceil_mode = 1 non pris en charge");
  if (w->dilations != Pair{1, 1}) return r.unsupported("dilatation non prise en charge");
  return PoolAttrs{.kernel = w->kernel,
                   .strides = w->strides,
                   .pads = w->pads,
                   .auto_pad = w->auto_pad,
                   .count_include_pad = *count_include_pad != 0};
}

std::expected<Attrs, Refusal> resize_attrs(const AttrReader& r) {
  auto mode = choose<ResizeMode>(r, "mode", "nearest",
                                 {{"nearest", ResizeMode::Nearest}, {"linear", ResizeMode::Linear}});
  auto coord = choose<CoordMode>(r, "coordinate_transformation_mode", "half_pixel",
                                 {{"half_pixel", CoordMode::HalfPixel},
                                  {"pytorch_half_pixel", CoordMode::PytorchHalfPixel},
                                  {"align_corners", CoordMode::AlignCorners},
                                  {"asymmetric", CoordMode::Asymmetric}});
  auto nearest = choose<NearestMode>(r, "nearest_mode", "round_prefer_floor",
                                     {{"round_prefer_floor", NearestMode::RoundPreferFloor},
                                      {"round_prefer_ceil", NearestMode::RoundPreferCeil},
                                      {"floor", NearestMode::Floor},
                                      {"ceil", NearestMode::Ceil}});
  auto policy = r.text("keep_aspect_ratio_policy", "stretch");
  auto antialias = r.integer("antialias", 0);
  auto axes = r.integers("axes");
  if (!mode) return std::unexpected(mode.error());
  if (!coord) return std::unexpected(coord.error());
  if (!nearest) return std::unexpected(nearest.error());
  if (!policy) return std::unexpected(policy.error());
  if (!antialias) return std::unexpected(antialias.error());
  if (!axes) return std::unexpected(axes.error());
  if (*policy != "stretch") return r.unsupported("keep_aspect_ratio_policy autre que stretch");
  if (*antialias != 0) return r.unsupported("antialias non pris en charge");
  return ResizeAttrs{.mode = *mode, .coord = *coord, .nearest = *nearest, .axes = std::move(*axes)};
}

std::expected<Attrs, Refusal> grid_sample_attrs(const AttrReader& r, std::int64_t opset) {
  // Opset 20 renomme bilinear en linear ; les deux noms sont lus.
  auto mode = choose<GridMode>(
      r, "mode", opset >= 20 ? "linear" : "bilinear",
      {{"bilinear", GridMode::Bilinear}, {"linear", GridMode::Bilinear}, {"nearest", GridMode::Nearest}});
  auto padding = choose<GridPadding>(r, "padding_mode", "zeros",
                                     {{"zeros", GridPadding::Zeros}, {"border", GridPadding::Border}});
  auto align = r.integer("align_corners", 0);
  if (!mode) return std::unexpected(mode.error());
  if (!padding) return std::unexpected(padding.error());
  if (!align) return std::unexpected(align.error());
  return GridSampleAttrs{.mode = *mode, .padding = *padding, .align_corners = *align != 0};
}

std::expected<Attrs, Refusal> block_attrs(const AttrReader& r, OpKind op) {
  auto blocksize = r.integer("blocksize", 0);
  if (!blocksize) return std::unexpected(blocksize.error());
  if (!r.has("blocksize")) return r.invalid("blocksize obligatoire");
  if (op == OpKind::SpaceToDepth) return BlockAttrs{.blocksize = *blocksize, .mode = BlockMode::Dcr};
  auto mode = choose<BlockMode>(r, "mode", "DCR", {{"DCR", BlockMode::Dcr}, {"CRD", BlockMode::Crd}});
  if (!mode) return std::unexpected(mode.error());
  return BlockAttrs{.blocksize = *blocksize, .mode = *mode};
}

// Attribut d'entiers d'un ancien opset, rendu en valeur I64 pour l'entrée 1.
std::expected<std::optional<Value>, Refusal> moved_ints(const AttrReader& r, std::string_view name) {
  if (!r.has(name)) return std::optional<Value>{};
  auto values = r.integers(name);
  if (!values) return std::unexpected(values.error());
  Value value{
      .type = DataType::I64, .dims = {static_cast<std::int64_t>(values->size())}, .i = std::move(*values)};
  return std::optional{std::move(value)};
}

std::expected<ImportedAttrs, Refusal> with_moved(Attrs attrs, const AttrReader& r, std::string_view name,
                                                 bool moved_form) {
  if (!moved_form) {
    if (r.has(name)) return r.invalid(std::format("{} est une entrée dans cet opset", name));
    return ImportedAttrs{.attrs = std::move(attrs)};
  }
  auto value = moved_ints(r, name);
  if (!value) return std::unexpected(value.error());
  return ImportedAttrs{.attrs = std::move(attrs), .input_from_attr = std::move(*value)};
}

template <class T>
std::expected<ImportedAttrs, Refusal> plain(std::expected<T, Refusal> attrs) {
  if (!attrs) return std::unexpected(attrs.error());
  return ImportedAttrs{.attrs = Attrs{std::move(*attrs)}};
}

}  // namespace

std::expected<ImportedAttrs, Refusal> import_attrs(OpKind op, const onnx::Node& node, std::int64_t opset) {
  const AttrReader r(node);
  switch (op) {
    case OpKind::Conv:
    case OpKind::ConvTranspose:
      return plain(conv_attrs(r, op));
    case OpKind::AveragePool:
      return plain(pool_attrs(r));
    case OpKind::LeakyRelu: {
      auto alpha = r.real("alpha", 0.01F);
      if (!alpha) return std::unexpected(alpha.error());
      return ImportedAttrs{.attrs = LeakyReluAttrs{.alpha = *alpha}};
    }
    case OpKind::Concat:
      if (!r.has("axis")) return r.invalid("axis obligatoire");
      [[fallthrough]];
    case OpKind::Gather: {
      auto axis = r.integer("axis", 0);
      if (!axis) return std::unexpected(axis.error());
      return ImportedAttrs{.attrs = AxisAttrs{.axis = *axis}};
    }
    case OpKind::Split: {
      auto axis = r.integer("axis", 0);
      auto num_outputs = r.integer("num_outputs", 0);
      if (!axis) return std::unexpected(axis.error());
      if (!num_outputs) return std::unexpected(num_outputs.error());
      return with_moved(SplitAttrs{.axis = *axis, .num_outputs = *num_outputs}, r, "split", opset < 13);
    }
    case OpKind::Reshape: {
      auto allowzero = r.integer("allowzero", 0);
      if (!allowzero) return std::unexpected(allowzero.error());
      return ImportedAttrs{.attrs = ReshapeAttrs{.allowzero = *allowzero != 0}};
    }
    case OpKind::Transpose: {
      auto perm = r.integers("perm");
      if (!perm) return std::unexpected(perm.error());
      return ImportedAttrs{.attrs = TransposeAttrs{.perm = std::move(*perm)}};
    }
    case OpKind::Unsqueeze:
    case OpKind::Squeeze:
      return with_moved(NoAttrs{}, r, "axes", opset < 13);
    case OpKind::ReduceMean: {
      auto keepdims = r.integer("keepdims", 1);
      auto noop = r.integer("noop_with_empty_axes", 0);
      if (!keepdims) return std::unexpected(keepdims.error());
      if (!noop) return std::unexpected(noop.error());
      return with_moved(ReduceAttrs{.keepdims = *keepdims != 0, .noop_with_empty_axes = *noop != 0}, r,
                        "axes", opset < 18);
    }
    case OpKind::Shape: {
      auto start = r.integer("start", 0);
      auto end = r.integer("end", ShapeAttrs{}.end);
      if (!start) return std::unexpected(start.error());
      if (!end) return std::unexpected(end.error());
      return ImportedAttrs{.attrs = ShapeAttrs{.start = *start, .end = *end}};
    }
    case OpKind::Cast: {
      if (!r.has("to")) return r.invalid("to obligatoire");
      auto to = r.integer("to", 0);
      if (!to) return std::unexpected(to.error());
      if (*to < 0 || *to > 64) return r.invalid(std::format("type {}", *to));
      auto type = data_type_of(static_cast<std::int32_t>(*to), node.name);
      if (!type) return std::unexpected(type.error());
      return ImportedAttrs{.attrs = CastAttrs{.to = *type}};
    }
    case OpKind::Mod: {
      auto fmod = r.integer("fmod", 0);
      if (!fmod) return std::unexpected(fmod.error());
      return ImportedAttrs{.attrs = ModAttrs{.fmod = *fmod != 0}};
    }
    case OpKind::DepthToSpace:
    case OpKind::SpaceToDepth:
      return plain(block_attrs(r, op));
    case OpKind::Pad:
      return plain(
          choose<PadMode>(
              r, "mode", "constant",
              {{"constant", PadMode::Constant}, {"reflect", PadMode::Reflect}, {"edge", PadMode::Edge}})
              .transform([](PadMode mode) { return PadAttrs{.mode = mode}; }));
    case OpKind::Resize:
      return plain(resize_attrs(r));
    case OpKind::GridSample:
      return plain(grid_sample_attrs(r, opset));
    default:
      return ImportedAttrs{.attrs = NoAttrs{}};
  }
}

std::expected<DataType, Refusal> data_type_of(std::int32_t onnx_type, std::string_view where) {
  switch (onnx_type) {
    case onnx::kFloat:
      return DataType::F32;
    case onnx::kFloat16:
      return DataType::F16;
    case onnx::kInt64:
    case onnx::kInt32:
      return DataType::I64;
    default:
      return std::unexpected(
          Refusal{RefusalCode::OnnxUnsupported,
                  std::format("{} : type ONNX {} hors de F32, F16, I64", where, onnx_type)});
  }
}

std::expected<Value, Refusal> tensor_value(const onnx::Tensor& tensor) {
  auto type = data_type_of(tensor.data_type, tensor.name);
  if (!type) return std::unexpected(type.error());
  if (tensor.dims.size() > kMaxRank ||
      std::ranges::any_of(tensor.dims, [](std::int64_t d) { return d < 0 || d > kMaxDim; })) {
    return std::unexpected(Refusal{RefusalCode::ShapeInvalid,
                                   std::format("{} : forme {}", tensor.name, format_dims(tensor.dims))});
  }
  return Value{.type = *type, .dims = tensor.dims, .f = tensor.floats, .i = tensor.ints};
}

std::expected<Value, Refusal> constant_value(const onnx::Node& node) {
  if (node.attributes.size() != 1) {
    return std::unexpected(
        Refusal{RefusalCode::GraphInvalid, std::format("Constant {} : un seul attribut", node.name)});
  }
  const onnx::Attribute& attr = node.attributes[0];
  if (attr.name == "value" && attr.tensors.size() == 1) return tensor_value(attr.tensors[0]);
  if (attr.name == "value_float") return Value{.type = DataType::F32, .dims = {}, .f = {attr.f}};
  if (attr.name == "value_floats") {
    return Value{
        .type = DataType::F32, .dims = {static_cast<std::int64_t>(attr.floats.size())}, .f = attr.floats};
  }
  if (attr.name == "value_int") return Value{.type = DataType::I64, .dims = {}, .i = {attr.i}};
  if (attr.name == "value_ints") {
    return Value{
        .type = DataType::I64, .dims = {static_cast<std::int64_t>(attr.ints.size())}, .i = attr.ints};
  }
  return std::unexpected(
      Refusal{RefusalCode::OnnxUnsupported,
              std::format("Constant {} : attribut {} non pris en charge", node.name, attr.name)});
}

}  // namespace ajp::gpu::ir::detail
