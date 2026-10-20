#include "gpu/vulkan/backend/net_artifact.h"

#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <type_traits>
#include <utility>
#include <variant>

namespace ajp::gpu::vk::net {
namespace {

constexpr std::array<char, 8> kMagic{'A', 'J', 'P', 'V', 'K', 'N', 'E', 'T'};

// Écriture petit-boutiste : entiers, flottants (bits), énumérations et booléens sur un octet,
// chaînes et vecteurs précédés de leur taille sur 64 bits.
class Writer {
 public:
  template <class T>
  void operator()(const T& value) {
    if constexpr (std::is_same_v<T, bool>) {
      raw(std::uint8_t{value ? std::uint8_t{1} : std::uint8_t{0}});
    } else if constexpr (std::is_enum_v<T>) {
      raw(static_cast<std::uint8_t>(value));
    } else if constexpr (std::is_same_v<T, float>) {
      raw(std::bit_cast<std::uint32_t>(value));
    } else if constexpr (std::is_integral_v<T>) {
      raw(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
      raw(static_cast<std::uint64_t>(value.size()));
      const auto* chars = reinterpret_cast<const std::byte*>(value.data());
      out_.insert(out_.end(), chars, chars + value.size());
    } else {
      fields(*this, value);
    }
  }

  template <class T>
  void operator()(const std::vector<T>& values) {
    raw(static_cast<std::uint64_t>(values.size()));
    for (const T& v : values) (*this)(v);
  }

  template <class T, std::size_t N>
  void operator()(const std::array<T, N>& values) {
    for (const T& v : values) (*this)(v);
  }

  std::vector<std::byte> take() { return std::move(out_); }
  std::vector<std::byte>& bytes() { return out_; }

 private:
  template <class T>
  void raw(T value) {
    for (std::size_t k = 0; k < sizeof(T); ++k) {
      out_.push_back(static_cast<std::byte>((static_cast<std::uint64_t>(value) >> (8 * k)) & 0xFFU));
    }
  }

  std::vector<std::byte> out_;
};

// Lecture symétrique ; la première lecture hors des octets, ou une énumération hors de sa
// borne, arrête tout (failed()).
class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  template <class T>
  void operator()(T& value) {
    if constexpr (std::is_same_v<T, bool>) {
      value = raw<std::uint8_t>() != 0;
    } else if constexpr (std::is_enum_v<T>) {
      value = static_cast<T>(raw<std::uint8_t>());
    } else if constexpr (std::is_same_v<T, float>) {
      value = std::bit_cast<float>(raw<std::uint32_t>());
    } else if constexpr (std::is_integral_v<T>) {
      value = raw<T>();
    } else if constexpr (std::is_same_v<T, std::string>) {
      const std::uint64_t size = count(1);
      if (failed_) return;
      value.assign(reinterpret_cast<const char*>(bytes_.data() + at_), size);
      at_ += size;
    } else {
      fields(*this, value);
    }
  }

  template <class T>
  void operator()(std::vector<T>& values) {
    const std::uint64_t size = count(1);
    values.clear();
    if (failed_) return;
    values.resize(size);
    for (T& v : values) (*this)(v);
  }

  template <class T, std::size_t N>
  void operator()(std::array<T, N>& values) {
    for (T& v : values) (*this)(v);
  }

  void fail() { failed_ = true; }
  bool failed() const { return failed_; }
  bool done() const { return at_ == bytes_.size(); }

 private:
  template <class T>
  T raw() {
    if (failed_ || bytes_.size() - at_ < sizeof(T)) {
      failed_ = true;
      return T{};
    }
    std::uint64_t v = 0;
    for (std::size_t k = 0; k < sizeof(T); ++k)
      v |= std::to_integer<std::uint64_t>(bytes_[at_ + k]) << (8 * k);
    at_ += sizeof(T);
    return static_cast<T>(v);
  }

  // Une taille de vecteur ou de chaîne : chaque élément occupe au moins un octet.
  std::uint64_t count(std::uint64_t min_element) {
    const auto size = raw<std::uint64_t>();
    if (failed_ || size > (bytes_.size() - at_) / min_element) {
      failed_ = true;
      return 0;
    }
    return size;
  }

  std::span<const std::byte> bytes_;
  std::size_t at_ = 0;
  bool failed_ = false;
};

// Champs de chaque structure, dans l'ordre : la même fonction écrit et lit.
template <class Ar, class S>
void fields(Ar& ar, S& v)
  requires std::is_same_v<std::remove_const_t<S>, ir::Value>
{
  ar(v.type);
  ar(v.dims);
  ar(v.f);
  ar(v.i);
}

template <class Ar, class S>
void fields(Ar& ar, S& t)
  requires std::is_same_v<std::remove_const_t<S>, ir::Tensor>
{
  ar(t.name);
  ar(t.kind);
  ar(t.type);
  ar(t.dims);
  ar(t.known);
  ar(t.value);
  ar(t.layout);
}

template <class Ar, class S>
void fields(Ar& ar, S& e)
  requires std::is_same_v<std::remove_const_t<S>, ir::Epilogue>
{
  ar(e.residual);
  ar(e.act);
  ar(e.alpha);
  ar(e.lo);
  ar(e.hi);
  ar(e.gate);
  ar(e.space);
  ar(e.space_mode);
}

// Attributs : chaque alternative de ir::Attrs.
template <class Ar, class S>
void attr_fields(Ar& ar, S& a) {
  using A = std::remove_const_t<S>;
  if constexpr (std::is_same_v<A, ir::ConvAttrs>) {
    ar(a.kernel), ar(a.strides), ar(a.dilations), ar(a.pads), ar(a.auto_pad), ar(a.group), ar(a.epilogue);
  } else if constexpr (std::is_same_v<A, ir::ConvTransposeAttrs>) {
    ar(a.kernel), ar(a.strides), ar(a.dilations), ar(a.pads), ar(a.output_padding), ar(a.output_shape);
    ar(a.auto_pad), ar(a.group);
  } else if constexpr (std::is_same_v<A, ir::LeakyReluAttrs>) {
    ar(a.alpha);
  } else if constexpr (std::is_same_v<A, ir::AxisAttrs>) {
    ar(a.axis);
  } else if constexpr (std::is_same_v<A, ir::SplitAttrs>) {
    ar(a.axis), ar(a.num_outputs);
  } else if constexpr (std::is_same_v<A, ir::ReshapeAttrs>) {
    ar(a.allowzero);
  } else if constexpr (std::is_same_v<A, ir::TransposeAttrs>) {
    ar(a.perm);
  } else if constexpr (std::is_same_v<A, ir::ShapeAttrs>) {
    ar(a.start), ar(a.end);
  } else if constexpr (std::is_same_v<A, ir::CastAttrs>) {
    ar(a.to);
  } else if constexpr (std::is_same_v<A, ir::ModAttrs>) {
    ar(a.fmod);
  } else if constexpr (std::is_same_v<A, ir::BlockAttrs>) {
    ar(a.blocksize), ar(a.mode);
  } else if constexpr (std::is_same_v<A, ir::PadAttrs>) {
    ar(a.mode);
  } else if constexpr (std::is_same_v<A, ir::ResizeAttrs>) {
    ar(a.mode), ar(a.coord), ar(a.nearest), ar(a.axes);
  } else if constexpr (std::is_same_v<A, ir::GridSampleAttrs>) {
    ar(a.mode), ar(a.padding), ar(a.align_corners), ar(a.grid_nchw);
  } else if constexpr (std::is_same_v<A, ir::ReduceAttrs>) {
    ar(a.keepdims), ar(a.noop_with_empty_axes);
  } else if constexpr (std::is_same_v<A, ir::PoolAttrs>) {
    ar(a.kernel), ar(a.strides), ar(a.pads), ar(a.auto_pad), ar(a.count_include_pad);
  }
}

template <std::size_t... I>
bool emplace_attrs(ir::Attrs& attrs, std::size_t index, std::index_sequence<I...> /*unused*/) {
  return ((index == I ? (attrs.emplace<I>(), true) : false) || ...);
}

void attrs_io(Writer& ar, const ir::Attrs& attrs) {
  ar(static_cast<std::uint8_t>(attrs.index()));
  std::visit([&](const auto& a) { attr_fields(ar, a); }, attrs);
}

void attrs_io(Reader& ar, ir::Attrs& attrs) {
  std::uint8_t index = 0;
  ar(index);
  if (!emplace_attrs(attrs, index, std::make_index_sequence<std::variant_size_v<ir::Attrs>>{})) {
    ar.fail();
    return;
  }
  std::visit([&](auto& a) { attr_fields(ar, a); }, attrs);
}

template <class Ar, class S>
void fields(Ar& ar, S& n)
  requires std::is_same_v<std::remove_const_t<S>, ir::Node>
{
  ar(n.name);
  ar(n.op);
  attrs_io(ar, n.attrs);
  ar(n.inputs);
  ar(n.outputs);
}

template <class Ar, class S>
void fields(Ar& ar, S& g)
  requires std::is_same_v<std::remove_const_t<S>, ir::Graph>
{
  ar(g.tensors);
  ar(g.nodes);
  ar(g.inputs);
  ar(g.outputs);
  ar(g.opset);
}

template <class Ar, class S>
void fields(Ar& ar, S& c)
  requires std::is_same_v<std::remove_const_t<S>, KernelChoice>
{
  ar(c.variant);
  ar(c.tuning);
}

template <class Ar, class S>
void fields(Ar& ar, S& b)
  requires std::is_same_v<std::remove_const_t<S>, Binding>
{
  ar(b.kind);
  ar(b.index);
}

template <class Ar, class S>
void fields(Ar& ar, S& d)
  requires std::is_same_v<std::remove_const_t<S>, Dispatch>
{
  ar(d.node);
  ar(d.shader);
  ar(d.constants);
  ar(d.bindings);
  ar(d.push);
  ar(d.groups);
  ar(d.subgroup);
}

template <class Ar, class S>
void fields(Ar& ar, S& h)
  requires std::is_same_v<std::remove_const_t<S>, ShaderHash>
{
  ar(h.shader);
  ar(h.hash);
}

template <class Ar, class S>
void fields(Ar& ar, S& a)
  requires std::is_same_v<std::remove_const_t<S>, ArtifactData>
{
  ar(a.lowering);
  ar(a.model_id);
  ar(a.model_sha256);
  ar(a.device_key);
  ar(a.fp16);
  ar(a.graph);
  ar(a.choices);
  ar(a.dispatches);
  ar(a.shaders);
}

// Les énumérations relues restent dans leurs bornes (la dernière valeur de chacune).
bool enums_valid(const ArtifactData& a) {
  for (const ir::Tensor& t : a.graph.tensors) {
    if (t.kind > ir::TensorKind::Activation || t.type > ir::DataType::I64 || t.layout > ir::Layout::Nhwc16 ||
        t.value.type > ir::DataType::I64) {
      return false;
    }
  }
  for (const ir::Node& n : a.graph.nodes) {
    if (n.op > ir::OpKind::AveragePool) return false;
  }
  for (const Dispatch& d : a.dispatches) {
    for (const Binding& b : d.bindings) {
      if (b.kind > BindingKind::Dummy) return false;
    }
  }
  return true;
}

}  // namespace

std::uint64_t fnv1a(std::span<const std::byte> bytes, std::uint64_t hash) {
  for (const std::byte b : bytes) {
    hash ^= std::to_integer<std::uint64_t>(b);
    hash *= 0x100000001B3ULL;
  }
  return hash;
}

std::uint64_t spirv_hash(std::span<const std::uint32_t> words) {
  return fnv1a(std::as_bytes(words));
}

std::vector<std::byte> serialize(const ArtifactData& data) {
  Writer w;
  for (const char c : kMagic) w(static_cast<std::uint8_t>(c));
  w(kArtifactVersion);
  w(data);
  const std::uint64_t sum = fnv1a(w.bytes());
  w(sum);
  return w.take();
}

std::expected<ArtifactData, Refusal> deserialize(std::span<const std::byte> bytes) {
  constexpr std::size_t kHeader = kMagic.size() + 4;
  if (bytes.size() < kHeader + 8) {
    return std::unexpected(
        Refusal{RefusalCode::CacheIo, std::format("artefact de {} o, tronqué", bytes.size())});
  }
  Reader header(bytes.first(kHeader));
  for (const char c : kMagic) {
    std::uint8_t got = 0;
    header(got);
    if (got != static_cast<std::uint8_t>(c)) {
      return std::unexpected(
          Refusal{RefusalCode::FormatVersion, "ce n'est pas un artefact du moteur Vulkan"});
    }
  }
  std::uint32_t version = 0;
  header(version);
  if (version != kArtifactVersion) {
    return std::unexpected(
        Refusal{RefusalCode::FormatVersion,
                std::format("artefact au format {}, {} attendu", version, kArtifactVersion)});
  }
  const std::span<const std::byte> body = bytes.first(bytes.size() - 8);
  Reader tail(bytes.last(8));
  std::uint64_t sum = 0;
  tail(sum);
  if (sum != fnv1a(body))
    return std::unexpected(Refusal{RefusalCode::HashMismatch, "somme de l'artefact fausse"});
  Reader reader(body.subspan(kHeader));
  ArtifactData data;
  reader(data);
  if (reader.failed() || !reader.done() || !enums_valid(data)) {
    return std::unexpected(
        Refusal{RefusalCode::CacheIo, "artefact illisible : champ tronqué ou hors de sa borne"});
  }
  return data;
}

}  // namespace ajp::gpu::vk::net
