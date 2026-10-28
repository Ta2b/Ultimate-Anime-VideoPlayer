#include "gpu/span_fp8/package.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <string_view>

namespace ajp::gpu::span_fp8 {
namespace {

constexpr std::string_view kMagic = "AJPSPAN8";
constexpr std::size_t kFirstWeights = std::size_t{kChannels} * 3 * 3 * kUnshuffledPadded;
constexpr std::size_t kBlockWeights = std::size_t{kChannels} * 3 * 3 * kChannels;
constexpr std::size_t kAttentionWeights = std::size_t{kChannels} * kChannels;
constexpr std::size_t kConcatWeights = std::size_t{kChannels} * kConcatChannels;
constexpr std::size_t kPixelWeights = std::size_t{kPixelChannels} * 3 * 3 * kChannels;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool finite_half(std::uint16_t bits) {
  return (bits & 0x7C00U) != 0x7C00U;
}

// Lecteur séquentiel de la charge : chaque lecture est bornée, rien ne dépasse jamais la fin.
class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
  bool take(std::size_t count, std::size_t size, void* out) {
    const std::size_t total = count * size;
    if (total > bytes_.size() - at_) return false;
    std::memcpy(out, bytes_.data() + at_, total);
    at_ += total;
    return true;
  }
  std::size_t left() const { return bytes_.size() - at_; }

 private:
  std::span<const std::byte> bytes_;
  std::size_t at_ = 0;
};

bool read_conv(Reader& in, Conv& conv, std::size_t weights, std::size_t bias) {
  conv.weights.resize(weights);
  conv.bias.resize(bias);
  if (!in.take(weights, sizeof(std::uint16_t), conv.weights.data())) return false;
  if (!in.take(bias, sizeof(float), conv.bias.data())) return false;
  for (const std::uint16_t w : conv.weights)
    if (!finite_half(w)) return false;
  for (const float b : conv.bias)
    if (!std::isfinite(b)) return false;
  return true;
}

void put_u32(std::vector<std::byte>& out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFFU));
}

void put_u64(std::vector<std::byte>& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFFU));
}

std::uint32_t get_u32(const std::byte* at) {
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(at[i]) << (8 * i);
  return value;
}

std::uint64_t get_u64(const std::byte* at) {
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(at[i]) << (8 * i);
  return value;
}

void put_conv(std::vector<std::byte>& out, const Conv& conv) {
  const auto* w = reinterpret_cast<const std::byte*>(conv.weights.data());
  out.insert(out.end(), w, w + conv.weights.size() * sizeof(std::uint16_t));
  const auto* b = reinterpret_cast<const std::byte*>(conv.bias.data());
  out.insert(out.end(), b, b + conv.bias.size() * sizeof(float));
}

Conv filled(std::size_t weights, std::size_t bias, float fill) {
  Conv conv;
  // 0x211E : 0,01 en demi-précision ; la valeur exacte n'importe pas aux tests, seulement qu'elle soit finie.
  conv.weights.assign(weights, fill > 0.0F ? std::uint16_t{0x211E} : std::uint16_t{0});
  conv.bias.assign(bias, fill);
  return conv;
}

}  // namespace

bool size_supported(int height, int width) {
  return size_refusal(height, width).empty();
}

std::string size_refusal(int height, int width) {
  constexpr long long kMaxSide = 16384;  // au-delà, les indices des noyaux (32 bits) ne tiennent plus
  if (height <= 0 || width <= 0 || height > kMaxSide || width > kMaxSide)
    return std::format("taille {}x{} hors de 1 à {}", width, height, kMaxSide);
  if (height % 2 != 0 || width % 2 != 0)
    return std::format("taille {}x{} : hauteur et largeur paires attendues (dégroupage 2x2)", width, height);
  const long long features = static_cast<long long>(height / 2) * (width / 2);
  if (features % 128 != 0)
    return std::format("taille {}x{} : {} pixels de caractéristiques, un multiple de 128 est attendu", width,
                       height, features);
  return {};
}

std::uint32_t crc32(std::span<const std::byte> bytes) {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
      t[i] = c;
    }
    return t;
  }();
  std::uint32_t crc = 0xFFFFFFFFU;
  for (const std::byte b : bytes) crc = table[(crc ^ static_cast<std::uint32_t>(b)) & 0xFFU] ^ (crc >> 8U);
  return crc ^ 0xFFFFFFFFU;
}

std::expected<Package, Refusal> parse(std::span<const std::byte> bytes) {
  if (bytes.size() < kHeaderBytes || bytes.size() > kMaxPackageBytes)
    return refuse(RefusalCode::FileUnreadable, std::format("paquet SPAN FP8 de {} octets", bytes.size()));
  if (std::memcmp(bytes.data(), kMagic.data(), kMagic.size()) != 0)
    return refuse(RefusalCode::FormatUnsupported, "paquet SPAN FP8 : signature absente");
  const std::byte* head = bytes.data() + kMagic.size();
  if (get_u32(head) != kVersion)
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("paquet SPAN FP8 : version {} inconnue", get_u32(head)));
  if (get_u32(head + 4) != static_cast<std::uint32_t>(kBlocks) ||
      get_u32(head + 8) != static_cast<std::uint32_t>(kChannels) || get_u32(head + 12) != 0)
    return refuse(RefusalCode::FormatUnsupported, "paquet SPAN FP8 : seuls 8 blocs de 64 canaux sont pris");
  const std::uint64_t payload = get_u64(head + 16);
  if (payload != bytes.size() - kHeaderBytes)
    return refuse(RefusalCode::FileUnreadable, "paquet SPAN FP8 tronqué ou trop long");
  const std::span<const std::byte> body = bytes.subspan(kHeaderBytes);
  if (crc32(body) != get_u32(head + 24))
    return refuse(RefusalCode::FileUnreadable, "paquet SPAN FP8 : somme de contrôle fausse");

  Package package;
  Reader in(body);
  const auto bad = [] {
    return refuse(RefusalCode::FormatUnsupported, "paquet SPAN FP8 : contenu incohérent");
  };
  if (!in.take(kScaleCount, sizeof(float), package.scales.data())) return bad();
  for (const float s : package.scales)
    if (!(s > 0.0F) || !std::isfinite(s)) return bad();
  if (!read_conv(in, package.first, kFirstWeights, kChannels)) return bad();
  for (Block& block : package.blocks) {
    for (Conv& conv : block.conv)
      if (!read_conv(in, conv, kBlockWeights, kChannels)) return bad();
    if (!read_conv(in, block.attention, kAttentionWeights, kChannels)) return bad();
  }
  if (!read_conv(in, package.concat, kConcatWeights, kChannels)) return bad();
  if (!read_conv(in, package.pixel, kPixelWeights, kPixelChannels)) return bad();
  if (in.left() != 0) return bad();
  return package;
}

std::vector<std::byte> serialize(const Package& package) {
  std::vector<std::byte> body;
  const auto* scales = reinterpret_cast<const std::byte*>(package.scales.data());
  body.insert(body.end(), scales, scales + sizeof(float) * kScaleCount);
  put_conv(body, package.first);
  for (const Block& block : package.blocks) {
    for (const Conv& conv : block.conv) put_conv(body, conv);
    put_conv(body, block.attention);
  }
  put_conv(body, package.concat);
  put_conv(body, package.pixel);

  std::vector<std::byte> out;
  out.reserve(kHeaderBytes + body.size());
  for (const char c : kMagic) out.push_back(static_cast<std::byte>(c));
  put_u32(out, kVersion);
  put_u32(out, static_cast<std::uint32_t>(kBlocks));
  put_u32(out, static_cast<std::uint32_t>(kChannels));
  put_u32(out, 0);
  put_u64(out, body.size());
  put_u32(out, crc32(body));
  put_u32(out, 0);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

Package synthetic(float fill) {
  Package package;
  package.scales.fill(0.05F);
  package.first = filled(kFirstWeights, kChannels, fill);
  for (Block& block : package.blocks) {
    for (Conv& conv : block.conv) conv = filled(kBlockWeights, kChannels, fill);
    block.attention = filled(kAttentionWeights, kChannels, fill);
  }
  package.concat = filled(kConcatWeights, kChannels, fill);
  package.pixel = filled(kPixelWeights, kPixelChannels, fill);
  return package;
}

}  // namespace ajp::gpu::span_fp8
