#include "app/play/ed25519.h"

#include <array>
#include <cstring>

extern "C" {
#include <libavutil/mem.h>
#include <libavutil/sha512.h>
}

namespace ajp::app::play {
namespace {

using i64 = std::int64_t;
using gf = std::array<i64, 16>;

void car25519(gf& o) {
  for (int i = 0; i < 16; ++i) {
    o[static_cast<std::size_t>(i)] += i64{1} << 16;
    const i64 c = o[static_cast<std::size_t>(i)] >> 16;
    o[static_cast<std::size_t>((i + 1) * (i < 15 ? 1 : 0))] += c - 1 + 37 * (c - 1) * (i == 15 ? 1 : 0);
    o[static_cast<std::size_t>(i)] -= c << 16;
  }
}

void sel25519(gf& p, gf& q, i64 b) {
  const i64 c = ~(b - 1);
  for (std::size_t i = 0; i < 16; ++i) {
    const i64 t = c & (p[i] ^ q[i]);
    p[i] ^= t;
    q[i] ^= t;
  }
}

void pack25519(std::uint8_t* o, const gf& n) {
  gf m{};
  gf t = n;
  car25519(t);
  car25519(t);
  car25519(t);
  for (int j = 0; j < 2; ++j) {
    m[0] = t[0] - 0xffed;
    for (std::size_t i = 1; i < 15; ++i) {
      m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
      m[i - 1] &= 0xffff;
    }
    m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
    const i64 b = (m[15] >> 16) & 1;
    m[14] &= 0xffff;
    sel25519(t, m, 1 - b);
  }
  for (std::size_t i = 0; i < 16; ++i) {
    o[2 * i] = static_cast<std::uint8_t>(t[i] & 0xff);
    o[2 * i + 1] = static_cast<std::uint8_t>(t[i] >> 8);
  }
}

void unpack25519(gf& o, const std::uint8_t* n) {
  for (std::size_t i = 0; i < 16; ++i) o[i] = n[2 * i] + (static_cast<i64>(n[2 * i + 1]) << 8);
  o[15] &= 0x7fff;
}

gf plus(const gf& a, const gf& b) {
  gf o;
  for (std::size_t i = 0; i < 16; ++i) o[i] = a[i] + b[i];
  return o;
}
gf minus(const gf& a, const gf& b) {
  gf o;
  for (std::size_t i = 0; i < 16; ++i) o[i] = a[i] - b[i];
  return o;
}
gf times(const gf& a, const gf& b) {
  std::array<i64, 31> t{};
  for (std::size_t i = 0; i < 16; ++i)
    for (std::size_t j = 0; j < 16; ++j) t[i + j] += a[i] * b[j];
  for (std::size_t i = 0; i < 15; ++i) t[i] += 38 * t[i + 16];
  gf o;
  for (std::size_t i = 0; i < 16; ++i) o[i] = t[i];
  car25519(o);
  car25519(o);
  return o;
}
gf square(const gf& a) {
  return times(a, a);
}

gf small(i64 value) {
  gf o{};
  o[0] = value;
  return o;
}

gf invert(const gf& i) {
  gf c = i;
  for (int a = 253; a >= 0; --a) {
    c = square(c);
    if (a != 2 && a != 4) c = times(c, i);
  }
  return c;
}

gf pow2523(const gf& i) {
  gf c = i;
  for (int a = 250; a >= 0; --a) {
    c = square(c);
    if (a != 1) c = times(c, i);
  }
  return c;
}

bool equal(const gf& a, const gf& b) {
  std::array<std::uint8_t, 32> c{};
  std::array<std::uint8_t, 32> d{};
  pack25519(c.data(), a);
  pack25519(d.data(), b);
  return std::memcmp(c.data(), d.data(), 32) == 0;
}

int parity(const gf& a) {
  std::array<std::uint8_t, 32> d{};
  pack25519(d.data(), a);
  return d[0] & 1;
}

// Les constantes de la courbe, calculées : d = -121665 / 121666, 2d, sqrt(-1) = 2^((p-1)/4), la base (4/5, x
// pair).
struct Curve {
  gf d;
  gf d2;
  gf sqrt_m1;
  gf base_x;
  gf base_y;
};

using Point = std::array<gf, 4>;

bool unpack_negated(Point& r, const std::uint8_t* p, const Curve& curve);

const Curve& curve() {
  static const Curve instance = [] {
    Curve c{};
    c.d = times(minus(small(0), small(121665)), invert(small(121666)));
    c.d2 = plus(c.d, c.d);
    // 2^(2^253 - 5) : 250 bits à 1, puis 0, 1, 1.
    gf r = small(1);
    for (int bit = 252; bit >= 0; --bit) {
      r = square(r);
      const bool one = bit >= 3 || bit < 2;
      if (one) r = times(r, small(2));
    }
    c.sqrt_m1 = r;
    c.base_y = times(small(4), invert(small(5)));
    std::array<std::uint8_t, 32> encoded{};
    pack25519(encoded.data(), c.base_y);  // bit de signe 0 : x pair
    Point negated{};
    unpack_negated(negated, encoded.data(), c);
    c.base_x = minus(small(0), negated[0]);
    return c;
  }();
  return instance;
}

void add(Point& p, const Point& q, const Curve& c) {
  const gf a = times(minus(p[1], p[0]), minus(q[1], q[0]));
  const gf b = times(plus(p[0], p[1]), plus(q[0], q[1]));
  const gf cc = times(times(p[3], q[3]), c.d2);
  gf dd = times(p[2], q[2]);
  dd = plus(dd, dd);
  const gf e = minus(b, a);
  const gf f = minus(dd, cc);
  const gf g = plus(dd, cc);
  const gf h = plus(b, a);
  p[0] = times(e, f);
  p[1] = times(h, g);
  p[2] = times(g, f);
  p[3] = times(e, h);
}

void cswap(Point& p, Point& q, i64 b) {
  for (std::size_t i = 0; i < 4; ++i) sel25519(p[i], q[i], b);
}

void pack_point(std::uint8_t* r, const Point& p) {
  const gf zi = invert(p[2]);
  const gf tx = times(p[0], zi);
  const gf ty = times(p[1], zi);
  pack25519(r, ty);
  r[31] = static_cast<std::uint8_t>(r[31] ^ (parity(tx) << 7));
}

Point scalarmult(Point q, const std::uint8_t* s, const Curve& c) {
  Point p{small(0), small(1), small(1), small(0)};
  for (int i = 255; i >= 0; --i) {
    const i64 b = (s[i / 8] >> (i & 7)) & 1;
    cswap(p, q, b);
    add(q, p, c);
    const Point same = p;
    add(p, same, c);
    cswap(p, q, b);
  }
  return p;
}

Point scalarbase(const std::uint8_t* s, const Curve& c) {
  const Point q{c.base_x, c.base_y, small(1), times(c.base_x, c.base_y)};
  return scalarmult(q, s, c);
}

bool unpack_negated(Point& r, const std::uint8_t* p, const Curve& c) {
  r[2] = small(1);
  unpack25519(r[1], p);
  gf num = square(r[1]);
  gf den = times(num, c.d);
  num = minus(num, r[2]);
  den = plus(r[2], den);
  const gf den2 = square(den);
  const gf den4 = square(den2);
  const gf den6 = times(den4, den2);
  gf t = times(times(den6, num), den);
  t = pow2523(t);
  t = times(times(times(t, num), den), den);
  r[0] = times(t, den);
  gf chk = times(square(r[0]), den);
  if (!equal(chk, num)) r[0] = times(r[0], c.sqrt_m1);
  chk = times(square(r[0]), den);
  if (!equal(chk, num)) return false;
  if (parity(r[0]) == (p[31] >> 7)) r[0] = minus(small(0), r[0]);
  r[3] = times(r[0], r[1]);
  return true;
}

constexpr std::array<std::uint8_t, 32> kL{0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7,
                                          0xa2, 0xde, 0xf9, 0xde, 0x14, 0,    0,    0,    0,    0,    0,
                                          0,    0,    0,    0,    0,    0,    0,    0,    0,    0x10};

void mod_l(std::uint8_t* r, std::array<i64, 64>& x) {
  for (int i = 63; i >= 32; --i) {
    i64 carry = 0;
    int j = i - 32;
    for (; j < i - 12; ++j) {
      x[static_cast<std::size_t>(j)] +=
          carry - 16 * x[static_cast<std::size_t>(i)] * kL[static_cast<std::size_t>(j - (i - 32))];
      carry = (x[static_cast<std::size_t>(j)] + 128) >> 8;
      x[static_cast<std::size_t>(j)] -= carry * 256;
    }
    x[static_cast<std::size_t>(j)] += carry;
    x[static_cast<std::size_t>(i)] = 0;
  }
  i64 carry = 0;
  for (std::size_t j = 0; j < 32; ++j) {
    x[j] += carry - (x[31] >> 4) * kL[j];
    carry = x[j] >> 8;
    x[j] &= 255;
  }
  for (std::size_t j = 0; j < 32; ++j) x[j] -= carry * kL[j];
  for (std::size_t i = 0; i < 32; ++i) {
    x[i + 1] += x[i] >> 8;
    r[i] = static_cast<std::uint8_t>(x[i] & 255);
  }
}

void reduce_64(std::uint8_t* r) {
  std::array<i64, 64> x{};
  for (std::size_t i = 0; i < 64; ++i) x[i] = r[i];
  for (std::size_t i = 0; i < 64; ++i) r[i] = 0;
  mod_l(r, x);
}

// S < L : refuse les signatures malléables.
bool canonical_s(const std::uint8_t* s) {
  for (int i = 31; i >= 0; --i) {
    if (s[i] < kL[static_cast<std::size_t>(i)]) return true;
    if (s[i] > kL[static_cast<std::size_t>(i)]) return false;
  }
  return false;  // S == L
}

std::array<std::uint8_t, 64> sha512(std::initializer_list<std::span<const std::uint8_t>> parts) {
  AVSHA512* context = av_sha512_alloc();
  std::array<std::uint8_t, 64> digest{};
  if (context == nullptr) return digest;
  av_sha512_init(context, 512);
  for (const auto part : parts) av_sha512_update(context, part.data(), part.size());
  av_sha512_final(context, digest.data());
  av_free(context);
  return digest;
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

}  // namespace

bool ed25519_verify(std::span<const std::uint8_t, 32> public_key, std::span<const std::uint8_t> message,
                    std::span<const std::uint8_t, 64> signature) {
  const Curve& c = curve();
  if (!canonical_s(signature.data() + 32)) return false;
  Point negated_a{};
  if (!unpack_negated(negated_a, public_key.data(), c)) return false;
  std::array<std::uint8_t, 64> h = sha512({signature.first<32>(), public_key, message});
  reduce_64(h.data());
  Point p = scalarmult(negated_a, h.data(), c);          // h * (-A)
  const Point q = scalarbase(signature.data() + 32, c);  // S * B
  add(p, q, c);
  std::array<std::uint8_t, 32> computed{};
  pack_point(computed.data(), p);
  return std::memcmp(computed.data(), signature.data(), 32) == 0;
}

std::optional<std::vector<std::uint8_t>> from_hex(std::string_view hex) {
  if (hex.size() % 2 != 0) return std::nullopt;
  std::vector<std::uint8_t> out;
  out.reserve(hex.size() / 2);
  for (std::size_t i = 0; i < hex.size(); i += 2) {
    const int high = hex_value(hex[i]);
    const int low = hex_value(hex[i + 1]);
    if (high < 0 || low < 0) return std::nullopt;
    out.push_back(static_cast<std::uint8_t>(high * 16 + low));
  }
  return out;
}

bool ed25519_verify_hex(std::string_view public_key_hex, std::string_view message,
                        std::string_view signature_hex) {
  const auto key = from_hex(public_key_hex);
  const auto sig = from_hex(signature_hex);
  if (!key || !sig || key->size() != 32 || sig->size() != 64) return false;
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(message.data());
  return ed25519_verify(std::span<const std::uint8_t, 32>(key->data(), 32), {bytes, message.size()},
                        std::span<const std::uint8_t, 64>(sig->data(), 64));
}

}  // namespace ajp::app::play
