#include "core/inflate.h"

#include <array>
#include <vector>

namespace ajp::core {
namespace {

constexpr int kMaxBits = 15;
constexpr std::size_t kWindow = 32768;
constexpr std::size_t kInputChunk = 1U << 16;

struct Stop {
  RefusalCode code;
  const char* what;
};

std::unexpected<Refusal> corrupt(const char* what) {
  return std::unexpected(Refusal{RefusalCode::DecodeFailed, what});
}

// Table de Huffman canonique : nombre de codes par longueur, symboles triés par code.
struct Huffman {
  std::array<std::uint16_t, kMaxBits + 1> count{};
  std::array<std::uint16_t, 288> symbol{};

  // false si l'ensemble de longueurs sur-souscrit le code ; un ensemble incomplet passe (un seul
  // code de distance est permis).
  bool build(std::span<const std::uint8_t> lengths) {
    count.fill(0);
    for (const std::uint8_t length : lengths) ++count[length];
    int left = 1;
    for (int len = 1; len <= kMaxBits; ++len) {
      left = (left << 1) - count[static_cast<std::size_t>(len)];
      if (left < 0) return false;
    }
    std::array<std::uint16_t, kMaxBits + 2> offsets{};
    for (int len = 1; len <= kMaxBits; ++len)
      offsets[static_cast<std::size_t>(len) + 1] = static_cast<std::uint16_t>(
          offsets[static_cast<std::size_t>(len)] + count[static_cast<std::size_t>(len)]);
    for (std::size_t sym = 0; sym < lengths.size(); ++sym) {
      if (lengths[sym] != 0) symbol[offsets[lengths[sym]]++] = static_cast<std::uint16_t>(sym);
    }
    return true;
  }
};

class Inflater {
 public:
  Inflater(const InflateRead& read, const InflateWrite& write)
      : read_(read), write_(write), input_(kInputChunk), window_(kWindow) {}

  std::expected<std::uint64_t, Refusal> run() {
    bool last = false;
    while (!last) {
      const int final_bit = bits(1);
      const int type = bits(2);
      if (failed_) return corrupt("deflate : flux tronqué");
      last = final_bit != 0;
      std::expected<void, Refusal> block;
      if (type == 0) {
        block = stored();
      } else if (type == 1) {
        block = fixed();
      } else if (type == 2) {
        block = dynamic();
      } else {
        return corrupt("deflate : type de bloc invalide");
      }
      if (!block) return std::unexpected(block.error());
    }
    if (auto flushed = flush(); !flushed) return std::unexpected(flushed.error());
    return total_;
  }

 private:
  int bits(int need) {
    std::uint32_t value = bit_buffer_;
    while (bit_count_ < need) {
      if (in_pos_ == in_len_) {
        in_len_ = read_(input_);
        in_pos_ = 0;
        if (in_len_ == 0) {
          failed_ = true;
          return 0;
        }
      }
      value |= static_cast<std::uint32_t>(input_[in_pos_++]) << bit_count_;
      bit_count_ += 8;
    }
    bit_buffer_ = value >> need;
    bit_count_ -= need;
    return static_cast<int>(value & ((1U << need) - 1U));
  }

  int decode(const Huffman& h) {
    int code = 0;
    int first = 0;
    int index = 0;
    for (int len = 1; len <= kMaxBits; ++len) {
      code |= bits(1);
      if (failed_) return -1;
      const int count = h.count[static_cast<std::size_t>(len)];
      if (code - count < first) return h.symbol[static_cast<std::size_t>(index + (code - first))];
      index += count;
      first += count;
      first <<= 1;
      code <<= 1;
    }
    return -1;
  }

  std::expected<void, Refusal> put(std::uint8_t byte) {
    window_[out_pos_++] = byte;
    ++total_;
    if (out_pos_ == kWindow) return flush();
    return {};
  }

  std::expected<void, Refusal> flush() {
    if (out_pos_ > flushed_) {
      if (!write_(std::span<const std::uint8_t>(window_).subspan(flushed_, out_pos_ - flushed_)))
        return std::unexpected(Refusal{RefusalCode::Cancelled, "décompression interrompue"});
    }
    if (out_pos_ == kWindow) out_pos_ = 0;
    flushed_ = out_pos_;
    return {};
  }

  std::expected<void, Refusal> stored() {
    bit_buffer_ = 0;
    bit_count_ = 0;
    std::array<std::uint8_t, 4> header{};
    for (std::uint8_t& b : header) b = static_cast<std::uint8_t>(bits(8));
    if (failed_) return corrupt("deflate : bloc stocké tronqué");
    const unsigned len = header[0] | (static_cast<unsigned>(header[1]) << 8);
    const unsigned nlen = header[2] | (static_cast<unsigned>(header[3]) << 8);
    if (len != (~nlen & 0xFFFFU)) return corrupt("deflate : longueur de bloc stocké");
    for (unsigned i = 0; i < len; ++i) {
      const int byte = bits(8);
      if (failed_) return corrupt("deflate : bloc stocké tronqué");
      if (auto ok = put(static_cast<std::uint8_t>(byte)); !ok) return ok;
    }
    return {};
  }

  std::expected<void, Refusal> codes(const Huffman& literals, const Huffman& distances) {
    static constexpr std::array<std::uint16_t, 29> kLengthBase{3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                                                               15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                                                               67, 83, 99, 115, 131, 163, 195, 227, 258};
    static constexpr std::array<std::uint8_t, 29> kLengthExtra{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                                               2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static constexpr std::array<std::uint16_t, 30> kDistBase{
        1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
        193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static constexpr std::array<std::uint8_t, 30> kDistExtra{
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    for (;;) {
      const int symbol = decode(literals);
      if (symbol < 0) return corrupt("deflate : code invalide");
      if (symbol < 256) {
        if (auto ok = put(static_cast<std::uint8_t>(symbol)); !ok) return ok;
      } else if (symbol == 256) {
        return {};
      } else {
        const std::size_t li = static_cast<std::size_t>(symbol - 257);
        if (li >= kLengthBase.size()) return corrupt("deflate : longueur invalide");
        const int length = kLengthBase[li] + bits(kLengthExtra[li]);
        const int ds = decode(distances);
        if (ds < 0 || ds >= 30 || failed_) return corrupt("deflate : distance invalide");
        const std::size_t di = static_cast<std::size_t>(ds);
        const std::uint64_t distance =
            static_cast<std::uint64_t>(kDistBase[di]) + static_cast<unsigned>(bits(kDistExtra[di]));
        if (failed_ || distance > total_ || distance > kWindow)
          return corrupt("deflate : distance trop grande");
        for (int i = 0; i < length; ++i) {
          const std::size_t from = (out_pos_ + kWindow - static_cast<std::size_t>(distance)) % kWindow;
          if (auto ok = put(window_[from]); !ok) return ok;
        }
      }
    }
  }

  std::expected<void, Refusal> fixed() {
    std::array<std::uint8_t, 288> lengths{};
    for (std::size_t s = 0; s < 144; ++s) lengths[s] = 8;
    for (std::size_t s = 144; s < 256; ++s) lengths[s] = 9;
    for (std::size_t s = 256; s < 280; ++s) lengths[s] = 7;
    for (std::size_t s = 280; s < 288; ++s) lengths[s] = 8;
    Huffman literals;
    literals.build(lengths);
    std::array<std::uint8_t, 30> dist_lengths{};
    dist_lengths.fill(5);
    Huffman distances;
    distances.build(dist_lengths);
    return codes(literals, distances);
  }

  std::expected<void, Refusal> dynamic() {
    static constexpr std::array<std::uint8_t, 19> kOrder{16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                                         11, 4,  12, 3, 13, 2, 14, 1, 15};
    const int nlen = bits(5) + 257;
    const int ndist = bits(5) + 1;
    const int ncode = bits(4) + 4;
    if (failed_ || nlen > 286 || ndist > 30) return corrupt("deflate : table dynamique invalide");
    std::array<std::uint8_t, 19> code_lengths{};
    for (int i = 0; i < ncode; ++i)
      code_lengths[kOrder[static_cast<std::size_t>(i)]] = static_cast<std::uint8_t>(bits(3));
    Huffman length_code;
    if (failed_ || !length_code.build(code_lengths)) return corrupt("deflate : table de longueurs");
    std::array<std::uint8_t, 320> lengths{};
    int index = 0;
    while (index < nlen + ndist) {
      int symbol = decode(length_code);
      if (symbol < 0) return corrupt("deflate : code de longueur invalide");
      if (symbol < 16) {
        lengths[static_cast<std::size_t>(index++)] = static_cast<std::uint8_t>(symbol);
        continue;
      }
      std::uint8_t previous = 0;
      int repeat = 0;
      if (symbol == 16) {
        if (index == 0) return corrupt("deflate : répétition sans longueur");
        previous = lengths[static_cast<std::size_t>(index - 1)];
        repeat = 3 + bits(2);
      } else if (symbol == 17) {
        repeat = 3 + bits(3);
      } else {
        repeat = 11 + bits(7);
      }
      if (failed_ || index + repeat > nlen + ndist) return corrupt("deflate : répétition trop longue");
      while (repeat-- > 0) lengths[static_cast<std::size_t>(index++)] = previous;
    }
    if (lengths[256] == 0) return corrupt("deflate : pas de fin de bloc");
    Huffman literals;
    if (!literals.build(std::span<const std::uint8_t>(lengths).first(static_cast<std::size_t>(nlen))))
      return corrupt("deflate : table des littéraux");
    Huffman distances;
    if (!distances.build(std::span<const std::uint8_t>(lengths).subspan(static_cast<std::size_t>(nlen),
                                                                        static_cast<std::size_t>(ndist))))
      return corrupt("deflate : table des distances");
    return codes(literals, distances);
  }

  const InflateRead& read_;
  const InflateWrite& write_;
  std::vector<std::uint8_t> input_;
  std::vector<std::uint8_t> window_;
  std::size_t in_pos_ = 0;
  std::size_t in_len_ = 0;
  std::uint32_t bit_buffer_ = 0;
  int bit_count_ = 0;
  bool failed_ = false;
  std::size_t out_pos_ = 0;
  std::size_t flushed_ = 0;
  std::uint64_t total_ = 0;
};

}  // namespace

std::expected<std::uint64_t, Refusal> inflate_raw(const InflateRead& read, const InflateWrite& write) {
  return Inflater(read, write).run();
}

}  // namespace ajp::core
