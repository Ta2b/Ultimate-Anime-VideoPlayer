#include "core/inflate.h"

#include <array>
#include <zlib.h>

namespace ajp::core {
namespace {

constexpr std::size_t kChunk = 1U << 16;

std::unexpected<Refusal> corrupt(const char* what) {
  return std::unexpected(Refusal{RefusalCode::DecodeFailed, what});
}

// Le flux de zlib, rendu à la sortie.
struct Stream {
  z_stream z{};
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() { inflateEnd(&z); }
};

}  // namespace

// zlib en deflate brut (fenêtre de 2^15 sans en-tête : -15), à la place du décodeur maison (P1.30) :
// les données viennent du réseau, et zlib est éprouvé et plus rapide.
std::expected<std::uint64_t, Refusal> inflate_raw(const InflateRead& read, const InflateWrite& write) {
  Stream stream;
  if (inflateInit2(&stream.z, -15) != Z_OK) return corrupt("deflate : zlib ne démarre pas");
  std::array<std::uint8_t, kChunk> input{};
  std::array<std::uint8_t, kChunk> output{};
  std::uint64_t produced = 0;
  for (;;) {
    if (stream.z.avail_in == 0) {
      const std::size_t got = read(input);
      if (got == 0) return corrupt("deflate : flux tronqué");
      stream.z.next_in = input.data();
      stream.z.avail_in = static_cast<uInt>(got);
    }
    stream.z.next_out = output.data();
    stream.z.avail_out = static_cast<uInt>(output.size());
    const int status = inflate(&stream.z, Z_NO_FLUSH);
    if (status != Z_OK && status != Z_STREAM_END && status != Z_BUF_ERROR) {
      return corrupt(stream.z.msg != nullptr ? stream.z.msg : "deflate : flux corrompu");
    }
    const std::size_t written = output.size() - stream.z.avail_out;
    if (written > 0) {
      if (!write(std::span<const std::uint8_t>(output.data(), written))) {
        return std::unexpected(Refusal{RefusalCode::Cancelled, "deflate : écriture interrompue"});
      }
      produced += written;
    }
    if (status == Z_STREAM_END) return produced;
  }
}

}  // namespace ajp::core
