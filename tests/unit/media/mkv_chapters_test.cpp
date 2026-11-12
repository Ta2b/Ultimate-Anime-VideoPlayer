// Les chapitres ordonnés et liés de Matroska, reconnus dans des fichiers écrits octet par octet (A14-03) : le
// muxeur de FFmpeg ne sait pas écrire EditionFlagOrdered ni ChapterSegmentUID.
#include "media/mkv_chapters.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::media::read_chapter_layout;

namespace {

using Bytes = std::vector<std::uint8_t>;

// Un élément : son identifiant (marque comprise), sa taille sur huit octets, ses données.
Bytes element(std::initializer_list<std::uint8_t> id, const Bytes& payload) {
  Bytes out(id);
  out.push_back(0x01);
  for (int shift = 48; shift >= 0; shift -= 8)
    out.push_back(static_cast<std::uint8_t>(payload.size() >> shift));
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

Bytes join(std::initializer_list<Bytes> parts) {
  Bytes out;
  for (const Bytes& part : parts) out.insert(out.end(), part.begin(), part.end());
  return out;
}

Bytes chapters(bool ordered, bool linked) {
  const Bytes uid = element({0x6E, 0x67}, Bytes(16, 0xAB));
  const Bytes atom = element({0xB6}, join({element({0x73, 0xC4}, {0x01}), linked ? uid : Bytes{}}));
  const Bytes edition = element(
      {0x45, 0xB9}, join({element({0x45, 0xDD}, {ordered ? std::uint8_t{1} : std::uint8_t{0}}), atom}));
  return element({0x10, 0x43, 0xA7, 0x70}, edition);
}

std::filesystem::path write(const std::string& name, const Bytes& segment_payload) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  const Bytes file =
      join({element({0x1A, 0x45, 0xDF, 0xA3}, {}), element({0x18, 0x53, 0x80, 0x67}, segment_payload)});
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
  return path;
}

}  // namespace

TEST_CASE("Chapitres Matroska : édition simple, ordonnée, liée (A14-03)", "[media][chapters]") {
  CHECK(read_chapter_layout(write("ajp-chap-plain.mkv", chapters(false, false))).plain());
  const auto ordered = read_chapter_layout(write("ajp-chap-ordered.mkv", chapters(true, false)));
  CHECK(ordered.ordered);
  CHECK_FALSE(ordered.linked);
  const auto linked = read_chapter_layout(write("ajp-chap-linked.mkv", chapters(true, true)));
  CHECK(linked.ordered);
  CHECK(linked.linked);
}

TEST_CASE("Chapitres Matroska : trouvés par le SeekHead après un premier Cluster", "[media][chapters]") {
  // SeekHead (taille fixe), Cluster, puis Chapters : la position du SeekHead est relative au segment.
  const Bytes cluster = element({0x1F, 0x43, 0xB6, 0x75}, Bytes(64, 0));
  const auto seek_head = [&](std::uint64_t position) {
    Bytes pos(8);
    for (int i = 0; i < 8; ++i)
      pos[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(position >> (56 - 8 * i));
    const Bytes seek = element(
        {0x4D, 0xBB}, join({element({0x53, 0xAB}, {0x10, 0x43, 0xA7, 0x70}), element({0x53, 0xAC}, pos)}));
    return element({0x11, 0x4D, 0x9B, 0x74}, seek);
  };
  const std::uint64_t at = seek_head(0).size() + cluster.size();
  const auto layout =
      read_chapter_layout(write("ajp-chap-seek.mkv", join({seek_head(at), cluster, chapters(false, true)})));
  CHECK(layout.linked);
  CHECK_FALSE(layout.ordered);
}

TEST_CASE("Chapitres Matroska : autre format ou fichier tronqué, disposition simple", "[media][chapters]") {
  CHECK(read_chapter_layout(std::filesystem::temp_directory_path() / "ajp-chap-absent.mkv").plain());
  const std::filesystem::path text = std::filesystem::temp_directory_path() / "ajp-chap-text.mkv";
  std::ofstream(text) << "pas du Matroska";
  CHECK(read_chapter_layout(text).plain());
  Bytes cut = chapters(true, false);
  cut.resize(cut.size() / 2);
  CHECK(read_chapter_layout(write("ajp-chap-cut.mkv", cut)).plain());
}

namespace {

Bytes ns(std::uint64_t value) {
  Bytes out(8);
  for (int i = 0; i < 8; ++i)
    out[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(value >> (56 - 8 * i));
  return out;
}

// Un chapitre de start_s à end_s secondes ; disabled : ChapterFlagEnabled = 0 ; linked : dans un autre
// segment.
Bytes atom(std::uint64_t start_s, std::uint64_t end_s, bool disabled = false, bool linked = false) {
  return element({0xB6}, join({element({0x73, 0xC4}, {0x01}), element({0x91}, ns(start_s * 1'000'000'000)),
                               element({0x92}, ns(end_s * 1'000'000'000)),
                               disabled ? element({0x45, 0x98}, {0x00}) : Bytes{},
                               linked ? element({0x6E, 0x67}, Bytes(16, 0xAB)) : Bytes{}}));
}

Bytes edition(bool ordered, bool by_default, std::initializer_list<Bytes> atoms) {
  Bytes body = join({element({0x45, 0xDD}, {ordered ? std::uint8_t{1} : std::uint8_t{0}}),
                     element({0x45, 0xDB}, {by_default ? std::uint8_t{1} : std::uint8_t{0}})});
  for (const Bytes& a : atoms) body.insert(body.end(), a.begin(), a.end());
  return element({0x45, 0xB9}, body);
}

}  // namespace

TEST_CASE("Chapitres Matroska : les parties de l'édition ordonnée jouée, dans son ordre",
          "[media][chapters]") {
  // La première édition, simple, n'est pas celle par défaut ; la seconde, ordonnée, l'est. Son chapitre
  // désactivé et son générique lié ne sont pas des parties de ce fichier.
  const Bytes both = element(
      {0x10, 0x43, 0xA7, 0x70},
      join({edition(false, false, {atom(0, 10)}),
            edition(true, true,
                    {atom(4, 6), atom(0, 2), atom(2, 4, true), atom(0, 90, false, true), atom(6, 10)})}));
  const auto layout = read_chapter_layout(write("ajp-chap-parts.mkv", both));
  CHECK(layout.ordered);
  CHECK(layout.linked);
  REQUIRE(layout.parts.size() == 3);
  CHECK(layout.parts[0].start_us == 4'000'000);
  CHECK(layout.parts[0].end_us == 6'000'000);
  CHECK(layout.parts[1].start_us == 0);
  CHECK(layout.parts[2].end_us == 10'000'000);

  // Sans édition par défaut, la première est jouée : simple, elle ne donne aucune partie.
  const Bytes first_plain =
      element({0x10, 0x43, 0xA7, 0x70},
              join({edition(false, false, {atom(0, 10)}), edition(true, false, {atom(4, 6)})}));
  const auto plain = read_chapter_layout(write("ajp-chap-first.mkv", first_plain));
  CHECK(plain.ordered);
  CHECK(plain.parts.empty());
}

namespace {

// Un fichier en mémoire : c'est ce que le démultiplexeur branche sur son AVIOContext (C14-09).
class MemorySource final : public ajp::media::ByteSource {
 public:
  explicit MemorySource(Bytes bytes) : bytes_(std::move(bytes)) {}
  std::uint64_t size() override { return bytes_.size(); }
  bool read(std::uint64_t offset, std::span<std::uint8_t> out) override {
    if (offset > bytes_.size() || out.size() > bytes_.size() - offset) return false;
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(offset), out.size(), out.begin());
    return true;
  }

 private:
  Bytes bytes_;
};

}  // namespace

TEST_CASE("Chapitres Matroska : lus par une source d'octets (C14-09)", "[media][chapters]") {
  const Bytes segment = chapters(true, true);
  MemorySource source(
      join({element({0x1A, 0x45, 0xDF, 0xA3}, {}), element({0x18, 0x53, 0x80, 0x67}, segment)}));
  const auto layout = read_chapter_layout(source);
  CHECK(layout.ordered);
  CHECK(layout.linked);
  MemorySource empty(Bytes{});
  CHECK(read_chapter_layout(empty).plain());
}
