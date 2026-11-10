// Lecture EBML minimale : en-tête, segment, SeekHead et Chapters, rien d'autre. Chaque taille est bornée par
// son parent et par la taille du fichier ; un octet faux rend une disposition simple, jamais un refus.
#include "media/mkv_chapters.h"

#include <cstddef>
#include <fstream>
#include <optional>
#include <vector>

namespace ajp::media {
namespace {

constexpr std::uint32_t kEbml = 0x1A45DFA3;
constexpr std::uint32_t kSegment = 0x18538067;
constexpr std::uint32_t kSeekHead = 0x114D9B74;
constexpr std::uint32_t kSeek = 0x4DBB;
constexpr std::uint32_t kSeekId = 0x53AB;
constexpr std::uint32_t kSeekPosition = 0x53AC;
constexpr std::uint32_t kCluster = 0x1F43B675;
constexpr std::uint32_t kChapters = 0x1043A770;
constexpr std::uint32_t kEditionEntry = 0x45B9;
constexpr std::uint32_t kEditionFlagOrdered = 0x45DD;
constexpr std::uint32_t kChapterAtom = 0xB6;
constexpr std::uint32_t kChapterSegmentUid = 0x6E67;
constexpr std::uint32_t kChapterSegmentEditionUid = 0x6EBC;
constexpr std::uint64_t kUnknownSize = ~std::uint64_t{0};
constexpr std::uint64_t kMaxChaptersBytes = std::uint64_t{16} << 20U;  // bien au-delà de mille chapitres
constexpr int kMaxTopElements = 64;                                    // avant le premier Cluster

// Un en-tête d'élément : identifiant (marque comprise), taille des données, où elles commencent.
struct Element {
  std::uint32_t id = 0;
  std::uint64_t size = 0;
  std::uint64_t data = 0;
};

// Lit un entier de longueur variable ; `keep_marker` pour un identifiant. std::nullopt sur un octet invalide.
template <class Read>
std::optional<std::uint64_t> vint(Read& read, bool keep_marker, int& length) {
  const std::optional<std::uint8_t> first = read();
  if (!first || *first == 0) return std::nullopt;
  length = 1;
  while ((*first & (0x80U >> (length - 1))) == 0) ++length;
  if (length > 8 || (keep_marker && length > 4)) return std::nullopt;
  std::uint64_t value = keep_marker ? *first : (*first & (0xFFU >> length));
  bool all_ones = value == (0xFFU >> length);
  for (int i = 1; i < length; ++i) {
    const std::optional<std::uint8_t> next = read();
    if (!next) return std::nullopt;
    value = (value << 8U) | *next;
    all_ones = all_ones && *next == 0xFF;
  }
  return !keep_marker && all_ones ? kUnknownSize : value;
}

class File {
 public:
  explicit File(const std::filesystem::path& path) : in_(path, std::ios::binary) {
    in_.seekg(0, std::ios::end);
    size_ = in_ ? static_cast<std::uint64_t>(in_.tellg()) : 0;
    in_.seekg(0);
  }
  std::uint64_t size() const { return size_; }

  std::optional<Element> element(std::uint64_t at) {
    in_.clear();
    in_.seekg(static_cast<std::streamoff>(at));
    const auto read = [this]() -> std::optional<std::uint8_t> {
      char c = 0;
      if (!in_.get(c)) return std::nullopt;
      return static_cast<std::uint8_t>(c);
    };
    int id_length = 0;
    int size_length = 0;
    const auto id = vint(read, true, id_length);
    if (!id) return std::nullopt;
    const auto size = vint(read, false, size_length);
    if (!size) return std::nullopt;
    const std::uint64_t data = at + static_cast<std::uint64_t>(id_length + size_length);
    if (*size != kUnknownSize && (data > size_ || *size > size_ - data)) return std::nullopt;
    return Element{.id = static_cast<std::uint32_t>(*id), .size = *size, .data = data};
  }

  std::optional<std::vector<std::uint8_t>> bytes(const Element& e) {
    if (e.size == kUnknownSize || e.size > kMaxChaptersBytes) return std::nullopt;
    std::vector<std::uint8_t> out(e.size);
    in_.clear();
    in_.seekg(static_cast<std::streamoff>(e.data));
    if (!in_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size())))
      return std::nullopt;
    return out;
  }

 private:
  std::ifstream in_;
  std::uint64_t size_ = 0;
};

// Les éléments d'un bloc déjà lu en mémoire, un par un.
class Block {
 public:
  explicit Block(const std::vector<std::uint8_t>& bytes, std::size_t from = 0, std::size_t to = 0)
      : bytes_(bytes), at_(from), end_(to == 0 ? bytes.size() : to) {}

  std::optional<Element> next() {
    if (at_ >= end_) return std::nullopt;
    const auto read = [this]() -> std::optional<std::uint8_t> {
      if (at_ >= end_) return std::nullopt;
      return bytes_[at_++];
    };
    int id_length = 0;
    int size_length = 0;
    const auto id = vint(read, true, id_length);
    const auto size = id ? vint(read, false, size_length) : std::nullopt;
    if (!size || *size == kUnknownSize || *size > end_ - at_) return std::nullopt;
    const Element e{.id = static_cast<std::uint32_t>(*id), .size = *size, .data = at_};
    at_ += static_cast<std::size_t>(*size);
    return e;
  }
  std::uint64_t uint(const Element& e) const {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < e.size && i < 8; ++i) value = (value << 8U) | bytes_[e.data + i];
    return value;
  }
  Block inside(const Element& e) const {
    return Block(bytes_, static_cast<std::size_t>(e.data), static_cast<std::size_t>(e.data + e.size));
  }

 private:
  const std::vector<std::uint8_t>& bytes_;
  std::size_t at_;
  std::size_t end_;
};

void read_atom(Block atom, ChapterLayout& layout, int depth) {
  while (const auto e = atom.next()) {
    if (e->id == kChapterSegmentUid || e->id == kChapterSegmentEditionUid) layout.linked = true;
    if (e->id == kChapterAtom && depth < 8) read_atom(atom.inside(*e), layout, depth + 1);
  }
}

ChapterLayout read_chapters(const std::vector<std::uint8_t>& bytes) {
  ChapterLayout layout;
  Block chapters(bytes);
  while (const auto edition = chapters.next()) {
    if (edition->id != kEditionEntry) continue;
    Block entry = chapters.inside(*edition);
    while (const auto e = entry.next()) {
      if (e->id == kEditionFlagOrdered && entry.uint(*e) != 0) layout.ordered = true;
      if (e->id == kChapterAtom) read_atom(entry.inside(*e), layout, 0);
    }
  }
  return layout;
}

// La position du Chapters que nomme un SeekHead, relative au début des données du segment.
std::optional<std::uint64_t> chapters_in_seek_head(const std::vector<std::uint8_t>& bytes) {
  Block head(bytes);
  while (const auto seek = head.next()) {
    if (seek->id != kSeek) continue;
    Block entry = head.inside(*seek);
    std::uint64_t id = 0;
    std::optional<std::uint64_t> position;
    while (const auto e = entry.next()) {
      if (e->id == kSeekId) id = entry.uint(*e);
      if (e->id == kSeekPosition) position = entry.uint(*e);
    }
    if (id == kChapters && position) return position;
  }
  return std::nullopt;
}

}  // namespace

ChapterLayout read_chapter_layout(const std::filesystem::path& path) {
  File file(path);
  const auto header = file.element(0);
  if (!header || header->id != kEbml || header->size == kUnknownSize) return {};
  const auto segment = file.element(header->data + header->size);
  if (!segment || segment->id != kSegment) return {};
  const std::uint64_t end = segment->size == kUnknownSize ? file.size() : segment->data + segment->size;
  std::uint64_t at = segment->data;
  for (int n = 0; n < kMaxTopElements && at < end; ++n) {
    const auto e = file.element(at);
    if (!e || e->size == kUnknownSize) return {};
    if (e->id == kChapters) {
      const auto bytes = file.bytes(*e);
      return bytes ? read_chapters(*bytes) : ChapterLayout{};
    }
    if (e->id == kSeekHead) {
      const auto bytes = file.bytes(*e);
      if (const auto position = bytes ? chapters_in_seek_head(*bytes) : std::nullopt) {
        const auto chapters = file.element(segment->data + *position);
        if (!chapters || chapters->id != kChapters) return {};
        const auto inside = file.bytes(*chapters);
        return inside ? read_chapters(*inside) : ChapterLayout{};
      }
    }
    if (e->id == kCluster) return {};  // les chapitres viennent avant les images, ou par le SeekHead
    at = e->data + e->size;
  }
  return {};
}

}  // namespace ajp::media
