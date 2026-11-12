#include "media/face_cascade.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <system_error>
#include <utility>

namespace ajp::media {
namespace {

// Le contenu de la prochaine balise <tag>…</tag> à partir de pos, pos passé derrière elle.
std::optional<std::string_view> next_tag(std::string_view xml, std::string_view tag, std::size_t& pos,
                                         std::size_t end = std::string_view::npos) {
  const std::string open = "<" + std::string(tag) + ">";
  const std::string close = "</" + std::string(tag) + ">";
  const std::size_t at = xml.find(open, pos);
  if (at == std::string_view::npos || (end != std::string_view::npos && at >= end)) return std::nullopt;
  const std::size_t from = at + open.size();
  const std::size_t to = xml.find(close, from);
  if (to == std::string_view::npos) return std::nullopt;
  pos = to + close.size();
  return xml.substr(from, to - from);
}

// Les nombres d'un texte séparés par des blancs ; rien si l'un ne se lit pas.
template <class T>
std::optional<std::vector<T>> numbers(std::string_view text) {
  std::vector<T> out;
  const char* p = text.data();
  const char* const last = text.data() + text.size();
  while (true) {
    while (p != last && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    if (p == last) return out;
    T value{};
    const auto [next, ec] = std::from_chars(p, last, value);
    if (ec != std::errc{}) return std::nullopt;
    out.push_back(value);
    p = next;
  }
}

template <class T>
std::optional<T> number(std::string_view text) {
  const auto all = numbers<T>(text);
  if (!all || all->size() != 1) return std::nullopt;
  return all->front();
}

template <class T>
std::optional<T> number_tag(std::string_view xml, std::string_view tag, std::size_t& pos) {
  const auto text = next_tag(xml, tag, pos);
  return text ? number<T>(*text) : std::nullopt;
}

struct Box {
  float x = 0.0F, y = 0.0F, w = 0.0F, h = 0.0F;
};

// Deux fenêtres du même visage : chaque bord à 20 % de la taille près (le regroupement des cascades).
bool similar(const Box& a, const Box& b) {
  constexpr float kEps = 0.2F;
  const float delta = kEps * 0.5F * (std::min(a.w, b.w) + std::min(a.h, b.h));
  return std::abs(a.x - b.x) <= delta && std::abs(a.y - b.y) <= delta &&
         std::abs(a.x + a.w - b.x - b.w) <= delta && std::abs(a.y + a.h - b.y - b.h) <= delta;
}

std::size_t root_of(std::vector<std::size_t>& parent, std::size_t i) {
  while (parent[i] != i) i = parent[i] = parent[parent[i]];
  return i;
}

// Les fenêtres regroupées : un groupe de plus de min_neighbors fenêtres devient un visage, son cadre la
// moyenne des leurs ; un groupe contenu dans un groupe plus fourni s'efface.
std::vector<std::pair<Box, int>> group(const std::vector<Box>& found, int min_neighbors) {
  std::vector<std::size_t> parent(found.size());
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  for (std::size_t i = 0; i < found.size(); ++i)
    for (std::size_t j = i + 1; j < found.size(); ++j)
      if (similar(found[i], found[j])) parent[root_of(parent, i)] = root_of(parent, j);
  std::vector<Box> sums(found.size());
  std::vector<int> counts(found.size(), 0);
  for (std::size_t i = 0; i < found.size(); ++i) {
    const std::size_t r = root_of(parent, i);
    sums[r].x += found[i].x;
    sums[r].y += found[i].y;
    sums[r].w += found[i].w;
    sums[r].h += found[i].h;
    ++counts[r];
  }
  std::vector<std::pair<Box, int>> groups;
  for (std::size_t r = 0; r < found.size(); ++r) {
    if (counts[r] <= min_neighbors) continue;
    const auto n = static_cast<float>(counts[r]);
    groups.emplace_back(Box{sums[r].x / n, sums[r].y / n, sums[r].w / n, sums[r].h / n}, counts[r]);
  }
  std::vector<std::pair<Box, int>> kept;
  for (const auto& [a, na] : groups) {
    const bool inside = std::ranges::any_of(groups, [&](const auto& other) {
      const auto& [b, nb] = other;
      if (&b == &a) return false;
      const float dx = 0.2F * b.w;
      const float dy = 0.2F * b.h;
      return a.x >= b.x - dx && a.y >= b.y - dy && a.x + a.w <= b.x + b.w + dx &&
             a.y + a.h <= b.y + b.h + dy && nb > std::max(3, na);
    });
    if (!inside) kept.emplace_back(a, na);
  }
  return kept;
}

}  // namespace

// L'image intégrale : sum(x, y) est la somme des points au-dessus et à gauche de (x, y), exclus.
struct FaceCascade::Integral {
  int width = 0;
  std::vector<std::int32_t> sum;
  explicit Integral(const GrayImage& image) : width(image.width + 1) {
    sum.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(image.height + 1), 0);
    for (int y = 0; y < image.height; ++y) {
      std::int32_t row = 0;
      for (int x = 0; x < image.width; ++x) {
        row += image.at(x, y);
        at(x + 1, y + 1) = at(x + 1, y) + row;
      }
    }
  }
  std::int32_t& at(int x, int y) {
    return sum[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)];
  }
  std::int32_t at(int x, int y) const {
    return sum[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)];
  }
  std::int32_t block(int x, int y, int w, int h) const {
    return at(x + w, y + h) - at(x + w, y) - at(x, y + h) + at(x, y);
  }
};

void equalize(GrayImage& image) {
  std::array<std::uint32_t, 256> hist{};
  for (const std::uint8_t v : image.pixels) ++hist.at(v);
  const auto total = static_cast<std::uint32_t>(image.pixels.size());
  std::size_t first = 0;
  while (first < hist.size() && hist.at(first) == 0) ++first;
  if (first == hist.size() || hist.at(first) == total) return;  // vide ou d'un seul niveau
  const double scale = 255.0 / static_cast<double>(total - hist.at(first));
  std::array<std::uint8_t, 256> lut{};
  std::uint32_t cdf = 0;
  for (std::size_t i = first + 1; i < hist.size(); ++i) {
    cdf += hist.at(i);
    lut.at(i) = static_cast<std::uint8_t>(std::lround(std::min(255.0, static_cast<double>(cdf) * scale)));
  }
  for (std::uint8_t& v : image.pixels) v = lut.at(v);
}

GrayImage resize_linear(const GrayImage& image, int width, int height) {
  GrayImage out{.width = width, .height = height};
  out.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  if (image.width <= 0 || image.height <= 0 || width <= 0 || height <= 0) return out;
  const float sx = static_cast<float>(image.width) / static_cast<float>(width);
  const float sy = static_cast<float>(image.height) / static_cast<float>(height);
  for (int y = 0; y < height; ++y) {
    const float fy =
        std::clamp(((static_cast<float>(y) + 0.5F) * sy) - 0.5F, 0.0F, static_cast<float>(image.height - 1));
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float ty = fy - static_cast<float>(y0);
    for (int x = 0; x < width; ++x) {
      const float fx =
          std::clamp(((static_cast<float>(x) + 0.5F) * sx) - 0.5F, 0.0F, static_cast<float>(image.width - 1));
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, image.width - 1);
      const float tx = fx - static_cast<float>(x0);
      const float top =
          std::lerp(static_cast<float>(image.at(x0, y0)), static_cast<float>(image.at(x1, y0)), tx);
      const float bottom =
          std::lerp(static_cast<float>(image.at(x0, y1)), static_cast<float>(image.at(x1, y1)), tx);
      out.pixels[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
                 static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(std::lround(std::lerp(top, bottom, ty)));
    }
  }
  return out;
}

std::optional<FaceCascade::Weak> FaceCascade::weak_of(std::string_view nodes_text,
                                                      std::optional<std::string_view> leaves_text) {
  const auto nodes = numbers<std::int64_t>(nodes_text);
  const auto leaves = leaves_text ? numbers<float>(*leaves_text) : std::nullopt;
  // Une souche : « 0 -1 trait » puis les huit mots de l'ensemble ; deux feuilles.
  if (!nodes || nodes->size() != 11 || (*nodes)[0] != 0 || (*nodes)[1] != -1 || !leaves ||
      leaves->size() != 2)
    return std::nullopt;
  Weak weak{.feature = static_cast<int>((*nodes)[2]), .left = (*leaves)[0], .right = (*leaves)[1]};
  for (std::size_t i = 0; i < 8; ++i) weak.subset.at(i) = static_cast<std::int32_t>((*nodes)[3 + i]);
  return weak;
}

bool FaceCascade::read_stage(std::string_view xml, std::size_t& pos) {
  const auto threshold = number_tag<float>(xml, "stageThreshold", pos);
  const std::size_t end = xml.find("</weakClassifiers>", pos);
  if (!threshold || end == std::string_view::npos) return false;
  Stage stage{.threshold = *threshold, .first = weak_.size()};
  while (const auto nodes_text = next_tag(xml, "internalNodes", pos, end)) {
    const auto weak = weak_of(*nodes_text, next_tag(xml, "leafValues", pos, end));
    if (!weak) return false;
    weak_.push_back(*weak);
  }
  pos = end;
  stage.count = weak_.size() - stage.first;
  if (stage.count == 0) return false;
  stages_.push_back(stage);
  return true;
}

bool FaceCascade::read_features(std::string_view xml, std::size_t pos) {
  pos = xml.find("<features>", pos);
  if (pos == std::string_view::npos) return false;
  while (const auto rect_text = next_tag(xml, "rect", pos)) {
    const auto r = numbers<int>(*rect_text);
    if (!r || r->size() != 4) return false;
    const Rect rect{.x = (*r)[0], .y = (*r)[1], .w = (*r)[2], .h = (*r)[3]};
    // Trois blocs de côté dans la fenêtre.
    if (rect.x < 0 || rect.y < 0 || rect.w <= 0 || rect.h <= 0 || rect.x + (3 * rect.w) > width_ ||
        rect.y + (3 * rect.h) > height_)
      return false;
    features_.push_back(rect);
  }
  return std::ranges::none_of(weak_, [&](const Weak& wk) {
    return wk.feature < 0 || static_cast<std::size_t>(wk.feature) >= features_.size();
  });
}

std::optional<FaceCascade> FaceCascade::parse(std::string_view xml) {
  FaceCascade c;
  std::size_t pos = 0;
  const auto stage_type = next_tag(xml, "stageType", pos);
  const auto feature_type = next_tag(xml, "featureType", pos);
  if (!stage_type || *stage_type != "BOOST" || !feature_type || *feature_type != "LBP") return std::nullopt;
  c.height_ = number_tag<int>(xml, "height", pos).value_or(0);
  c.width_ = number_tag<int>(xml, "width", pos).value_or(0);
  if (c.height_ < 3 || c.width_ < 3) return std::nullopt;
  const int stage_count = number_tag<int>(xml, "stageNum", pos).value_or(0);
  if (stage_count <= 0) return std::nullopt;
  for (int s = 0; s < stage_count; ++s)
    if (!c.read_stage(xml, pos)) return std::nullopt;
  if (!c.read_features(xml, pos)) return std::nullopt;
  return c;
}

const FaceCascade* FaceCascade::anime() {
  static const std::optional<FaceCascade> cascade = parse(kAnimeFaceCascade);
  return cascade ? &*cascade : nullptr;
}

bool FaceCascade::accepts(const Integral& sum, int x, int y) const {
  for (const Stage& stage : stages_) {
    float total = 0.0F;
    for (std::size_t k = stage.first; k < stage.first + stage.count; ++k) {
      const Weak& weak = weak_[k];
      const Rect& f = features_[static_cast<std::size_t>(weak.feature)];
      const int bx = x + f.x;
      const int by = y + f.y;
      const auto b = [&](int col, int row) {
        return sum.block(bx + (col * f.w), by + (row * f.h), f.w, f.h);
      };
      const std::int32_t c = b(1, 1);
      // Les huit voisins dans le sens des aiguilles d'une montre depuis le coin haut gauche, bit 7 d'abord.
      const unsigned code = (b(0, 0) >= c ? 128U : 0U) | (b(1, 0) >= c ? 64U : 0U) |
                            (b(2, 0) >= c ? 32U : 0U) | (b(2, 1) >= c ? 16U : 0U) | (b(2, 2) >= c ? 8U : 0U) |
                            (b(1, 2) >= c ? 4U : 0U) | (b(0, 2) >= c ? 2U : 0U) | (b(0, 1) >= c ? 1U : 0U);
      const auto word = static_cast<std::uint32_t>(weak.subset.at(code >> 5U));
      total += (word & (1U << (code & 31U))) != 0 ? weak.left : weak.right;
    }
    if (total < stage.threshold) return false;
  }
  return true;
}

std::vector<FaceBox> FaceCascade::detect(const GrayImage& gray, const CascadeOptions& options) const {
  std::vector<FaceBox> faces;
  if (gray.width < width_ || gray.height < height_ || stages_.empty()) return faces;
  GrayImage image = gray;
  equalize(image);
  std::vector<Box> found;
  const float step = std::max(options.scale_step, 1.01F);
  for (float factor = 1.0F;; factor *= step) {
    const int sw = static_cast<int>(std::lround(static_cast<float>(image.width) / factor));
    const int sh = static_cast<int>(std::lround(static_cast<float>(image.height) / factor));
    if (sw < width_ || sh < height_) break;
    if (static_cast<float>(width_) * factor < static_cast<float>(options.min_size)) continue;
    const GrayImage scaled = factor == 1.0F ? image : resize_linear(image, sw, sh);
    const Integral sum(scaled);
    const int stride = factor > 2.0F ? 1 : 2;
    for (int y = 0; y + height_ <= sh; y += stride)
      for (int x = 0; x + width_ <= sw; x += stride)
        if (accepts(sum, x, y))
          found.push_back(Box{static_cast<float>(x) * factor, static_cast<float>(y) * factor,
                              static_cast<float>(width_) * factor, static_cast<float>(height_) * factor});
  }
  const auto fw = static_cast<float>(gray.width);
  const auto fh = static_cast<float>(gray.height);
  for (const auto& [b, n] : group(found, options.min_neighbors)) {
    faces.push_back(FaceBox{
        .x0 = std::clamp(b.x / fw, 0.0F, 1.0F),
        .y0 = std::clamp(b.y / fh, 0.0F, 1.0F),
        .x1 = std::clamp((b.x + b.w) / fw, 0.0F, 1.0F),
        .y1 = std::clamp((b.y + b.h) / fh, 0.0F, 1.0F),
        .score = std::min(
            1.0F, static_cast<float>(n) / static_cast<float>(4 * std::max(options.min_neighbors, 1)))});
  }
  return suppress_faces(std::move(faces));
}

}  // namespace ajp::media
