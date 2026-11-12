#include "media/field_alignment.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

namespace ajp::media {
namespace {

// Une triplette de lignes (y, y+1, y+2) tous les kRowStep, une colonne sur kColumnStep.
constexpr int kRowStep = 8;
constexpr int kColumnStep = 2;
// En unités de 8 bits : en deçà d'une différence moyenne de kMoving entre deux images, rien n'a bougé ; sous
// kFlat d'écart moyen entre lignes de même parité, l'image est trop plate pour juger.
constexpr double kMoving = 1.0;
constexpr double kFlat = 1.5;
// Rapport (lignes voisines) / (lignes de même parité) : sous kAligned, les champs se prolongent ; au-dessus
// de kComb, ils se contredisent.
constexpr double kAligned = 0.85;
constexpr double kComb = 1.0;
// Au moins kMinMoving images en mouvement pour conclure ; aligné si presque aucune n'est peignée, peigné si
// une sur dix l'est.
constexpr int kMinMoving = 20;
constexpr double kAlignedShare = 0.9;
constexpr double kStrayCombs = 0.02;
constexpr double kCombedShare = 0.1;

}  // namespace

void FieldAlignmentMeter::push(const std::uint8_t* luma, std::ptrdiff_t stride, int width, int height,
                               int sample_bytes, int bits) {
  if (luma == nullptr || width < 16 || height < 2 * kRowStep || (sample_bytes != 1 && sample_bytes != 2))
    return;
  const int shift = std::max(bits - 8, 0);
  const auto sample = [&](int y, int x) {
    const std::uint8_t* row = luma + static_cast<std::ptrdiff_t>(y) * stride;
    const auto index = static_cast<std::size_t>(x) * static_cast<std::size_t>(sample_bytes);
    const int value = sample_bytes == 1 ? row[index] : (row[index] | (row[index + 1] << 8));
    return value >> shift;
  };
  std::vector<std::uint16_t> same_parity;
  std::int64_t neighbor_sum = 0;
  std::int64_t parity_sum = 0;
  std::int64_t count = 0;
  for (int y = 0; y + 2 < height; y += kRowStep) {
    for (int x = 0; x < width; x += kColumnStep) {
      const int p0 = sample(y, x);
      const int p1 = sample(y + 1, x);
      const int p2 = sample(y + 2, x);
      neighbor_sum += std::abs(p1 - p0);
      parity_sum += std::abs(p2 - p0);
      ++count;
      if (x % (2 * kColumnStep) == 0) same_parity.push_back(static_cast<std::uint16_t>(p0));
    }
  }
  if (count == 0) return;
  if (previous_.size() == same_parity.size() && !previous_.empty()) {
    std::int64_t moved = 0;
    for (std::size_t i = 0; i < same_parity.size(); ++i)
      moved += std::abs(static_cast<int>(same_parity[i]) - static_cast<int>(previous_[i]));
    ++compared_;
    const double motion = static_cast<double>(moved) / static_cast<double>(same_parity.size());
    const double detail = static_cast<double>(parity_sum) / static_cast<double>(count);
    if (motion >= kMoving && detail >= kFlat) {
      ++moving_;
      const double ratio = static_cast<double>(neighbor_sum) / static_cast<double>(parity_sum);
      if (ratio <= kAligned) ++aligned_;
      if (ratio > kComb) ++combed_;
    }
  }
  previous_ = std::move(same_parity);
}

FieldAlignmentMeter::Verdict FieldAlignmentMeter::verdict() const {
  if (moving_ < kMinMoving) return Verdict::Unknown;
  const double moving = moving_;
  if (aligned_ >= kAlignedShare * moving && combed_ <= kStrayCombs * moving) return Verdict::Aligned;
  if (combed_ >= kCombedShare * moving) return Verdict::Combed;
  return Verdict::Unknown;
}

}  // namespace ajp::media
