#include "subs/atlas_packer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numeric>

namespace ajp::subs {
namespace {

constexpr int kMaxSideLimit = 65536;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

int next_power_of_two(int value) {
  int power = 1;
  while (power < value) power *= 2;
  return power;
}

// Étagères de gauche à droite, de haut en bas, dans l'ordre donné. Renvoie
// false si le contenu dépasse max_height.
bool fill_shelves(std::span<const PackSize> sizes, std::span<const std::size_t> order, int width,
                  int max_height, Packing& out) {
  int x = 0;
  int y = 0;
  int shelf_height = 0;
  out.width = 0;
  for (const std::size_t i : order) {
    const PackSize size = sizes[i];
    if (x > 0 && x + size.width > width) {  // étagère pleine : on descend
      y += shelf_height + kAtlasPadding;
      x = 0;
      shelf_height = 0;
    }
    if (y + size.height > max_height) return false;
    out.positions[i] = {x, y};
    out.width = std::max(out.width, x + size.width);
    shelf_height = std::max(shelf_height, size.height);
    x += size.width + kAtlasPadding;
  }
  out.height = y + shelf_height;
  return true;
}

}  // namespace

std::expected<Packing, Refusal> pack(std::span<const PackSize> sizes, int max_side) {
  if (max_side < 1 || max_side > kMaxSideLimit) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("atlas : côté maximal {}", max_side));
  }
  Packing packing;
  packing.positions.resize(sizes.size());
  if (sizes.empty()) return packing;

  int widest = 0;
  double area = 0.0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const PackSize size = sizes[i];
    if (size.width < 1 || size.height < 1) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("atlas : rectangle {} de {}×{}", i, size.width, size.height));
    }
    if (size.width > max_side || size.height > max_side) {
      return refuse(RefusalCode::SubtitleAtlasFull,
                    std::format("atlas : rectangle {}×{} au-delà de {}", size.width, size.height, max_side));
    }
    widest = std::max(widest, size.width);
    area +=
        static_cast<double>(size.width + kAtlasPadding) * static_cast<double>(size.height + kAtlasPadding);
  }

  // Les plus hauts d'abord : chaque étagère se remplit de hauteurs voisines.
  std::vector<std::size_t> order(sizes.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
    if (sizes[a].height != sizes[b].height) return sizes[a].height > sizes[b].height;
    return sizes[a].width > sizes[b].width;
  });

  // Largeur de départ : le carré de l'aire, puis on double jusqu'à ce que tout tienne.
  const int square = static_cast<int>(std::min(std::ceil(std::sqrt(area)), static_cast<double>(max_side)));
  int width = std::min(std::max(widest, next_power_of_two(square)), max_side);
  while (!fill_shelves(sizes, order, width, max_side, packing)) {
    if (width == max_side) {
      return refuse(RefusalCode::SubtitleAtlasFull,
                    std::format("atlas : {} rectangles au-delà de {}×{}", sizes.size(), max_side, max_side));
    }
    width = std::min(width * 2, max_side);
  }
  return packing;
}

}  // namespace ajp::subs
