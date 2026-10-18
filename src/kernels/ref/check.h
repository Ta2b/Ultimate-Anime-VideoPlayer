// Vérifications partagées par la référence : une entrée hors contrat est refusée, jamais
// corrigée. Interne à kernels/ref/.
#pragma once

#include <format>
#include <string_view>

#include "kernels/image.h"
#include "kernels/spec/resample.h"

namespace ajp::kernels::ref {

// Une image de la chaîne : dimensions paires dans [2, kMaxDimension].
inline Result<void> check_frame(int width, int height, std::string_view what) {
  for (const int n : {width, height}) {
    if (n < 2 || n > kMaxDimension) {
      return fail(ErrorCode::DimensionOutOfRange,
                  std::format("{} : {}x{} hors de [2, {}]", what, width, height, kMaxDimension));
    }
    if (n % 2 != 0) {
      return fail(ErrorCode::OddDimension, std::format("{} : {}x{} impair", what, width, height));
    }
  }
  return {};
}

// Un plan quelconque : dimensions dans [1, kMaxDimension].
template <class T>
Result<void> check_plane(const Plane<T>& p, std::string_view what) {
  if (!valid_plane_dimension(p.width) || !valid_plane_dimension(p.height)) {
    return fail(ErrorCode::DimensionOutOfRange,
                std::format("{} : {}x{} hors de [1, {}]", what, p.width, p.height, kMaxDimension));
  }
  return {};
}

template <class A, class B>
Result<void> check_same_size(const Plane<A>& a, const Plane<B>& b, std::string_view what) {
  if (a.width != b.width || a.height != b.height) {
    return fail(ErrorCode::SizeMismatch,
                std::format("{} : {}x{} contre {}x{}", what, a.width, a.height, b.width, b.height));
  }
  return {};
}

// Les trois plans d'une image RGB ont la même taille, dans la plage des plans.
inline Result<void> check_rgb(const RgbImage& image, std::string_view what) {
  if (auto ok = check_same_size(image.planes[0], image.planes[1], what); !ok) return ok;
  if (auto ok = check_same_size(image.planes[0], image.planes[2], what); !ok) return ok;
  return check_plane(image.planes[0], what);
}

inline Result<void> check_same_size(const RgbImage& a, const RgbImage& b, std::string_view what) {
  if (auto ok = check_rgb(a, what); !ok) return ok;
  if (auto ok = check_rgb(b, what); !ok) return ok;
  return check_same_size(a.planes[0], b.planes[0], what);
}

}  // namespace ajp::kernels::ref
