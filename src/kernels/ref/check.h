// Vérifications partagées par la référence : une entrée hors contrat est refusée, jamais
// corrigée. Interne à kernels/ref/.
#pragma once

#include <cstdint>
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

// Un plan quelconque : dimensions dans [1, kMaxDimension], et exactement width × height
// échantillons (les champs sont publics : un plan peut mentir sur sa taille).
template <class T>
Result<void> check_plane(const Plane<T>& p, std::string_view what) {
  if (!valid_plane_dimension(p.width) || !valid_plane_dimension(p.height)) {
    return fail(ErrorCode::DimensionOutOfRange,
                std::format("{} : {}x{} hors de [1, {}]", what, p.width, p.height, kMaxDimension));
  }
  if (p.data.size() != p.area()) {
    return fail(ErrorCode::SizeMismatch,
                std::format("{} : {} échantillons pour {}x{}", what, p.data.size(), p.width, p.height));
  }
  return {};
}

// Chaque code dans son conteneur : au plus max_code et, en P010, ses 6 bits bas à zéro.
inline Result<void> check_samples(const PlaneU16& p, Format f, std::string_view what) {
  const std::uint32_t low_bits = (1U << info(f).shift) - 1U;
  for (const std::uint16_t code : p.data) {
    if (code > max_code(f) || (code & low_bits) != 0) {
      return fail(ErrorCode::SampleOutOfRange, std::format("{} : code {} hors du conteneur", what, code));
    }
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

// Les trois plans d'une image RGB ont la même taille, et chacun est un plan conforme.
inline Result<void> check_rgb(const RgbImage& image, std::string_view what) {
  for (const PlaneF& p : image.planes) {
    if (auto ok = check_same_size(image.planes[0], p, what); !ok) return ok;
    if (auto ok = check_plane(p, what); !ok) return ok;
  }
  return {};
}

inline Result<void> check_same_size(const RgbImage& a, const RgbImage& b, std::string_view what) {
  if (auto ok = check_rgb(a, what); !ok) return ok;
  if (auto ok = check_rgb(b, what); !ok) return ok;
  return check_same_size(a.planes[0], b.planes[0], what);
}

}  // namespace ajp::kernels::ref
