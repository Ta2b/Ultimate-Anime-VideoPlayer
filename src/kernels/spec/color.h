// Contrat des conversions YUV ↔ RGB : formats, matrices, plages, siting et quantification.
//
// Repris de animejanai-inference-rtx@118c272 :
//   src/resample.h:100-129 (make_csp : échelles en unités du conteneur),
//   src/kernels.cu:94-97 (quant), 141-165 (k_pre_combine), 221-242 (k_post_matrix),
//   src/kernels.cu:846-875 (k_rtx_pack : clamp, × (2^n − 1), + 0,5, troncature) ;
// et de the-database/mpv video/filter/vf_animejanai.c:337-366 (matrice, plage, siting).
// Tests : tests/unit/kernels/color_test.cpp.
//
// Toutes les formules s'évaluent en float, dans l'ordre écrit, sans contraction FMA.
#pragma once

#include <cstdint>
#include <optional>

namespace ajp::kernels {

enum class Format : std::uint8_t {
  Nv12,       // 8 bits 4:2:0, chroma CbCr entrelacée
  P010,       // 10 bits dans 16, alignés sur le poids fort (6 bits bas à zéro), 4:2:0 entrelacé
  P016,       // 16 bits 4:2:0, chroma entrelacée
  Yuv420p,    // 8 bits 4:2:0 planaire
  Yuv420p10,  // 10 bits dans 16, alignés sur le poids faible, 4:2:0 planaire
  Yuv444p16,  // 16 bits 4:4:4 planaire
};

enum class Matrix : std::uint8_t { Bt601, Bt709, Bt2020Ncl };
enum class Range : std::uint8_t { Limited, Full };
// Position des échantillons de chroma 4:2:0 par rapport à la luma.
enum class Siting : std::uint8_t {
  Left,     // MPEG-2/4, H.264 : alignée sur la colonne paire, centrée verticalement
  Center,   // MPEG-1, JPEG : au centre des quatre lumas
  TopLeft,  // HEVC/UHD BT.2020 : alignée sur la colonne et la ligne paires
};

// Ce qu'un format range dans sa mémoire.
struct FormatInfo {
  int bits = 8;             // profondeur significative : 8, 10 ou 16
  int shift = 0;            // alignement sur le poids fort : 6 pour P010, 0 sinon
  int bytes = 1;            // taille d'un échantillon
  bool subsampled = true;   // chroma 4:2:0, de dimensions (w/2, h/2)
  bool interleaved = true;  // CbCr entrelacés dans un seul plan
};

constexpr FormatInfo info(Format f) {
  switch (f) {
    case Format::Nv12:
      return {.bits = 8, .shift = 0, .bytes = 1, .subsampled = true, .interleaved = true};
    case Format::P010:
      return {.bits = 10, .shift = 6, .bytes = 2, .subsampled = true, .interleaved = true};
    case Format::P016:
      return {.bits = 16, .shift = 0, .bytes = 2, .subsampled = true, .interleaved = true};
    case Format::Yuv420p:
      return {.bits = 8, .shift = 0, .bytes = 1, .subsampled = true, .interleaved = false};
    case Format::Yuv420p10:
      return {.bits = 10, .shift = 0, .bytes = 2, .subsampled = true, .interleaved = false};
    case Format::Yuv444p16:
      return {.bits = 16, .shift = 0, .bytes = 2, .subsampled = false, .interleaved = false};
  }
  return {};
}

// Plus grand code valide du conteneur : (2^bits − 1) << shift.
constexpr std::uint32_t max_code(Format f) {
  const FormatInfo i = info(f);
  return ((1U << i.bits) - 1U) << i.shift;
}

// Unités du conteneur qui valent un niveau 8 bits : 2^(bits − 8) << shift.
constexpr std::uint32_t level_unit(Format f) {
  const FormatInfo i = info(f);
  return (1U << (i.bits - 8)) << i.shift;
}

// Coefficients de luma de la matrice : Y = kr R + (1 − kr − kb) G + kb B.
struct Coefficients {
  float kr = 0.0F;
  float kb = 0.0F;
};

constexpr Coefficients coefficients(Matrix m) {
  switch (m) {
    case Matrix::Bt601:
      return {.kr = 0.299F, .kb = 0.114F};
    case Matrix::Bt709:
      return {.kr = 0.2126F, .kb = 0.0722F};
    case Matrix::Bt2020Ncl:
      return {.kr = 0.2627F, .kb = 0.0593F};
  }
  return {};
}

// Normalisation en unités brutes du conteneur (P010 garde son décalage de 6 bits) :
//   Y = (brut − yoff) / yscale,  Cb, Cr = (brut − coff) / cscale.
// Plage limitée : 16, 219, 128, 224 niveaux 8 bits, mis à l'échelle de level_unit.
// Plage pleine : 0, 2^bits − 1, 2^(bits − 1), 2^bits − 1, décalés de shift.
struct YuvScale {
  float yoff = 0.0F;
  float yscale = 1.0F;
  float coff = 0.0F;
  float cscale = 1.0F;
};

constexpr YuvScale yuv_scale(Format f, Range r) {
  const FormatInfo i = info(f);
  if (r == Range::Limited) {
    const auto unit = static_cast<float>(level_unit(f));
    return {.yoff = 16.0F * unit, .yscale = 219.0F * unit, .coff = 128.0F * unit, .cscale = 224.0F * unit};
  }
  const auto peak = static_cast<float>(max_code(f));
  const auto middle = static_cast<float>((1U << (i.bits - 1)) << i.shift);
  return {.yoff = 0.0F, .yscale = peak, .coff = middle, .cscale = peak};
}

// Quantification exacte (kernels.cu quant) :
//   code = clamp(arrondi_pair(v / qdiv), 0, qmax) × qdiv,
// avec qdiv = 2^shift et qmax = 2^bits − 1. P010 se quantifie donc à 10 bits vrais, puis se
// décale. L'arrondi va au plus proche, les égalités au pair (rintf en CUDA, roundEven en GLSL).
// NaN et valeurs négatives donnent 0, +∞ donne qmax. Aucun tramage.
struct Quantizer {
  float qdiv = 1.0F;
  float qmax = 255.0F;
};

constexpr Quantizer quantizer(Format f) {
  const FormatInfo i = info(f);
  return {.qdiv = static_cast<float>(1U << i.shift), .qmax = static_cast<float>((1U << i.bits) - 1U)};
}

// Arrondi au plus proche, égalités au pair, pour 0 ≤ x < 2^24 (exact en float).
constexpr float round_half_even(float x) {
  const auto whole = static_cast<std::int64_t>(x);
  const auto lower = static_cast<float>(whole);
  const float fraction = x - lower;
  if (fraction > 0.5F) return lower + 1.0F;
  if (fraction < 0.5F) return lower;
  return whole % 2 == 0 ? lower : lower + 1.0F;
}

constexpr std::uint16_t quantize(float raw, Format f) {
  const Quantizer q = quantizer(f);
  const float x = raw / q.qdiv;
  float code = 0.0F;
  if (x >= q.qmax) {
    code = q.qmax;
  } else if (x > 0.0F) {
    code = round_half_even(x);
  }
  return static_cast<std::uint16_t>(code * q.qdiv);
}

// Tramage de la sortie quantifiée. None : quantize, l'arrondi au plus proche. Bayer8 : un seuil
// ordonné 8 × 8 par position (quantize_dithered), pour l'export (le tramage que libplacebo y
// appliquait) : les dégradés lents ne font pas de marches visibles en 8 bits.
enum class Dither : std::uint8_t { None, Bayer8 };

// Matrice de Bayer d'ordre 8 à la position (x, y) : 16 m(x₀, y₀) + 4 m(x₁, y₁) + m(x₂, y₂), où
// x_k et y_k sont les bits de x et y modulo 8 et m(a, b) = 2 (a ⊕ b) + b (matrice d'ordre 2
// [[0, 2], [3, 1]]). Chaque valeur de 0 à 63 une fois par bloc 8 × 8 ; première ligne 0 32 8 40 2
// 34 10 42.
constexpr int bayer8(int x, int y) {
  const auto m = [](int a, int b) { return 2 * (a ^ b) + b; };
  return 16 * m(x & 1, y & 1) + 4 * m((x >> 1) & 1, (y >> 1) & 1) + m((x >> 2) & 1, (y >> 2) & 1);
}

// Seuil de la position : (bayer8 + 1/2) / 64, dans ]0, 1[, de moyenne 1/2 sur un bloc.
constexpr float bayer8_threshold(int x, int y) {
  return (static_cast<float>(bayer8(x, y)) + 0.5F) / 64.0F;
}

// Quantification tramée : v = raw / qdiv + seuil ; code = qmax si v ≥ qmax, ⌊v⌋ si v > 0, 0 sinon
// (NaN compris), puis × qdiv. Un seuil de 1/2 arrondit au plus proche, égalités vers le haut ; la
// moyenne des codes d'un aplat sur un bloc 8 × 8 vaut raw / qdiv à 1/64 près.
constexpr std::uint16_t quantize_dithered(float raw, Format f, float threshold) {
  const Quantizer q = quantizer(f);
  const float v = raw / q.qdiv + threshold;
  float code = 0.0F;
  if (v >= q.qmax) {
    code = q.qmax;
  } else if (v > 0.0F) {
    code = static_cast<float>(static_cast<std::int64_t>(v));
  }
  return static_cast<std::uint16_t>(code * q.qdiv);
}

// Pixel normalisé : Y dans [0, 1], Cb et Cr dans [−1/2, 1/2] pour un pixel dans le gamut.
struct Yuv {
  float y = 0.0F;
  float u = 0.0F;
  float v = 0.0F;
};

// RGB non linéaire, [0, 1] dans le gamut. Rien n'est borné avant la quantification finale :
// une couleur hors gamut passe telle quelle vers les réseaux, comme dans zimg.
struct Rgb {
  float r = 0.0F;
  float g = 0.0F;
  float b = 0.0F;
};

constexpr Yuv normalize(float y, float u, float v, const YuvScale& s) {
  return {.y = (y - s.yoff) / s.yscale, .u = (u - s.coff) / s.cscale, .v = (v - s.coff) / s.cscale};
}

// k_pre_combine : R = Y + 2(1 − kr) Cr, B = Y + 2(1 − kb) Cb,
// G = Y − (2 kb (1 − kb) Cb + 2 kr (1 − kr) Cr) / (1 − kr − kb).
constexpr Rgb to_rgb(const Yuv& p, const Coefficients& k) {
  const float kg = 1.0F - k.kr - k.kb;
  return {
      .r = p.y + 2.0F * (1.0F - k.kr) * p.v,
      .g = p.y - (2.0F * k.kb * (1.0F - k.kb) * p.u + 2.0F * k.kr * (1.0F - k.kr) * p.v) / kg,
      .b = p.y + 2.0F * (1.0F - k.kb) * p.u,
  };
}

// k_post_matrix : Y = kr R + (1 − kr − kb) G + kb B, Cb = (B − Y) / (2(1 − kb)),
// Cr = (R − Y) / (2(1 − kr)).
constexpr Yuv to_yuv(const Rgb& p, const Coefficients& k) {
  const float y = k.kr * p.r + (1.0F - k.kr - k.kb) * p.g + k.kb * p.b;
  return {.y = y, .u = (p.b - y) / (2.0F * (1.0F - k.kb)), .v = (p.r - y) / (2.0F * (1.0F - k.kr))};
}

constexpr std::uint16_t encode_luma(float y, Format f, const YuvScale& s) {
  return quantize(y * s.yscale + s.yoff, f);
}

constexpr std::uint16_t encode_chroma(float c, Format f, const YuvScale& s) {
  return quantize(c * s.cscale + s.coff, f);
}

// Les mêmes, tramées au seuil de leur position (bayer8_threshold) : la luma à celle du pixel, Cb
// et Cr à celle de l'échantillon de chroma, le même seuil pour les deux.
constexpr std::uint16_t encode_luma_dithered(float y, Format f, const YuvScale& s, float threshold) {
  return quantize_dithered(y * s.yscale + s.yoff, f, threshold);
}

constexpr std::uint16_t encode_chroma_dithered(float c, Format f, const YuvScale& s, float threshold) {
  return quantize_dithered(c * s.cscale + s.coff, f, threshold);
}

// Sortie RGBA16 UNORM du ring : code = tronque(clamp(v, 0, 1) × 65535 + 0,5), alpha = 65535.
// Le clamp suit fminf(fmaxf(v, 0), 1) : NaN donne 0 (un port GLSL le traite explicitement,
// clamp() y étant indéfini pour NaN).
inline constexpr float kUnorm16Max = 65535.0F;
inline constexpr std::uint16_t kOpaqueAlpha16 = 65535;

constexpr float clamp_unit(float v) {
  if (!(v > 0.0F)) return 0.0F;
  return v < 1.0F ? v : 1.0F;
}

constexpr std::uint16_t unorm16(float v) {
  // La valeur est positive : tronquer après + 0,5 arrondit au plus proche, égalités vers le haut.
  const float biased = clamp_unit(v) * kUnorm16Max + 0.5F;
  return static_cast<std::uint16_t>(biased);
}

// Étiquettes de la source (vf_animejanai.c:337-366). std::nullopt : absente ou inconnue.
// Matrice non étiquetée : BT.601 sous 720 lignes, BT.709 sinon (heuristique SD du fork).
inline constexpr int kSdHeightLimit = 720;

constexpr Matrix effective_matrix(std::optional<Matrix> tagged, int height) {
  if (tagged) return *tagged;
  return height < kSdHeightLimit ? Matrix::Bt601 : Matrix::Bt709;
}

// Plage non étiquetée : limitée.
constexpr Range effective_range(std::optional<Range> tagged) {
  return tagged.value_or(Range::Limited);
}

// Siting non étiqueté : gauche.
constexpr Siting effective_siting(std::optional<Siting> tagged) {
  return tagged.value_or(Siting::Left);
}

// La sortie 4:2:0 se sous-échantillonne toujours en siting gauche, quelle que soit l'étiquette
// de la source : resize.Spline36(format=YUV420) de la chaîne de référence ne propage pas la
// position de chroma depuis RGB (aji_trt.cpp:2940-2945).
inline constexpr Siting kOutputSiting = Siting::Left;

}  // namespace ajp::kernels
