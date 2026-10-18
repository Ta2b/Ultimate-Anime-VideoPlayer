// Formats YUV des noyaux : constantes de spécialisation, choix des constantes de spec.h et
// accès aux codes rangés en mots de 32 bits. Un plan commence à un octet multiple de 4 de la
// vue liée et son pas de ligne aussi : un échantillon de 16 bits ne chevauche jamais deux mots.
// Les numéros suivent kernels::Format, Matrix et Range (gpu/vulkan/compute/kernel_types.h).
// S'inclut après ajp_kernels_spec.glsl et common.glsl.
#ifndef AJP_YUV_GLSL
#define AJP_YUV_GLSL

const uint AJP_FORMAT_NV12 = 0u;
const uint AJP_FORMAT_P010 = 1u;
const uint AJP_FORMAT_P016 = 2u;
const uint AJP_FORMAT_YUV420P = 3u;
const uint AJP_FORMAT_YUV420P10 = 4u;
const uint AJP_FORMAT_YUV444P16 = 5u;

const uint AJP_MATRIX_BT601 = 0u;
const uint AJP_MATRIX_BT709 = 1u;
const uint AJP_MATRIX_BT2020 = 2u;

// Fixées par l'hôte depuis kernels::info(format) : rien de la structure n'est recopié ici.
layout(constant_id = 0) const uint AJP_FORMAT = AJP_FORMAT_NV12;
layout(constant_id = 1) const uint AJP_MATRIX = AJP_MATRIX_BT709;
layout(constant_id = 2) const uint AJP_RANGE = 0u;  // 0 limitée, 1 pleine
layout(constant_id = 3) const uint AJP_SAMPLE_BYTES = 1u;
layout(constant_id = 4) const bool AJP_INTERLEAVED = true;
layout(constant_id = 5) const bool AJP_SUBSAMPLED = true;

const uint AJP_SAMPLES_PER_WORD = 4u / AJP_SAMPLE_BYTES;

// yuv_scale(format, plage) : (yoff, yscale, coff, cscale).
vec4 yuv_scale() {
  bool full = AJP_RANGE == 1u;
  if (AJP_FORMAT == AJP_FORMAT_NV12) return full ? AJP_YUV_SCALE_NV12_FULL : AJP_YUV_SCALE_NV12_LIMITED;
  if (AJP_FORMAT == AJP_FORMAT_P010) return full ? AJP_YUV_SCALE_P010_FULL : AJP_YUV_SCALE_P010_LIMITED;
  if (AJP_FORMAT == AJP_FORMAT_P016) return full ? AJP_YUV_SCALE_P016_FULL : AJP_YUV_SCALE_P016_LIMITED;
  if (AJP_FORMAT == AJP_FORMAT_YUV420P) {
    return full ? AJP_YUV_SCALE_YUV420P_FULL : AJP_YUV_SCALE_YUV420P_LIMITED;
  }
  if (AJP_FORMAT == AJP_FORMAT_YUV420P10) {
    return full ? AJP_YUV_SCALE_YUV420P10_FULL : AJP_YUV_SCALE_YUV420P10_LIMITED;
  }
  return full ? AJP_YUV_SCALE_YUV444P16_FULL : AJP_YUV_SCALE_YUV444P16_LIMITED;
}

// quantizer(format) : (qdiv, qmax).
vec2 quantizer() {
  if (AJP_FORMAT == AJP_FORMAT_NV12) return AJP_QUANT_NV12;
  if (AJP_FORMAT == AJP_FORMAT_P010) return AJP_QUANT_P010;
  if (AJP_FORMAT == AJP_FORMAT_P016) return AJP_QUANT_P016;
  if (AJP_FORMAT == AJP_FORMAT_YUV420P) return AJP_QUANT_YUV420P;
  if (AJP_FORMAT == AJP_FORMAT_YUV420P10) return AJP_QUANT_YUV420P10;
  return AJP_QUANT_YUV444P16;
}

// coefficients(matrice) : (kr, kb).
vec2 coefficients() {
  if (AJP_MATRIX == AJP_MATRIX_BT601) return vec2(AJP_KR_BT601, AJP_KB_BT601);
  if (AJP_MATRIX == AJP_MATRIX_BT709) return vec2(AJP_KR_BT709, AJP_KB_BT709);
  return vec2(AJP_KR_BT2020, AJP_KB_BT2020);
}

// Octet de l'échantillon i de la ligne y d'un plan, depuis le début de la vue.
uint sample_byte(uint offset, uint pitch, uint i, uint y) {
  return offset + y * pitch + i * AJP_SAMPLE_BYTES;
}

// Le code rangé à l'octet byte, extrait du mot qui le contient.
uint sample_in_word(uint word, uint byte) {
  uint mask = AJP_SAMPLE_BYTES == 1u ? 0xFFu : 0xFFFFu;
  return (word >> ((byte & 3u) * 8u)) & mask;
}

// Place le code de l'échantillon s (0 en tête) dans un mot.
uint sample_to_word(uint code, uint s) {
  return code << (s * 8u * AJP_SAMPLE_BYTES);
}

// Échantillon de Cb (ou Cr) de la colonne x dans un plan entrelacé ou planaire.
uint chroma_sample_index(uint x, bool cr) {
  return AJP_INTERLEAVED ? 2u * x + (cr ? 1u : 0u) : x;
}

// normalize puis to_rgb (spec/color.h, k_pre_combine), y, u et v en codes du conteneur.
vec3 yuv_to_rgb(float y, float u, float v) {
  vec4 s = yuv_scale();
  vec2 k = coefficients();
  precise float luma = (y - s.x) / s.y;
  precise float cb = (u - s.z) / s.w;
  precise float cr = (v - s.z) / s.w;
  precise float kg = 1.0 - k.x - k.y;
  precise float r = luma + 2.0 * (1.0 - k.x) * cr;
  precise float g = luma - (2.0 * k.y * (1.0 - k.y) * cb + 2.0 * k.x * (1.0 - k.x) * cr) / kg;
  precise float b = luma + 2.0 * (1.0 - k.y) * cb;
  return vec3(r, g, b);
}

// to_yuv (spec/color.h, k_post_matrix) : Y, Cb et Cr normalisés.
vec3 rgb_to_yuv(vec3 p) {
  vec2 k = coefficients();
  precise float luma = k.x * p.r + (1.0 - k.x - k.y) * p.g + k.y * p.b;
  precise float cb = (p.b - luma) / (2.0 * (1.0 - k.y));
  precise float cr = (p.r - luma) / (2.0 * (1.0 - k.x));
  return vec3(luma, cb, cr);
}

// encode_luma et encode_chroma (spec/color.h).
uint encode_luma(float luma) {
  vec4 s = yuv_scale();
  precise float raw = luma * s.y + s.x;
  return quantize(raw, quantizer());
}

uint encode_chroma(float c) {
  vec4 s = yuv_scale();
  precise float raw = c * s.w + s.z;
  return quantize(raw, quantizer());
}

#endif  // AJP_YUV_GLSL
