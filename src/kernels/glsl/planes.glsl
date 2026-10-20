// Lecture directe des plans d'une image décodée : la VkImage multiplan de FFmpeg (NV12, P010,
// P016 : luma au plan 0, CbCr entrelacés au plan 1), vue plan par plan en UNORM et lue sans
// filtrage par texelFetch. Le code du conteneur revient en multipliant par le maximum du plan,
// et P010 remet ses 10 bits en poids fort (kernels/spec/color.h : shift = 6). Mêmes résultats
// que les lectures de yuv.glsl dans un tampon de disposition serrée.
// S'inclut après yuv.glsl ; AJP_INTERLEAVED et AJP_SUBSAMPLED valent toujours ici.
#ifndef AJP_PLANES_GLSL
#define AJP_PLANES_GLSL

float plane_max() {
  if (AJP_FORMAT == AJP_FORMAT_P010) return 1023.0;
  if (AJP_SAMPLE_BYTES == 1u) return 255.0;
  return 65535.0;
}

uint plane_shift() {
  return AJP_FORMAT == AJP_FORMAT_P010 ? 6u : 0u;
}

// Code du conteneur d'une valeur UNORM lue dans le plan : exacte, l'arrondi rattrape la
// conversion flottante du matériel.
uint plane_code(float unorm) {
  return uint(round(unorm * plane_max())) << plane_shift();
}

// Luma (x, y) du plan 0.
uint luma_code(sampler2D luma, uint x, uint y) {
  return plane_code(texelFetch(luma, ivec2(int(x), int(y)), 0).r);
}

// Échantillon sample_x (chroma_sample_index : 2x pour Cb, 2x + 1 pour Cr) de la ligne y du
// plan 1, dont chaque texel porte Cb puis Cr.
uint chroma_code_at(sampler2D chroma, uint sample_x, uint y) {
  vec4 t = texelFetch(chroma, ivec2(int(sample_x >> 1u), int(y)), 0);
  return plane_code((sample_x & 1u) == 0u ? t.r : t.g);
}

#endif  // AJP_PLANES_GLSL
