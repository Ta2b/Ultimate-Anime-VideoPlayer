// Élément des tenseurs RGB planaires des réseaux (NCHW sans lot) : float, ou float16_t quand le
// shader est compilé avec KIRARI_TENSOR_F16 (variante « _f16 », stockage 16 bits, calcul en float).
// Les tampons intermédiaires des passes séparables restent en float dans les deux variantes.
// S'inclut en premier, avant toute déclaration : il active une extension.
#ifndef KIRARI_TENSOR_GLSL
#define KIRARI_TENSOR_GLSL

#ifdef KIRARI_TENSOR_F16
#extension GL_EXT_shader_16bit_storage : require
#define KIRARI_TENSOR float16_t
#else
#define KIRARI_TENSOR float
#endif

#endif  // KIRARI_TENSOR_GLSL
