// Élément des tenseurs RGB planaires des réseaux (NCHW sans lot) : float, ou float16_t quand le
// shader est compilé avec AJP_TENSOR_F16 (variante « _f16 », stockage 16 bits, calcul en float).
// Les tampons intermédiaires des passes séparables restent en float dans les deux variantes.
// S'inclut en premier, avant toute déclaration : il active une extension.
#ifndef AJP_TENSOR_GLSL
#define AJP_TENSOR_GLSL

#ifdef AJP_TENSOR_F16
#extension GL_EXT_shader_16bit_storage : require
#define AJP_TENSOR float16_t
#else
#define AJP_TENSOR float
#endif

#endif  // AJP_TENSOR_GLSL
