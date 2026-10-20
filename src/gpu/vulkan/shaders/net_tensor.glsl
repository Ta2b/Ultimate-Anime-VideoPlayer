// Tenseurs des réseaux exécutés par le moteur Vulkan (gpu_ir) : rangement, lecture et écriture.
//
// Un tenseur est vu en rang 4 (n, c, h, w), les rangs plus petits alignés à droite. Deux
// dispositions, celles de gpu/ir/ir.h : 0 Nchw (ordre ligne), 1 Nhwc16 (canaux en dernier, leur
// nombre arrondi à cp, multiple de 16). Dans la variante « _f16 » (AJP_NET_F16), chaque tampon est
// lié deux fois au même binding, en float et en float16_t, et une constante de spécialisation par
// tenseur dit lequel lire : 0 float, 1 float16. Le calcul se fait en float.
// S'inclut en premier, avant toute déclaration : il active une extension.
#ifndef AJP_NET_TENSOR_GLSL
#define AJP_NET_TENSOR_GLSL

#ifdef AJP_NET_F16
#extension GL_EXT_shader_16bit_storage : require
#endif

struct TDesc {
  uint n;
  uint c;
  uint h;
  uint w;
  uint order;  // disposition : 0 Nchw, 1 Nhwc16
  uint cp;
};

uint t_index(TDesc d, uint n, uint c, uint y, uint x) {
  if (d.order == 0u) return ((n * d.c + c) * d.h + y) * d.w + x;
  return ((n * d.h + y) * d.w + x) * d.cp + c;
}

uint t_count(TDesc d) {
  return d.n * d.c * d.h * d.w;
}

// Coordonnées logiques de l'élément i, dans l'ordre de la mémoire de d : écritures contiguës.
uvec4 t_coords(TDesc d, uint i) {
  uvec4 o;  // (n, c, y, x)
  if (d.order == 0u) {
    o.w = i % d.w;
    i /= d.w;
    o.z = i % d.h;
    i /= d.h;
    o.y = i % d.c;
    o.x = i / d.c;
  } else {
    o.y = i % d.c;
    i /= d.c;
    o.w = i % d.w;
    i /= d.w;
    o.z = i % d.h;
    o.x = i / d.h;
  }
  return o;
}

// Indice linéaire d'un groupe de 256 invocations dans une grille 2D (gl_NumWorkGroups.x groupes
// par ligne), pour dépasser la limite de 65535 groupes d'un axe.
uint linear_invocation() {
  return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u + gl_LocalInvocationIndex;
}

#ifdef AJP_NET_F16
#define AJP_NET_BUFFER(b, name, access)                                       \
  layout(std430, binding = b) access buffer name##F { float v[]; } name##_f; \
  layout(std430, binding = b) access buffer name##H { float16_t v[]; } name##_h;
#define AJP_LOAD(name, type, i) ((type) == 1u ? float(name##_h.v[i]) : name##_f.v[i])
#define AJP_STORE(name, type, i, value) \
  if ((type) == 1u) {                   \
    name##_h.v[i] = float16_t(value);   \
  } else {                              \
    name##_f.v[i] = (value);            \
  }
#else
#define AJP_NET_BUFFER(b, name, access) layout(std430, binding = b) access buffer name##F { float v[]; } name##_f;
#define AJP_LOAD(name, type, i) (name##_f.v[i])
#define AJP_STORE(name, type, i, value) name##_f.v[i] = (value);
#endif

#endif  // AJP_NET_TENSOR_GLSL
