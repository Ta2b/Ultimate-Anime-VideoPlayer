// Faux SDK Maxine VFX : les fonctions NvVFX_* et NvCVImage_* que charge maxine_vfx.cpp, avec
// les prototypes de maxine_abi.h. Les « pointeurs CUDA » sont de la mémoire hôte. Le faux
// vérifie ce que le vrai SDK exige ou ce que le chargeur promet :
//   - flux 0 hérité (SetCudaStream avec nullptr) et exécution synchrone (async = 0) ;
//   - modèle chargé une seule fois, avant la première exécution ;
//   - descripteurs aux formats RGBA/U8 en entrée et RGB10A2/P32 en sortie, en mémoire GPU,
//     lus au moment de Run à l'adresse donnée à SetImage (ils doivent donc encore vivre).
// Sortie : R10 = R8 × 4 × contraste / 100, G10 = G8 × 4 × saturation / 100,
// B10 = B8 × 4 × gris moyen / 50, chacune bornée à 1023, alpha 3.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "vendor_hdr/maxine_abi.h"

using namespace ajp::vendor_hdr::maxine;

namespace {

constexpr NvCV_Status kErrGeneral = -1;
constexpr NvCV_Status kErrSelector = -3;
constexpr NvCV_Status kErrPixelFormat = -4;
constexpr NvCV_Status kErrInitialization = -5;
constexpr NvCV_Status kErrParameter = -6;

}  // namespace

struct ajp::vendor_hdr::maxine::NvVFX_Object {
  std::string model_directory;
  bool stream_set = false;
  bool legacy_stream = false;
  unsigned contrast = 100;
  unsigned saturation = 100;
  unsigned middle_gray = 50;
  unsigned luminance = 1000;
  NvCVImage* source = nullptr;
  NvCVImage* target = nullptr;
  int loads = 0;
};

namespace {

std::uint32_t scale(std::uint32_t value8, unsigned numerator, unsigned denominator) {
  return std::min<std::uint32_t>(1023, value8 * 4 * numerator / denominator);
}

bool same(const char* selector, const char* expected) {
  return std::string_view(selector) == expected;
}

}  // namespace

extern "C" {

NvCV_Status NvVFX_CreateEffect(const char* code, NvVFX_Handle* effect) {
  if (code == nullptr || effect == nullptr) return kErrParameter;
  if (!same(code, kFxTrueHdr)) return kErrSelector;
  *effect = new NvVFX_Object();
  return kSuccess;
}

void NvVFX_DestroyEffect(NvVFX_Handle effect) {
  delete effect;
}

NvCV_Status NvVFX_SetU32(NvVFX_Handle effect, const char* parameter, unsigned int value) {
  if (same(parameter, kContrast)) {
    effect->contrast = value;
  } else if (same(parameter, kSaturation)) {
    effect->saturation = value;
  } else if (same(parameter, kMiddleGray)) {
    effect->middle_gray = value;
  } else if (same(parameter, kLuminance)) {
    effect->luminance = value;
  } else {
    return kErrSelector;
  }
  return kSuccess;
}

// TrueHDR 1.3 ne lit pas de dossier de modèles : comme le vrai, il répond UNIMPLEMENTED.
NvCV_Status NvVFX_SetString(NvVFX_Handle effect, const char* parameter, const char* value) {
  if (!same(parameter, kModelDirectory)) return kErrSelector;
  effect->model_directory = value;
  return kErrUnimplemented;
}

NvCV_Status NvVFX_SetImage(NvVFX_Handle effect, const char* parameter, NvCVImage* image) {
  if (same(parameter, kInputImage)) {
    effect->source = image;
  } else if (same(parameter, kOutputImage)) {
    effect->target = image;
  } else {
    return kErrSelector;
  }
  return kSuccess;
}

NvCV_Status NvVFX_SetCudaStream(NvVFX_Handle effect, const char* parameter, CUstream stream) {
  if (!same(parameter, kCudaStream)) return kErrSelector;
  effect->stream_set = true;
  effect->legacy_stream = stream == nullptr;
  return kSuccess;
}

NvCV_Status NvVFX_Load(NvVFX_Handle effect) {
  if (++effect->loads > 1) return kErrInitialization;  // le chargeur promet un seul chargement
  return kSuccess;
}

NvCV_Status NvVFX_Run(NvVFX_Handle effect, int async) {
  if (effect->loads != 1 || async != 0 || !effect->stream_set || !effect->legacy_stream)
    return kErrInitialization;
  const NvCVImage* in = effect->source;
  const NvCVImage* out = effect->target;
  if (in == nullptr || out == nullptr) return kErrParameter;
  if (in->pixel_format != kRgba || in->component_type != kU8 || out->pixel_format != kRgb10A2 ||
      out->component_type != kP32 || in->gpu_mem != kGpu || out->gpu_mem != kGpu) {
    return kErrPixelFormat;
  }
  if (in->width != out->width || in->height != out->height) return kErrParameter;
  const auto* source = static_cast<const unsigned char*>(in->pixels);
  auto* target = static_cast<unsigned char*>(out->pixels);
  for (unsigned y = 0; y < in->height; ++y) {
    for (unsigned x = 0; x < in->width; ++x) {
      const unsigned char* rgba = source + static_cast<std::ptrdiff_t>(y) * in->pitch + x * 4;
      const std::uint32_t packed = scale(rgba[0], effect->contrast, 100) |
                                   (scale(rgba[1], effect->saturation, 100) << 10U) |
                                   (scale(rgba[2], effect->middle_gray, 50) << 20U) | (3U << 30U);
      std::memcpy(target + static_cast<std::ptrdiff_t>(y) * out->pitch + x * 4, &packed, sizeof packed);
    }
  }
  return kSuccess;
}

NvCV_Status NvCVImage_Init(NvCVImage* image, unsigned width, unsigned height, int pitch, void* pixels,
                           int format, int type, unsigned layout, unsigned memory) {
  const bool rgba8 = format == kRgba && type == kU8;
  const bool packed10 = format == kRgb10A2 && type == kP32;
  if (!rgba8 && !packed10) return kErrPixelFormat;
  *image = NvCVImage{};
  image->width = width;
  image->height = height;
  image->pitch = pitch;
  image->pixel_format = format;
  image->component_type = type;
  image->pixel_bytes = 4;
  image->component_bytes = rgba8 ? 1 : 4;
  image->num_components = rgba8 ? 4 : 1;
  image->planar = layout == kInterleaved ? 0 : 1;
  image->gpu_mem = static_cast<unsigned char>(memory);
  image->pixels = pixels;
  image->buffer_bytes = static_cast<unsigned long long>(pitch) * height;
  return kSuccess;
}

const char* NvCV_GetErrorStringFromCode(NvCV_Status status) {
  switch (status) {
    case kSuccess:
      return "succès";
    case kErrGeneral:
      return "erreur générale (faux SDK)";
    case kErrUnimplemented:
      return "non implémenté (faux SDK)";
    case kErrSelector:
      return "sélecteur inconnu (faux SDK)";
    case kErrPixelFormat:
      return "format de pixel refusé (faux SDK)";
    case kErrInitialization:
      return "effet mal initialisé (faux SDK)";
    default:
      return "paramètre refusé (faux SDK)";
  }
}

}  // extern "C"
