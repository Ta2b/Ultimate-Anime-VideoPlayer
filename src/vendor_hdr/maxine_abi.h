// Prototypes minimaux du SDK NVIDIA Maxine VFX 1.3, déclarés ici : les en-têtes du SDK
// (nvVideoEffects.h, nvCVImage.h, nvCVStatus.h, nvVFXTrueHDR.h) ne sont pas dans le dépôt.
// Vérifiés le 19 octobre 2026 contre VFXSDK_linux 1.3.0.0 et sa fonction nvvfxtruehdr 1.3.0.0
// installés dans /usr/local/VideoFX :
//   - valeurs : nvCVStatus.h:47,49 ; nvCVImage.h:49,61,68,78,92,140 ; nvVideoEffects.h:303-314 ;
//     features/nvvfxtruehdr/include/nvVFXTrueHDR.h:15-20 ;
//   - signatures : nvVideoEffects.h:75-185 et nvCVImage.h:151-166,261 (enum passés en int, de
//     même taille sur x86-64) ; nvCVStatus.h:130 pour NvCV_GetErrorStringFromCode ;
//   - symboles : nm -D, NvVFX_* dans lib/libVideoFX.so, NvCVImage_Init et
//     NvCV_GetErrorStringFromCode dans lib/libNVCVImage.so.
// check_maxine_abi() les compare encore aux en-têtes de chaque SDK importé avant tout
// chargement : une valeur qui changerait refuse le SDK en la nommant, sans l'atteindre.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace ajp::vendor_hdr::maxine {

using NvCV_Status = int;  // enum NvCV_Status, de la taille d'un int
struct NvVFX_Object;
using NvVFX_Handle = NvVFX_Object*;
struct CUstream_st;
using CUstream = CUstream_st*;

// nvCVImage.h, partie C de la structure : 64 octets sur les ABI 64 bits.
struct NvCVImage {
  unsigned int width;
  unsigned int height;
  int pitch;
  int pixel_format;    // NvCVImage_PixelFormat
  int component_type;  // NvCVImage_ComponentType
  unsigned char pixel_bytes;
  unsigned char component_bytes;
  unsigned char num_components;
  unsigned char planar;
  unsigned char gpu_mem;
  unsigned char colorspace;
  std::array<unsigned char, 2> reserved;
  void* pixels;
  void* delete_ptr;
  void (*delete_proc)(void*);
  unsigned long long buffer_bytes;
};
static_assert(sizeof(NvCVImage) == 64 && std::is_standard_layout_v<NvCVImage>);

inline constexpr NvCV_Status kSuccess = 0;            // NVCV_SUCCESS
inline constexpr NvCV_Status kErrUnimplemented = -2;  // NVCV_ERR_UNIMPLEMENTED
inline constexpr int kRgba = 6;                       // NVCV_RGBA
inline constexpr int kRgb10A2 = 13;                   // NVCV_RGB10A2, utilisé avec NVCV_P32
inline constexpr int kU8 = 1;                         // NVCV_U8
inline constexpr int kP32 = 11;                       // NVCV_P32
inline constexpr unsigned kInterleaved = 0;           // NVCV_INTERLEAVED
inline constexpr unsigned kGpu = 1;                   // NVCV_GPU, alias de NVCV_CUDA

// Sélecteurs d'effet et de paramètres.
inline constexpr const char* kFxTrueHdr = "TrueHDR";        // NVVFX_FX_TRUE_HDR
inline constexpr const char* kInputImage = "SrcImage0";     // NVVFX_INPUT_IMAGE
inline constexpr const char* kOutputImage = "DstImage0";    // NVVFX_OUTPUT_IMAGE
inline constexpr const char* kModelDirectory = "ModelDir";  // NVVFX_MODEL_DIRECTORY
inline constexpr const char* kCudaStream = "CudaStream";    // NVVFX_CUDA_STREAM
inline constexpr const char* kContrast = "Contrast";        // NVVFXTRUEHDR_CONTRAST
inline constexpr const char* kSaturation = "Saturation";    // NVVFXTRUEHDR_SATURATION
inline constexpr const char* kMiddleGray = "MiddleGray";    // NVVFXTRUEHDR_MIDDLE_GRAY
inline constexpr const char* kLuminance = "Luminance";      // NVVFXTRUEHDR_LUMINANCE

// Une constante telle que ses en-têtes doivent la définir : nombre ou chaîne.
struct AbiConstant {
  std::string_view name;
  std::variant<long long, std::string_view> value;
};

inline constexpr std::array<AbiConstant, 17> kAbi{{
    {"NVCV_SUCCESS", kSuccess},
    {"NVCV_ERR_UNIMPLEMENTED", kErrUnimplemented},
    {"NVCV_RGBA", kRgba},
    {"NVCV_RGB10A2", kRgb10A2},
    {"NVCV_U8", kU8},
    {"NVCV_P32", kP32},
    {"NVCV_INTERLEAVED", kInterleaved},
    {"NVCV_GPU", kGpu},
    {"NVVFX_FX_TRUE_HDR", kFxTrueHdr},
    {"NVVFX_INPUT_IMAGE", kInputImage},
    {"NVVFX_OUTPUT_IMAGE", kOutputImage},
    {"NVVFX_MODEL_DIRECTORY", kModelDirectory},
    {"NVVFX_CUDA_STREAM", kCudaStream},
    {"NVVFXTRUEHDR_CONTRAST", kContrast},
    {"NVVFXTRUEHDR_SATURATION", kSaturation},
    {"NVVFXTRUEHDR_MIDDLE_GRAY", kMiddleGray},
    {"NVVFXTRUEHDR_LUMINANCE", kLuminance},
}};

// Fonctions résolues par nom ; NvVFX_API vaut la convention C par défaut sous Linux.
using CreateEffectFn = NvCV_Status(const char* code, NvVFX_Handle* effect);
using DestroyEffectFn = void(NvVFX_Handle effect);
using SetU32Fn = NvCV_Status(NvVFX_Handle effect, const char* parameter, unsigned int value);
using SetStringFn = NvCV_Status(NvVFX_Handle effect, const char* parameter, const char* value);
using SetImageFn = NvCV_Status(NvVFX_Handle effect, const char* parameter, NvCVImage* image);
using SetCudaStreamFn = NvCV_Status(NvVFX_Handle effect, const char* parameter, CUstream stream);
using LoadFn = NvCV_Status(NvVFX_Handle effect);
using RunFn = NvCV_Status(NvVFX_Handle effect, int async);
using ImageInitFn = NvCV_Status(NvCVImage* image, unsigned width, unsigned height, int pitch, void* pixels,
                                int format, int type, unsigned layout, unsigned memory);
using ErrorStringFn = const char*(NvCV_Status status);
using GetVersionFn = NvCV_Status(unsigned int* version);

// NvVFX_GetVersion rend (majeur << 24) | (mineur << 16) | (build << 8) (nvVideoEffects.h:71-72) :
// 0x01030000 pour le SDK 1.3.0.0.
constexpr unsigned version_major(unsigned version) {
  return version >> 24U;
}
constexpr unsigned version_minor(unsigned version) {
  return (version >> 16U) & 0xFFU;
}

}  // namespace ajp::vendor_hdr::maxine
