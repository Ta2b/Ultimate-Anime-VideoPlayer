// Chemin CUDA du NVIDIA RTX Video SDK 1.1 (Windows), déclaré ici : le SDK n'est pas dans le
// dépôt. Origines :
//   - noms et signatures : rtx_video_api.h du SDK (rtx_video_sdk_v1.1.0.zip, « Native CUDA
//     support », developer.nvidia.com/rtx-video-sdk/getting-started), tels que retenus par le
//     plan ; non revus sur le SDK, à confirmer à M3 sur le PC de référence ;
//   - sémantique des réglages et formats : pont aji_rtx.dll appelé par
//     animejanai-inference-rtx@118c272 src/aji_trt.cpp:2197-2254 et 2375-2425 (création avec le
//     contexte et le flux 0, entrée RGB 10 bits tassée, R dans les bits de poids faible, sortie
//     HDR10 au même format, réglages contraste, saturation, gris moyen et pic).
// L'API n'a pas de poignée : une seule instance existe à la fois dans le processus.
#pragma once

#include <cstdint>

namespace ajp::vendor_hdr::rtx_video {

struct CUctx_st;
using CUcontext = CUctx_st*;
struct CUstream_st;
using CUstream = CUstream_st*;
using CUdeviceptr = unsigned long long;

struct ApiRect {  // API_RECT : bornes exclusives à droite et en bas
  int left;
  int top;
  int right;
  int bottom;
};

struct VsrSetting {  // API_VSR_Setting
  int quality_level;
};

struct ThdrSetting {  // API_THDR_Setting
  int contrast;
  int saturation;
  int middle_gray;
  int max_luminance;
};

using CudaCreateFn = bool(CUcontext* context, CUstream stream, bool thdr, bool vsr);
using CudaEvaluateFn = bool(CUdeviceptr input, CUdeviceptr output, ApiRect input_rect, ApiRect output_rect,
                            VsrSetting* vsr, ThdrSetting* thdr);
using CudaShutdownFn = void();

inline constexpr const char* kCreate = "rtx_video_api_cuda_create";
inline constexpr const char* kEvaluate = "rtx_video_api_cuda_evaluate";
inline constexpr const char* kShutdown = "rtx_video_api_cuda_shutdown";

}  // namespace ajp::vendor_hdr::rtx_video
