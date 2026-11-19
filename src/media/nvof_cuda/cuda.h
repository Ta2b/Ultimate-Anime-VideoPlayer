// Ce que nvOpticalFlowCuda.h (third_party/nvidia-optical-flow) attend de <cuda.h> : quatre types du pilote,
// déclarés comme le toolkit CUDA les déclare. media charge le pilote à l'exécution (media/cuda_driver.h) et
// se construit sans le toolkit ; ce dossier n'est sur le chemin des inclusions que de probe_flow_nvof.cpp.
#pragma once

using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;
using CUdeviceptr = unsigned long long;  // NOLINT(google-runtime-int) : le type du pilote
using CUarray = struct CUarray_st*;
