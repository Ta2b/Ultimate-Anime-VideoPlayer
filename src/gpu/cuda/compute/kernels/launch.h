// Lancements des noyaux CUDA, implémentés dans les .cu du dossier. Chacun soumet ses noyaux
// sur le flux donné et rend cudaGetLastError() : le plan hôte qui l'appelle en fait un refus
// qui cite le noyau. C++20, sans std::expected : ces déclarations sont partagées avec nvcc.
// Aucune vérification ici : les plans hôtes (gpu/cuda/compute/*.h) valident avant de lancer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "gpu/cuda/compute/kernels/views.h"

namespace ajp::gpu::cuda::dev {

// Le binaire contient du code pour l'appareil courant (sinon cudaErrorNoKernelImageForDevice).
cudaError_t probe_kernel_image();

// Ingestion 4:2:0 : passe horizontale de la chroma brute (codes du conteneur), puis passe
// verticale et matrice. tmp : 2 plans float de padded_width × (padded_height / 2).
cudaError_t ingest_chroma_rows(const YuvSource& src, Weights wx, float* tmp, cudaStream_t stream);
cudaError_t ingest_combine_420(const YuvSource& src, const float* tmp, Weights wy, YuvToRgb m, Tensor dst,
                               cudaStream_t stream);
// Ingestion 4:4:4 : matrice seule.
cudaError_t ingest_combine_444(const YuvSource& src, YuvToRgb m, Tensor dst, cudaStream_t stream);

// Passe horizontale de planes plans d'un tenseur (F16 ou F32) vers tmp float, wx.dst × height.
cudaError_t rows(Tensor src, int planes, Weights wx, float* tmp, cudaStream_t stream);
// Passe verticale de planes plans float (width × wy.src) vers un tenseur width × wy.dst.
cudaError_t columns(const float* tmp, int width, int planes, Weights wy, Tensor dst, cudaStream_t stream);

// Sortie 4:2:0 : Y quantifié dans la fenêtre, Cb et Cr normalisés dans tmp (2 plans float de la
// taille de src) ; puis, après rows(), passe verticale, quantification et rangement.
cudaError_t output_matrix_420(Tensor src, RgbToYuv m, YuvTarget dst, float* tmp, cudaStream_t stream);
cudaError_t output_chroma_store(const float* tmp, int width, Weights wy, RgbToYuv m, YuvTarget dst,
                                cudaStream_t stream);
// Sortie 4:4:4 : matrice et quantification seules.
cudaError_t output_matrix_444(Tensor src, RgbToYuv m, YuvTarget dst, cudaStream_t stream);

// RGBA16 UNORM, alpha opaque : vers une surface (tableau CUDA, image Vulkan importée) ou une
// mémoire linéaire à pas de ligne.
cudaError_t rgba16_to_surface(Tensor src, cudaSurfaceObject_t surface, cudaStream_t stream);
cudaError_t rgba16_to_linear(Tensor src, Plane dst, cudaStream_t stream);

// RIFE : les quatre plans constants (grilles et pas) à partir de first, de width × height.
cudaError_t rife_constants(Tensor first, cudaStream_t stream);
// count éléments à partir de ptr, tous à value (l'instant t de RIFE).
cudaError_t fill(std::uintptr_t ptr, Elem elem, std::size_t count, float value, cudaStream_t stream);
// Copie de la fenêtre (left, top, dst.width, dst.height) des trois plans de src.
cudaError_t crop(Tensor src, int left, int top, Tensor dst, cudaStream_t stream);

// Warp arrière et mélange à l'instant t ; t = 1/2 : le mouvement tel quel.
cudaError_t warp_blend(Tensor a, Tensor b, Motion motion, float t, BlendLaw law, Tensor dst,
                       cudaStream_t stream);
// Plus grand |déplacement| de la fenêtre, en bits de float, par atomicMax dans *result (à zéro).
cudaError_t flow_max(Motion motion, std::uint32_t* result, cudaStream_t stream);

// Mesures d'une paire de lumas : par tuile de 32 × 32, somme des écarts normalisés (double) dans
// partials[tuile] et pire écart moyen par atomicMax dans result->worst (à zéro) ; puis somme
// finale dans un ordre fixe, divisée par area, dans result->score et result->identical.
int tile_count(int width, int height);
cudaError_t pair_tiles(Luma a, Luma b, float norm, std::uint32_t scale, double* partials, PairResult* result,
                       cudaStream_t stream);
cudaError_t pair_finish(const double* partials, int count, double area, PairResult* result,
                        cudaStream_t stream);

// Fonctions de transfert et primaires, pixel par pixel, sur trois plans.
cudaError_t apply_transfer(Tensor src, TransferParams params, Tensor dst, cudaStream_t stream);

}  // namespace ajp::gpu::cuda::dev
