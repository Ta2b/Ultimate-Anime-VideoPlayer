// Le réseau SPAN FP8 d'une largeur donnée (64 canaux, ou 48 sans rembourrage), derrière SpanFp8Net. Les
// largeurs sont des instanciations d'un même gabarit (span_fp8_body.cuh) réparties sur deux unités de
// compilation (span_fp8_net.cu pour 64, span_fp8_net48.cu pour 48) : chacune reste sous la taille visée et se
// compile en parallèle. Comme span_fp8_net.h, sans std::expected ni <format> (nvcc).
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "gpu/cuda/backend/span_fp8_net.h"
#include "kernels/spec/span_tiles.h"

namespace kirari::gpu::cuda {

class SpanFp8Body {
 public:
  SpanFp8Body() = default;
  virtual ~SpanFp8Body() = default;
  SpanFp8Body(const SpanFp8Body&) = delete;
  SpanFp8Body& operator=(const SpanFp8Body&) = delete;

  // Envoie les poids (déjà réduits à la largeur du corps) ; faux avec status et error si un envoi échoue.
  virtual bool upload(const SpanFp8Weights& weights, SpanFp8Status& status, std::string& error) = 0;
  virtual std::int64_t workspace_bytes() const = 0;
  virtual void bind(void* workspace) = 0;
  virtual bool launch_body(cudaStream_t stream, const void* input, std::string& error) = 0;
  virtual bool launch_tail(cudaStream_t stream, void* output, const SpanFp8Ring& ring,
                           std::string& error) = 0;
  virtual SpanFp8AccChange poll_accumulator() = 0;  // SpanFp8Net::poll_accumulator
  virtual int overflow_frames() const = 0;
  // Tuiles qui bougent (kernels/spec/span_tiles.h). Un corps qui ne sait calculer que l'image entière
  // (CUTLASS) garde ces défauts.
  virtual kernels::SpanTileGrid tile_grid() const { return {}; }
  // Masques d'appareil (kSpanTileMaskWords mots chacun) des lancements suivants : region pour le corps, core
  // pour la queue ; nuls : l'image entière. Sans appel CUDA.
  virtual void pack_tile_layers(std::span<const std::uint8_t> /*core*/, std::uint32_t* /*body*/) const {}
  virtual void set_tile_masks(const std::uint32_t* /*region*/, const std::uint32_t* /*core*/,
                              bool /*layered*/) {}
  // La sortie FP16 NCHW (3 × 4fh × 4fw) from copiée dans to, décalée de (dx, dy) pixels de sortie : to(x, y)
  // = from(x − dx, y − dy) là où ce point est dans l'image, rien ailleurs. from et to distincts.
  virtual bool shift_output(cudaStream_t /*stream*/, const void* /*from*/, void* /*to*/, int /*dx*/,
                            int /*dy*/, std::string& error) {
    error = "ce corps ne décale pas sa sortie";
    return false;
  }
};

// height, width : l'image d'entrée, déjà jugée par span_fp8::size_refusal.
std::unique_ptr<SpanFp8Body> make_span_fp8_body64(int height, int width);
// flat : K mis à plat (kFewChannels, sans queue masquée par position du filtre) ; sinon K par position
// (kOptimized : seules les colonnes de sortie sont réduites à 48). La mesure sur la carte tranche.
std::unique_ptr<SpanFp8Body> make_span_fp8_body48(int height, int width, bool flat);
// Le SPAN FP16 natif (span_native16.cu, noyaux écrits sans CUTLASS) : 64 ou 48 canaux ; sommes en f16, ou
// en f32 (SpanNativeSums).
std::unique_ptr<SpanFp8Body> make_span_fp8_body16(int height, int width, int channels, SpanNativeSums sums);

}  // namespace kirari::gpu::cuda
