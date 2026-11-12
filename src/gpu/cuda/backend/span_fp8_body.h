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

namespace ajp::gpu::cuda {

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
};

// height, width : l'image d'entrée, déjà jugée par span_fp8::size_refusal.
std::unique_ptr<SpanFp8Body> make_span_fp8_body64(int height, int width);
// flat : K mis à plat (kFewChannels, sans queue masquée par position du filtre) ; sinon K par position
// (kOptimized : seules les colonnes de sortie sont réduites à 48). La mesure sur la carte tranche.
std::unique_ptr<SpanFp8Body> make_span_fp8_body48(int height, int width, bool flat);
// Les blocs en FP16 sans quantification (span_fp8_net16.cu) : 64 canaux, ou 48 avec K à plat.
std::unique_ptr<SpanFp8Body> make_span_fp8_body16(int height, int width, int channels);

}  // namespace ajp::gpu::cuda
