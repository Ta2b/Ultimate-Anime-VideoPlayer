// SpanFp8Net d'un backend construit sans CUTLASS (AJP_WITH_SPAN_FP8 = OFF : plateforme autre que Linux, ou
// third_party/cutlass absent) : l'appareil n'est jamais pris en charge, le backend ne l'annonce pas
// (BackendCaps::span_fp8_cutlass) et la chaîne reste sur TensorRT.
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/span_fp8_ref.h"

namespace ajp::gpu::cuda {

struct SpanFp8Net::Impl {};

SpanFp8Net::SpanFp8Net(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SpanFp8Net::~SpanFp8Net() = default;

bool SpanFp8Net::supported_device(int /*major*/, int /*minor*/) {
  return false;
}

std::unique_ptr<SpanFp8Net> SpanFp8Net::create(const SpanFp8Weights& /*weights*/, int /*height*/,
                                               int /*width*/, SpanFp8Status& status, std::string& error) {
  status = SpanFp8Status::Failed;
  error = "ce backend est construit sans les noyaux CUTLASS";
  return nullptr;
}

std::int64_t SpanFp8Net::workspace_bytes() const {
  return 0;
}

void SpanFp8Net::bind(void* /*workspace*/) {}

bool SpanFp8Net::launch(cudaStream_t /*stream*/, const void* /*input*/, void* /*output*/,
                        std::string& error) {
  error = "noyaux absents";
  return false;
}

struct SpanFp8Reference::Impl {};

SpanFp8Reference::SpanFp8Reference(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SpanFp8Reference::~SpanFp8Reference() = default;

std::unique_ptr<SpanFp8Reference> SpanFp8Reference::create(const SpanFp8Weights& /*weights*/, int /*height*/,
                                                           int /*width*/, SpanFp8Status& status,
                                                           std::string& error) {
  status = SpanFp8Status::Failed;
  error = "ce backend est construit sans les noyaux CUTLASS";
  return nullptr;
}

bool SpanFp8Reference::run(cudaStream_t /*stream*/, const void* /*input*/, void* /*output*/, bool /*track*/,
                           std::string& error) {
  error = "noyaux absents";
  return false;
}

bool SpanFp8Reference::peaks(float (& /*out*/)[kPeaks], std::string& error) {
  error = "noyaux absents";
  return false;
}

}  // namespace ajp::gpu::cuda
