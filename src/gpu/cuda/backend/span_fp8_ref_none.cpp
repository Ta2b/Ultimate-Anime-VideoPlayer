// SpanFp8Reference sans son noyau (span_fp8_ref.cu) : un backend construit sans CUTLASS, ou sous Windows, où
// nvcc 13.4 s'arrête sous MSVC sur span_fp8_ref.cu (« Internal Compiler Error (codegen): unexpected
// expression with aggregate type! »). La référence ne sert qu'à la dérivation du SPAN FP8 : ici, la
// dérivation est refusée et la chaîne garde TensorRT ; les noyaux du SPAN (span_fp8_net*.cu) restent
// construits.
#include "gpu/cuda/backend/span_fp8_ref.h"

namespace ajp::gpu::cuda {

struct SpanFp8Reference::Impl {};

SpanFp8Reference::SpanFp8Reference(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SpanFp8Reference::~SpanFp8Reference() = default;

std::unique_ptr<SpanFp8Reference> SpanFp8Reference::create(const SpanFp8Weights& /*weights*/, int /*height*/,
                                                           int /*width*/, SpanFp8Status& status,
                                                           std::string& error) {
  status = SpanFp8Status::Failed;
  error = "ce backend est construit sans la référence de la dérivation";
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
