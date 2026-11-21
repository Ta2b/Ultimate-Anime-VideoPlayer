// Backend construit sans noyaux SPAN personnalisés. La chaîne choisit les autres voies TensorRT.
#include "gpu/cuda/backend/span_fp8_net.h"

namespace kirari::gpu::cuda {

struct SpanFp8Net::Impl {};

SpanFp8Net::SpanFp8Net(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SpanFp8Net::~SpanFp8Net() = default;

bool SpanFp8Net::supported_device(int /*major*/, int /*minor*/) {
  return false;
}

bool SpanFp8Net::supported_fp16_device(int /*major*/, int /*minor*/) {
  return false;
}

std::unique_ptr<SpanFp8Net> SpanFp8Net::create(const SpanFp8Weights& /*weights*/, int /*height*/,
                                               int /*width*/, SpanFp8Status& status, std::string& error) {
  status = SpanFp8Status::Failed;
  error = "ce backend est construit sans noyaux SPAN natifs";
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

bool SpanFp8Net::launch_body(cudaStream_t /*stream*/, const void* /*input*/, std::string& error) {
  error = "noyaux absents";
  return false;
}

bool SpanFp8Net::launch_tail(cudaStream_t /*stream*/, void* /*output*/, const SpanFp8Ring& /*ring*/,
                             std::string& error) {
  error = "noyaux absents";
  return false;
}

SpanFp8AccChange SpanFp8Net::poll_accumulator() {
  return SpanFp8AccChange::None;
}

int SpanFp8Net::overflow_frames() const {
  return 0;
}

kernels::SpanTileGrid SpanFp8Net::tile_grid() const {
  return {};
}

void SpanFp8Net::pack_tile_layers(std::span<const std::uint8_t> /*core*/, std::uint32_t* /*body*/) const {}

void SpanFp8Net::set_tile_masks(const std::uint32_t* /*region*/, const std::uint32_t* /*core*/,
                                bool /*layered*/) {}

bool SpanFp8Net::shift_output(cudaStream_t /*stream*/, const void* /*from*/, void* /*to*/, int /*dx*/,
                              int /*dy*/, std::string& error) {
  error = "noyaux absents";
  return false;
}

}  // namespace kirari::gpu::cuda
