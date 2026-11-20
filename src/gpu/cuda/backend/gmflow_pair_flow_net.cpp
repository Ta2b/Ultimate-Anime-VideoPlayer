#include "gpu/cuda/backend/gmflow_pair_flow_net.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gmflow_backbone_net.h"
#include "gpu/cuda/backend/gmflow_convex_net.h"
#include "gpu/cuda/backend/gmflow_global_kernels.h"
#include "gpu/cuda/backend/gmflow_local_kernels.h"
#include "gpu/cuda/backend/gmflow_propagation_net.h"
#include "gpu/cuda/backend/gmflow_scale_kernels.h"
#include "gpu/cuda/backend/gmflow_transformer_net.h"
#include "gpu/cuda/backend/gmfss_pair_flow_glue.h"
#include "prepare/gmflow_attention_native.h"
#include "prepare/gmflow_mlp.h"
#include "prepare/gmflow_native.h"
#include "prepare/gmflow_onnx_view.h"
#include "prepare/gmflow_propagation.h"

namespace kirari::gpu::cuda {
namespace {

namespace onnx = gpu::ir::onnx;

bool follows_half_resize(const prepare::OnnxView& g, std::string_view input, std::string_view from) {
  const auto* resize = g.made_by(input, "Resize");
  if (resize == nullptr || resize->inputs.size() != 3 || g.base(resize->inputs[0]) != from) return false;
  const auto* scales = g.constant(resize->inputs[2]);
  const auto* coordinate = prepare::onnx_attribute(*resize, "coordinate_transformation_mode");
  const auto* mode = prepare::onnx_attribute(*resize, "mode");
  return scales != nullptr && scales->floats == std::vector<float>{1.F, 1.F, .5F, .5F} &&
         coordinate != nullptr && coordinate->s == "half_pixel" && mode != nullptr && mode->s == "linear";
}

std::expected<bool, Refusal> uses_half_flow(std::span<const std::byte> pair) {
  const auto model = onnx::read_onnx(pair);
  if (!model) return refuse(RefusalCode::ModelUnreadable, model.error().detail);
  const prepare::OnnxView g(*model);
  const auto* a = g.node("/flownet/Sub", "Sub");
  const auto* b = g.node("/flownet/Sub_1", "Sub");
  if (a != nullptr && b != nullptr && a->inputs.size() == 2 && b->inputs.size() == 2) {
    if (g.base(a->inputs[0]) == "half0" && g.base(b->inputs[0]) == "half1") return false;
    if (follows_half_resize(g, a->inputs[0], "half0") && follows_half_resize(g, b->inputs[0], "half1"))
      return true;
  }
  return refuse(RefusalCode::FieldMissing, "GMFlow : échelle des entrées du flot inconnue");
}

struct Geometry {
  int height, width;
  bool half_flow;

  int h(int level = 0) const { return height / (half_flow ? 2 : 1) / (1 << level); }
  int w(int level = 0) const { return width / (half_flow ? 2 : 1) / (1 << level); }
  std::size_t area(int level = 0) const {
    return static_cast<std::size_t>(h(level)) * static_cast<std::size_t>(w(level));
  }
};

struct Layout {
  std::size_t pictures, features8, features4, input, result, flow0, flow1, pos8, pos4, scratch, size = 0;
};

constexpr std::size_t kTokenBytes = prepare::kGmflowModel * sizeof(std::uint16_t);

Layout plan(Geometry g, std::size_t pos8, std::size_t pos4, std::size_t scratch) {
  Layout p{};
  const auto region = [&](std::size_t bytes) {
    const auto at = p.size;
    p.size += (bytes + 255U) / 256U * 256U;
    return at;
  };
  // Les images [2][3] FP16 ne sont plus lues quand l'agrandissement écrit le flot [2][2] FP32.
  p.pictures = region(g.area() * 4 * sizeof(float));
  p.features8 = region(2 * g.area(3) * kTokenBytes);
  p.features4 = region(2 * g.area(2) * kTokenBytes);
  p.input = region(4 * g.area(2) * kTokenBytes);
  p.result = region(4 * g.area(2) * kTokenBytes);
  p.flow0 = region(4 * g.area(2) * sizeof(float));
  p.flow1 = region(4 * g.area(2) * sizeof(float));
  p.pos8 = region(pos8);
  p.pos4 = region(pos4);
  p.scratch = region(scratch);
  return p;
}

class Net final : public GmflowPairFlowNet {
 public:
  explicit Net(Geometry g) : g_(g) {}
  ~Net() override {
    if (buffer_ != nullptr) static_cast<void>(cudaFree(buffer_));
  }

  std::expected<void, Refusal> build(std::span<const std::byte> pair) {
    auto backbone = prepare::gmflow_backbone(pair, g_.half_flow ? prepare::GmflowStemInput::kAlreadyNormalized
                                                                : prepare::GmflowStemInput::kNormalizeF16);
    if (!backbone) return std::unexpected(std::move(backbone.error()));
    std::ranges::copy(backbone->pad, normalization_.mean);
    std::ranges::copy(backbone->deviation, normalization_.deviation);
    auto attention = prepare::gmflow_attention(pair);
    if (!attention) return std::unexpected(std::move(attention.error()));
    auto mlp = prepare::gmflow_mlp(pair);
    if (!mlp) return std::unexpected(std::move(mlp.error()));
    auto proj = prepare::gmflow_propagation(pair);
    if (!proj) return std::unexpected(std::move(proj.error()));
    auto up = prepare::gmflow_upsampler(pair);
    if (!up) return std::unexpected(std::move(up.error()));
    std::string error;
    backbone_ = GmflowBackboneNet::create(*backbone, g_.h(), g_.w(), error);
    if (!backbone_) return build_error(error);
    eighth_ = GmflowTransformerNet::create(*attention, *mlp, 0, g_.h(3), g_.w(3), GmflowMlpBlock::k64, error);
    if (!eighth_) return build_error(error);
    quarter_ =
        GmflowTransformerNet::create(*attention, *mlp, 1, g_.h(2), g_.w(2), GmflowMlpBlock::k64, error);
    if (!quarter_) return build_error(error);
    projection_ = GmflowPropagationNet::create(*proj, error);
    if (!projection_) return build_error(error);
    convex_ = GmflowConvexNet::create(*up, g_.h(2), g_.w(2), error);
    if (!convex_) return build_error(error);
    splits_ = attention->splits;
    layout_ = plan(g_, gmflow_position_bytes(g_.h(3), g_.w(3), splits_[0]),
                   gmflow_position_bytes(g_.h(2), g_.w(2), splits_[1]),
                   std::max({eighth_->scratch_bytes(), quarter_->scratch_bytes(),
                             static_cast<std::size_t>(convex_->workspace_bytes())}));
    if (auto ok = cuda_check(cudaMalloc(&buffer_, layout_.size), "GMFlow : tampons"); !ok) return ok;
    if (auto ok = cuda_check(launch_gmflow_position(nullptr, g_.h(3), g_.w(3), splits_[0], at(layout_.pos8)),
                             "GMFlow : positions au huitième");
        !ok)
      return ok;
    if (auto ok = cuda_check(launch_gmflow_position(nullptr, g_.h(2), g_.w(2), splits_[1], at(layout_.pos4)),
                             "GMFlow : positions au quart");
        !ok)
      return ok;
    return cuda_check(cudaStreamSynchronize(nullptr), "GMFlow : envoi des positions");
  }

  cudaError_t run(cudaStream_t stream, const void* frames, const GmflowPairOutput& out,
                  bool follows) const override {
    const bool reuse_left = std::exchange(features_ready_, false) && follows && features_stream_ == stream;
    void* pictures = at(layout_.pictures);
    if (const auto e =
            launch_gmfss_pair_flow_input(stream, frames, g_.height, g_.width, g_.half_flow, pictures,
                                         out.half, g_.half_flow ? &normalization_ : nullptr);
        e != cudaSuccess)
      return e;
    GmflowFeatures features{
        .quarter = at(layout_.features4), .eighth = at(layout_.features8), .quarter_swapped = quarter_swapped_};
    if (const auto e = backbone_->run(stream, pictures, features, reuse_left); e != cudaSuccess) return e;
    if (const auto e = match_eighth(stream, features); e != cudaSuccess) return e;
    if (const auto e = match_quarter(stream, features); e != cudaSuccess) return e;
    auto* const planar = static_cast<float*>(pictures);
    if (const auto e = convex_->run(stream, at(layout_.result), static_cast<const float*>(at(layout_.flow1)),
                                    at(layout_.scratch), planar);
        e != cudaSuccess)
      return e;
    const auto status = launch_gmfss_pair_flow_output(stream, planar, g_.height, g_.width, g_.half_flow,
                                                      {out.half[0], out.half[1]}, out.flow, out.metric);
    features_stream_ = stream;
    quarter_swapped_ = features.quarter_swapped;
    features_ready_ = status == cudaSuccess;
    return status;
  }

 private:
  cudaError_t match_eighth(cudaStream_t stream, const GmflowFeatures& features) const {
    const int h = g_.h(3), w = g_.w(3);
    void* result = at(layout_.result);
    const void* input = features.eighth_from_quarter ? features.quarter : features.eighth;
    if (const auto e =
            launch_gmflow_eighth_input(stream, input, at(layout_.pos8), h, w, splits_[0], features.eighth,
                                       features.eighth_from_quarter, features.quarter_swapped);
        e != cudaSuccess)
      return e;
    if (const auto e = eighth_->run(stream, features.eighth, result, at(layout_.scratch)); e != cudaSuccess)
      return e;
    if (const auto e = launch_gmflow_global_match(stream, result, h, w, at(layout_.flow0)); e != cudaSuccess)
      return e;
    const auto qk = projections(3);
    if (const auto e =
            projection_->project(stream, result, static_cast<long>(g_.area(3)), true, qk[0], qk[1]);
        e != cudaSuccess)
      return e;
    return launch_gmflow_global_propagate(stream, qk[0], qk[1], at(layout_.flow0), h, w, at(layout_.flow1));
  }

  cudaError_t match_quarter(cudaStream_t stream, const GmflowFeatures& features) const {
    const int h = g_.h(2), w = g_.w(2);
    void* input = at(layout_.input);
    void* result = at(layout_.result);
    void* flow = at(layout_.flow0);
    if (const auto e = launch_gmflow_quarter_input(stream, features.quarter, at(layout_.flow1),
                                                   at(layout_.pos4), h, w, splits_[1], input, flow,
                                                   features.quarter_swapped);
        e != cudaSuccess)
      return e;
    if (const auto e = quarter_->run(stream, input, result, at(layout_.scratch)); e != cudaSuccess) return e;
    if (const auto e = launch_gmflow_local_match(stream, result, flow, h, w, flow); e != cudaSuccess)
      return e;
    const auto qk = projections(2);
    if (const auto e =
            projection_->project(stream, result, static_cast<long>(g_.area(2)), false, qk[0], qk[1]);
        e != cudaSuccess)
      return e;
    return launch_gmflow_local_propagate(stream, qk[0], qk[1], flow, h, w, at(layout_.flow1));
  }

  // Après le transformeur, la zone de son entrée est libre et assez grande pour les deux projections.
  std::array<void*, 2> projections(int level) const {
    return {at(layout_.input), at(layout_.input + 2 * g_.area(level) * kTokenBytes)};
  }

  void* at(std::size_t bytes) const { return static_cast<std::byte*>(buffer_) + bytes; }

  static std::unexpected<Refusal> build_error(const std::string& error) {
    if (const auto lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "GMFlow natif"));
    return refuse(RefusalCode::FormatUnsupported, "GMFlow natif : " + error);
  }

  Geometry g_;
  GmfssImageNormalization normalization_;
  std::array<int, 2> splits_{};
  Layout layout_{};
  void* buffer_ = nullptr;
  mutable cudaStream_t features_stream_ = nullptr;
  mutable bool features_ready_ = false;
  mutable bool quarter_swapped_ = false;
  std::unique_ptr<GmflowBackboneNet> backbone_;
  std::unique_ptr<GmflowTransformerNet> eighth_, quarter_;
  std::unique_ptr<GmflowPropagationNet> projection_;
  std::unique_ptr<GmflowConvexNet> convex_;
};

}  // namespace

std::expected<std::unique_ptr<GmflowPairFlowNet>, Refusal> GmflowPairFlowNet::create(
    std::span<const std::byte> pair, int height, int width) {
  const auto half = uses_half_flow(pair);
  if (!half) return std::unexpected(half.error());
  const Geometry g{height, width, *half};
  if (height < 32 || width < 32 || height > 2176 || width > 3840 || height % (g.half_flow ? 64 : 32) != 0 ||
      width % (g.half_flow ? 64 : 32) != 0)
    return refuse(RefusalCode::ShapeInvalid, "GMFlow natif : taille du réseau non multiple de 32");
  auto net = std::make_unique<Net>(g);
  if (auto ok = net->build(pair); !ok) return std::unexpected(std::move(ok.error()));
  return net;
}

}  // namespace kirari::gpu::cuda
