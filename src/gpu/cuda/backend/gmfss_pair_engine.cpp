#include "gpu/cuda/backend/gmfss_pair_engine.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <string_view>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/fused_graphs.h"

namespace kirari::gpu::cuda {
namespace {

// Les sorties de la paire exportée par tools/gmfss/export_gmfss.py, dans l'ordre de GmfssPairEngine::Output.
constexpr std::array<std::string_view, GmfssPairEngine::kOutputs> kOutputNames{
    "half0",     "half1",     "feature0_1", "source0_2", "source0_3", "feature1_1", "source1_2",
    "source1_3", "motion0_1", "motion0_2",  "motion0_3", "motion1_1", "motion1_2",  "motion1_3"};

// Les plans des caractéristiques d'une image à chaque niveau (feature_1, source_2, source_3).
constexpr std::array<std::int64_t, prepare::kGmfssFeatureLevels> kFeaturePlanes{64, 128, 192};
constexpr std::int64_t kFlowPlanes = 2;    // le flot, les deux premiers plans d'un mouvement
constexpr std::int64_t kMotionPlanes = 3;  // flot et métrique
constexpr std::int64_t kImagePlanes = 3;

Shape planes_at(std::int64_t planes, int height, int width, int level) {
  return {.dims = {1, planes, height >> level, width >> level}};
}

// La forme de la sortie o de la paire dont la fusion est de height × width.
Shape output_shape_of(std::size_t o, int height, int width) {
  if (o < GmfssPairEngine::kFeature0) return planes_at(kImagePlanes, height, width, 0);
  if (o < GmfssPairEngine::kMotion) {
    const auto level = static_cast<int>((o - GmfssPairEngine::kFeature0) % 3);
    return planes_at(kFeaturePlanes[static_cast<std::size_t>(level)], height, width, level);
  }
  return planes_at(kMotionPlanes, height, width, static_cast<int>((o - GmfssPairEngine::kMotion) % 3));
}

std::expected<std::size_t, Refusal> find_name(std::span<const std::string> names, std::string_view name,
                                              std::string_view engine) {
  const auto found = std::ranges::find(names, name);
  if (found == names.end())
    return refuse(RefusalCode::GraphInvalid,
                  std::format("GMFSS natif : sortie « {} » absente du moteur {}", name, engine));
  return static_cast<std::size_t>(found - names.begin());
}

bool planar(const TrtEngine& engine) {
  for (std::size_t i = 0; i < engine.input_names().size(); ++i) {
    if (engine.input_pitch(i) != 0) return false;
  }
  for (std::size_t i = 0; i < engine.output_names().size(); ++i) {
    if (engine.output_pitch(i) != 0) return false;
  }
  return true;
}

}  // namespace

GmfssPairEngine::~GmfssPairEngine() {
  if (metric_in_ != nullptr) static_cast<void>(cudaFree(metric_in_));
}

std::expected<std::unique_ptr<GmfssPairEngine>, Refusal> GmfssPairEngine::load(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const GmfssFusedKey& key,
    const prepare::GmfssPairCut& cut, const std::shared_ptr<SharedActivations>& shared) {
  if (key.engines.size() != 2) return refuse(RefusalCode::GraphInvalid, "GMFSS natif : clé invalide");
  std::unique_ptr<GmfssPairEngine> engine(new GmfssPairEngine());
  const int h = key.height;
  const int w = key.width;
  engine->height_ = h;
  engine->width_ = w;
  auto full = TrtEngine::load(runtime, cache_dir / core::path_from_utf8(key.engines[0]), shared);
  if (!full) return std::unexpected(std::move(full.error()));
  auto flow = TrtEngine::load(runtime, cache_dir / core::path_from_utf8(key.engines[1]), shared);
  if (!flow) return std::unexpected(std::move(flow.error()));
  engine->full_ = std::move(*full);
  engine->flow_ = std::move(*flow);
  const TrtEngine& f = *engine->full_;
  const TrtEngine& r = *engine->flow_;
  const auto bad = [](std::string_view what) {
    return refuse(RefusalCode::ShapeInvalid, std::format("GMFSS natif : {} de forme inattendue", what));
  };
  const Shape frames{.dims = {1, 2 * kImagePlanes, 2 * h, 2 * w}};
  if (f.input_names().size() != 1 || r.input_names().size() != 1 || f.output_names().size() != kOutputs ||
      r.output_names().size() != kFlowOutputs || !planar(f) || !planar(r) || !(f.input_shape(0) == frames) ||
      !(r.input_shape(0) == frames))
    return bad("paire ou flot");
  for (std::size_t o = 0; o < kOutputs; ++o) {
    auto at = find_name(f.output_names(), kOutputNames[o], "de la paire");
    if (!at) return std::unexpected(std::move(at.error()));
    engine->at_[o] = *at;
    if (!(f.output_shape(*at) == output_shape_of(o, h, w))) return bad(kOutputNames[o]);
  }
  // Le flot : l'entrée du MetricNet, les six flots (rangés k · 3 + l - 1), half0, half1.
  for (std::size_t j = 0; j < kFlowOutputs; ++j) {
    const bool flow_j = j >= 1 && j <= prepare::kGmfssPairFlows;
    const std::string_view name = j == 0   ? std::string_view(cut.metric_input)
                                  : flow_j ? std::string_view(cut.flows[j - 1])
                                           : kOutputNames[j - 1 - prepare::kGmfssPairFlows];
    auto at = find_name(r.output_names(), name, "du flot");
    if (!at) return std::unexpected(std::move(at.error()));
    engine->flow_at_[j] = *at;
    const Shape want = j == 0   ? planes_at(prepare::kGmfssMetricPlanes, h, w, 0)
                       : flow_j ? planes_at(kFlowPlanes, h, w, static_cast<int>((j - 1) % 3))
                                : planes_at(kImagePlanes, h, w, 0);
    if (!(r.output_shape(*at) == want)) return bad(std::format("flot ({})", name));
  }
  const auto bytes = static_cast<std::size_t>(prepare::kGmfssMetricPlanes) * static_cast<std::size_t>(h) *
                     static_cast<std::size_t>(w) * 2;
  if (auto ok = cuda_check(cudaMalloc(&engine->metric_in_, bytes), "GMFSS natif : entrée du MetricNet");
      !ok) {
    engine->metric_in_ = nullptr;
    return std::unexpected(std::move(ok.error()));
  }
  return engine;
}

std::expected<void, Refusal> GmfssPairEngine::enqueue(cudaStream_t stream,
                                                      const std::shared_ptr<SharedActivations>& shared,
                                                      std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
  if (inputs.size() != 1 || outputs.size() != kOutputs)
    return refuse(RefusalCode::ShapeInvalid, "GMFSS natif : une entrée et quatorze sorties attendues");
  if (!native_on_) return full_->enqueue(stream, shared, inputs, outputs);
  return enqueue_native(stream, shared, inputs[0], outputs);
}

std::expected<void, Refusal> GmfssPairEngine::enqueue_native(cudaStream_t stream,
                                                             const std::shared_ptr<SharedActivations>& shared,
                                                             const Tensor& frames,
                                                             std::span<const Tensor> outputs) {
  // Les noyaux lisent et écrivent les plans sans autre garde : chaque tenseur à la forme de la paire entière.
  const auto matches = [](const Tensor& t, const Shape& shape) {
    return t.type == TensorType::F16 && t.shape == shape && t.handle != 0;
  };
  bool good = matches(frames, full_->input_shape(0));
  for (std::size_t i = 0; i < outputs.size(); ++i) good = good && matches(outputs[i], full_->output_shape(i));
  if (!good) return refuse(RefusalCode::ShapeInvalid, "GMFSS natif : forme d'entrée ou de sortie");
  const auto out = [&](std::size_t o) -> const Tensor& { return outputs[at_[o]]; };
  // Le flot écrit l'entrée du MetricNet, les deux premiers plans de chaque mouvement, half0 et half1.
  std::array<Tensor, kFlowOutputs> flow_out;
  flow_out[flow_at_[0]] = fused_tensor(reinterpret_cast<std::uintptr_t>(metric_in_), 0,
                                       prepare::kGmfssMetricPlanes, height_, width_);
  for (std::size_t i = 0; i < prepare::kGmfssPairFlows; ++i) {
    const Tensor& motion = out(kMotion + i);
    const auto level = static_cast<int>(i % 3);
    flow_out[flow_at_[1 + i]] = Tensor{.type = TensorType::F16,
                                       .shape = planes_at(kFlowPlanes, height_, width_, level),
                                       .handle = motion.handle,
                                       .offset = motion.offset};
  }
  flow_out[flow_at_[kFlowOutputs - 2]] = out(kHalf0);
  flow_out[flow_at_[kFlowOutputs - 1]] = out(kHalf1);
  const Tensor flow_in[]{frames};
  if (auto ok = flow_->enqueue(stream, shared, flow_in, flow_out); !ok) return ok;
  // Le FeatureNet de chaque image (les trois premiers plans de frames, puis les trois suivants), puis le
  // MetricNet, qui écrit le troisième plan des mouvements.
  const auto* images = static_cast<const std::uint8_t*>(fused_address(frames));
  const auto image_bytes = static_cast<std::size_t>(kImagePlanes) * 4U * static_cast<std::size_t>(height_) *
                           static_cast<std::size_t>(width_) * 2U;
  for (std::size_t side = 0; side < 2; ++side) {
    GmfssPairFeatures levels;
    for (std::size_t j = 0; j < levels.levels.size(); ++j)
      levels.levels[j] = fused_address(out(kFeature0 + 3 * side + j));
    if (auto ok = cuda_check(native_->features(stream, images + side * image_bytes, levels),
                             "GMFSS natif : FeatureNet");
        !ok)
      return ok;
  }
  GmfssPairMotion motion;
  for (std::size_t i = 0; i < motion.motion.size(); ++i) motion.motion[i] = fused_address(out(kMotion + i));
  return cuda_check(native_->metric(stream, metric_in_, motion), "GMFSS natif : MetricNet");
}

}  // namespace kirari::gpu::cuda
