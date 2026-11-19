#include "gpu/cuda/backend/gmfss_fused_engine.h"

#include <algorithm>
#include <format>
#include <string_view>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/fused_graphs.h"

namespace kirari::gpu::cuda {
namespace {

// Les entrées de la fusion exportée par tools/gmfss/export_gmfss.py, dans l'ordre de GmfssFusedEngine::Input.
constexpr std::array<std::string_view, 9> kNames{
    "half0", "half1", "time", "warped0_1", "warped0_2", "warped0_3", "warped1_1", "warped1_2", "warped1_3"};

// Les plans de chaque entrée et sa taille, en fraction de la fusion (0 : 1 × 1, l'instant).
struct Expected {
  std::int64_t planes = 0;
  int shift = 0;
};
constexpr std::array<Expected, 9> kExpected{{{3, 0},
                                             {3, 0},
                                             {1, -1},
                                             {3 + prepare::kGmfssFeatureChannels, 0},
                                             {128, 1},
                                             {192, 2},
                                             {3 + prepare::kGmfssFeatureChannels, 0},
                                             {128, 1},
                                             {192, 2}}};

Shape expected_shape(std::size_t input, int height, int width) {
  const Expected& e = kExpected[input];
  if (e.shift < 0) return {.dims = {1, 1, 1, 1}};
  return {.dims = {1, e.planes, height >> e.shift, width >> e.shift}};
}

std::expected<std::size_t, Refusal> find_input(const TrtEngine& engine, std::string_view name) {
  const std::span<const std::string> names = engine.input_names();
  const auto found = std::ranges::find(names, name);
  if (found == names.end())
    return refuse(RefusalCode::GraphInvalid, std::format("GMFSS natif : entrée « {} » absente", name));
  return static_cast<std::size_t>(found - names.begin());
}

bool planar(const TrtEngine& engine) {
  for (std::size_t i = 0; i < engine.input_names().size(); ++i) {
    if (engine.input_pitch(i) != 0) return false;
  }
  return engine.output_names().size() == 1 && engine.output_pitch(0) == 0;
}

}  // namespace

GmfssFusedEngine::~GmfssFusedEngine() {
  if (rife_ != nullptr) static_cast<void>(cudaFree(rife_));
}

std::expected<std::unique_ptr<GmfssFusedEngine>, Refusal> GmfssFusedEngine::load(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const GmfssFusedKey& key,
    const std::shared_ptr<SharedActivations>& shared) {
  if (key.engines.size() != 2) return refuse(RefusalCode::GraphInvalid, "GMFSS natif : clé invalide");
  std::unique_ptr<GmfssFusedEngine> engine(new GmfssFusedEngine());
  engine->height_ = key.height;
  engine->width_ = key.width;
  auto full = TrtEngine::load(runtime, cache_dir / core::path_from_utf8(key.engines[0]), shared);
  if (!full) return std::unexpected(std::move(full.error()));
  auto ifnet = TrtEngine::load(runtime, cache_dir / core::path_from_utf8(key.engines[1]), shared);
  if (!ifnet) return std::unexpected(std::move(ifnet.error()));
  engine->full_ = std::move(*full);
  engine->ifnet_ = std::move(*ifnet);
  const TrtEngine& f = *engine->full_;
  const TrtEngine& r = *engine->ifnet_;
  const auto bad = [](std::string_view what) {
    return refuse(RefusalCode::ShapeInvalid, std::format("GMFSS natif : {} de forme inattendue", what));
  };
  if (f.input_names().size() != kInputs || r.input_names().size() != 3 || !planar(f) || !planar(r))
    return bad("moteurs");
  for (std::size_t i = 0; i < kInputs; ++i) {
    auto at = find_input(f, kNames[i]);
    if (!at) return std::unexpected(std::move(at.error()));
    engine->at_[i] = *at;
    if (!(f.input_shape(*at) == expected_shape(i, key.height, key.width))) return bad(kNames[i]);
  }
  for (std::size_t j = 0; j < 3; ++j) {
    const auto found = std::ranges::find(kNames.begin(), kNames.begin() + 3, r.input_names()[j]);
    if (found == kNames.begin() + 3) return bad("IFNet (entrées)");
    const auto input = static_cast<std::size_t>(found - kNames.begin());
    engine->ifnet_from_[j] = input;
    if (!(r.input_shape(j) == expected_shape(input, key.height, key.width))) return bad("IFNet (entrées)");
  }
  if (!(f.output_shape(0) == Shape{.dims = {1, 3, 2 * key.height, 2 * key.width}})) return bad("sortie");
  if (!(r.output_shape(0) == Shape{.dims = {1, 3, key.height, key.width}})) return bad("IFNet (sortie)");
  const auto bytes =
      std::size_t{3} * static_cast<std::size_t>(key.height) * static_cast<std::size_t>(key.width) * 2;
  if (auto ok = cuda_check(cudaMalloc(&engine->rife_, bytes), "GMFSS natif : image de l'IFNet"); !ok) {
    engine->rife_ = nullptr;
    return std::unexpected(std::move(ok.error()));
  }
  return engine;
}

std::expected<void, Refusal> GmfssFusedEngine::enqueue(cudaStream_t stream,
                                                       const std::shared_ptr<SharedActivations>& shared,
                                                       std::span<const Tensor> inputs,
                                                       std::span<const Tensor> outputs) {
  if (inputs.size() != kInputs || outputs.size() != 1)
    return refuse(RefusalCode::ShapeInvalid, "GMFSS natif : neuf entrées et une sortie attendues");
  if (!native_on_) return full_->enqueue(stream, shared, inputs, outputs);
  return enqueue_native(stream, shared, inputs, outputs[0]);
}

std::expected<void, Refusal> GmfssFusedEngine::enqueue_native(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared, std::span<const Tensor> inputs,
    const Tensor& out) {
  // Les noyaux lisent et écrivent les plans sans autre garde : chaque tenseur à la forme du moteur entier.
  const auto matches = [](const Tensor& t, const Shape& shape) {
    return t.type == TensorType::F16 && t.shape == shape && t.handle != 0;
  };
  bool good = matches(out, full_->output_shape(0));
  for (std::size_t i = 0; i < inputs.size(); ++i) good = good && matches(inputs[i], full_->input_shape(i));
  if (!good) return refuse(RefusalCode::ShapeInvalid, "GMFSS natif : forme d'entrée ou de sortie");
  std::array<Tensor, 3> ifnet_in;
  for (std::size_t j = 0; j < 3; ++j) ifnet_in[j] = inputs[at_[ifnet_from_[j]]];
  const Tensor rife = fused_tensor(reinterpret_cast<std::uintptr_t>(rife_), 0, 3, height_, width_);
  const Tensor ifnet_out[]{rife};
  if (auto ok = ifnet_->enqueue(stream, shared, ifnet_in, ifnet_out); !ok) return ok;
  GmfssNativeInputs in;
  for (std::size_t level = 0; level < 3; ++level) {
    in.warped0[level] = fused_address(inputs[at_[kW01 + level]]);
    in.warped1[level] = fused_address(inputs[at_[kW11 + level]]);
  }
  in.rife = rife_;
  return cuda_check(native_->run(stream, in, fused_address(out)), "GMFSS natif : GridNet");
}

}  // namespace kirari::gpu::cuda
