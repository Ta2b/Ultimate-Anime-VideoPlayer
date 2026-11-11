// Le temps du SPAN FP16 de TensorRT, auquel la préparation du mode Performance compare ses noyaux FP8
// (B02-01, span_derive.h : time_span_fp16).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <vector>

#include "app/play/span_derive.h"
#include "core/utf8_path.h"
#include "prepare/sha256.h"

namespace ajp::app::play {

std::expected<double, Refusal> time_span_fp16(gpu::Backend& backend, const std::filesystem::path& model,
                                              int width, int height, gpu::Progress& progress) {
  if (width <= 0 || height <= 0)
    return refuse(RefusalCode::ShapeInvalid, "temps du SPAN FP16 : taille d'image nulle");
  auto sha = prepare::sha256_file_known(model);
  if (!sha) return std::unexpected(std::move(sha.error()));
  // La même demande que la chaîne (chain_setup.cpp, prepare_span) : le moteur de la lecture, lu dans le
  // cache.
  const gpu::Shape in_shape{.dims = {1, 3, height, width}};
  auto artifact =
      backend.build(gpu::ModelSpec{.id = core::to_utf8(model.stem()), .onnx_path = model, .sha256 = *sha},
                    gpu::ShapeSet{.min = in_shape, .opt = in_shape, .max = in_shape}, progress);
  if (!artifact) return std::unexpected(std::move(artifact.error()));
  backend.use_lane(gpu::Lane::Upscale);
  auto engine = backend.load(*artifact);
  if (!engine) return std::unexpected(std::move(engine.error()));
  auto input = backend.alloc_tensor(gpu::TensorType::F16, in_shape);
  if (!input) return std::unexpected(std::move(input.error()));
  auto output = backend.alloc_tensor(
      gpu::TensorType::F16, gpu::Shape{.dims = {1, 3, std::int64_t{height} * 2, std::int64_t{width} * 2}});
  if (!output) return std::unexpected(std::move(output.error()));
  const std::array inputs{(*input)->tensor()};
  std::array outputs{(*output)->tensor()};
  const auto run = [&](int times) -> std::expected<void, Refusal> {
    for (int i = 0; i < times; ++i) {
      if (progress.stop.stop_requested())
        return refuse(RefusalCode::Cancelled, "temps du SPAN FP16 interrompu");
      if (auto ok = backend.enqueue(**engine, inputs, outputs); !ok) return ok;
    }
    backend.drain();  // l'hôte attend la fin des passes enfilées
    return {};
  };
  if (auto ok = run(kSpanTimingWarmup); !ok) return std::unexpected(std::move(ok.error()));
  // Des lots de passes enchaînées, comme la lecture les enfile : le temps d'un lot divisé par ses passes ne
  // compte pas l'attente de l'hôte entre deux passes ; la médiane des lots écarte un lot dérangé.
  std::vector<double> per_pass;
  for (int batch = 0; batch < kSpanTimingBatches; ++batch) {
    const auto start = std::chrono::steady_clock::now();
    if (auto ok = run(kSpanTimingBatchRuns); !ok) return std::unexpected(std::move(ok.error()));
    const std::chrono::duration<double, std::milli> spent = std::chrono::steady_clock::now() - start;
    per_pass.push_back(spent.count() / kSpanTimingBatchRuns);
  }
  std::ranges::sort(per_pass);
  return per_pass[per_pass.size() / 2];
}

}  // namespace ajp::app::play
