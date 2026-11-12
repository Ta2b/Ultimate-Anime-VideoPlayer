// Le moteur TensorRT du SPAN FP16 auquel la préparation du mode Performance compare ses noyaux FP8 (B02-01,
// span_derive.h : build_span_fp16) ; le backend le chronomètre avec eux (C02-01).
#include <expected>
#include <filesystem>

#include "app/play/span_derive.h"
#include "core/utf8_path.h"
#include "prepare/sha256.h"

namespace ajp::app::play {

std::expected<gpu::Artifact, Refusal> build_span_fp16(gpu::Backend& backend,
                                                      const std::filesystem::path& model, int width,
                                                      int height, gpu::Progress& progress) {
  if (width <= 0 || height <= 0)
    return refuse(RefusalCode::ShapeInvalid, "moteur du SPAN FP16 : taille d'image nulle");
  auto sha = prepare::sha256_file_known(model);
  if (!sha) return std::unexpected(std::move(sha.error()));
  // La même demande que la chaîne (chain_setup.cpp, prepare_span) : le moteur de la lecture, lu dans le
  // cache.
  const gpu::Shape in_shape{.dims = {1, 3, height, width}};
  return backend.build(gpu::ModelSpec{.id = core::to_utf8(model.stem()), .onnx_path = model, .sha256 = *sha},
                       gpu::ShapeSet{.min = in_shape, .opt = in_shape, .max = in_shape}, progress);
}

}  // namespace ajp::app::play
