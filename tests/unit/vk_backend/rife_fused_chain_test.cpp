// RIFE fusionné de bout en bout sur le backend Vulkan, en test caché (un vrai RIFE lu en place, quelques
// secondes sur un GPU, des minutes sur lavapipe) : le manifeste dérivé (prepare/rife_split.h) construit et
// chargé, contre le RIFE entier sur le même backend, sur une paire d'images synthétiques décalées : PSNR de
// l'image d'au moins 50 dB ; la variante mouvement, un flot borné.
//   ajp_vk_backend_tests "[.rife_fused]"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "prepare/rife_split.h"
#include "prepare/sha256.h"

namespace ajp::gpu::vk::test {
namespace {

namespace fs = std::filesystem;

// Un dessin d'anime synthétique : aplats, dégradé, traits nets ; shift le décale vers la droite.
kernels::RgbImage drawing(int h, int w, int shift) {
  kernels::RgbImage image;
  for (int c = 0; c < 3; ++c) {
    kernels::PlaneF plane(w, h, 0.0F);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int u = std::max(0, x - shift);
        float v = 0.25F + 0.1F * static_cast<float>(c);
        if (u > w / 3 && u < 2 * w / 3 && y > h / 4) v = 0.7F - 0.1F * static_cast<float>(c);
        if (y > h / 2) v += 0.25F * static_cast<float>(u) / static_cast<float>(w);
        if ((u + 2 * y) % 37 < 2) v = 0.05F;
        plane.at(x, y) = std::clamp(v, 0.F, 1.F);
      }
    }
    image.planes[static_cast<std::size_t>(c)] = std::move(plane);
  }
  return image;
}

ShapeSet fixed(const Shape& shape) {
  return {.min = shape, .opt = shape, .max = shape};
}

ModelSpec spec_of(const fs::path& file) {
  auto sha = prepare::sha256_file(file);
  REQUIRE(sha);
  return {.id = file.stem().string(), .onnx_path = file, .sha256 = *sha, .precision = TensorType::F16};
}

void compare_chain(const std::string& file, bool motion) {
  const fs::path onnx = fs::path(AJP_RIFE_MODEL_DIR) / file;
  if (!fs::is_regular_file(onnx)) SKIP("modèle absent : " << onnx.string());
  TempDir dir("ajp-rife-fused-vk");
  std::unique_ptr<VulkanBackend> backend = make_backend(dir.path() / "cache");
  if (!backend->capabilities().rife_fused) SKIP("appareil sans stockage 16 bits");
  constexpr int h = 64;
  constexpr int w = 128;
  const Shape input_shape{.dims = {1, 11, h, w}};
  Progress progress;
  const auto manifest = prepare::derive_rife_fused(
      onnx, dir.path(), h, w, motion ? prepare::RifeFusedOutput::Motion : prepare::RifeFusedOutput::Image);
  REQUIRE(manifest);
  auto fused_artifact = backend->build(spec_of(*manifest), fixed(input_shape), progress);
  if (!fused_artifact) FAIL(fused_artifact.error().detail);
  auto fused = backend->load(*fused_artifact);
  if (!fused) FAIL(fused.error().detail);

  // Le RIFE entier, référence de l'image ; la variante mouvement vérifie seulement que son flot reste borné.
  auto whole_artifact = backend->build(spec_of(onnx), fixed(input_shape), progress);
  if (!whole_artifact) FAIL(whole_artifact.error().detail);
  auto whole = backend->load(*whole_artifact);
  if (!whole) FAIL(whole.error().detail);

  auto a = rgb_tensor(*backend, w, h, TensorType::F16);
  auto b = rgb_tensor(*backend, w, h, TensorType::F16);
  write_tensor(*backend, a->tensor(), rgb_bytes(drawing(h, w, 0), TensorType::F16));
  write_tensor(*backend, b->tensor(), rgb_bytes(drawing(h, w, 6), TensorType::F16));
  auto input = backend->alloc_tensor(TensorType::F16, input_shape);
  REQUIRE(input);
  REQUIRE(backend->kernels()->rife_input(a->tensor(), b->tensor(), 0.5F, (*input)->tensor()));
  const std::array inputs{(*input)->tensor()};

  const auto run = [&](Engine& engine, int planes) {
    auto out = rgb_tensor(*backend, w, h, TensorType::F16, planes);
    std::unique_ptr<TensorBuffer> mask;
    std::vector<Tensor> all{out->tensor()};
    if (planes == 4) {
      mask = rgb_tensor(*backend, w, h, TensorType::F16, 1);
      all.push_back(mask->tensor());
    }
    for (int i = 0; i < 2; ++i) REQUIRE(backend->enqueue(engine, inputs, all));  // le second rejoue
    return planes_from(read_tensor(*backend, out->tensor(), static_cast<std::size_t>(planes * h * w * 2)), w,
                       h, planes, TensorType::F16);
  };
  if (motion) {
    // Le flot du chemin fusionné, dans les bornes d'un mouvement de quelques pixels.
    const auto flow = run(**fused, 4);
    float largest = 0.0F;
    for (const auto& plane : flow)
      for (const float v : plane.data) largest = std::max(largest, std::abs(v));
    CHECK(largest < 64.0F);
    return;
  }
  const auto want = run(**whole, 3);
  const auto got = run(**fused, 3);
  double square = 0.0;
  double worst = 0.0;
  std::size_t count = 0;
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < want[c].data.size(); ++i) {
      const double d = static_cast<double>(got[c].data[i]) - static_cast<double>(want[c].data[i]);
      square += d * d;
      worst = std::max(worst, std::abs(d));
      ++count;
    }
  }
  const double mse = square / static_cast<double>(count);
  const double psnr = mse == 0.0 ? 200.0 : 10.0 * std::log10(1.0 / mse);
  WARN(file << " : PSNR " << psnr << " dB contre le RIFE entier, écart maximal " << worst);
  CHECK(psnr >= 50.0);
}

}  // namespace

TEST_CASE("RIFE fusionné : la chaîne Vulkan contre le RIFE entier", "[vk_backend][.rife_fused]") {
  SECTION("4.22 lite, image") {
    compare_chain("rife_v4.22_lite.onnx", false);
  }
  SECTION("4.22 lite, mouvement") {
    compare_chain("rife_v4.22_lite.onnx", true);
  }
  SECTION("4.25, image") {
    compare_chain("rife_v4.25.onnx", false);
  }
}

}  // namespace ajp::gpu::vk::test
