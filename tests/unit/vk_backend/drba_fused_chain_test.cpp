// DistilDRBA fusionné de bout en bout sur le backend Vulkan (D7), en test caché (le vrai DistilDRBA v2 lite
// lu en place ; des minutes sur lavapipe) : le manifeste dérivé (prepare/drba_split.h) construit et chargé,
// contre le DistilDRBA entier sur le même backend, sur trois images synthétiques décalées, des deux côtés de
// l'image du milieu (tau 0,5 et 1,25) : PSNR d'au moins 50 dB ; la variante mouvement, un flot borné.
//   ajp_vk_backend_tests "[.drba_fused]"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "prepare/drba_split.h"
#include "prepare/model_source.h"
#include "prepare/onnx_dims.h"
#include "prepare/sha256.h"

namespace ajp::gpu::vk::test {
namespace {

namespace fs = std::filesystem;

// Un dessin d'anime synthétique : aplats, dégradé, traits nets ; shift le décale vers la droite.
std::vector<kernels::PlaneF> drawing(int h, int w, int shift) {
  std::vector<kernels::PlaneF> planes;
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
    planes.push_back(std::move(plane));
  }
  return planes;
}

// Une paire en 7 plans, comme rife_input : la première image, la seconde, tau / 2.
std::vector<kernels::PlaneF> pair_of(const std::vector<kernels::PlaneF>& a,
                                     const std::vector<kernels::PlaneF>& b, int h, int w, float tau) {
  std::vector<kernels::PlaneF> planes(a);
  planes.insert(planes.end(), b.begin(), b.end());
  planes.emplace_back(w, h, tau / 2.0F);
  return planes;
}

ShapeSet fixed(const Shape& shape) {
  return {.min = shape, .opt = shape, .max = shape};
}

ModelSpec spec_of(const fs::path& file) {
  auto sha = prepare::sha256_file(file);
  REQUIRE(sha);
  return {.id = file.stem().string(), .onnx_path = file, .sha256 = *sha, .precision = TensorType::F16};
}

}  // namespace

TEST_CASE("DistilDRBA fusionné : la chaîne Vulkan contre le DistilDRBA entier", "[vk_backend][.drba_fused]") {
  const fs::path onnx = fs::path(AJP_RIFE_MODEL_DIR) / "distil_drba_v2_lite.onnx";
  if (!fs::is_regular_file(onnx)) SKIP("modèle absent : " << onnx.string());
  TempDir dir("ajp-drba-fused-vk");
  std::unique_ptr<VulkanBackend> backend = make_backend(dir.path() / "cache");
  if (!backend->capabilities().drba_fused) SKIP("appareil sans stockage 16 bits");
  constexpr int h = 64;
  constexpr int w = 128;
  const Shape input_shape{.dims = {1, 7, h, w}};
  Progress progress;
  const auto manifest = prepare::derive_drba_fused(onnx, dir.path(), h, w);
  REQUIRE(manifest);
  auto fused_artifact = backend->build(spec_of(*manifest), fixed(input_shape), progress);
  if (!fused_artifact) FAIL(fused_artifact.error().detail);
  auto fused = backend->load(*fused_artifact);
  if (!fused) FAIL(fused.error().detail);
  REQUIRE((*fused)->input_names().size() == 2);
  // Le DistilDRBA entier, aux dimensions fixées comme le lecteur le charge (deux entrées : figé).
  const auto bytes = prepare::read_onnx_bytes(onnx);
  REQUIRE(bytes);
  const std::vector<prepare::NamedDim> dims{{"H", h}, {"W", w}};
  const auto static_bytes = prepare::with_fixed_dims(*bytes, dims);
  REQUIRE(static_bytes);
  const fs::path static_onnx = dir.path() / "distil_drba_v2_lite-fixe.onnx";
  {
    std::ofstream out(static_onnx, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(static_bytes->data()),
              static_cast<std::streamsize>(static_bytes->size()));
  }
  auto whole_artifact = backend->build(spec_of(static_onnx), fixed(input_shape), progress);
  if (!whole_artifact) FAIL(whole_artifact.error().detail);
  auto whole = backend->load(*whole_artifact);
  if (!whole) FAIL(whole.error().detail);

  const auto img0 = drawing(h, w, 0);
  const auto img1 = drawing(h, w, 4);
  const auto img2 = drawing(h, w, 8);
  auto in01 = rgb_tensor(*backend, w, h, TensorType::F16, 7);
  auto in12 = rgb_tensor(*backend, w, h, TensorType::F16, 7);
  const std::array inputs{in01->tensor(), in12->tensor()};
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
  // D6 : les mêmes images à un autre instant, annoncé par next_times_only : l'encodeur et post sautés.
  const auto times_only = [&](float tau) {
    write_tensor(*backend, in01->tensor(), planes_bytes(pair_of(img0, img1, h, w, tau), TensorType::F16));
    write_tensor(*backend, in12->tensor(), planes_bytes(pair_of(img1, img2, h, w, tau), TensorType::F16));
    auto out = rgb_tensor(*backend, w, h, TensorType::F16, 3);
    std::array all{out->tensor()};
    (*fused)->next_times_only();
    REQUIRE(backend->enqueue(**fused, inputs, all));
    return planes_from(read_tensor(*backend, out->tensor(), static_cast<std::size_t>(3 * h * w * 2)), w, h, 3,
                       TensorType::F16);
  };
  for (const auto& [tau, reuse] :
       {std::pair{0.5F, false}, std::pair{1.25F, false}, std::pair{0.75F, true}, std::pair{0.25F, true}}) {
    write_tensor(*backend, in01->tensor(), planes_bytes(pair_of(img0, img1, h, w, tau), TensorType::F16));
    write_tensor(*backend, in12->tensor(), planes_bytes(pair_of(img1, img2, h, w, tau), TensorType::F16));
    const auto want = run(**whole, 3);
    const auto got = reuse ? times_only(tau) : run(**fused, 3);
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
    WARN("tau " << tau << (reuse ? " (temps seul)" : "") << " : PSNR " << psnr
                << " dB contre le DistilDRBA entier, écart maximal " << worst);
    CHECK(psnr >= 50.0);
  }
  // La variante mouvement : le flot du warp final, dans les bornes d'un mouvement de quelques pixels.
  const auto flow = run(**fused, 4);
  float largest = 0.0F;
  for (const auto& plane : flow)
    for (const float v : plane.data) largest = std::max(largest, std::abs(v));
  CHECK(largest < 64.0F);
}

}  // namespace ajp::gpu::vk::test
