// RIFE fusionné de bout en bout sur le backend Vulkan, en test caché (un vrai RIFE lu en place, quelques
// secondes sur un GPU, des minutes sur lavapipe) : le manifeste dérivé (prepare/rife_split.h) construit et
// chargé, contre le RIFE entier sur le même backend, sur une paire d'images synthétiques décalées : PSNR de
// l'image d'au moins 50 dB, et d'au moins 55 dB contre l'interpréteur en float, qui ne partage aucun noyau
// avec le moteur (A45-11) ; la variante mouvement, un flot borné. La CI le lance la nuit et à la demande, sur
// les modèles téléchargés (ci.yml) ; sans modèle il est sauté. Et l'encodage gardé d'un appel à l'autre
// (même paire à un autre instant, paire qui suit), bit à bit contre la chaîne qui encode tout.
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
#include "fused_reference.h"
#include "gpu/vulkan/backend/rife_fused_engine.h"
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
  // Contre une référence qui ne passe par aucun noyau Vulkan : l'interpréteur, en float, sur l'entrée
  // telle que rife_input l'a écrite (A45-11).
  const auto fed =
      planes_from(read_tensor(*backend, (*input)->tensor(), static_cast<std::size_t>(11 * h * w * 2)), w, h,
                  11, TensorType::F16);
  const double fidelity = planes_psnr(got, interpreter_planes(onnx, {fed}, h, w, 3));
  WARN(file << " : PSNR " << fidelity << " dB contre l'interpréteur en float");
  CHECK(fidelity >= 55.0);
}

// L'encodage gardé d'un appel à l'autre (next_times_only, next_pair_follows) rend les octets exacts d'une
// chaîne qui encode tout à chaque appel : trois images, deux paires dans deux tampons d'entrée à tour de
// rôle, deux instants pour la première, sur la variante image et la variante mouvement.
void kept_encoding_is_exact(const std::string& file, bool motion) {
  const fs::path onnx = fs::path(AJP_RIFE_MODEL_DIR) / file;
  if (!fs::is_regular_file(onnx)) SKIP("modèle absent : " << onnx.string());
  TempDir dir("ajp-rife-fused-encode-vk");
  std::unique_ptr<VulkanBackend> backend = make_backend(dir.path() / "cache");
  if (!backend->capabilities().rife_fused) SKIP("appareil sans stockage 16 bits");
  constexpr int h = 64;
  constexpr int w = 128;
  const Shape input_shape{.dims = {1, 11, h, w}};
  Progress progress;
  const auto manifest = prepare::derive_rife_fused(
      onnx, dir.path(), h, w, motion ? prepare::RifeFusedOutput::Motion : prepare::RifeFusedOutput::Image);
  REQUIRE(manifest);
  auto artifact = backend->build(spec_of(*manifest), fixed(input_shape), progress);
  if (!artifact) FAIL(artifact.error().detail);
  auto loaded = backend->load(*artifact);
  if (!loaded) FAIL(loaded.error().detail);
  auto* fused = dynamic_cast<RifeFusedEngine*>(loaded->get());
  REQUIRE(fused != nullptr);

  std::array<std::unique_ptr<TensorBuffer>, 3> images;
  for (std::size_t i = 0; i < images.size(); ++i) {
    images[i] = rgb_tensor(*backend, w, h, TensorType::F16);
    write_tensor(*backend, images[i]->tensor(),
                 rgb_bytes(drawing(h, w, 4 * static_cast<int>(i)), TensorType::F16));
  }
  std::array<std::unique_ptr<TensorBuffer>, 2> inputs;
  for (auto& input : inputs) {
    auto made = backend->alloc_tensor(TensorType::F16, input_shape);
    REQUIRE(made);
    input = std::move(*made);
  }
  const int planes = motion ? 4 : 3;
  auto out = rgb_tensor(*backend, w, h, TensorType::F16, planes);
  auto mask = rgb_tensor(*backend, w, h, TensorType::F16, 1);
  std::vector<Tensor> outs{out->tensor()};
  if (motion) outs.push_back(mask->tensor());
  const auto read = [&] {
    std::vector<std::byte> bytes =
        read_tensor(*backend, out->tensor(), static_cast<std::size_t>(planes * h * w * 2));
    if (motion) {
      const std::vector<std::byte> m =
          read_tensor(*backend, mask->tensor(), static_cast<std::size_t>(h * w * 2));
      bytes.insert(bytes.end(), m.begin(), m.end());
    }
    return bytes;
  };
  // Un appel : la paire (first, first + 1) dans le tampon buffer à l'instant t ; whole réécrit tout, sinon
  // rife_time n'écrit que l'instant.
  enum class Say { Nothing, TimesOnly, Follows };
  const auto call = [&](std::size_t first, std::size_t buffer, float t, bool whole, Say say) {
    const Tensor& input = inputs[buffer]->tensor();
    if (whole) {
      REQUIRE(backend->kernels()->rife_input(images[first]->tensor(), images[first + 1]->tensor(), t, input));
    } else {
      REQUIRE(backend->kernels()->rife_time(input, t));
    }
    if (say == Say::TimesOnly) fused->next_times_only();
    if (say == Say::Follows) fused->next_pair_follows();
    const std::array ins{input};
    REQUIRE(backend->enqueue(*fused, ins, outs));
    return read();
  };

  // Tout encodé à chaque appel : la référence.
  const auto want_a = call(0, 0, 0.25F, true, Say::Nothing);
  const auto want_b = call(0, 0, 0.75F, true, Say::Nothing);
  const auto want_c = call(1, 1, 0.5F, true, Say::Nothing);
  const auto want_d = call(1, 1, 0.25F, true, Say::Nothing);
  // Le même chemin en gardant l'encodage : la paire à froid, l'instant seul, la paire qui suit, l'instant
  // seul.
  const std::uint64_t before = fused->encodes();
  const auto got_a = call(0, 0, 0.25F, true, Say::Nothing);
  const auto got_b = call(0, 0, 0.75F, false, Say::TimesOnly);
  const auto got_c = call(1, 1, 0.5F, true, Say::Follows);
  const auto got_d = call(1, 1, 0.25F, false, Say::TimesOnly);
  CHECK(fused->encodes() - before == 3);  // A et B de la première paire, B de la seconde
  CHECK(got_a == want_a);
  CHECK(got_b == want_b);
  CHECK(got_c == want_c);
  CHECK(got_d == want_d);
  CHECK(want_a != want_b);  // l'instant change bien la sortie

  // Un instant annoncé sur une autre entrée que celle du dernier appel : l'encodeur repasse, rien de faux.
  const std::uint64_t again = fused->encodes();
  REQUIRE(
      backend->kernels()->rife_input(images[0]->tensor(), images[1]->tensor(), 0.25F, inputs[0]->tensor()));
  fused->next_times_only();
  const std::array ins{inputs[0]->tensor()};
  REQUIRE(backend->enqueue(*fused, ins, outs));
  CHECK(read() == want_a);
  CHECK(fused->encodes() - again == 2);
}

}  // namespace

TEST_CASE("RIFE fusionné : l'encodage repris d'un appel à l'autre rend les mêmes octets",
          "[vk_backend][.rife_fused]") {
  SECTION("4.22 lite, image") {
    kept_encoding_is_exact("rife_v4.22_lite.onnx", false);
  }
  SECTION("4.22 lite, mouvement") {
    kept_encoding_is_exact("rife_v4.22_lite.onnx", true);
  }
  SECTION("4.25, image") {
    kept_encoding_is_exact("rife_v4.25.onnx", false);
  }
}

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
