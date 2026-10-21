// Backend TensorRT chargé comme le lecteur le charge (dlopen), sur la RTX : warp et mélange par le
// mouvement, entrée de RIFE, mesures d'une paire, RTX HDR, et le vrai modèle SPAN d'AnimeJaNai
// s'il est installé (lu en place, jamais copié). Sans GPU NVIDIA, chaque test se SKIP.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "backend_frames.h"
#include "gpu/backend.h"
#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/tolerance.h"
#include "prepare/sha256.h"

namespace {

using namespace ajp;
using namespace ajp::test;

// Mouvement dans des plans pw × ph, image dans la fenêtre area ; images 1,5 fois plus grandes
// (mouvement estimé à 720p, warp à 1080p). tolerance : l'écart admis à la référence, pour warp_blend
// et warp_blend_poses.
struct WarpCase {
  int pw = 0;
  int ph = 0;
  int w = 0;
  int h = 0;
  kernels::Window area;
  double tolerance = 0.0;
};

void check_warp(const WarpCase& wc) {
  const int pw = wc.pw;
  const int ph = wc.ph;
  const int w = wc.w;
  const int h = wc.h;
  const kernels::Window area = wc.area;
  const auto q = [](float v) { return kernels::ref::half_to_float(kernels::ref::float_to_half(v)); };
  kernels::ref::Motion motion;
  for (auto& plane : motion.flow) plane = kernels::PlaneF(pw, ph);
  motion.mask = kernels::PlaneF(pw, ph);
  motion.area = area;
  std::vector<float> flow(static_cast<std::size_t>(4 * pw * ph));
  std::vector<float> mask(static_cast<std::size_t>(pw * ph));
  for (int y = 0; y < ph; ++y) {
    for (int x = 0; x < pw; ++x) {
      for (int c = 0; c < 4; ++c) {
        const float v =
            q(4.0F * std::sin(0.21F * static_cast<float>(x + 3 * c) + 0.37F * static_cast<float>(y)));
        motion.flow[static_cast<std::size_t>(c)].at(x, y) = v;
        flow[static_cast<std::size_t>((c * ph + y) * pw + x)] = v;
      }
      const float m =
          q(0.5F + 0.3F * std::cos(0.17F * static_cast<float>(x) - 0.23F * static_cast<float>(y)));
      motion.mask.at(x, y) = m;
      mask[static_cast<std::size_t>(y * pw + x)] = m;
    }
  }
  kernels::RgbImage a(w, h);
  kernels::RgbImage b(w, h);
  std::vector<float> va(static_cast<std::size_t>(3 * w * h));
  std::vector<float> vb(va.size());
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const std::size_t i = static_cast<std::size_t>((c * h + y) * w + x);
        va[i] = q(0.5F + 0.4F * std::sin(0.3F * static_cast<float>(x) + 0.5F * static_cast<float>(c) +
                                         0.1F * static_cast<float>(y)));
        vb[i] = q(0.5F + 0.4F * std::cos(0.2F * static_cast<float>(y) - 0.4F * static_cast<float>(c) +
                                         0.15F * static_cast<float>(x)));
        a.planes[static_cast<std::size_t>(c)].at(x, y) = va[i];
        b.planes[static_cast<std::size_t>(c)].at(x, y) = vb[i];
      }
    }
  }
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto alloc = [&](std::int64_t c, std::int64_t hh, std::int64_t ww) {
    auto t = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, c, hh, ww}});
    REQUIRE(t);
    return std::move(*t);
  };
  auto ta = alloc(3, h, w);
  auto tb = alloc(3, h, w);
  auto out = alloc(3, h, w);
  auto tflow = alloc(4, ph, pw);
  auto tmask = alloc(1, ph, pw);
  upload(ta->tensor(), va);
  upload(tb->tensor(), vb);
  upload(tflow->tensor(), flow);
  upload(tmask->tensor(), mask);
  const gpu::Motion gm{.flow = tflow->tensor(), .mask = tmask->tensor(), .area = area};
  for (const float t : {0.5F, 0.25F, 0.75F}) {
    CAPTURE(t);
    REQUIRE(loaded.backend->kernels()->warp_blend(ta->tensor(), tb->tensor(), gm, t, out->tensor()));
    const auto got = download(out->tensor());
    const auto want = kernels::ref::warp_blend_at(a, b, motion, t);
    REQUIRE(want);
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          worst = std::max(
              worst, static_cast<double>(std::fabs(want->planes[static_cast<std::size_t>(c)].at(x, y) -
                                                   got[static_cast<std::size_t>((c * h + y) * w + x)])));
        }
      }
    }
    CHECK(worst < wc.tolerance);
  }
  // Garder les poses : la paire précédente bougeait à 30 % (valeurs FP16 exactes), puis était tenue.
  kernels::ref::Motion before = motion;
  std::vector<float> before_flow(flow.size());
  for (std::size_t i = 0; i < flow.size(); ++i) before_flow[i] = q(0.3F * flow[i]);
  for (int c = 0; c < 4; ++c) {
    for (int y = 0; y < ph; ++y) {
      for (int x = 0; x < pw; ++x)
        before.flow[static_cast<std::size_t>(c)].at(x, y) =
            before_flow[static_cast<std::size_t>((c * ph + y) * pw + x)];
    }
  }
  auto tbefore = alloc(4, ph, pw);
  upload(tbefore->tensor(), before_flow);
  const gpu::Motion gbefore{.flow = tbefore->tensor(), .mask = tmask->tensor(), .area = area};
  for (const bool held : {false, true}) {
    for (const float t : {0.25F, 0.5F, 0.75F}) {
      CAPTURE(held, t);
      REQUIRE(loaded.backend->kernels()->warp_blend_poses(ta->tensor(), tb->tensor(), gm,
                                                          held ? nullptr : &gbefore, t, out->tensor()));
      const auto got = download(out->tensor());
      const auto want = kernels::ref::warp_blend_poses(a, b, motion, held ? nullptr : &before, t);
      REQUIRE(want);
      double worst = 0.0;
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
          for (int x = 0; x < w; ++x) {
            worst = std::max(
                worst, static_cast<double>(std::fabs(want->planes[static_cast<std::size_t>(c)].at(x, y) -
                                                     got[static_cast<std::size_t>((c * h + y) * w + x)])));
          }
        }
      }
      CHECK(worst < wc.tolerance);
    }
  }
  const auto max = loaded.backend->kernels()->flow_max_now(gm);
  REQUIRE(max);
  const auto want_max = kernels::ref::flow_max(motion);
  REQUIRE(want_max);
  CHECK(*max == *want_max);
  // Un instant hors de ]0, 1[ et un mouvement mal formé sont refusés par leur nom.
  CHECK(loaded.backend->kernels()
            ->warp_blend(ta->tensor(), tb->tensor(), gm, 1.0F, out->tensor())
            .error()
            .code == RefusalCode::FieldOutOfRange);
  const gpu::Motion outside{
      .flow = tflow->tensor(),
      .mask = tmask->tensor(),
      .area = {.x = 0, .y = ph - area.height + 1, .width = area.width, .height = area.height}};
  CHECK(loaded.backend->kernels()
            ->warp_blend(ta->tensor(), tb->tensor(), outside, 0.5F, out->tensor())
            .error()
            .code == RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend TensorRT : warp et mélange par le mouvement, contre la référence",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  // Largeurs de 80 et 120 octets, hors du pas de 32 des objets texture : la lecture exacte, à
  // l'arrondi FP16 de la sortie près.
  check_warp({.pw = 40,
              .ph = 32,
              .w = 60,
              .h = 36,
              .area = {.x = 0, .y = 4, .width = 40, .height = 24},
              .tolerance = 2e-3});
}

TEST_CASE("Backend TensorRT : warp par objets texture, deux pixels par fil, contre la référence",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  // Largeurs de 128 et 192 octets : le filtre bilinéaire matériel et la sortie half2.
  check_warp({.pw = 64,
              .ph = 48,
              .w = 96,
              .h = 48,
              .area = {.x = 0, .y = 8, .width = 64, .height = 32},
              .tolerance = kernels::tolerance::kWarpFiltered});
}

TEST_CASE("Backend TensorRT : entrée de RIFE, 11 plans remplis, puis recadrage de la fenêtre",
          "[backend_tensorrt][cuda][rife]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 64;
  constexpr int h = 36;  // rempli à 64 : 14 lignes au-dessus (nombre pair), 14 en dessous
  const kernels::RifeGeometry g = kernels::rife_geometry(w, h);
  REQUIRE(g.padded_height == 64);
  REQUIRE(g.top == 14);
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto b = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto input =
      loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 11, g.padded_height, g.padded_width}});
  auto back = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(a);
  REQUIRE(b);
  REQUIRE(input);
  REQUIRE(back);
  const std::vector<float> va = ramp(w, h);
  std::vector<float> vb = ramp(w, h);
  for (float& v : vb) v += 0.25F;
  upload((*a)->tensor(), va);
  upload((*b)->tensor(), vb);
  REQUIRE(loaded.backend->kernels()->rife_input((*a)->tensor(), (*b)->tensor(), 0.25F, (*input)->tensor()));
  const auto got = download((*input)->tensor());
  const int pw = g.padded_width;
  const int ph = g.padded_height;
  const auto plane = [&](int c, int x, int y) {
    return got[static_cast<std::size_t>((c * ph + y) * pw + x)];
  };
  const float tol = 1e-3F;
  CHECK(std::fabs(plane(0, 5, g.top + 3) - va[static_cast<std::size_t>(3 * w + 5)]) < tol);
  CHECK(std::fabs(plane(5, 7, g.top) - vb[static_cast<std::size_t>(2 * h * w + 7)]) < tol);
  CHECK(plane(1, 5, 0) == 0.0F);       // noir de studio au-dessus
  CHECK(plane(4, 5, ph - 1) == 0.0F);  // et en dessous
  CHECK(std::fabs(plane(6, 9, 9) - 0.25F) < tol);
  CHECK(std::fabs(plane(7, 0, 0) - kernels::rife_mesh(0, pw)) < tol);
  CHECK(std::fabs(plane(8, 0, ph - 1) - kernels::rife_mesh(ph - 1, ph)) < tol);
  CHECK(std::fabs(plane(9, 3, 3) - kernels::rife_mesh_step(pw)) < 1e-4F);
  CHECK(std::fabs(plane(10, 3, 3) - kernels::rife_mesh_step(ph)) < 1e-4F);

  // Recadrage : les plans 0 à 2 de l'entrée redonnent a, au bit près.
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, ph, pw}});
  REQUIRE(rgb);
  REQUIRE(cudaMemcpy(reinterpret_cast<void*>((*rgb)->tensor().handle),
                     reinterpret_cast<const void*>((*input)->tensor().handle),
                     static_cast<std::size_t>(3 * ph * pw) * 2, cudaMemcpyDeviceToDevice) == cudaSuccess);
  REQUIRE(loaded.backend->kernels()->rife_crop((*rgb)->tensor(), (*back)->tensor()));
  const auto cropped = download((*back)->tensor());
  double worst = 0.0;
  for (std::size_t i = 0; i < cropped.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::fabs(cropped[i] - kernels::ref::round_to_half(va[i]))));
  }
  CHECK(worst == 0.0);

  // Refus nommés : instant hors de ]0, 1[, tailles qui ne vont pas ensemble.
  CHECK(loaded.backend->kernels()
            ->rife_input((*a)->tensor(), (*b)->tensor(), 1.0F, (*input)->tensor())
            .error()
            .code == RefusalCode::FieldOutOfRange);
  CHECK(loaded.backend->kernels()->rife_crop((*back)->tensor(), (*back)->tensor()).error().code ==
        RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend TensorRT : mesures d'une paire, coupe de scène et pire bloc",
          "[backend_tensorrt][cuda][rife]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 64;
  constexpr int h = 36;
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto b = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(a);
  REQUIRE(b);
  const std::vector<float> gray(static_cast<std::size_t>(3 * w * h), 0.5F);
  upload((*a)->tensor(), gray);
  upload((*b)->tensor(), gray);
  auto same = loaded.backend->kernels()->scores_now((*a)->tensor(), (*b)->tensor());
  REQUIRE(same);
  CHECK(same->identical);
  CHECK(same->worst_block == 0);
  CHECK_FALSE(kernels::is_scene_cut(same->scene_score));

  // Le bloc 32 × 32 en haut à gauche plus clair de 0,25 en R, G et B : luma +0,25.
  std::vector<float> lit = gray;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) lit[static_cast<std::size_t>((c * h + y) * w + x)] = 0.75F;
    }
  }
  upload((*b)->tensor(), lit);
  auto moved = loaded.backend->kernels()->scores_now((*a)->tensor(), (*b)->tensor());
  REQUIRE(moved);
  CHECK_FALSE(moved->identical);
  CHECK(moved->worst_block == static_cast<std::uint32_t>(0.25F * 219.0F * 256.0F));
  const double expected = 32.0 * 32.0 * 0.25 * (219.0 / 255.0) / (64.0 * 64.0);
  CHECK(std::fabs(moved->scene_score - expected) < 1e-4);
}

TEST_CASE("Backend TensorRT : RTX HDR refusé par son nom tant qu'il n'est pas ouvert",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 64, 64}});
  REQUIRE(rgb);
  const RingSlotDesc slot{.index = 0, .generation = 1, .width = 64, .height = 64};
  auto refused = loaded.backend->kernels()->write_ring_true_hdr((*rgb)->tensor(), {}, slot,
                                                                {.wait_value = 0, .signal_value = 1});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::HdrUnavailable);
}

TEST_CASE("Backend TensorRT : le vrai SPAN d'AnimeJaNai, construit puis exécuté",
          "[backend_tensorrt][cuda][span]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const std::filesystem::path model(AJP_SPAN_MODEL);
  if (!std::filesystem::is_regular_file(model)) SKIP("modèle AnimeJaNai absent : " + model.string());
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  constexpr int w = 96;
  constexpr int h = 64;
  gpu::Progress progress;
  const auto sha256 = prepare::sha256_file(model);
  REQUIRE(sha256);
  const gpu::ModelSpec spec{.id = model.stem().string(), .onnx_path = model, .sha256 = *sha256};
  const gpu::Shape in_shape{.dims = {1, 3, h, w}};
  auto artifact = loaded.backend->build(spec, {.min = in_shape, .opt = in_shape, .max = in_shape}, progress);
  REQUIRE(artifact);
  auto engine = loaded.backend->load(*artifact);
  REQUIRE(engine);
  auto input = loaded.backend->alloc_tensor(gpu::TensorType::F16, in_shape);
  auto output = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 2 * h, 2 * w}});
  REQUIRE(input);
  REQUIRE(output);
  // Gris uniforme 0,5 : SPAN le rend presque tel quel.
  std::vector<std::uint16_t> gray(3 * w * h, kernels::ref::float_to_half(0.5F));
  REQUIRE(cudaMemcpy(reinterpret_cast<void*>((*input)->tensor().handle), gray.data(), gray.size() * 2,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  const std::array inputs{(*input)->tensor()};
  std::array outputs{(*output)->tensor()};
  REQUIRE(loaded.backend->enqueue(**engine, inputs, outputs));
  const auto got = download((*output)->tensor());
  REQUIRE(got.size() == static_cast<std::size_t>(3 * 4 * w * h));
  double sum = 0.0;
  for (const float v : got) sum += v;
  CHECK(std::fabs(sum / static_cast<double>(got.size()) - 0.5) < 0.05);
}

}  // namespace
