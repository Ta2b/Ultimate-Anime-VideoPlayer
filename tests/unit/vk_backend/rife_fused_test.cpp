// RIFE fusionné (kernels/spec/rife_fused.h) dans le backend Vulkan : les trois noyaux de colle GLSL contre la
// référence CPU en double (kernels/ref/rife_fused.h), sur des données aléatoires, pour les deux découpages du
// catalogue (4.25 : 8, 8, 4, 2, 1 ; 4.22 lite : 8, 4, 2, 1) ; et la clé d'artefact du moteur.
#include "kernels/ref/rife_fused.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/backend/rife_fused_engine.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/recording.h"
#include "kernels/ref/half.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/rife_fused.h"
#include "net_rig.h"

namespace ajp::gpu::vk::test {
namespace {

std::vector<double> rounded(const std::vector<double>& values) {
  std::vector<double> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    out[i] = static_cast<double>(
        kernels::ref::half_to_float(kernels::ref::float_to_half(static_cast<float>(values[i]))));
  return out;
}

Buffer half_buffer(NetRig& rig, std::size_t count) {
  return require_value(Buffer::create(*rig.context, *rig.memory, (count * 2 + 3) / 4 * 4, MemoryUse::Device));
}

Buffer upload_half(NetRig& rig, const std::vector<double>& values) {
  Buffer buffer = half_buffer(rig, values.size());
  std::vector<std::byte> bytes((values.size() * 2 + 3) / 4 * 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint16_t h = kernels::ref::float_to_half(static_cast<float>(values[i]));
    std::memcpy(bytes.data() + i * 2, &h, 2);
  }
  require_ok(upload(*rig.context, *rig.memory, buffer, 0, bytes));
  return buffer;
}

std::vector<double> download_half(NetRig& rig, const Buffer& buffer, std::size_t count) {
  std::vector<std::byte> bytes((count * 2 + 3) / 4 * 4);
  require_ok(download(*rig.context, *rig.memory, buffer, 0, bytes));
  std::vector<double> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t h = 0;
    std::memcpy(&h, bytes.data() + i * 2, 2);
    out[i] = static_cast<double>(kernels::ref::half_to_float(h));
  }
  return out;
}

void run(NetRig& rig, const std::function<Done(Recorder&)>& body) {
  Recorder recorder = require_value(Recorder::begin(*rig.context, *rig.memory));
  require_ok(body(recorder));
  std::unique_ptr<Recording> recording = require_value(recorder.finish());
  require_ok(rig.context->submit(*recording));
  require_ok(rig.context->wait_on_host(recording->last_run()));
}

double psnr_of(const std::vector<double>& got, const std::vector<double>& want, double& worst) {
  REQUIRE(got.size() == want.size());
  double square = 0.0;
  double peak = 0.0;
  worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = got[i] - want[i];
    square += d * d;
    worst = std::max(worst, std::abs(d));
    peak = std::max(peak, std::abs(want[i]));
  }
  const double mse = square / static_cast<double>(got.size());
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
}

kernels::ref::RifeFusedState random_state(const kernels::RifeFusedPlan& plan, int h, int w,
                                          std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> signed_unit(-1.0, 1.0);
  const std::size_t area = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  kernels::ref::RifeFusedState state{.plan = plan, .height = h, .width = w};
  state.input.assign(11 * area, 0.0);
  for (std::size_t i = 0; i < 6 * area; ++i) state.input[i] = unit(rng);
  for (std::size_t i = 0; i < area; ++i) state.input[6 * area + i] = 0.5;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
      state.input[7 * area + i] = kernels::rife_mesh(x, w);
      state.input[8 * area + i] = kernels::rife_mesh(y, h);
      state.input[9 * area + i] = kernels::rife_mesh_step(w);
      state.input[10 * area + i] = kernels::rife_mesh_step(h);
    }
  }
  state.input = rounded(state.input);
  state.encode.resize(8 * area);
  for (double& v : state.encode) v = signed_unit(rng);
  state.encode = rounded(state.encode);
  for (int k = 0; k < plan.blocks; ++k) {
    std::vector<double> values(static_cast<std::size_t>(kernels::rife_fused_y(plan, k, h, w).elements()));
    for (double& v : values)
      v = 0.6 * signed_unit(rng);  // des flots de quelques pixels une fois multipliés par d
    state.y.push_back(rounded(values));
  }
  return state;
}

void compare_kernels(NetRig& rig, const kernels::RifeFusedPlan& plan, int h, int w, std::uint32_t seed) {
  const kernels::ref::RifeFusedState state = random_state(plan, h, w, seed);
  const Buffer input = upload_half(rig, state.input);
  const Buffer encode = upload_half(rig, state.encode);
  std::vector<Buffer> y;
  std::vector<Buffer> x;
  std::vector<BufferView> y_views;
  for (int k = 0; k < plan.blocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    y.push_back(upload_half(rig, state.y[i]));
    y_views.push_back(y.back().view());
    x.push_back(half_buffer(rig, static_cast<std::size_t>(kernels::rife_fused_x(plan, k, h, w).elements())));
  }
  INFO("plan " << plan.reduce[0] << "…" << plan.blocks << " blocs, " << w << "x" << h);
  double worst = 0.0;
  const auto check_block = [&](int k) {
    const auto dims = kernels::rife_fused_x(plan, k, h, w);
    const std::vector<double> got =
        download_half(rig, x[static_cast<std::size_t>(k)], static_cast<std::size_t>(dims.elements()));
    const double score = psnr_of(got, kernels::ref::rife_fused_block_input(state, k), worst);
    INFO("X" << k << " : PSNR " << score << " dB, écart maximal " << worst);
    CHECK(score >= 55.0);
  };
  run(rig, [&](Recorder& r) {
    return record_rife_fused_entry(*rig.pipelines, r, plan, h, w, input.view(), encode.view(), x[0].view());
  });
  check_block(0);
  for (int k = 1; k < plan.blocks; ++k) {
    run(rig, [&](Recorder& r) {
      return record_rife_fused_glue(*rig.pipelines, r, plan, h, w, k, input.view(), encode.view(),
                                    x[static_cast<std::size_t>(k)].view(),
                                    std::span<const BufferView>(y_views).first(static_cast<std::size_t>(k)));
    });
    check_block(k);
  }
  const kernels::ref::RifeFusedResult want = kernels::ref::rife_fused_finish(state);
  const auto last = static_cast<std::size_t>(plan.blocks - 1);
  const std::size_t area = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  const Buffer image = half_buffer(rig, 3 * area);
  const Buffer flow = half_buffer(rig, 4 * area);
  const Buffer mask = half_buffer(rig, area);
  run(rig, [&](Recorder& r) {
    return record_rife_fused_finish(*rig.pipelines, r, plan, h, w, false, input.view(), x[last].view(),
                                    y[last].view(), image.view(), image.view());
  });
  double score = psnr_of(download_half(rig, image, want.image.size()), want.image, worst);
  INFO("image : PSNR " << score << " dB, écart maximal " << worst);
  CHECK(score >= 55.0);
  run(rig, [&](Recorder& r) {
    return record_rife_fused_finish(*rig.pipelines, r, plan, h, w, true, input.view(), x[last].view(),
                                    y[last].view(), flow.view(), mask.view());
  });
  score = psnr_of(download_half(rig, flow, want.flow.size()), want.flow, worst);
  INFO("flot : PSNR " << score << " dB, écart maximal " << worst);
  CHECK(score >= 55.0);
  score = psnr_of(download_half(rig, mask, want.mask.size()), want.mask, worst);
  INFO("masque : PSNR " << score << " dB, écart maximal " << worst);
  CHECK(score >= 55.0);
}

}  // namespace

TEST_CASE("RIFE fusionné : les noyaux de colle Vulkan suivent la référence CPU en double",
          "[vk_backend][rife_fused]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  const kernels::RifeFusedPlan v425{.blocks = 5, .reduce = {8, 8, 4, 2, 1}};
  const kernels::RifeFusedPlan v422{.blocks = 4, .reduce = {8, 4, 2, 1}};
  SECTION("4.25, 96x160") {
    compare_kernels(rig, v425, 96, 160, 1);
  }
  SECTION("4.22 lite, 96x160") {
    compare_kernels(rig, v422, 96, 160, 2);
  }
  SECTION("4.25, taille non carrée et lignes impaires de blocs") {
    compare_kernels(rig, v425, 144, 208, 3);
  }
  SECTION("4.22 lite, 64x128") {
    compare_kernels(rig, v422, 64, 128, 4);
  }
  SECTION("4.26 (premier bloc au 1/16), 128x192") {
    compare_kernels(rig, kernels::RifeFusedPlan{.blocks = 5, .reduce = {16, 8, 4, 2, 1}}, 128, 192, 5);
  }
  SECTION("4.25 lite (premier bloc au 1/32), 128x256") {
    compare_kernels(rig, kernels::RifeFusedPlan{.blocks = 5, .reduce = {32, 16, 8, 4, 1}}, 128, 256, 6);
  }
}

TEST_CASE("RIFE fusionné : la clé d'artefact Vulkan se relit", "[vk_backend][rife_fused]") {
  const RifeFusedKey key{
      .height = 1088,
      .width = 1920,
      .motion = true,
      .plan = {.blocks = 4, .reduce = {8, 4, 2, 1}},
      .artifacts = {"C:/cache/vulkan/enc.ajpvk", "/c/b0.ajpvk", "/c/b1.ajpvk", "/c/b2.ajpvk", "/c/b3.ajpvk"}};
  const std::string text = rife_fused_artifact_key(key);
  CHECK(rife_fused_artifact_key_parse(text) == key);
  CHECK_FALSE(rife_fused_artifact_key_parse("0123456789abcdef"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused-vk:1088x1920:image:8,4,2,1:/a|/b"));
  CHECK_FALSE(rife_fused_artifact_key_parse("rife-fused-vk:1000x1920:image:8,4,2,1:/a|/b|/c|/d|/e"));
}

}  // namespace ajp::gpu::vk::test
