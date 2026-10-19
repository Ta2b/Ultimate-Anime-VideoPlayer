#include "support.h"

#include <cmath>
#include <cstddef>
#include <utility>

#include <libplacebo/vulkan.h>

namespace ajp::render::test {

namespace {

// LeakSanitizer : le chargeur Vulkan décharge l'ICD (lavapipe) quand la dernière instance
// meurt ; ce que l'ICD garde dans ses variables globales paraît alors fuir, sans module pour
// l'attribuer (vu sous ASan avec lavapipe, invisible quand l'ICD reste chargé). Une instance
// tenue jusqu'à la fin du processus garde l'ICD chargé ; elle reste atteignable, donc propre.
void keep_vulkan_driver_loaded() {
  static const pl_vk_inst instance = pl_vk_inst_create(nullptr, &pl_vk_inst_default_params);
  static_cast<void>(instance);
}

}  // namespace

std::optional<RenderDevice> make_device(std::string& reason) {
  keep_vulkan_driver_loaded();
  auto device = RenderDevice::create(RenderDeviceOptions{.allow_software = true});
  if (!device) {
    reason = std::string(name(device.error().code)) + " : " + device.error().detail;
    return std::nullopt;
  }
  return std::move(*device);
}

core::DisplayInfo sdr_display() {
  return core::DisplayInfo{};
}

core::DisplayInfo pq_display(double peak_nits) {
  core::DisplayInfo display;
  display.hdr = true;
  display.space = core::OutputSpace::Pq;
  display.peak_nits = peak_nits;
  display.min_nits = 0.0;
  display.provenance.hdr = core::Provenance::System;
  display.provenance.peak = core::Provenance::System;
  display.provenance.space = core::Provenance::System;
  return display;
}

pl_tex make_texture(pl_gpu gpu, int width, int height, std::span<const std::uint16_t> pixels) {
  pl_tex_params params{};
  params.w = width;
  params.h = height;
  params.format = pl_find_named_fmt(gpu, "rgba16");
  params.sampleable = true;
  params.initial_data = pixels.data();
  params.debug_tag = "test";
  return pl_tex_create(gpu, &params);
}

std::vector<std::uint16_t> solid(int width, int height, float r, float g, float b) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
  const auto code = [](float v) { return static_cast<std::uint16_t>(std::lround(v * 65535.0F)); };
  for (std::size_t i = 0; i < out.size(); i += 4) {
    out[i] = code(r);
    out[i + 1] = code(g);
    out[i + 2] = code(b);
    out[i + 3] = 65535;
  }
  return out;
}

std::vector<std::uint16_t> read_back(OffscreenTarget& target) {
  if (!target.start_readback()) return {};
  while (!target.readback_ready(UINT64_MAX)) {
  }
  auto pixels = target.pixels();
  if (!pixels) return {};
  return {pixels->begin(), pixels->end()};
}

float channel(std::span<const std::uint16_t> pixels, int width, int x, int y, int c) {
  const auto index =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4 +
      static_cast<std::size_t>(c);
  return static_cast<float>(pixels[index]) / 65535.0F;
}

double pq_encode(double nits) {
  constexpr double m1 = 0.1593017578125;
  constexpr double m2 = 78.84375;
  constexpr double c1 = 0.8359375;
  constexpr double c2 = 18.8515625;
  constexpr double c3 = 18.6875;
  const double y = std::pow(nits / 10000.0, m1);
  return std::pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

}  // namespace ajp::render::test
