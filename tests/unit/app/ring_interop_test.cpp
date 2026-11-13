// Étape 5.5c : l'anneau de bout en bout, sur la vraie interopérabilité Vulkan → CUDA. Une rampe ingérée et
// écrite dans un emplacement par le backend TensorRT, relue par libplacebo (pl_tex_download) : au plus un
// code d'écart avec la valeur attendue en RGB10A2. Test caché : il demande une carte NVIDIA et le backend
// construit (AJP_BACKEND_TENSORRT de CMake) :
//   ajp_app_tests "[.ring_interop]"   ou   ctest -L nvidia
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/vulkan.h>

#include "app/play/backend_library.h"
#include "app/play/placebo_impl.h"
#include "core/frame.h"
#include "core/ring.h"
#include "gpu/backend.h"
#include "kernels/ref/half.h"
#include "kernels/spec/color.h"
#include "kernels/spec/ring_pack.h"

namespace ajp::app::play {
namespace {

class HostFrame final : public FrameData {
 public:
  HostFrame(std::vector<std::vector<std::byte>> planes, std::vector<std::size_t> strides)
      : planes_(std::move(planes)), strides_(std::move(strides)) {}
  FrameMemory memory() const override { return FrameMemory::Host; }
  int plane_count() const override { return static_cast<int>(planes_.size()); }
  std::span<const std::byte> host_plane(int plane) const override {
    return planes_[static_cast<std::size_t>(plane)];
  }
  std::size_t host_stride(int plane) const override { return strides_[static_cast<std::size_t>(plane)]; }
  std::uintptr_t native_plane(int /*plane*/) const override { return 0; }
  std::size_t native_pitch(int /*plane*/) const override { return 0; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }

 private:
  std::vector<std::vector<std::byte>> planes_;
  std::vector<std::size_t> strides_;
};

void put16(std::vector<std::byte>& out, std::size_t at, std::uint16_t code) {
  out[2 * at] = static_cast<std::byte>(code & 0xFFU);
  out[2 * at + 1] = static_cast<std::byte>(code >> 8U);
}

// Une rampe de gris en YUV 4:2:0 10 bits, plage limitée : luma de 64 à 940 sur la largeur, chroma neutre.
DecodedFrame ramp(int w, int h) {
  const auto ww = static_cast<std::size_t>(w), hh = static_cast<std::size_t>(h);
  std::vector<std::byte> y(2 * ww * hh), u(2 * (ww / 2) * (hh / 2)), v(u.size());
  for (std::size_t j = 0; j < hh; ++j)
    for (std::size_t i = 0; i < ww; ++i)
      put16(y, j * ww + i, static_cast<std::uint16_t>(64 + (876 * i) / (ww - 1)));
  for (std::size_t i = 0; i < u.size() / 2; ++i) {
    put16(u, i, 512);
    put16(v, i, 512);
  }
  DecodedFrame frame{.width = w, .height = h, .format = PixelFormat::Yuv420p10};
  frame.color = ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited};
  frame.data = std::make_shared<HostFrame>(std::vector{std::move(y), std::move(u), std::move(v)},
                                           std::vector<std::size_t>{2 * ww, ww, ww});
  return frame;
}

}  // namespace

TEST_CASE("Anneau : une rampe écrite par le backend TensorRT se relit par libplacebo à un code près",
          "[.ring_interop][app][ring]") {
  // Le backend construit avec ce binaire (macro de CMake, vide sans CUDA) : pas de variable d'environnement,
  // le test ne se saute plus à tort sur un binaire construit avec CUDA (B40-02).
  const std::string path = AJP_BACKEND_TENSORRT;
  if (path.empty()) SKIP("binaire construit sans CUDA : pas de backend TensorRT");
  const RingFormat wanted = GENERATE(RingFormat::Rgba16Unorm, RingFormat::Rgb10a2Unorm);
  const bool ten_bits = wanted == RingFormat::Rgb10a2Unorm;
  CAPTURE(ten_bits);

  pl_log_params log_params{};
  log_params.log_level = PL_LOG_WARN;
  pl_log log = pl_log_create(PL_API_VER, &log_params);
  pl_vulkan_params params = pl_vulkan_default_params;
  params.allow_software = false;
  pl_vulkan vk = pl_vulkan_create(log, &params);
  if (vk == nullptr) {
    pl_log_destroy(&log);
    SKIP("pas d'appareil Vulkan matériel");
  }
  {
    constexpr int w = 1024, h = 8;
    PlaceboRing ring;
    auto slots = ring.create(vk->gpu, w, h, 1, true, wanted);
    REQUIRE(slots);
    REQUIRE(slots->size() == 1);
    const RingSlotDesc& slot = slots->front();
    if (slot.memory.kind == HandleKind::None) SKIP("l'appareil n'exporte pas la mémoire de l'anneau");
    if (ten_bits && slot.format != RingFormat::Rgb10a2Unorm) SKIP("rgb10a2 n'est pas pris par cet appareil");

    const std::filesystem::path candidates[]{path};
    auto library = BackendLibrary::open(candidates);
    REQUIRE(library);
    Refusal refusal{};
    const gpu::BackendInit init{.cache_dir = std::filesystem::temp_directory_path() / "ajp-ring-interop",
                                .refusal = &refusal};
    auto backend = BackendLibrary::create(*library, init);
    REQUIRE(backend);
    REQUIRE((*backend)->capabilities().ring_rgb10a2);
    gpu::KernelOps& ops = *(*backend)->kernels();
    auto rgb = (*backend)->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
    REQUIRE(rgb);
    const DecodedFrame frame = ramp(w, h);
    REQUIRE(ops.ingest(frame, {}, (*rgb)->tensor()));
    REQUIRE(ops.write_ring((*rgb)->tensor(), slot, {.wait_value = 0, .signal_value = 1}));

    const RingFrame written{.slot = 0, .generation = slot.generation, .ready_value = 1};
    auto tex = ring.acquire(vk->gpu, written);
    REQUIRE(tex);
    const std::size_t texel = ten_bits ? 4 : 8;
    std::vector<std::byte> host(static_cast<std::size_t>(w) * h * texel);
    pl_tex_transfer_params transfer{};
    transfer.tex = *tex;
    transfer.ptr = host.data();
    transfer.row_pitch = static_cast<std::size_t>(w) * texel;
    REQUIRE(pl_tex_download(vk->gpu, &transfer));
    pl_gpu_finish(vk->gpu);

    int worst = 0;
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        // Gris : R = G = B = (Y − 64) / 876, arrondi en FP16 par l'ingestion.
        const float value = kernels::ref::half_to_float(
            kernels::ref::float_to_half(static_cast<float>((876 * i) / (w - 1)) / 876.0F));
        const std::size_t at = (static_cast<std::size_t>(j) * w + static_cast<std::size_t>(i)) * texel;
        int code = 0, want = 0;
        if (ten_bits) {
          std::uint32_t packed = 0;
          std::copy_n(host.data() + at, 4, reinterpret_cast<std::byte*>(&packed));
          code = static_cast<int>(packed & 0x3FFU);
          want = static_cast<int>(kernels::unorm10(value));
          CHECK((packed >> 30U) == 3U);
        } else {
          std::uint16_t red = 0;
          std::copy_n(host.data() + at, 2, reinterpret_cast<std::byte*>(&red));
          code = red;
          want = kernels::unorm16(value);
        }
        worst = std::max(worst, code > want ? code - want : want - code);
      }
    }
    WARN((ten_bits ? "RGB10A2" : "RGBA16") << " : écart maximal " << worst << " code(s)");
    // Un code près en 10 bits ; en 16 bits, l'ingestion (FP32 → FP16) et la conversion de la rampe donnent
    // au plus quelques codes de 65 535.
    CHECK(worst <= (ten_bits ? 1 : 64));
    if (!ten_bits) {
      // Le grade avant le warp : le slot relu par CUDA (read_ring) puis réécrit rend les mêmes codes, au
      // FP16 du tenseur près. libplacebo rend le slot (2), le reprend et le rend prêt à relire (3).
      ring.retire(vk->gpu, written);
      const RingFrame rendered{.slot = 0, .generation = slot.generation, .ready_value = 2};
      REQUIRE(ring.acquire(vk->gpu, rendered));
      ring.retire(vk->gpu, rendered);
      auto back = (*backend)->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
      REQUIRE(back);
      REQUIRE(ops.read_ring(slot, {.wait_value = 3, .signal_value = 4}, (*back)->tensor()));
      REQUIRE(ops.write_ring((*back)->tensor(), slot, {.wait_value = 4, .signal_value = 5}));
      const RingFrame again{.slot = 0, .generation = slot.generation, .ready_value = 5};
      auto tex_again = ring.acquire(vk->gpu, again);
      REQUIRE(tex_again);
      std::vector<std::byte> host_again(host.size());
      transfer.tex = *tex_again;
      transfer.ptr = host_again.data();
      REQUIRE(pl_tex_download(vk->gpu, &transfer));
      pl_gpu_finish(vk->gpu);
      int drift = 0;
      for (std::size_t at = 0; at < host.size(); at += 2) {
        std::uint16_t first = 0, second = 0;
        std::copy_n(host.data() + at, 2, reinterpret_cast<std::byte*>(&first));
        std::copy_n(host_again.data() + at, 2, reinterpret_cast<std::byte*>(&second));
        drift = std::max(drift, std::abs(static_cast<int>(first) - static_cast<int>(second)));
      }
      WARN("RGBA16 relu puis réécrit : écart maximal " << drift << " code(s)");
      CHECK(drift <= 16);  // un demi-ulp FP16 près de 1 : 2^-12 × 65 535
    }
    ring.clear(vk->gpu);
  }
  pl_vulkan_destroy(&vk);
  pl_log_destroy(&log);
}

}  // namespace ajp::app::play
