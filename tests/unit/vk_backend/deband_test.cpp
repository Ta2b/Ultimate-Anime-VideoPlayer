// Le débandage de la source sous Vulkan (KernelOps::deband) contre la référence kernels/ref/deband.h : au
// tenseur près là où le décalage tiré est le même, à moins de kDebandSpread ailleurs (sin et cos du pilote).
#include "kernels/ref/deband.h"

#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "kernels/spec/tolerance.h"
#include "net_rig.h"

using namespace kirari;
using namespace kirari::gpu;
using namespace kirari::gpu::vk::test;
namespace k = kirari::kernels;

TEST_CASE("KernelOps : débandage de la source, contre kernels/ref", "[vk_backend][kernels][deband]") {
  TempDir dir("deband");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  std::vector<TensorType> types{TensorType::F32};
  if (backend->profile().target.f16_storage) types.push_back(TensorType::F16);
  constexpr int kW = 80;
  constexpr int kH = 48;
  for (const TensorType type : types) {
    CAPTURE(static_cast<int>(type));
    // Un aplat en paliers fins (lissé) traversé d'un bruit plus fort (gardé).
    k::RgbImage image = synthetic_rgb(kW, kH, 81, 0.0F, 0.004F);
    for (k::PlaneF& p : image.planes)
      for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) p.at(x, y) += 0.4F + (0.002F * static_cast<float>(x / 10));
    image = stored(image, type);
    auto t = rgb_tensor(*backend, kW, kH, type);
    write_tensor(*backend, t->tensor(), rgb_bytes(image, type));
    // Deux fois : l'enregistrement rejoué donne la même image depuis la même source.
    for (int run = 0; run < 2; ++run) {
      if (run == 1) write_tensor(*backend, t->tensor(), rgb_bytes(image, type));
      require_ok(ops.deband(t->tensor()));
      const std::size_t element = type == TensorType::F16 ? 2 : 4;
      const k::RgbImage got =
          rgb_from(read_tensor(*backend, t->tensor(), element * 3 * std::size_t{kW * kH}), kW, kH, type);
      const auto want = k::ref::deband(image);
      REQUIRE(want);
      const k::RgbImage expected = stored(*want, type);
      std::size_t off = 0;
      std::size_t changed = 0;
      for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < got.planes[c].data.size(); ++i) {
          const float diff = std::fabs(got.planes[c].data[i] - expected.planes[c].data[i]);
          CHECK(diff <= k::ref::kDebandSpread + k::tolerance::kRgbF16);
          if (diff > k::tolerance::kRgbF16) ++off;
          if (got.planes[c].data[i] != image.planes[c].data[i]) ++changed;
        }
      }
      const std::size_t total = 3 * std::size_t{kW * kH};
      CHECK(off * 100 <= total);    // au plus 1 % des points tirés autrement
      CHECK(changed * 2 >= total);  // l'aplat est bien débandé
    }
  }
}

TEST_CASE("KernelOps : l'ingestion qui débande rend l'image de l'ingestion puis du débandage",
          "[vk_backend][kernels][deband]") {
  TempDir dir("deband-ingest");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  REQUIRE(ops.debands_on_ingest());
  std::vector<TensorType> types{TensorType::F32};
  if (backend->profile().target.f16_storage) types.push_back(TensorType::F16);
  const ColorInfo color{
      .matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  constexpr int kW = 80;
  constexpr int kH = 48;
  const k::YuvImage yuv = synthetic_yuv(k::Format::Nv12, kW, kH, 83);
  const DecodedFrame frame = host_frame(yuv, color, 1);
  for (const TensorType type : types) {
    CAPTURE(static_cast<int>(type));
    const std::size_t element = type == TensorType::F16 ? 2 : 4;
    // À taille égale, puis réduite de moitié : les deux chemins de l'ingestion.
    for (const int scale : {1, 2}) {
      CAPTURE(scale);
      const int w = kW / scale;
      const int h = kH / scale;
      const std::size_t bytes = element * 3 * std::size_t{static_cast<std::size_t>(w * h)};
      auto separate = rgb_tensor(*backend, w, h, type);
      auto fused = rgb_tensor(*backend, w, h, type);
      require_ok(ops.ingest(frame, {}, separate->tensor()));
      require_ok(ops.deband(separate->tensor()));
      require_ok(ops.ingest(frame, {.deband = true}, fused->tensor()));
      CHECK(read_tensor(*backend, fused->tensor(), bytes) ==
            read_tensor(*backend, separate->tensor(), bytes));
      // Sans débandage, l'ingestion rend toujours l'image non débandée (l'enregistrement est distinct).
      require_ok(ops.ingest(frame, {}, fused->tensor()));
      auto plain = rgb_tensor(*backend, w, h, type);
      require_ok(ops.ingest(frame, {}, plain->tensor()));
      CHECK(read_tensor(*backend, fused->tensor(), bytes) == read_tensor(*backend, plain->tensor(), bytes));
    }
  }
}
