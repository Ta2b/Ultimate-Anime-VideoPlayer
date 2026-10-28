// KernelOps du backend Vulkan contre la référence kernels/ref, par l'interface du contrat :
// tenseurs du backend, images décodées en mémoire hôte, slot de ring créé sur le même appareil.
// Tolérances de kernels/spec/tolerance.h ; refus nommés du contrat avant toute soumission.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/image.h"
#include "gpu/vulkan/compute/sync.h"
#include "kernels/ref/color.h"
#include "kernels/ref/detect.h"
#include "kernels/ref/resample.h"
#include "kernels/ref/rife.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/tolerance.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

std::vector<TensorType> types_of(const vk::VulkanBackend& backend) {
  std::vector<TensorType> types{TensorType::F32};
  if (backend.profile().target.f16_storage) types.push_back(TensorType::F16);
  return types;
}

k::RgbImage download(vk::VulkanBackend& backend, const TensorBuffer& t, int w, int h) {
  const std::size_t element = t.tensor().type == TensorType::F16 ? 2 : 4;
  return rgb_from(read_tensor(backend, t.tensor(), element * 3 * static_cast<std::size_t>(w * h)), w, h,
                  t.tensor().type);
}

ColorInfo color_of(ColorMatrix m, ColorRange r, ChromaSiting s) {
  return ColorInfo{.matrix = m, .range = r, .siting = s};
}

k::ref::ColorSpec ref_color(const ColorInfo& c) {
  const k::Matrix m = c.matrix == ColorMatrix::Bt601       ? k::Matrix::Bt601
                      : c.matrix == ColorMatrix::Bt2020Ncl ? k::Matrix::Bt2020Ncl
                                                           : k::Matrix::Bt709;
  const k::Siting s = c.siting == ChromaSiting::Center    ? k::Siting::Center
                      : c.siting == ChromaSiting::TopLeft ? k::Siting::TopLeft
                                                          : k::Siting::Left;
  return {
      .matrix = m, .range = c.range == ColorRange::Full ? k::Range::Full : k::Range::Limited, .siting = s};
}

}  // namespace

TEST_CASE("KernelOps : ingestion d'images hôte contre kernels/ref", "[vk_backend][kernels]") {
  TempDir dir("kernels");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  const std::array<ColorInfo, 3> colors{
      color_of(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left),
      color_of(ColorMatrix::Bt2020Ncl, ColorRange::Full, ChromaSiting::TopLeft),
      color_of(ColorMatrix::Bt601, ColorRange::Limited, ChromaSiting::Center)};
  int n = 0;
  for (const k::Format f : {k::Format::Nv12, k::Format::P010, k::Format::Yuv420p, k::Format::Yuv444p16}) {
    for (const TensorType type : types_of(*backend)) {
      const ColorInfo color = colors[static_cast<std::size_t>(n++) % colors.size()];
      INFO("format " << static_cast<int>(f) << ", type " << static_cast<int>(type));
      const k::YuvImage image = synthetic_yuv(f, 22, 14, static_cast<std::uint32_t>(n));
      const DecodedFrame frame = host_frame(image, color, static_cast<std::uint32_t>(n));
      auto rgb = rgb_tensor(*backend, 22, 14, type);
      for (const bool rife : {false, true}) {
        const IngestSpec spec{.rife_conversion = rife};
        require_ok(ops.ingest(frame, spec, rgb->tensor()));
        k::ref::ColorSpec want_color = ref_color(color);
        k::Filter filter = spec.chroma;
        if (rife) {
          want_color = {.matrix = k::kRifeMatrix, .range = want_color.range, .siting = k::kRifeSiting};
          filter = k::kRifeChromaFilter;
        }
        const auto want = k::ref::yuv_to_rgb(image, want_color, filter);
        REQUIRE(want);
        CHECK(max_abs_diff(download(*backend, *rgb, 22, 14), stored(*want, type)) <= k::tolerance::kRgbF16);
      }
    }
  }
  // Le recadrage hôte n'alloue plus rien au deuxième passage : mêmes enregistrements. La lecture
  // attend la fin du premier passage, sinon sa mémoire de transfert, encore occupée, en ferait
  // prendre une autre (et un autre enregistrement) selon la vitesse de l'appareil.
  const k::YuvImage image = synthetic_yuv(k::Format::Nv12, 22, 14, 99);
  auto rgb = rgb_tensor(*backend, 22, 14, TensorType::F32);
  const DecodedFrame frame = host_frame(image, colors[0], 99);
  require_ok(ops.ingest(frame, {}, rgb->tensor()));
  (void)download(*backend, *rgb, 22, 14);
  const std::size_t before = dynamic_cast<vk::VulkanKernelOps&>(ops).recordings();
  require_ok(ops.ingest(frame, {}, rgb->tensor()));
  CHECK(dynamic_cast<vk::VulkanKernelOps&>(ops).recordings() == before);
}

TEST_CASE("KernelOps : les copies d'images hôte passent par le fil d'envoi", "[vk_backend][kernels]") {
  TempDir dir("envoi");
  auto backend = make_backend(dir.path());
  auto& ops = dynamic_cast<vk::VulkanKernelOps&>(*backend->kernels());
  const ColorInfo color = color_of(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
  auto rgb = rgb_tensor(*backend, 48, 32, TensorType::F32);
  // Chaque ingestion met sa copie en file sans l'attendre ; le lot ne part qu'une fois toutes
  // faites (la lecture le fait partir), et l'image est rendue après sa copie.
  std::weak_ptr<const FrameData> held;
  for (std::uint32_t n = 1; n <= 4; ++n) {
    const k::YuvImage image = synthetic_yuv(k::Format::Nv12, 48, 32, n);
    const DecodedFrame frame = host_frame(image, color, n);
    held = frame.data;
    require_ok(ops.ingest(frame, {}, rgb->tensor()));
    const auto want = k::ref::yuv_to_rgb(image, ref_color(color), IngestSpec{}.chroma);
    REQUIRE(want);
    CHECK(max_abs_diff(download(*backend, *rgb, 48, 32), stored(*want, TensorType::F32)) <=
          k::tolerance::kRgbF16);
    CHECK(ops.uploads_pending() == 0);
  }
  require_ok(backend->context().wait_on_host(backend->queue().last()));
  require_ok(backend->kernels()->fill(rgb->tensor(), 0.0F));  // reclaim : l'image retenue est rendue
  (void)download(*backend, *rgb, 48, 32);
  CHECK(held.expired());
}

TEST_CASE("KernelOps : refus nommés du contrat", "[vk_backend][kernels]") {
  TempDir dir("refus");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  auto small = rgb_tensor(*backend, 16, 8, TensorType::F32);
  auto big = rgb_tensor(*backend, 32, 16, TensorType::F32);
  require_refusal(ops.resample(small->tensor(), k::Filter::Area, big->tensor()), RefusalCode::ShapeInvalid);
  require_refusal(ops.resample({}, k::Filter::Area, {}), RefusalCode::ShapeInvalid);
  Tensor bytes = small->tensor();
  bytes.type = TensorType::U8;
  require_refusal(ops.resample(bytes, k::Filter::Bilinear, big->tensor()), RefusalCode::FormatUnsupported);
  const Motion motion{.flow = big->tensor(), .mask = big->tensor(), .area = {0, 0, 32, 16}};
  require_refusal(ops.warp_blend(big->tensor(), big->tensor(), motion, 1.0F, big->tensor()),
                  RefusalCode::FieldOutOfRange);
  const DecodedFrame frame =
      host_frame(synthetic_yuv(k::Format::Nv12, 18, 10, 3),
                 color_of(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left), 3);
  // Plus petit, le tenseur demande une réduction ; plus grand, il est refusé.
  require_refusal(ops.ingest(frame, {}, big->tensor()), RefusalCode::ShapeInvalid);
  CHECK_FALSE(ops.previous_scores());
}

TEST_CASE("KernelOps : rééchantillonnage, entrée et sortie de RIFE, warp", "[vk_backend][kernels]") {
  TempDir dir("chaine");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  for (const TensorType type : types_of(*backend)) {
    INFO("type " << static_cast<int>(type));
    // Rééchantillonnage Spline36 vers le haut, Area vers le bas.
    const k::RgbImage image = stored(synthetic_rgb(40, 24, 5), type);
    auto in = rgb_tensor(*backend, 40, 24, type);
    write_tensor(*backend, in->tensor(), rgb_bytes(image, type));
    for (const auto& [w, h, filter] :
         {std::tuple{64, 36, k::Filter::Spline36}, std::tuple{20, 12, k::Filter::Area}}) {
      auto out = rgb_tensor(*backend, w, h, type);
      require_ok(ops.resample(in->tensor(), filter, out->tensor()));
      const auto want = k::ref::resize(image, w, h, filter);
      REQUIRE(want);
      CHECK(max_abs_diff(download(*backend, *out, w, h), stored(*want, type)) <= k::tolerance::kRgbF16);
    }

    // Entrée de RIFE : remplissage mod 64, conversion BT.709 ; puis le recadrage inverse.
    const k::YuvImage yuv = synthetic_yuv(k::Format::Nv12, 70, 38, 6);
    const DecodedFrame frame =
        host_frame(yuv, color_of(ColorMatrix::Bt2020Ncl, ColorRange::Full, ChromaSiting::TopLeft), 6);
    const k::RifeGeometry g = k::rife_geometry(70, 38);
    auto padded = rgb_tensor(*backend, g.padded_width, g.padded_height, type);
    require_ok(ops.rife_pad(frame, padded->tensor()));
    const auto prepared = k::ref::rife_prepare(yuv, k::Range::Full);
    REQUIRE(prepared);
    const k::RgbImage got_padded = download(*backend, *padded, g.padded_width, g.padded_height);
    CHECK(max_abs_diff(got_padded, stored(*prepared, type)) <= k::tolerance::kRgbF16);
    auto cropped = rgb_tensor(*backend, 70, 38, type);
    require_ok(ops.rife_crop(padded->tensor(), cropped->tensor()));
    const auto crop = k::ref::rife_crop(got_padded, g);
    REQUIRE(crop);
    CHECK(max_abs_diff(download(*backend, *cropped, 70, 38), *crop) == 0.0F);

    // Warp et mélange : mouvement dans une fenêtre de plans remplis plus grands.
    const k::RgbImage a = stored(synthetic_rgb(24, 16, 7), type);
    const k::RgbImage b = stored(synthetic_rgb(24, 16, 8), type);
    k::ref::Motion motion{.area = {4, 2, 24, 16}};
    for (std::size_t c = 0; c < 4; ++c)
      motion.flow[c] =
          stored(synthetic_rgb(32, 20, 9 + static_cast<std::uint32_t>(c), -3.0F, 3.0F), type).planes[0];
    motion.mask = stored(synthetic_rgb(32, 20, 13), type).planes[0];
    auto ta = rgb_tensor(*backend, 24, 16, type);
    auto tb = rgb_tensor(*backend, 24, 16, type);
    auto flow = rgb_tensor(*backend, 32, 20, type, 4);
    auto mask = rgb_tensor(*backend, 32, 20, type, 1);
    auto out = rgb_tensor(*backend, 24, 16, type);
    write_tensor(*backend, ta->tensor(), rgb_bytes(a, type));
    write_tensor(*backend, tb->tensor(), rgb_bytes(b, type));
    write_tensor(*backend, flow->tensor(), planes_bytes(motion.flow, type));
    write_tensor(*backend, mask->tensor(), planes_bytes(std::span(&motion.mask, 1), type));
    const Motion m{.flow = flow->tensor(), .mask = mask->tensor(), .area = motion.area};
    for (const float t : {0.5F, 0.25F}) {
      require_ok(ops.warp_blend(ta->tensor(), tb->tensor(), m, t, out->tensor()));
      const auto want = t == 0.5F ? k::ref::warp_blend(a, b, motion) : k::ref::warp_blend_at(a, b, motion, t);
      REQUIRE(want);
      CHECK(max_abs_diff(download(*backend, *out, 24, 16), stored(*want, type)) <= k::tolerance::kWarp);
    }
  }
}

TEST_CASE("KernelOps : un tampon détruit fait oublier ses enregistrements", "[vk_backend][kernels]") {
  // Une valeur de poignée se réutilise après destruction : un enregistrement qui liait l'ancien
  // tampon ne se rejoue plus jamais, ses descripteurs portant l'adresse d'une mémoire rendue.
  TempDir dir("oubli");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  auto& vulkan = dynamic_cast<vk::VulkanKernelOps&>(ops);
  const k::RgbImage image = stored(synthetic_rgb(40, 24, 5), TensorType::F32);
  auto in = rgb_tensor(*backend, 40, 24, TensorType::F32);
  write_tensor(*backend, in->tensor(), rgb_bytes(image, TensorType::F32));
  auto kept = rgb_tensor(*backend, 20, 12, TensorType::F32);
  auto dropped = rgb_tensor(*backend, 20, 12, TensorType::F32);
  require_ok(ops.resample(in->tensor(), k::Filter::Area, kept->tensor()));
  require_ok(ops.resample(in->tensor(), k::Filter::Area, dropped->tensor()));
  const std::size_t both = vulkan.recordings();
  dropped.reset();
  // Recréé de même taille : souvent la même poignée, et le résultat doit rester juste.
  auto again = rgb_tensor(*backend, 20, 12, TensorType::F32);
  require_ok(ops.resample(in->tensor(), k::Filter::Area, kept->tensor()));
  CHECK(vulkan.recordings() == both - 1);
  require_ok(ops.resample(in->tensor(), k::Filter::Area, again->tensor()));
  CHECK(vulkan.recordings() == both);
  const auto want = k::ref::resize(image, 20, 12, k::Filter::Area);
  REQUIRE(want);
  CHECK(max_abs_diff(download(*backend, *again, 20, 12), stored(*want, TensorType::F32)) <=
        k::tolerance::kRgbF16);
}

TEST_CASE("KernelOps : écriture d'un slot du ring", "[vk_backend][kernels]") {
  TempDir dir("ring");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  auto image = vk::StorageImage::create(backend->context(), backend->memory(), 20, 12);
  if (!image && image.error().code == RefusalCode::FeatureUnavailable) SKIP(describe(image.error()));
  REQUIRE(image);
  auto semaphore = require_value(vk::TimelineSemaphore::create(backend->context(), 0));
  const auto bits = [](auto handle) { return vk::detail::to_bits(handle); };
  const RingSlotDesc slot{.index = 0,
                          .generation = 1,
                          .width = 20,
                          .height = 12,
                          .image = bits(image->image()),
                          .semaphore = bits(semaphore.handle())};
  for (const TensorType type : types_of(*backend)) {
    const k::RgbImage rgb = stored(synthetic_rgb(20, 12, 21, -0.2F, 1.2F), type);
    auto t = rgb_tensor(*backend, 20, 12, type);
    write_tensor(*backend, t->tensor(), rgb_bytes(rgb, type));
    const std::uint64_t wait = type == TensorType::F32 ? 0 : 2;
    if (wait > 0) require_ok(semaphore.signal(wait));  // render rend le slot
    require_ok(ops.write_ring(t->tensor(), slot, {.wait_value = wait, .signal_value = wait + 1}));
    require_ok(backend->context().wait_on_host(semaphore.at(wait + 1)));
    const auto codes = require_value(vk::read_rgba16(backend->context(), backend->memory(), *image));
    const auto want = k::ref::to_rgba16(rgb);
    REQUIRE(want);
    CHECK(codes == *want);
  }
  // Signal pair ou décroissant, autre taille : refusés.
  auto t = rgb_tensor(*backend, 20, 12, TensorType::F32);
  require_refusal(ops.write_ring(t->tensor(), slot, {.wait_value = 4, .signal_value = 4}),
                  RefusalCode::FieldOutOfRange);
  auto other = rgb_tensor(*backend, 22, 12, TensorType::F32);
  require_refusal(ops.write_ring(other->tensor(), slot, {.wait_value = 4, .signal_value = 5}),
                  RefusalCode::ShapeInvalid);
  // Une génération plus ancienne que la dernière écrite est refusée.
  RingSlotDesc old = slot;
  old.generation = 0;
  require_refusal(ops.write_ring(t->tensor(), old, {.wait_value = 4, .signal_value = 5}),
                  RefusalCode::FieldOutOfRange);
}

TEST_CASE("KernelOps : mesure d'une paire relue sans attente", "[vk_backend][kernels]") {
  TempDir dir("mesure");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  const ColorInfo color = color_of(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
  for (const k::Format f : {k::Format::Nv12, k::Format::P010}) {
    const k::YuvImage a = synthetic_yuv(f, 70, 38, 31);
    const k::YuvImage b = synthetic_yuv(f, 70, 38, 32);
    require_ok(ops.measure(host_frame(a, color, 1), host_frame(b, color, 2)));
    require_ok(backend->context().wait_on_host(backend->queue().last()));
    const auto scores = ops.previous_scores();
    REQUIRE(scores);
    CHECK_FALSE(ops.previous_scores());  // une seule fois
    CHECK(scores->left == SourceId{1});
    CHECK(scores->right == SourceId{2});
    const auto want = k::ref::scene_score(a.y, b.y, f, k::rife_geometry(70, 38));
    REQUIRE(want);
    CHECK(std::fabs(scores->scene_score - *want) <= k::tolerance::kSceneScoreRelative * *want);
    const auto worst = k::ref::block_diff_max(a.y, b.y, f);
    REQUIRE(worst);
    CHECK(scores->worst_block == *worst);
    CHECK_FALSE(scores->identical);

    require_ok(ops.measure(host_frame(a, color, 3), host_frame(a, color, 4)));
    require_ok(backend->context().wait_on_host(backend->queue().last()));
    const auto same = ops.previous_scores();
    REQUIRE(same);
    CHECK(same->identical);
    CHECK(same->scene_score == 0.0);
    CHECK(same->worst_block == 0);
  }
  // Les images lues sont gardées jusqu'à la fin de leur lecture, puis rendues.
  require_ok(backend->context().wait_on_host(backend->queue().last()));
  (void)ops.previous_scores();
  CHECK(dynamic_cast<vk::VulkanKernelOps&>(ops).retained_frames() == 0);
}
