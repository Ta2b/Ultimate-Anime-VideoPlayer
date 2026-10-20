// Images décodées en mémoire Vulkan lues par KernelOps : une VkImage multiplan (NV12, P010, P016)
// remplie plan par plan et tenue par un FrameData factice (sémaphore timeline, lot), lue en
// place quand elle s'échantillonne, recopiée sinon ; mêmes résultats que son double en mémoire
// hôte. Et le mémo de copie : une image déjà dans un tampon ne se recopie pas.
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/sync.h"
#include "kernels/ref/color.h"
#include "kernels/spec/tolerance.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

constexpr VkImageUsageFlags kSampled =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageUsageFlags kCopiedOnly = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageCreateFlags kFlags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;

// Une VkImage multiplan du format, remplie des codes de image, en GENERAL ; nulle si l'appareil
// ne l'échantillonne ni ne la copie.
class PlanarImage {
 public:
  static std::optional<PlanarImage> create(VulkanBackend& backend, const k::YuvImage& image,
                                           VkImageUsageFlags usage) {
    VkContext& context = backend.context();
    const vk::DeviceFunctions& fn = context.fn();
    const vk::detail::PlaneFormats* formats = vk::detail::plane_formats(image.format);
    REQUIRE(formats != nullptr);
    VkFormatProperties properties{};
    context.instance_functions().vkGetPhysicalDeviceFormatProperties(context.physical_device(),
                                                                     formats->image, &properties);
    constexpr VkFormatFeatureFlags kNeeded = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                             VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                             VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((properties.optimalTilingFeatures & kNeeded) != kNeeded) return std::nullopt;
    PlanarImage made(context);
    made.format_ = formats->image;
    made.usage_ = usage;
    const auto w = static_cast<std::uint32_t>(image.width());
    const auto h = static_cast<std::uint32_t>(image.height());
    const VkImageCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                 .flags = kFlags,
                                 .imageType = VK_IMAGE_TYPE_2D,
                                 .format = formats->image,
                                 .extent = {.width = w, .height = h, .depth = 1},
                                 .mipLevels = 1,
                                 .arrayLayers = 1,
                                 .samples = VK_SAMPLE_COUNT_1_BIT,
                                 .tiling = VK_IMAGE_TILING_OPTIMAL,
                                 .usage = usage,
                                 .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    require_ok(vk::check(fn.vkCreateImage(context.device(), &info, nullptr, &made.image_), "vkCreateImage"));
    VkMemoryRequirements requirements{};
    fn.vkGetImageMemoryRequirements(context.device(), made.image_, &requirements);
    auto allocation = backend.memory().allocate(requirements, vk::MemoryUse::Device, vk::Tiling::Optimal);
    REQUIRE(allocation);
    made.allocation_ = std::move(*allocation);
    require_ok(vk::check(fn.vkBindImageMemory(context.device(), made.image_, made.allocation_.memory(),
                                              made.allocation_.offset()),
                         "vkBindImageMemory"));

    // Les deux plans, serrés, dans un tampon d'envoi : luma, puis CbCr entrelacés.
    const k::FormatInfo fi = k::info(image.format);
    const auto bytes = static_cast<std::size_t>(fi.bytes);
    const std::size_t luma_bytes = std::size_t{w} * h * bytes;
    const auto cw = static_cast<std::size_t>(image.u.width);
    const auto ch = static_cast<std::size_t>(image.u.height);
    const std::size_t chroma_bytes = cw * ch * 2 * bytes;
    auto upload =
        vk::Buffer::create(context, backend.memory(), luma_bytes + chroma_bytes, vk::MemoryUse::Upload);
    REQUIRE(upload);
    std::byte* out = upload->mapped();
    const auto put = [&](std::size_t at, std::uint16_t code) {
      out[at] = static_cast<std::byte>(code & 0xFFU);
      if (bytes == 2) out[at + 1] = static_cast<std::byte>(code >> 8U);
    };
    for (int y = 0; y < image.height(); ++y) {
      for (int x = 0; x < image.width(); ++x) {
        put((static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)) * bytes, image.y.at(x, y));
      }
    }
    for (int y = 0; y < image.u.height; ++y) {
      for (int x = 0; x < image.u.width; ++x) {
        const std::size_t at =
            luma_bytes + ((static_cast<std::size_t>(y) * cw + static_cast<std::size_t>(x)) * 2 * bytes);
        put(at, image.u.at(x, y));
        put(at + bytes, image.v.at(x, y));
      }
    }
    const VkImage target = made.image_;
    const VkBuffer source = upload->handle();
    require_ok(context.run_once([&](VkCommandBuffer cb) {
      const VkImageMemoryBarrier to_general{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_GENERAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = target,
          .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}};
      fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                              nullptr, 0, nullptr, 1, &to_general);
      const std::array regions{
          VkBufferImageCopy{.bufferOffset = 0,
                            .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT, .layerCount = 1},
                            .imageExtent = {.width = w, .height = h, .depth = 1}},
          VkBufferImageCopy{.bufferOffset = luma_bytes,
                            .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT, .layerCount = 1},
                            .imageExtent = {.width = static_cast<std::uint32_t>(cw),
                                            .height = static_cast<std::uint32_t>(ch),
                                            .depth = 1}}};
      fn.vkCmdCopyBufferToImage(cb, source, target, VK_IMAGE_LAYOUT_GENERAL,
                                static_cast<std::uint32_t>(regions.size()), regions.data());
    }));
    return made;
  }

  PlanarImage(PlanarImage&& other) noexcept
      : context_(other.context_),
        image_(std::exchange(other.image_, VK_NULL_HANDLE)),
        allocation_(std::move(other.allocation_)),
        format_(other.format_),
        usage_(other.usage_) {}
  PlanarImage& operator=(PlanarImage&&) = delete;
  PlanarImage(const PlanarImage&) = delete;
  PlanarImage& operator=(const PlanarImage&) = delete;
  ~PlanarImage() {
    if (image_ == VK_NULL_HANDLE) return;
    (void)context_->settle();
    context_->fn().vkDestroyImage(context_->device(), image_, nullptr);
  }

  VkImage image() const { return image_; }
  VkFormat format() const { return format_; }
  VkImageUsageFlags usage() const { return usage_; }

 private:
  explicit PlanarImage(vk::VkContext& context) : context_(&context) {}
  vk::VkContext* context_;
  VkImage image_ = VK_NULL_HANDLE;
  vk::Allocation allocation_;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  VkImageUsageFlags usage_ = 0;
};

// Le lot que les images factices partagent, comme un AVHWFramesContext.
struct FakePool {};

// Une image décodée Vulkan factice : son sémaphore et l'état de l'AVVkFrame (valeur,
// disposition) tenus ici.
class VulkanFrame final : public FrameData {
 public:
  VulkanFrame(const PlanarImage& image, vk::TimelineSemaphore semaphore, std::shared_ptr<const FakePool> pool)
      : image_(image.image()),
        format_(image.format()),
        usage_(image.usage()),
        semaphore_(std::move(semaphore)),
        pool_(std::move(pool)) {}

  FrameMemory memory() const override { return FrameMemory::Vulkan; }
  int plane_count() const override { return 2; }
  std::span<const std::byte> host_plane(int /*plane*/) const override { return {}; }
  std::size_t host_stride(int /*plane*/) const override { return 0; }
  std::uintptr_t native_plane(int plane) const override {
    return plane == 0 ? static_cast<std::uintptr_t>(vk::detail::to_bits(image_)) : 0;
  }
  std::size_t native_pitch(int /*plane*/) const override { return 0; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }
  std::expected<VulkanImageAccess, Refusal> begin_vulkan_access() const override {
    ++opened_;
    return VulkanImageAccess{
        .semaphore = static_cast<std::uintptr_t>(vk::detail::to_bits(semaphore_.handle())),
        .wait_value = value_,
        .signal_value = value_ + 1,
        .state = {.layout = layout_, .queue_family = VK_QUEUE_FAMILY_IGNORED},
        .usage = static_cast<std::uint32_t>(usage_),
        .flags = static_cast<std::uint32_t>(kFlags),
        .format = static_cast<std::uint32_t>(format_)};
  }
  void end_vulkan_access(const VulkanImageState& after, bool submitted) const override {
    ++closed_;
    if (!submitted) return;
    ++value_;
    layout_ = after.layout;
  }
  std::uintptr_t native_pool_id() const override { return reinterpret_cast<std::uintptr_t>(pool_.get()); }
  std::shared_ptr<const void> native_pool() const override { return pool_; }

  std::uint64_t value() const { return value_; }  // soumissions qui ont lu l'image
  std::uint64_t opened() const { return opened_; }
  std::uint64_t closed() const { return closed_; }
  const vk::TimelineSemaphore& semaphore() const { return semaphore_; }

 private:
  VkImage image_;
  VkFormat format_;
  VkImageUsageFlags usage_;
  vk::TimelineSemaphore semaphore_;
  std::shared_ptr<const FakePool> pool_;
  mutable std::uint64_t value_ = 0;
  mutable std::uint32_t layout_ = VK_IMAGE_LAYOUT_GENERAL;
  mutable std::uint64_t opened_ = 0;
  mutable std::uint64_t closed_ = 0;
};

std::shared_ptr<const VulkanFrame> vulkan_data(VulkanBackend& backend, const PlanarImage& image,
                                               std::shared_ptr<const FakePool> pool) {
  auto semaphore = vk::TimelineSemaphore::create(backend.context(), 0);
  REQUIRE(semaphore);
  return std::make_shared<const VulkanFrame>(image, std::move(*semaphore), std::move(pool));
}

DecodedFrame vulkan_frame(const k::YuvImage& image, std::shared_ptr<const VulkanFrame> data,
                          const ColorInfo& color, std::uint32_t id) {
  return DecodedFrame{.id = SourceId{id},
                      .width = image.width(),
                      .height = image.height(),
                      .format = pixel_format(image.format),
                      .color = color,
                      .pts_us = 0,
                      .data = std::move(data)};
}

k::RgbImage download(VulkanBackend& backend, const TensorBuffer& t, int w, int h) {
  return rgb_from(read_tensor(backend, t.tensor(), 4 * 3 * static_cast<std::size_t>(w * h)), w, h,
                  TensorType::F32);
}

// Tout ce que les noyaux font d'une image décodée, depuis frame et depuis son double hôte : mêmes
// tenseurs et mêmes mesures.
void same_as_host(VulkanBackend& backend, vk::VulkanKernelOps& ops, const k::YuvImage& image,
                  const DecodedFrame& frame, const DecodedFrame& other, const DecodedFrame& host,
                  const DecodedFrame& host_other) {
  const int w = image.width();
  const int h = image.height();
  auto rgb = rgb_tensor(backend, w, h, TensorType::F32);
  auto rgb_host = rgb_tensor(backend, w, h, TensorType::F32);
  require_ok(ops.ingest(frame, {}, rgb->tensor()));
  require_ok(ops.ingest(host, {}, rgb_host->tensor()));
  const auto want = k::ref::yuv_to_rgb(image, {.matrix = k::Matrix::Bt709}, IngestSpec{}.chroma);
  REQUIRE(want);
  CHECK(max_abs_diff(download(backend, *rgb, w, h), *want) <= k::tolerance::kRgbF16);
  CHECK(max_abs_diff(download(backend, *rgb, w, h), download(backend, *rgb_host, w, h)) <= 1e-6F);

  const k::RifeGeometry g = k::rife_geometry(w, h);
  auto padded = rgb_tensor(backend, g.padded_width, g.padded_height, TensorType::F32);
  auto padded_host = rgb_tensor(backend, g.padded_width, g.padded_height, TensorType::F32);
  require_ok(ops.rife_pad(frame, padded->tensor()));
  require_ok(ops.rife_pad(host, padded_host->tensor()));
  CHECK(max_abs_diff(download(backend, *padded, g.padded_width, g.padded_height),
                     download(backend, *padded_host, g.padded_width, g.padded_height)) <= 1e-6F);

  require_ok(ops.measure(frame, other));
  require_ok(backend.context().wait_on_host(backend.queue().last()));
  const auto scores = ops.previous_scores();
  REQUIRE(scores);
  require_ok(ops.measure(host, host_other));
  require_ok(backend.context().wait_on_host(backend.queue().last()));
  const auto scores_host = ops.previous_scores();
  REQUIRE(scores_host);
  CHECK(scores->scene_score == scores_host->scene_score);
  CHECK(scores->worst_block == scores_host->worst_block);
  CHECK(scores->identical == scores_host->identical);
}

}  // namespace

TEST_CASE("KernelOps : les plans d'une image décodée se lisent en place", "[vk_backend][kernels]") {
  TempDir dir("plans");
  auto backend = make_backend(dir.path());
  auto& ops = dynamic_cast<vk::VulkanKernelOps&>(*backend->kernels());
  const ColorInfo color{
      .matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  std::size_t tested = 0;
  for (const k::Format f : {k::Format::Nv12, k::Format::P010, k::Format::P016}) {
    // Un lot par format, comme un AVHWFramesContext par flux : les images d'un lot vivent tant
    // que lui (AVBufferPool), et un lot neuf rend les vues de l'ancien. Un même lot dont les
    // images disparaissent recyclerait leurs poignées sous des vues gardées.
    auto pool = std::make_shared<const FakePool>();
    const k::YuvImage a = synthetic_yuv(f, 70, 38, 41);
    const k::YuvImage b = synthetic_yuv(f, 70, 38, 42);
    auto image_a = PlanarImage::create(*backend, a, kSampled);
    auto image_b = PlanarImage::create(*backend, b, kSampled);
    if (!image_a || !image_b) continue;  // format multiplan hors de l'appareil
    ++tested;
    auto data_a = vulkan_data(*backend, *image_a, pool);
    auto data_b = vulkan_data(*backend, *image_b, pool);
    const DecodedFrame frame_a = vulkan_frame(a, data_a, color, 1);
    const DecodedFrame frame_b = vulkan_frame(b, data_b, color, 2);
    same_as_host(*backend, ops, a, frame_a, frame_b, host_frame(a, color, 3), host_frame(b, color, 4));
    // Deux images lues en place, chacune par ses deux vues, créées une fois (celles du lot
    // précédent rendues) ; chaque lecture a attendu et signalé le sémaphore de l'image, sans la
    // copier dans un tampon.
    CHECK(ops.plane_views() == 2);
    CHECK(data_a->value() == 3);  // ingestion, remplissage RIFE, mesure
    CHECK(data_b->value() == 1);
    CHECK(data_a->opened() == data_a->closed());
    require_ok(backend->context().wait_on_host(backend->queue().last()));
    const auto signalled = data_a->semaphore().value();
    REQUIRE(signalled);
    CHECK(*signalled == data_a->value());
    // Fin de vie des images : elles ne sont plus retenues une fois la file passée.
    require_ok(ops.fill(rgb_tensor(*backend, 8, 8, TensorType::F32)->tensor(), 0.0F));
    require_ok(backend->context().wait_on_host(backend->queue().last()));
  }
  if (tested == 0) SKIP("aucun format multiplan échantillonnable sur cet appareil");
}

TEST_CASE("KernelOps : une image décodée Vulkan sans échantillonnage se copie une fois",
          "[vk_backend][kernels]") {
  TempDir dir("copie");
  auto backend = make_backend(dir.path());
  auto& ops = dynamic_cast<vk::VulkanKernelOps&>(*backend->kernels());
  const ColorInfo color{
      .matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  auto pool = std::make_shared<const FakePool>();
  const k::YuvImage a = synthetic_yuv(k::Format::Nv12, 70, 38, 51);
  const k::YuvImage b = synthetic_yuv(k::Format::Nv12, 70, 38, 52);
  auto image_a = PlanarImage::create(*backend, a, kCopiedOnly);
  auto image_b = PlanarImage::create(*backend, b, kCopiedOnly);
  if (!image_a || !image_b) SKIP("NV12 multiplan hors de l'appareil");
  auto data_a = vulkan_data(*backend, *image_a, pool);
  auto data_b = vulkan_data(*backend, *image_b, pool);
  const DecodedFrame frame_a = vulkan_frame(a, data_a, color, 1);
  const DecodedFrame frame_b = vulkan_frame(b, data_b, color, 2);
  same_as_host(*backend, ops, a, frame_a, frame_b, host_frame(a, color, 3), host_frame(b, color, 4));
  // Pas de vue : copiée par le GPU, une seule fois (ingestion) ; le remplissage RIFE et la mesure
  // relisent le tampon (mémo) sans rouvrir l'image.
  CHECK(ops.plane_views() == 0);
  CHECK(data_a->value() == 1);
  CHECK(data_b->value() == 1);
  CHECK(data_a->opened() == 1);
}

TEST_CASE("KernelOps : une image hôte s'envoie une fois pour toutes ses lectures", "[vk_backend][kernels]") {
  TempDir dir("memo");
  auto backend = make_backend(dir.path());
  auto& ops = dynamic_cast<vk::VulkanKernelOps&>(*backend->kernels());
  const ColorInfo color{
      .matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left};
  const k::YuvImage a = synthetic_yuv(k::Format::Nv12, 48, 32, 61);
  const k::YuvImage b = synthetic_yuv(k::Format::Nv12, 48, 32, 62);
  const k::YuvImage c = synthetic_yuv(k::Format::Nv12, 48, 32, 63);
  const DecodedFrame frame_a = host_frame(a, color, 1);
  const DecodedFrame frame_b = host_frame(b, color, 2);
  const DecodedFrame frame_c = host_frame(c, color, 3);
  auto rgb = rgb_tensor(*backend, 48, 32, TensorType::F32);
  const k::RifeGeometry g = k::rife_geometry(48, 32);
  auto padded = rgb_tensor(*backend, g.padded_width, g.padded_height, TensorType::F32);
  // La chaîne d'une image : mesure avec la précédente, remplissage RIFE, ingestion ; chaque image
  // part une fois.
  require_ok(ops.measure(frame_a, frame_b));
  CHECK(ops.uploads_queued() == 2);
  require_ok(ops.rife_pad(frame_b, padded->tensor()));
  require_ok(ops.ingest(frame_b, {}, rgb->tensor()));
  CHECK(ops.uploads_queued() == 2);
  require_ok(ops.measure(frame_b, frame_c));
  CHECK(ops.uploads_queued() == 3);
  require_ok(ops.ingest(frame_c, {}, rgb->tensor()));
  CHECK(ops.uploads_queued() == 3);
  const auto want = k::ref::yuv_to_rgb(c, {.matrix = k::Matrix::Bt709}, IngestSpec{}.chroma);
  REQUIRE(want);
  CHECK(max_abs_diff(download(*backend, *rgb, 48, 32), *want) <= k::tolerance::kRgbF16);
  // Une image relâchée puis une autre à la même adresse : le mémo ne la prend pas pour elle.
  const k::YuvImage d = synthetic_yuv(k::Format::Nv12, 48, 32, 64);
  require_ok(backend->context().wait_on_host(backend->queue().last()));
  require_ok(ops.fill(rgb->tensor(), 0.0F));  // reclaim : les images retenues sont rendues
  const std::uint64_t before = ops.uploads_queued();
  for (int n = 0; n < 4; ++n) {
    const DecodedFrame frame_d = host_frame(d, color, 5);
    require_ok(ops.ingest(frame_d, {}, rgb->tensor()));
    require_ok(backend->context().wait_on_host(backend->queue().last()));
    require_ok(ops.fill(padded->tensor(), 0.0F));
  }
  CHECK(ops.uploads_queued() == before + 4);
  const auto want_d = k::ref::yuv_to_rgb(d, {.matrix = k::Matrix::Bt709}, IngestSpec{}.chroma);
  REQUIRE(want_d);
  CHECK(max_abs_diff(download(*backend, *rgb, 48, 32), *want_d) <= k::tolerance::kRgbF16);
}
