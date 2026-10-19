// Noyaux de la chaîne sur tenseurs : enregistrements mis en cache, file, rééchantillonnage,
// recadrage RIFE, warp et mélange. Les images décodées sont dans kernel_frames.cpp, le ring
// dans kernel_ring.cpp.
#include "gpu/vulkan/backend/kernel_ops.h"

#include <algorithm>
#include <bit>
#include <format>
#include <type_traits>

#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::vk {

namespace detail {

VkBuffer to_buffer(std::uintptr_t handle) {
  return from_bits<VkBuffer>(handle);
}

std::uint64_t handle_bits(VkBuffer buffer) {
  return to_bits(buffer);
}

std::unexpected<Refusal> contract_refusal(Refusal refusal) {
  // Le socle refuse une taille ou une disposition par InputMismatch ; le contrat de KernelOps
  // les nomme ShapeInvalid.
  if (refusal.code == RefusalCode::InputMismatch) refusal.code = RefusalCode::ShapeInvalid;
  return std::unexpected(std::move(refusal));
}

std::expected<TensorView, Refusal> tensor_view(const Tensor& t, std::uint32_t planes, std::string_view what) {
  if (t.type != TensorType::F16 && t.type != TensorType::F32) {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported, std::format("{} : tenseur ni F16 ni F32", what)});
  }
  const auto& d = t.shape.dims;
  const auto fits = [](std::int64_t v) { return v >= 1 && v <= kernels::kMaxDimension; };
  if (d[0] != 1 || (planes != 0 && d[1] != planes) || d[1] < 1 || d[1] > 64 || !fits(d[2]) || !fits(d[3])) {
    return std::unexpected(Refusal{RefusalCode::ShapeInvalid,
                                   std::format("{} : forme {}x{}x{}x{}, 1 x {} x h x w attendu", what, d[0], d[1],
                                               d[2], d[3], planes)});
  }
  if (t.handle == 0) return std::unexpected(Refusal{RefusalCode::ShapeInvalid, std::format("{} : sans tampon", what)});
  const Precision precision = t.type == TensorType::F16 ? Precision::F16 : Precision::F32;
  const auto w = static_cast<std::uint32_t>(d[3]);
  const auto h = static_cast<std::uint32_t>(d[2]);
  const auto c = static_cast<std::uint32_t>(d[1]);
  return TensorView{.buffer = {.buffer = to_buffer(t.handle), .offset = t.offset, .size = tensor_bytes(w, h, c, precision)},
                    .width = w,
                    .height = h,
                    .planes = c,
                    .precision = precision};
}

void add_view(OpKey& key, const TensorView& v) {
  key.add(handle_bits(v.buffer.buffer)).add(v.buffer.offset).add(v.width).add(v.height);
  key.add(v.planes).add(static_cast<std::uint64_t>(v.precision));
}

}  // namespace detail

using detail::add_view;
using detail::contract_refusal;
using detail::tensor_view;

VulkanKernelOps::VulkanKernelOps(VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines,
                                 BackendQueue& queue)
    : context_(context), memory_(memory), pipelines_(pipelines), queue_(queue), runner_(context, pipelines) {}

VulkanKernelOps::~VulkanKernelOps() {
  // Tout ce que la file a reçu s'achève avant de rendre images, vues et tampons.
  (void)context_.wait_on_host(queue_.last());
  for (const RingView& v : ring_views_) context_.fn().vkDestroyImageView(context_.device(), v.view, nullptr);
}

std::expected<void, Refusal> VulkanKernelOps::submit(Recording& recording, std::span<const TimelinePoint> waits,
                                                     std::span<const TimelinePoint> signals) {
  if (auto ok = queue_.submit(recording, waits, signals); !ok) return ok;
  reclaim();
  return {};
}

void VulkanKernelOps::reclaim() {
  const auto completed = queue_.completed();
  if (!completed) return;
  std::erase_if(retained_, [&](const Retained& r) { return r.until <= *completed; });
  std::erase_if(ring_views_, [&](const RingView& v) {
    if (v.generation >= ring_generation_ || v.last_write > *completed) return false;
    context_.fn().vkDestroyImageView(context_.device(), v.view, nullptr);
    return true;
  });
}

std::expected<void, Refusal> VulkanKernelOps::resample(const Tensor& in, kernels::Filter filter, const Tensor& out) {
  auto src = tensor_view(in, 3, "rééchantillonnage : source");
  if (!src) return std::unexpected(src.error());
  auto dst = tensor_view(out, 3, "rééchantillonnage : cible");
  if (!dst) return std::unexpected(dst.error());
  if (filter == kernels::Filter::Area && (dst->width > src->width || dst->height > src->height)) {
    return std::unexpected(Refusal{RefusalCode::ShapeInvalid, "rééchantillonnage : Area n'agrandit pas"});
  }
  OpKey key;
  key.add(1);
  add_view(key, *src);
  add_view(key, *dst);
  key.add(static_cast<std::uint64_t>(filter));
  auto rec = recording(key, [&](Recorder& r) { return runner_.resize(r, *src, *dst, filter); });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<void, Refusal> VulkanKernelOps::rife_crop(const Tensor& padded, const Tensor& out) {
  auto src = tensor_view(padded, 3, "recadrage RIFE : tenseur rempli");
  if (!src) return std::unexpected(src.error());
  auto dst = tensor_view(out, 3, "recadrage RIFE : sortie");
  if (!dst) return std::unexpected(dst.error());
  const kernels::RifeGeometry g = kernels::rife_geometry(static_cast<int>(dst->width), static_cast<int>(dst->height));
  if (std::cmp_not_equal(src->width, g.padded_width) || std::cmp_not_equal(src->height, g.padded_height)) {
    return std::unexpected(Refusal{RefusalCode::ShapeInvalid,
                                   std::format("recadrage RIFE : {}x{} rempli, {}x{} attendu pour {}x{}", src->width,
                                               src->height, g.padded_width, g.padded_height, dst->width,
                                               dst->height)});
  }
  OpKey key;
  key.add(2);
  add_view(key, *src);
  add_view(key, *dst);
  auto rec = recording(key, [&](Recorder& r) { return runner_.rife_crop(r, *src, *dst); });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<void, Refusal> VulkanKernelOps::warp_blend(const Tensor& a, const Tensor& b, const Motion& motion,
                                                         float t, const Tensor& out) {
  if (!(t > 0.0F && t < 1.0F)) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, std::format("warp : instant {} hors de ]0, 1[", t)});
  }
  auto va = tensor_view(a, 3, "warp : image a");
  if (!va) return std::unexpected(va.error());
  auto vb = tensor_view(b, 3, "warp : image b");
  if (!vb) return std::unexpected(vb.error());
  auto vo = tensor_view(out, 3, "warp : sortie");
  if (!vo) return std::unexpected(vo.error());
  auto flow = tensor_view(motion.flow, 4, "warp : flot");
  if (!flow) return std::unexpected(flow.error());
  auto mask = tensor_view(motion.mask, 1, "warp : masque");
  if (!mask) return std::unexpected(mask.error());
  const MotionView mv{.flow = *flow, .mask = *mask, .area = motion.area};
  OpKey key;
  key.add(3);
  for (const TensorView* v : {&*va, &*vb, &*vo, &*flow, &*mask}) add_view(key, *v);
  key.add(static_cast<std::uint64_t>(motion.area.x)).add(static_cast<std::uint64_t>(motion.area.y));
  key.add(static_cast<std::uint64_t>(motion.area.width)).add(static_cast<std::uint64_t>(motion.area.height));
  key.add(std::bit_cast<std::uint32_t>(t));
  auto rec = recording(key, [&](Recorder& r) { return runner_.warp_blend(r, *va, *vb, mv, t, *vo); });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

}  // namespace ajp::gpu::vk
