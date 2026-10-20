// Noyaux sur une ou deux images décodées : entrée de RIFE (remplissage YUV puis conversion), et
// mesure d'une paire (coupe de scène, dessins tenus) relue sans attente par previous_scores.
#include <cstring>
#include <format>

#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::vk {
namespace {

std::unexpected<Refusal> refuse_pair(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

}  // namespace

std::expected<void, Refusal> VulkanKernelOps::rife_pad(const DecodedFrame& frame, const Tensor& padded) {
  auto dst = detail::tensor_view(padded, 3, "entrée RIFE : tenseur rempli");
  if (!dst) return std::unexpected(dst.error());
  auto format = detail::kernel_format(frame.format);
  if (!format) return std::unexpected(format.error());
  auto color = detail::color_spec(frame.color);
  if (!color) return std::unexpected(color.error());
  const kernels::RifeGeometry g = kernels::rife_geometry(frame.width, frame.height);
  if (std::cmp_not_equal(dst->width, g.padded_width) || std::cmp_not_equal(dst->height, g.padded_height)) {
    return refuse_pair(RefusalCode::ShapeInvalid,
                       std::format("entrée RIFE : tenseur {}x{}, {}x{} attendu pour {}x{}", dst->width,
                                   dst->height, g.padded_width, g.padded_height, frame.width, frame.height));
  }
  auto src = source(frame);
  if (!src) return std::unexpected(src.error());
  detail::AccessGuard pending;
  if (src->access) pending.hold(frame.data.get(), src->access->state);
  // L'image remplie, en YUV, dans un tampon de travail du même format.
  auto pad = staging_for(*format, static_cast<std::uint32_t>(g.padded_width),
                         static_cast<std::uint32_t>(g.padded_height), MemoryUse::Device, src->staging);
  if (!pad) return std::unexpected(pad.error());
  const Staging& p = staging_[*pad];
  const YuvView padded_yuv{
      .buffer = p.buffer.view(), .format = *format, .width = p.width, .height = p.height, .layout = p.layout};
  const ColorSpec rife{.matrix = kernels::kRifeMatrix, .range = color->range, .siting = kernels::kRifeSiting};
  OpKey key;
  key.add(20);
  add_source(key, *src);
  key.add(detail::handle_bits(padded_yuv.buffer.buffer));
  detail::add_view(key, *dst);
  key.add(static_cast<std::uint64_t>(rife.range));
  auto rec = recording(key, [&](Recorder& r) -> Done {
    if (src->access) {
      if (auto ok = r.commands([&](VkCommandBuffer cb) { copy_image(cb, *src); }); !ok) return ok;
    }
    if (auto ok = runner_.rife_pad(r, src->view, padded_yuv); !ok) return ok;
    return runner_.ingest(r, padded_yuv, rife, *dst, kernels::kRifeChromaFilter);
  });
  if (!rec) return std::unexpected(rec.error());
  pending.dismiss();
  const std::array frames{&frame};
  std::array sources{*src};
  if (auto ok = submit_frames(**rec, sources, frames); !ok) return ok;
  staging_[*pad].busy_until = queue_.last().value;
  return {};
}

std::expected<std::size_t, Refusal> VulkanKernelOps::score_slot(std::uint32_t width, std::uint32_t height) {
  const auto completed = queue_.completed();
  if (!completed) return std::unexpected(completed.error());
  const std::uint32_t partials = KernelRunner::scene_partial_count(width, height);
  const std::uint32_t blocks = KernelRunner::block_count(width, height) + 1;
  std::size_t same = 0;
  std::size_t oldest = SIZE_MAX;
  for (std::size_t k = 0; k < scores_.size(); ++k) {
    const ScoreSlot& s = scores_[k];
    if (s.partials.size() != VkDeviceSize{partials} * sizeof(float) ||
        s.blocks.size() != VkDeviceSize{blocks} * sizeof(std::uint32_t)) {
      continue;
    }
    ++same;
    if (s.value <= *completed) return k;
    if (oldest == SIZE_MAX || s.value < scores_[oldest].value) oldest = k;
  }
  if (same >= kScoreSlots && oldest != SIZE_MAX) {
    if (auto ok = context_.wait_on_host(queue_.at(scores_[oldest].value)); !ok)
      return std::unexpected(ok.error());
    return oldest;
  }
  auto p = Buffer::create(context_, memory_, VkDeviceSize{partials} * sizeof(float), MemoryUse::Readback);
  if (!p) return std::unexpected(p.error());
  auto b =
      Buffer::create(context_, memory_, VkDeviceSize{blocks} * sizeof(std::uint32_t), MemoryUse::Readback);
  if (!b) return std::unexpected(b.error());
  scores_.push_back({.partials = std::move(*p), .blocks = std::move(*b)});
  return scores_.size() - 1;
}

std::expected<void, Refusal> VulkanKernelOps::measure(const DecodedFrame& a, const DecodedFrame& b) {
  if (a.format != b.format || a.width != b.width || a.height != b.height) {
    return refuse_pair(RefusalCode::ShapeInvalid,
                       std::format("mesure : {}x{} et {}x{}, même format et même taille attendus", a.width,
                                   a.height, b.width, b.height));
  }
  auto sa = source(a);
  if (!sa) return std::unexpected(sa.error());
  detail::AccessGuard pending;
  if (sa->access) pending.hold(a.data.get(), sa->access->state);
  auto sb = source(b, sa->staging);
  if (!sb) return std::unexpected(sb.error());
  if (sb->access) pending.hold(b.data.get(), sb->access->state);
  const auto w = static_cast<std::uint32_t>(a.width);
  const auto h = static_cast<std::uint32_t>(a.height);
  auto slot = score_slot(w, h);
  if (!slot) return std::unexpected(slot.error());
  const BufferView partials = scores_[*slot].partials.view();
  const BufferView blocks = scores_[*slot].blocks.view();
  OpKey key;
  key.add(30);
  add_source(key, *sa);
  add_source(key, *sb);
  key.add(detail::handle_bits(partials.buffer)).add(detail::handle_bits(blocks.buffer));
  auto rec = recording(key, [&](Recorder& r) -> Done {
    for (const Source* s : {&*sa, &*sb}) {
      if (!s->access) continue;
      if (auto ok = r.commands([&](VkCommandBuffer cb) { copy_image(cb, *s); }); !ok) return ok;
    }
    if (auto ok = runner_.scene_diff(r, sa->view, sb->view, partials); !ok) return ok;
    return runner_.block_diff(r, sa->view, sb->view, blocks);
  });
  if (!rec) return std::unexpected(rec.error());
  pending.dismiss();
  const std::array frames{&a, &b};
  std::array sources{*sa, *sb};
  if (auto ok = submit_frames(**rec, sources, frames); !ok) return ok;
  const kernels::RifeGeometry g = kernels::rife_geometry(a.width, a.height);
  ScoreSlot& s = scores_[*slot];
  s.value = queue_.last().value;
  s.scores = {.left = a.id, .right = b.id};
  s.area = static_cast<std::uint32_t>(g.padded_width * g.padded_height);
  s.count = KernelRunner::scene_partial_count(w, h);
  s.order = ++score_order_;
  return {};
}

std::optional<PairScores> VulkanKernelOps::previous_scores() {
  reclaim();
  const auto completed = queue_.completed();
  if (!completed) return std::nullopt;
  ScoreSlot* newest = nullptr;
  for (ScoreSlot& s : scores_) {
    if (s.order > returned_order_ && s.value != 0 && s.value <= *completed &&
        (newest == nullptr || s.order > newest->order)) {
      newest = &s;
    }
  }
  if (newest == nullptr) return std::nullopt;
  returned_order_ = newest->order;
  const auto* partials = reinterpret_cast<const float*>(newest->partials.mapped());
  const double sum = scene_sum(std::span(partials, newest->count));
  std::uint32_t worst = 0;
  std::memcpy(&worst, newest->blocks.mapped(), sizeof(worst));
  PairScores out = newest->scores;
  out.scene_score = sum / static_cast<double>(newest->area);
  out.worst_block = worst;
  out.identical = sum == 0.0;
  return out;
}

}  // namespace ajp::gpu::vk
