// Extensions de la tranche verticale (gpu/kernel_ops.h) sur des tenseurs RGB déjà ingérés :
// entrée de RIFE depuis deux images, puis les deux mesures relues par l'hôte (mesures de la paire,
// plus grand déplacement), enfilées sans attente puis attendues seules, comme le backend TensorRT.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/moving_area.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::vk {
namespace {

using detail::add_view;
using detail::tensor_view;

// La chaîne ingère en plage limitée : le noir de studio y vaut 0 en RGB, les pas de luma 8 bits
// entre noir et blanc sont 219 (comme tensorrt_rife.cpp).
constexpr float kStudioBlack = 0.0F;
constexpr float kLumaLevels = 219.0F;

std::unexpected<Refusal> refuse_slice(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

// L'écart de chaque bloc comme rgb_pair_diff.comp le prend pour le pire (moyenne × pas × 256), puis la zone
// qui bouge au seuil des dessins tenus (P6.6, P6.18).
core::MovingArea moving_of(std::span<const float> sums, std::uint32_t width, std::uint32_t height) {
  constexpr auto kBlock = static_cast<std::uint32_t>(kernels::kHeldBlock);
  const std::uint32_t bx = (width + kBlock - 1) / kBlock;
  if (bx == 0) return {};
  std::vector<std::uint32_t> deviation(sums.size());
  for (std::size_t i = 0; i < sums.size(); ++i) {
    const auto x = static_cast<std::uint32_t>(i % bx) * kBlock;
    const auto y = static_cast<std::uint32_t>(i / bx) * kBlock;
    const float pixels =
        static_cast<float>((std::min(x + kBlock, width) - x) * (std::min(y + kBlock, height) - y));
    deviation[i] = static_cast<std::uint32_t>(sums[i] / pixels * kLumaLevels * 256.0F);
  }
  return core::moving_area(deviation, bx, core::moving_limit(kernels::kHeldThreshold));
}

}  // namespace

std::expected<Buffer*, Refusal> VulkanKernelOps::readback(std::optional<Buffer>& buffer, VkDeviceSize bytes,
                                                          std::uint64_t busy_until) {
  if (buffer && buffer->size() >= bytes) return &*buffer;
  if (buffer && busy_until != 0) {  // une mesure enfilée l'écrit peut-être encore
    if (auto ok = context_.wait_on_host(queue_.at(busy_until)); !ok) return std::unexpected(ok.error());
  }
  buffer.reset();
  auto made = Buffer::create(context_, memory_, bytes, MemoryUse::Readback);
  if (!made) return std::unexpected(made.error());
  return &buffer.emplace(std::move(*made));
}

namespace {

// Les vues de rife_input et de rife_pair, vérifiées : a, b et l'entrée.
struct RifeViews {
  TensorView a;
  TensorView b;
  TensorView input;
};

std::expected<RifeViews, Refusal> rife_views(const Tensor& a, const Tensor& b, float t, const Tensor& input,
                                             std::string_view op) {
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse_slice(RefusalCode::FieldOutOfRange, std::format("{} : instant {} hors de ]0, 1[", op, t));
  }
  auto va = tensor_view(a, 3, std::format("{} : image a", op));
  if (!va) return std::unexpected(va.error());
  auto vb = tensor_view(b, 3, std::format("{} : image b", op));
  if (!vb) return std::unexpected(vb.error());
  auto vi = tensor_view(input, 0, std::format("{} : entrée", op));
  if (!vi) return std::unexpected(vi.error());
  if (!kernels::is_rife_input_planes(vi->planes)) {
    return refuse_slice(RefusalCode::ShapeInvalid,
                        std::format("{} : {} plans, 11 (v1) ou 7 (v2) attendus", op, vi->planes));
  }
  if (va->width != vb->width || va->height != vb->height) {
    return refuse_slice(RefusalCode::ShapeInvalid,
                        std::format("{} : deux images RGB de même taille attendues", op));
  }
  if (!kernels::rife_fits(static_cast<int>(va->width), static_cast<int>(va->height),
                          static_cast<int>(vi->width), static_cast<int>(vi->height))) {
    return refuse_slice(RefusalCode::ShapeInvalid,
                        std::format("{} : tenseur 1×{}×H×W attendu, multiples de {} d'au moins {}x{}", op,
                                    vi->planes, kernels::kRifeAlign, va->width, va->height));
  }
  return RifeViews{.a = *va, .b = *vb, .input = *vi};
}

}  // namespace

std::expected<void, Refusal> VulkanKernelOps::rife_input(const Tensor& a, const Tensor& b, float t,
                                                         const Tensor& input) {
  auto v = rife_views(a, b, t, input, "rife_input");
  if (!v) return std::unexpected(v.error());
  OpKey key;
  key.add(OpCode::RifeInput);
  for (const TensorView* view : {&v->a, &v->b, &v->input}) add_view(key, *view);
  const std::array time{std::bit_cast<std::uint32_t>(t)};
  auto rec = recording_with(key, time, [&](Recorder& r) -> Done {
    if (auto ok = runner_.rife_frames(r, v->a, v->b, v->input, kStudioBlack, v->input.planes); !ok) return ok;
    return runner_.rife_constants(r, v->input, t);
  });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<void, Refusal> VulkanKernelOps::rife_pair(const Tensor& a, const Tensor& b, float t,
                                                        const Tensor& input) {
  auto v = rife_views(a, b, t, input, "rife_pair");
  if (!v) return std::unexpected(v.error());
  OpKey key;
  key.add(OpCode::RifePair);
  for (const TensorView* view : {&v->a, &v->b, &v->input}) add_view(key, *view);
  const std::array time{std::bit_cast<std::uint32_t>(t)};
  // Les images, puis le plan de l'instant ; le maillage écrit par rife_input reste.
  auto rec = recording_with(key, time, [&](Recorder& r) -> Done {
    if (auto ok = runner_.rife_frames(r, v->a, v->b, v->input, kStudioBlack, v->input.planes); !ok) return ok;
    return runner_.rife_time(r, v->input, t);
  });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<void, Refusal> VulkanKernelOps::rife_time(const Tensor& input, float t) {
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse_slice(RefusalCode::FieldOutOfRange,
                        std::format("rife_time : instant {} hors de ]0, 1[", t));
  }
  auto vi = tensor_view(input, 0, "rife_time : entrée");
  if (!vi) return std::unexpected(vi.error());
  if (!kernels::is_rife_input_planes(vi->planes)) {
    return refuse_slice(RefusalCode::ShapeInvalid,
                        std::format("rife_time : {} plans, 11 (v1) ou 7 (v2) attendus", vi->planes));
  }
  OpKey key;
  key.add(OpCode::RifeTime);
  add_view(key, *vi);
  const std::array time{std::bit_cast<std::uint32_t>(t)};
  auto rec = recording_with(key, time, [&](Recorder& r) { return runner_.rife_time(r, *vi, t); });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<PairScores, Refusal> VulkanKernelOps::scores_now(const Tensor& a, const Tensor& b) {
  return scores_start(a, b).and_then([&](std::uint64_t ticket) { return scores_take(ticket); });
}

std::expected<std::uint64_t, Refusal> VulkanKernelOps::scores_start(const Tensor& a, const Tensor& b) {
  auto va = tensor_view(a, 3, "scores_start : image a");
  if (!va) return std::unexpected(va.error());
  auto vb = tensor_view(b, 3, "scores_start : image b");
  if (!vb) return std::unexpected(vb.error());
  if (va->width != vb->width || va->height != vb->height) {
    return refuse_slice(RefusalCode::ShapeInvalid, "scores_start : deux images RGB de même taille attendues");
  }
  const std::uint64_t ticket = pair_tickets_ + 1;
  PairRead& slot = pair_reads_[ticket % pair_reads_.size()];
  const std::uint32_t blocks = KernelRunner::block_count(va->width, va->height);
  auto sums_buffer = readback(slot.sums, VkDeviceSize{blocks} * sizeof(float), slot.value);
  if (!sums_buffer) return std::unexpected(sums_buffer.error());
  auto worst_buffer = readback(slot.worst, sizeof(std::uint32_t), slot.value);
  if (!worst_buffer) return std::unexpected(worst_buffer.error());
  const BufferView sums = (*sums_buffer)->view();
  const BufferView worst = (*worst_buffer)->view();
  OpKey key;
  key.add(OpCode::ScoresStart);
  for (const TensorView* v : {&*va, &*vb}) add_view(key, *v);
  key.add(detail::handle_bits(sums.buffer)).add(detail::handle_bits(worst.buffer));
  auto rec = recording(
      key, [&](Recorder& r) { return runner_.rgb_pair_diff(r, *va, *vb, sums, worst, kLumaLevels); });
  if (!rec) return std::unexpected(rec.error());
  if (auto ok = submit(**rec); !ok) return std::unexpected(ok.error());
  // Le lot part aussitôt : la mesure tourne pendant que l'hôte prépare la suite, au lieu
  // d'attendre l'attente de scores_take pour être soumise.
  if (auto ok = queue_.flush(); !ok) return std::unexpected(ok.error());
  const kernels::RifeGeometry g =
      kernels::rife_geometry(static_cast<int>(va->width), static_cast<int>(va->height));
  pair_tickets_ = ticket;
  slot.ticket = ticket;
  slot.value = queue_.ticket();
  slot.blocks = blocks;
  slot.width = va->width;
  slot.height = va->height;
  slot.area = static_cast<double>(g.padded_width) * static_cast<double>(g.padded_height);
  return ticket;
}

std::expected<PairScores, Refusal> VulkanKernelOps::scores_take(std::uint64_t ticket) {
  PairRead& slot = pair_reads_[ticket % pair_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket || !slot.sums || !slot.worst) {
    return refuse_slice(RefusalCode::SequenceOrder, std::format("scores_take : ticket {} périmé", ticket));
  }
  // Attente hôte voulue : la décision de lancer RIFE dépend de cette paire (kernels/spec/detect.h).
  // Elle ne couvre que la mesure, pas ce que la file a reçu depuis.
  if (auto ok = context_.wait_on_host(queue_.at(slot.value)); !ok) return std::unexpected(ok.error());
  slot.ticket = 0;
  std::vector<float> partials(slot.blocks);
  std::memcpy(partials.data(), slot.sums->mapped(), partials.size() * sizeof(float));
  double total = 0.0;
  for (const float s : partials) total += s;
  std::uint32_t worst_block = 0;
  std::memcpy(&worst_block, slot.worst->mapped(), sizeof(worst_block));
  // Σ |ΔY| × (pas 8 bits / 255) : l'écart en codes rapporté à la crête, sur l'aire remplie de RIFE.
  const double scene = total * (kLumaLevels / 255.0) / slot.area;
  return PairScores{.scene_score = scene,
                    .worst_block = worst_block,
                    .identical = total == 0.0,
                    .moving = moving_of(partials, slot.width, slot.height)};
}

std::expected<float, Refusal> VulkanKernelOps::flow_max_now(const Motion& motion) {
  return flow_max_start(motion).and_then([&](std::uint64_t ticket) { return flow_max_take(ticket); });
}

std::expected<std::uint64_t, Refusal> VulkanKernelOps::flow_max_start(const Motion& motion) {
  auto flow = tensor_view(motion.flow, 4, "flow_max_start : flot");
  if (!flow) return std::unexpected(flow.error());
  const std::uint64_t ticket = flow_tickets_ + 1;
  FlowRead& slot = flow_reads_[ticket % flow_reads_.size()];
  auto result_buffer = readback(slot.result, kernels::kFlowBins * sizeof(std::uint32_t), slot.value);
  if (!result_buffer) return std::unexpected(result_buffer.error());
  const BufferView result = (*result_buffer)->view();
  const MotionView mv{.flow = *flow, .mask = *flow, .area = motion.area};
  OpKey key;
  key.add(OpCode::FlowMaxStart);
  add_view(key, *flow);
  key.add(static_cast<std::uint64_t>(motion.area.x)).add(static_cast<std::uint64_t>(motion.area.y));
  key.add(static_cast<std::uint64_t>(motion.area.width)).add(static_cast<std::uint64_t>(motion.area.height));
  key.add(detail::handle_bits(result.buffer));
  auto rec = recording(key, [&](Recorder& r) { return runner_.flow_max(r, mv, result); });
  if (!rec) return std::unexpected(rec.error());
  if (auto ok = submit(**rec); !ok) return std::unexpected(ok.error());
  if (auto ok = queue_.flush(); !ok) return std::unexpected(ok.error());  // comme scores_start
  flow_tickets_ = ticket;
  slot.ticket = ticket;
  slot.value = queue_.ticket();
  slot.points =
      static_cast<std::uint64_t>(motion.area.width) * static_cast<std::uint64_t>(motion.area.height);
  return ticket;
}

std::expected<float, Refusal> VulkanKernelOps::flow_max_take(std::uint64_t ticket) {
  FlowRead& slot = flow_reads_[ticket % flow_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket || !slot.result) {
    return refuse_slice(RefusalCode::SequenceOrder, std::format("flow_max_take : ticket {} périmé", ticket));
  }
  // Attente hôte voulue : relancer RIFE aux autres instants dépend de ce maximum.
  if (auto ok = context_.wait_on_host(queue_.at(slot.value)); !ok) return std::unexpected(ok.error());
  slot.ticket = 0;
  std::array<std::uint32_t, kernels::kFlowBins> bins{};
  std::memcpy(bins.data(), slot.result->mapped(), sizeof(bins));
  return kernels::flow_percentile(bins.data(), slot.points);
}

}  // namespace ajp::gpu::vk
