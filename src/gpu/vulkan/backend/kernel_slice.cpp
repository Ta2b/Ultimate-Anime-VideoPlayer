// Extensions de la tranche verticale (gpu/kernel_ops.h) sur des tenseurs RGB déjà ingérés :
// entrée de RIFE depuis deux images, puis les deux mesures relues par l'hôte (mesures de la paire,
// plus grand déplacement), enfilées sans attente puis attendues seules, comme le backend TensorRT ;
// entrée et softmax splatting de GMFSS.
#include <bit>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"

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

std::expected<void, Refusal> VulkanKernelOps::rife_input(const Tensor& a, const Tensor& b, float t,
                                                         const Tensor& input) {
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse_slice(RefusalCode::FieldOutOfRange,
                        std::format("rife_input : instant {} hors de ]0, 1[", t));
  }
  auto va = tensor_view(a, 3, "rife_input : image a");
  if (!va) return std::unexpected(va.error());
  auto vb = tensor_view(b, 3, "rife_input : image b");
  if (!vb) return std::unexpected(vb.error());
  auto vi = tensor_view(input, 0, "rife_input : entrée");
  if (!vi) return std::unexpected(vi.error());
  if (!kernels::is_rife_input_planes(vi->planes)) {
    return refuse_slice(RefusalCode::ShapeInvalid,
                        std::format("rife_input : {} plans, 11 (v1) ou 7 (v2) attendus", vi->planes));
  }
  if (va->width != vb->width || va->height != vb->height) {
    return refuse_slice(RefusalCode::ShapeInvalid, "rife_input : deux images RGB de même taille attendues");
  }
  if (!kernels::rife_fits(static_cast<int>(va->width), static_cast<int>(va->height),
                          static_cast<int>(vi->width), static_cast<int>(vi->height))) {
    return refuse_slice(RefusalCode::ShapeInvalid,
                        std::format("rife_input : tenseur 1×{}×H×W attendu, multiples de {} d'au moins {}x{}",
                                    vi->planes, kernels::kRifeAlign, va->width, va->height));
  }
  OpKey key;
  key.add(40);
  for (const TensorView* v : {&*va, &*vb, &*vi}) add_view(key, *v);
  key.add(std::bit_cast<std::uint32_t>(t));
  auto rec = recording(key, [&](Recorder& r) -> Done {
    if (auto ok = runner_.rife_frames(r, *va, *vb, *vi, kStudioBlack, vi->planes); !ok) return ok;
    return runner_.rife_constants(r, *vi, t);
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
  key.add(46);
  add_view(key, *vi);
  key.add(std::bit_cast<std::uint32_t>(t));
  auto rec = recording(key, [&](Recorder& r) { return runner_.rife_constants(r, *vi, t); });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<void, Refusal> VulkanKernelOps::gmfss_frames(const Tensor& a, const Tensor& b,
                                                           const Tensor& frames) {
  constexpr auto planes = static_cast<std::uint32_t>(kernels::kGmfssFramePlanes);
  auto va = tensor_view(a, 3, "gmfss_frames : image a");
  if (!va) return std::unexpected(va.error());
  auto vb = tensor_view(b, 3, "gmfss_frames : image b");
  if (!vb) return std::unexpected(vb.error());
  auto vf = tensor_view(frames, planes, "gmfss_frames : entrée");
  if (!vf) return std::unexpected(vf.error());
  OpKey key;
  key.add(43);
  for (const TensorView* v : {&*va, &*vb, &*vf}) add_view(key, *v);
  auto rec = recording(
      key, [&](Recorder& r) { return runner_.rife_frames(r, *va, *vb, *vf, kStudioBlack, planes); });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<void, Refusal> VulkanKernelOps::soft_splat(const Tensor& source, const Tensor& motion, float s,
                                                         const Tensor& out) {
  auto vs = tensor_view(source, 0, "soft_splat : source");
  if (!vs) return std::unexpected(vs.error());
  auto vm = tensor_view(motion, kernels::kGmfssMotionChannels, "soft_splat : mouvement");
  if (!vm) return std::unexpected(vm.error());
  auto vo = tensor_view(out, vs->planes, "soft_splat : sortie");
  if (!vo) return std::unexpected(vo.error());
  const VkDeviceSize bytes = VkDeviceSize{vs->planes + 1U} * vs->width * vs->height * sizeof(float);
  if (!splat_sums_ || splat_sums_->size() < bytes) {
    // Les enregistrements faits sur l'ancien tampon ne servent plus (sa poignée est dans leur clé),
    // mais une exécution en cours peut encore l'écrire.
    if (splat_sums_) {
      if (auto ok = context_.wait_on_host(queue_.last()); !ok) return std::unexpected(ok.error());
      splat_sums_.reset();
    }
    auto made = Buffer::create(context_, memory_, bytes, MemoryUse::Device);
    if (!made) return std::unexpected(made.error());
    splat_sums_.emplace(std::move(*made));
  }
  const BufferView sums = splat_sums_->view();
  OpKey key;
  key.add(44);
  for (const TensorView* v : {&*vs, &*vm, &*vo}) add_view(key, *v);
  key.add(std::bit_cast<std::uint32_t>(s)).add(detail::handle_bits(sums.buffer));
  auto rec = recording(key, [&](Recorder& r) { return runner_.soft_splat(r, *vs, *vm, s, sums, *vo); });
  if (!rec) return std::unexpected(rec.error());
  return submit(**rec);
}

std::expected<void, Refusal> VulkanKernelOps::fill(const Tensor& tensor, float value) {
  auto view = tensor_view(tensor, 0, "fill");
  if (!view) return std::unexpected(view.error());
  OpKey key;
  key.add(45);
  add_view(key, *view);
  key.add(std::bit_cast<std::uint32_t>(value));
  auto rec = recording(key, [&](Recorder& r) { return runner_.fill(r, *view, value); });
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
  key.add(41);
  for (const TensorView* v : {&*va, &*vb}) add_view(key, *v);
  key.add(detail::handle_bits(sums.buffer)).add(detail::handle_bits(worst.buffer));
  auto rec = recording(
      key, [&](Recorder& r) { return runner_.rgb_pair_diff(r, *va, *vb, sums, worst, kLumaLevels); });
  if (!rec) return std::unexpected(rec.error());
  if (auto ok = submit(**rec); !ok) return std::unexpected(ok.error());
  const kernels::RifeGeometry g =
      kernels::rife_geometry(static_cast<int>(va->width), static_cast<int>(va->height));
  pair_tickets_ = ticket;
  slot.ticket = ticket;
  slot.value = queue_.last().value;
  slot.blocks = blocks;
  slot.area = static_cast<double>(g.padded_width) * static_cast<double>(g.padded_height);
  return ticket;
}

std::expected<PairScores, Refusal> VulkanKernelOps::scores_take(std::uint64_t ticket) {
  PairRead& slot = pair_reads_[ticket % pair_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket) {
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
  return PairScores{.scene_score = scene, .worst_block = worst_block, .identical = total == 0.0};
}

std::expected<float, Refusal> VulkanKernelOps::flow_max_now(const Motion& motion) {
  return flow_max_start(motion).and_then([&](std::uint64_t ticket) { return flow_max_take(ticket); });
}

std::expected<std::uint64_t, Refusal> VulkanKernelOps::flow_max_start(const Motion& motion) {
  auto flow = tensor_view(motion.flow, 4, "flow_max_start : flot");
  if (!flow) return std::unexpected(flow.error());
  const std::uint64_t ticket = flow_tickets_ + 1;
  FlowRead& slot = flow_reads_[ticket % flow_reads_.size()];
  auto result_buffer = readback(slot.result, sizeof(std::uint32_t), slot.value);
  if (!result_buffer) return std::unexpected(result_buffer.error());
  const BufferView result = (*result_buffer)->view();
  const MotionView mv{.flow = *flow, .mask = *flow, .area = motion.area};
  OpKey key;
  key.add(42);
  add_view(key, *flow);
  key.add(static_cast<std::uint64_t>(motion.area.x)).add(static_cast<std::uint64_t>(motion.area.y));
  key.add(static_cast<std::uint64_t>(motion.area.width)).add(static_cast<std::uint64_t>(motion.area.height));
  key.add(detail::handle_bits(result.buffer));
  auto rec = recording(key, [&](Recorder& r) { return runner_.flow_max(r, mv, result); });
  if (!rec) return std::unexpected(rec.error());
  if (auto ok = submit(**rec); !ok) return std::unexpected(ok.error());
  flow_tickets_ = ticket;
  slot.ticket = ticket;
  slot.value = queue_.last().value;
  return ticket;
}

std::expected<float, Refusal> VulkanKernelOps::flow_max_take(std::uint64_t ticket) {
  FlowRead& slot = flow_reads_[ticket % flow_reads_.size()];
  if (ticket == 0 || slot.ticket != ticket) {
    return refuse_slice(RefusalCode::SequenceOrder, std::format("flow_max_take : ticket {} périmé", ticket));
  }
  // Attente hôte voulue : relancer RIFE aux autres instants dépend de ce maximum.
  if (auto ok = context_.wait_on_host(queue_.at(slot.value)); !ok) return std::unexpected(ok.error());
  slot.ticket = 0;
  float value = 0.0F;
  std::memcpy(&value, slot.result->mapped(), sizeof(value));
  return value;
}

}  // namespace ajp::gpu::vk
