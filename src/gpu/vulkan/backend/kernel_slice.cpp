// Extensions de la tranche verticale (gpu/kernel_ops.h) sur des tenseurs RGB déjà ingérés :
// entrée de RIFE depuis deux images, puis les deux mesures rendues tout de suite au prix d'une
// attente hôte (mesures de la paire, plus grand déplacement), comme le backend TensorRT ; entrée
// et softmax splatting de GMFSS.
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

std::expected<Buffer*, Refusal> VulkanKernelOps::readback(std::optional<Buffer>& buffer, VkDeviceSize bytes) {
  if (buffer && buffer->size() >= bytes) return &*buffer;
  buffer.reset();  // l'ancien n'est plus lu : chaque appel attend la fin de son exécution
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
  auto vi = tensor_view(input, kernels::kRifeInputChannels, "rife_input : entrée");
  if (!vi) return std::unexpected(vi.error());
  if (va->width != vb->width || va->height != vb->height) {
    return refuse_slice(RefusalCode::ShapeInvalid, "rife_input : deux images RGB de même taille attendues");
  }
  const kernels::RifeGeometry g =
      kernels::rife_geometry(static_cast<int>(va->width), static_cast<int>(va->height));
  if (std::cmp_not_equal(vi->width, g.padded_width) || std::cmp_not_equal(vi->height, g.padded_height)) {
    return refuse_slice(RefusalCode::ShapeInvalid,
                        std::format("rife_input : tenseur 1×{}×{}×{} attendu", kernels::kRifeInputChannels,
                                    g.padded_height, g.padded_width));
  }
  OpKey key;
  key.add(40);
  for (const TensorView* v : {&*va, &*vb, &*vi}) add_view(key, *v);
  key.add(std::bit_cast<std::uint32_t>(t));
  auto rec = recording(key, [&](Recorder& r) -> Done {
    if (auto ok = runner_.rife_frames(r, *va, *vb, *vi, kStudioBlack); !ok) return ok;
    return runner_.rife_constants(r, *vi, t);
  });
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

std::expected<PairScores, Refusal> VulkanKernelOps::scores_now(const Tensor& a, const Tensor& b) {
  auto va = tensor_view(a, 3, "scores_now : image a");
  if (!va) return std::unexpected(va.error());
  auto vb = tensor_view(b, 3, "scores_now : image b");
  if (!vb) return std::unexpected(vb.error());
  if (va->width != vb->width || va->height != vb->height) {
    return refuse_slice(RefusalCode::ShapeInvalid, "scores_now : deux images RGB de même taille attendues");
  }
  const std::uint32_t blocks = KernelRunner::block_count(va->width, va->height);
  auto sums_buffer = readback(pair_sums_, VkDeviceSize{blocks} * sizeof(float));
  if (!sums_buffer) return std::unexpected(sums_buffer.error());
  auto worst_buffer = readback(pair_worst_, sizeof(std::uint32_t));
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
  // Attente hôte voulue : la décision de lancer RIFE dépend de cette paire (kernels/spec/detect.h).
  if (auto ok = context_.wait_on_host(queue_.last()); !ok) return std::unexpected(ok.error());
  std::vector<float> partials(blocks);
  std::memcpy(partials.data(), (*sums_buffer)->mapped(), partials.size() * sizeof(float));
  double total = 0.0;
  for (const float s : partials) total += s;
  std::uint32_t worst_block = 0;
  std::memcpy(&worst_block, (*worst_buffer)->mapped(), sizeof(worst_block));
  const kernels::RifeGeometry g =
      kernels::rife_geometry(static_cast<int>(va->width), static_cast<int>(va->height));
  // Σ |ΔY| × (pas 8 bits / 255) : l'écart en codes rapporté à la crête, sur l'aire remplie de RIFE.
  const double scene = total * (kLumaLevels / 255.0) /
                       (static_cast<double>(g.padded_width) * static_cast<double>(g.padded_height));
  return PairScores{.scene_score = scene, .worst_block = worst_block, .identical = total == 0.0};
}

std::expected<float, Refusal> VulkanKernelOps::flow_max_now(const Motion& motion) {
  auto flow = tensor_view(motion.flow, 4, "flow_max_now : flot");
  if (!flow) return std::unexpected(flow.error());
  auto result_buffer = readback(flow_result_, sizeof(std::uint32_t));
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
  // Attente hôte voulue : relancer RIFE aux autres instants dépend de ce maximum.
  if (auto ok = context_.wait_on_host(queue_.last()); !ok) return std::unexpected(ok.error());
  float value = 0.0F;
  std::memcpy(&value, (*result_buffer)->mapped(), sizeof(value));
  return value;
}

}  // namespace ajp::gpu::vk
