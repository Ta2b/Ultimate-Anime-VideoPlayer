// Entrée constante de RIFE, warp et mélange, fonctions de transfert. Les autres noyaux sont dans
// kernels_color.cpp (couleur, rééchantillonnage) et kernels_window.cpp (fenêtres, détections).
#include "gpu/vulkan/compute/kernel_runner.h"

#include <array>
#include <bit>
#include <cstdint>
#include <format>

#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/weights.h"

namespace ajp::gpu::vk {
namespace {

using detail::bytes_of;
using detail::groups_for;

bool is_hlg_display(TransferOp op) {
  return op == TransferOp::HlgOotf;
}

bool is_bt1886(TransferOp op) {
  return op == TransferOp::Bt1886Eotf || op == TransferOp::Bt1886InverseEotf;
}

// La fenêtre du mouvement tient dans son plan ; en 64 bits, x + width ne déborde pas.
Done check_window(const kernels::Window& area, const TensorView& flow) {
  const std::int64_t right = std::int64_t{area.x} + area.width;
  const std::int64_t bottom = std::int64_t{area.y} + area.height;
  if (area.x >= 0 && area.y >= 0 && area.width >= 1 && area.height >= 1 && right <= flow.width &&
      bottom <= flow.height)
    return {};
  return refuse(RefusalCode::InputMismatch,
                std::format("warp : fenêtre {}x{} en ({}, {}) hors du mouvement {}x{}", area.width,
                            area.height, area.x, area.y, flow.width, flow.height));
}

// Les entrées du warp : images a et b, mouvement et masque, mouvement précédent des poses, de même précision.
Done check_warp_inputs(const TensorView& a, const TensorView& b, const MotionView& motion,
                       const PoseView& poses, const Features& enabled) {
  if (auto ok = check_tensor(a, 3, enabled, "warp : image a"); !ok) return ok;
  if (auto ok = check_tensor(b, 3, enabled, "warp : image b"); !ok) return ok;
  if (auto ok = check_tensor(motion.flow, 4, enabled, "warp : mouvement"); !ok) return ok;
  if (auto ok = check_tensor(motion.mask, 1, enabled, "warp : masque"); !ok) return ok;
  if (auto ok = check_same_size(a.width, a.height, b.width, b.height, "warp : images a et b"); !ok) return ok;
  if (auto ok = check_same_size(motion.flow.width, motion.flow.height, motion.mask.width, motion.mask.height,
                                "warp : mouvement et masque");
      !ok) {
    return ok;
  }
  for (const Precision p : {b.precision, motion.flow.precision, motion.mask.precision}) {
    if (p != a.precision)
      return refuse(RefusalCode::InputMismatch, "warp : tenseurs de précisions différentes");
  }
  if (poses.previous != nullptr) {
    const TensorView& p = *poses.previous;
    if (auto ok = check_tensor(p, 4, enabled, "warp : mouvement précédent"); !ok) return ok;
    if (p.width != motion.flow.width || p.height != motion.flow.height || p.precision != a.precision) {
      return refuse(RefusalCode::InputMismatch,
                    "warp : mouvement précédent d'autres dimensions ou précision");
    }
  }
  return check_window(motion.area, motion.flow);
}

// Mouvement d'ensemble (spec/warp_pan.h) : estimé par pan_global.comp dans un tampon de l'enregistrement,
// relu par le warp ; éteint (on faux), l'instant est lié à sa place (jamais lu).
std::expected<BufferView, Refusal> pan_buffer(PipelineCache& pipelines, Recorder& recorder,
                                              const TensorView& a, const TensorView& b,
                                              const BufferView& time, bool on) {
  if (!on) return time;
  auto scratch = recorder.scratch(4 * sizeof(float));
  if (!scratch) return std::unexpected(scratch.error());
  const BufferView pan = *scratch;
  if (a.width < 16 || a.height < 16) {
    if (auto ok = recorder.fill(pan, 0); !ok) return std::unexpected(ok.error());
    return pan;
  }
  const detail::PanPush pan_push{a.width, a.height};
  const std::array pan_buffers{a.buffer, b.buffer, pan};
  if (auto ok = detail::run(pipelines, recorder, "pan_global", a.precision, {}, pan_buffers, {},
                            bytes_of(pan_push), {1, 1, 1});
      !ok) {
    return std::unexpected(ok.error());
  }
  return pan;
}

}  // namespace

Done KernelRunner::rife_constants(Recorder& recorder, const TensorView& input, float t) {
  if (!kernels::is_rife_input_planes(input.planes)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("entrée RIFE : {} plans, 11 ou 7 attendus", input.planes));
  }
  if (auto ok = check_tensor(input, input.planes, context_->caps().enabled, "entrée RIFE"); !ok) return ok;
  if (input.width % kernels::kRifeAlign != 0 || input.height % kernels::kRifeAlign != 0) {
    return refuse(RefusalCode::InputMismatch, std::format("entrée RIFE {}x{} : multiples de {} attendus",
                                                          input.width, input.height, kernels::kRifeAlign));
  }
  if (!(t >= 0.0F && t <= 1.0F)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("instant RIFE {} hors de [0, 1]", t));
  }
  return rife_constant_planes(recorder, input, t, input.planes);
}

Done KernelRunner::rife_constant_planes(Recorder& recorder, const TensorView& input, float t,
                                        std::uint32_t planes) {
  auto time = recorder.parameter(std::bit_cast<std::uint32_t>(t));
  if (!time) return std::unexpected(time.error());
  const detail::RifeConstantsPush push{input.width, input.height, planes};
  const std::array buffers{input.buffer, *time};
  return detail::run(*pipelines_, recorder, "rife_constants", input.precision, {}, buffers, {},
                     bytes_of(push), {groups_for(input.width), groups_for(input.height), 1});
}

Done KernelRunner::rife_time(Recorder& recorder, const TensorView& input, float t) {
  if (!kernels::is_rife_input_planes(input.planes)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("instant RIFE : {} plans, 11 ou 7 attendus", input.planes));
  }
  if (auto ok = check_tensor(input, input.planes, context_->caps().enabled, "instant RIFE"); !ok) return ok;
  if (input.width % kernels::kRifeAlign != 0 || input.height % kernels::kRifeAlign != 0) {
    return refuse(RefusalCode::InputMismatch, std::format("instant RIFE {}x{} : multiples de {} attendus",
                                                          input.width, input.height, kernels::kRifeAlign));
  }
  if (!(t > 0.0F && t < 1.0F)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("instant RIFE {} hors de ]0, 1[", t));
  }
  // Le plan de l'instant seul ; un remplissage de tampon garderait t dans l'enregistrement.
  return rife_constant_planes(recorder, input, t, static_cast<std::uint32_t>(kernels::kRifeV2InputChannels));
}

Done KernelRunner::warp_blend(Recorder& recorder, const TensorView& a, const TensorView& b,
                              const MotionView& motion, float t, const TensorView& dst,
                              const PoseView& poses) {
  if (auto ok = check_tensor(dst, 3, context_->caps().enabled, "warp : sortie"); !ok) return ok;
  if (auto ok = check_same_size(a.width, a.height, dst.width, dst.height, "warp : sortie"); !ok) return ok;
  if (dst.precision != a.precision)
    return refuse(RefusalCode::InputMismatch, "warp : tenseurs de précisions différentes");
  return warp(recorder, a, b, motion, t, &dst, VK_NULL_HANDLE, poses);
}

Done KernelRunner::warp_blend_ring(Recorder& recorder, const TensorView& a, const TensorView& b,
                                   const MotionView& motion, float t, VkImageView slot,
                                   const PoseView& poses) {
  if (slot == VK_NULL_HANDLE) return refuse(RefusalCode::InputMismatch, "warp : slot du ring sans vue");
  return warp(recorder, a, b, motion, t, nullptr, slot, poses);
}

Done KernelRunner::warp(Recorder& recorder, const TensorView& a, const TensorView& b,
                        const MotionView& motion, float t, const TensorView* dst, VkImageView slot,
                        const PoseView& poses) {
  if (!(t > 0.0F && t < 1.0F))
    return refuse(RefusalCode::FieldOutOfRange, std::format("instant {} hors de ]0, 1[", t));
  if (auto ok = check_warp_inputs(a, b, motion, poses, context_->caps().enabled); !ok) return ok;
  const kernels::Window& area = motion.area;
  const detail::WarpPush push{a.width,
                              a.height,
                              motion.flow.width,
                              motion.flow.height,
                              static_cast<std::uint32_t>(area.x),
                              static_cast<std::uint32_t>(area.y),
                              static_cast<std::uint32_t>(area.width),
                              static_cast<std::uint32_t>(area.height),
                              poses.keep ? 1U : 0U,
                              poses.previous != nullptr ? 1U : 0U,
                              motion.pan_lock && !poses.keep ? 1U : 0U,
                              motion.at_instant && !poses.keep ? 1U : 0U};
  const BufferView previous = poses.previous != nullptr ? poses.previous->buffer : motion.flow.buffer;
  auto time = recorder.parameter(std::bit_cast<std::uint32_t>(t));
  if (!time) return std::unexpected(time.error());
  auto pan_view = pan_buffer(*pipelines_, recorder, a, b, *time, push.pan_lock != 0U);
  if (!pan_view) return std::unexpected(pan_view.error());
  const BufferView pan = *pan_view;
  const std::array constants{motion.warp == kernels::RifeWarp::CatmullRom ? 1U : 0U};
  const std::array<std::uint32_t, 3> groups{groups_for(a.width), groups_for(a.height), 1};
  if (dst == nullptr) {  // warp_blend_ring.comp : le slot lié après les tampons
    const std::array buffers{a.buffer, b.buffer, motion.flow.buffer, motion.mask.buffer, previous,
                             *time,    pan};
    const std::array images{slot};
    return detail::run(*pipelines_, recorder, "warp_blend_ring", a.precision, constants, buffers, images,
                       bytes_of(push), groups);
  }
  const std::array buffers{a.buffer, b.buffer, motion.flow.buffer, motion.mask.buffer, dst->buffer, previous,
                           *time,    pan};
  return detail::run(*pipelines_, recorder, "warp_blend", a.precision, constants, buffers, {}, bytes_of(push),
                     groups);
}

Done KernelRunner::transfer(Recorder& recorder, TransferOp op, const TensorView& src, const TensorView& dst,
                            const DisplayParams& display) {
  if (static_cast<int>(op) > static_cast<int>(TransferOp::Bt2020ToBt709)) {
    return refuse(RefusalCode::InputMismatch, std::format("transfert {} inconnu", static_cast<int>(op)));
  }
  const Features& enabled = context_->caps().enabled;
  if (auto ok = check_tensor(src, 3, enabled, "transfert : source"); !ok) return ok;
  if (auto ok = check_tensor(dst, 3, enabled, "transfert : cible"); !ok) return ok;
  if (auto ok = check_same_size(src.width, src.height, dst.width, dst.height, "transfert"); !ok) return ok;
  if (src.precision != dst.precision)
    return refuse(RefusalCode::InputMismatch, "transfert : précisions différentes");
  HlgDisplay hlg;
  if (is_hlg_display(op)) {
    auto checked = hlg_display(display.hlg_peak);
    if (!checked) return std::unexpected(checked.error());
    hlg = *checked;
  }
  Bt1886Display bt1886;
  if (is_bt1886(op)) {
    auto checked = bt1886_display(display.bt1886_white, display.bt1886_black);
    if (!checked) return std::unexpected(checked.error());
    bt1886 = *checked;
  }
  const detail::TransferPush push{src.width, src.height, hlg.peak,     hlg.gamma,
                                  bt1886.a,  bt1886.b,   bt1886.black, bt1886.white};
  const std::array constants{static_cast<std::uint32_t>(op)};
  const std::array buffers{src.buffer, dst.buffer};
  return detail::run(*pipelines_, recorder, "transfer", src.precision, constants, buffers, {}, bytes_of(push),
                     {groups_for(src.width), groups_for(src.height), 1});
}

double scene_sum(std::span<const float> partials) {
  double sum = 0.0;
  for (const float partial : partials) sum += static_cast<double>(partial);
  return sum;
}

}  // namespace ajp::gpu::vk
