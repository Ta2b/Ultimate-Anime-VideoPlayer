#include "gpu/cpu/cpu_kernels.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <iterator>
#include <string>
#include <utility>

#include "core/thread.h"
#include "kernels/ref/color.h"
#include "kernels/ref/detect.h"
#include "kernels/ref/resample.h"
#include "kernels/ref/rife.h"
#include "kernels/ref/warp.h"

namespace ajp::gpu::cpu {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Une exception (mémoire) ne sort pas du backend : elle devient le refus de l'opération.
template <class Body>
std::expected<void, Refusal> guarded(std::string_view operation, Body&& body) noexcept {
  try {
    return std::forward<Body>(body)();
  } catch (...) {
    return std::unexpected(refusal_from_exception(RefusalCode::ShapeInvalid, operation));
  }
}

// Le résultat d'une fonction de la référence, ou son erreur traduite en refus nommé.
template <class T>
std::expected<T, Refusal> checked(kernels::Result<T> result, std::string_view operation) {
  if (!result) return std::unexpected(refusal_of(result.error(), operation));
  return std::move(*result);
}

}  // namespace

std::expected<void, Refusal> CpuKernels::ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                                const Tensor& rgb) {
  return guarded("ingest", [&] -> std::expected<void, Refusal> {
    if (auto ok = expect_image(rgb, 3, frame.height, frame.width, "ingest"); !ok) return ok;
    auto yuv = read_frame(frame);
    auto color = color_of(frame);
    if (!yuv || !color) return std::unexpected(!yuv ? yuv.error() : color.error());
    kernels::Filter chroma = spec.chroma;
    if (spec.rife_conversion) {
      *color = {.matrix = kernels::kRifeMatrix, .range = color->range, .siting = kernels::kRifeSiting};
      chroma = kernels::kRifeChromaFilter;
    }
    auto image = checked(kernels::ref::yuv_to_rgb(*yuv, *color, chroma), "ingest");
    if (!image) return std::unexpected(std::move(image.error()));
    return write_rgb(*memory_, rgb, *image);
  });
}

std::expected<void, Refusal> CpuKernels::resample(const Tensor& in, kernels::Filter filter,
                                                  const Tensor& out) {
  return guarded("resample", [&] -> std::expected<void, Refusal> {
    auto target = image_shape(out, "resample");
    if (!target) return std::unexpected(std::move(target.error()));
    auto image = read_rgb(*memory_, in);
    if (!image) return std::unexpected(std::move(image.error()));
    auto resized = checked(kernels::ref::resize(*image, target->width, target->height, filter), "resample");
    if (!resized) return std::unexpected(std::move(resized.error()));
    return write_rgb(*memory_, out, *resized);
  });
}

std::expected<void, Refusal> CpuKernels::rife_pad(const DecodedFrame& frame, const Tensor& padded) {
  return guarded("rife_pad", [&] -> std::expected<void, Refusal> {
    const kernels::RifeGeometry geometry = kernels::rife_geometry(frame.width, frame.height);
    if (auto ok = expect_image(padded, 3, geometry.padded_height, geometry.padded_width, "rife_pad"); !ok)
      return ok;
    auto yuv = read_frame(frame);
    auto color = color_of(frame);
    if (!yuv || !color) return std::unexpected(!yuv ? yuv.error() : color.error());
    auto image = checked(kernels::ref::rife_prepare(*yuv, color->range), "rife_pad");
    if (!image) return std::unexpected(std::move(image.error()));
    return write_rgb(*memory_, padded, *image);
  });
}

std::expected<void, Refusal> CpuKernels::rife_crop(const Tensor& padded, const Tensor& out) {
  return guarded("rife_crop", [&] -> std::expected<void, Refusal> {
    auto target = image_shape(out, "rife_crop");
    if (!target) return std::unexpected(std::move(target.error()));
    const kernels::RifeGeometry geometry = kernels::rife_geometry(target->width, target->height);
    if (auto ok = expect_image(padded, 3, geometry.padded_height, geometry.padded_width, "rife_crop"); !ok)
      return ok;
    auto image = read_rgb(*memory_, padded);
    if (!image) return std::unexpected(std::move(image.error()));
    auto cropped = checked(kernels::ref::rife_crop(*image, geometry), "rife_crop");
    if (!cropped) return std::unexpected(std::move(cropped.error()));
    return write_rgb(*memory_, out, *cropped);
  });
}

std::expected<void, Refusal> CpuKernels::warp_blend(const Tensor& a, const Tensor& b, const Motion& motion,
                                                    float t, const Tensor& out) {
  return guarded("warp_blend", [&] -> std::expected<void, Refusal> {
    auto ia = read_rgb(*memory_, a);
    auto ib = read_rgb(*memory_, b);
    auto flow = read_planes(*memory_, motion.flow);
    auto mask = read_planes(*memory_, motion.mask);
    if (!ia || !ib || !flow || !mask) {
      return std::unexpected(!ia ? ia.error() : !ib ? ib.error() : !flow ? flow.error() : mask.error());
    }
    if (flow->size() != 4 || mask->size() != 1) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("warp_blend : {} plans de flot et {} de masque, 4 et 1 attendus",
                                flow->size(), mask->size()));
    }
    if (auto ok = expect_image(out, 3, ia->height(), ia->width(), "warp_blend"); !ok) return ok;
    const kernels::ref::Motion reference{
        .flow = {std::move((*flow)[0]), std::move((*flow)[1]), std::move((*flow)[2]), std::move((*flow)[3])},
        .mask = std::move(mask->front()),
        .area = motion.area};
    auto blended = checked(kernels::ref::warp_blend_at(*ia, *ib, reference, t), "warp_blend");
    if (!blended) return std::unexpected(std::move(blended.error()));
    return write_rgb(*memory_, out, *blended);
  });
}

std::expected<void, Refusal> CpuKernels::write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                                    const RingWrite& sync) {
  return guarded("write_ring", [&] -> std::expected<void, Refusal> {
    if (slot.format != RingFormat::Rgba16Unorm || slot.memory.kind != HandleKind::None || slot.image == 0) {
      return refuse(RefusalCode::FormatUnsupported,
                    std::format("slot {} : le backend CPU n'écrit qu'un slot hôte RGBA16 UNORM", slot.index));
    }
    if (sync.signal_value <= sync.wait_value || sync.signal_value % 2 == 0) {
      return refuse(RefusalCode::SequenceOrder,
                    std::format("slot {} : attente {}, signal {} : impair et plus grand attendu", slot.index,
                                sync.wait_value, sync.signal_value));
    }
    auto image = read_rgb(*memory_, rgb);
    if (!image) return std::unexpected(std::move(image.error()));
    const auto row = static_cast<std::uint64_t>(slot.width) * 8U;
    if (image->width() != slot.width || image->height() != slot.height ||
        slot.memory_offset + row * static_cast<std::uint64_t>(slot.height) > slot.memory_size) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("slot {} {}×{} ({} octets) pour une image {}×{}", slot.index, slot.width,
                                slot.height, slot.memory_size, image->width(), image->height()));
    }
    // Le premier appel d'une génération plus récente relâche les précédentes (kernel_ops.h).
    if (slot.generation < newest_generation_) {
      return refuse(RefusalCode::SequenceOrder,
                    std::format("slot {} : génération {} déjà relâchée pour la {}", slot.index,
                                slot.generation, newest_generation_));
    }
    if (slot.generation > newest_generation_) {
      std::erase_if(signals_, [&](const SlotSignal& s) { return s.generation < slot.generation; });
      newest_generation_ = slot.generation;
    }
    auto signal = std::ranges::find_if(signals_, [&](const SlotSignal& s) {
      return s.generation == slot.generation && s.index == slot.index;
    });
    if (signal == signals_.end()) {
      signals_.push_back({.generation = slot.generation, .index = slot.index});
      signal = std::prev(signals_.end());
    }
    if (sync.signal_value <= signal->value) {
      return refuse(RefusalCode::SequenceOrder, std::format("slot {} : signal {} après {}", slot.index,
                                                            sync.signal_value, signal->value));
    }
    auto pixels = checked(kernels::ref::to_rgba16(*image), "write_ring");
    if (!pixels) return std::unexpected(std::move(pixels.error()));
    // Slot hôte : image porte l'adresse de sa mémoire (en-tête de cpu_kernels.h).
    auto* target = reinterpret_cast<std::byte*>(  // NOLINT(performance-no-int-to-ptr)
        static_cast<std::uintptr_t>(slot.image));
    std::memcpy(target + slot.memory_offset, pixels->data(), pixels->size() * sizeof(std::uint16_t));
    signal->value = sync.signal_value;
    return {};
  });
}

std::expected<void, Refusal> CpuKernels::measure(const DecodedFrame& a, const DecodedFrame& b) {
  return guarded("measure", [&] -> std::expected<void, Refusal> {
    auto ya = read_frame(a);
    auto yb = read_frame(b);
    if (!ya || !yb) return std::unexpected(!ya ? ya.error() : yb.error());
    if (ya->format != yb->format || ya->width() != yb->width() || ya->height() != yb->height()) {
      return refuse(
          RefusalCode::ShapeInvalid,
          std::format("measure : images {} et {} de formats ou tailles différents", a.id.value, b.id.value));
    }
    const kernels::RifeGeometry geometry = kernels::rife_geometry(ya->width(), ya->height());
    auto score = checked(kernels::ref::scene_score(ya->y, yb->y, ya->format, geometry), "measure");
    auto worst = checked(kernels::ref::block_diff_max(ya->y, yb->y, ya->format), "measure");
    auto sum = checked(kernels::ref::scene_diff_sum(ya->y, yb->y, ya->format), "measure");
    if (!score || !worst || !sum)
      return std::unexpected(!score ? score.error() : !worst ? worst.error() : sum.error());
    last_scores_ = PairScores{
        .left = a.id, .right = b.id, .scene_score = *score, .worst_block = *worst, .identical = *sum == 0.0};
    return {};
  });
}

std::optional<PairScores> CpuKernels::previous_scores() {
  return std::exchange(last_scores_, std::nullopt);
}

std::expected<void, Refusal> CpuKernels::rife_input(const Tensor& a, const Tensor& b, float t,
                                                    const Tensor& input) {
  return guarded("rife_input", [&] -> std::expected<void, Refusal> {
    auto ia = read_rgb(*memory_, a);
    auto ib = read_rgb(*memory_, b);
    if (!ia || !ib) return std::unexpected(!ia ? ia.error() : ib.error());
    auto planes = checked(kernels::ref::rife_input(*ia, *ib, t), "rife_input");
    if (!planes) return std::unexpected(std::move(planes.error()));
    return write_planes(*memory_, input, *planes);
  });
}

}  // namespace ajp::gpu::cpu
