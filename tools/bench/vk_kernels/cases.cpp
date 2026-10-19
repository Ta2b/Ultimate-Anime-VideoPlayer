#include "cases.h"

#include <array>
#include <bit>
#include <format>
#include <utility>

#include "kernels/ref/half.h"
#include "kernels/spec/color.h"
#include "kernels/spec/resample.h"
#include "kernels/spec/rife.h"

namespace ajp::bench {

namespace {

namespace k = ajp::kernels;
using gpu::vk::BufferView;
using gpu::vk::Done;
using gpu::vk::Precision;
using gpu::vk::Recorder;
using gpu::vk::TensorView;
using gpu::vk::YuvView;

struct Size {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

constexpr std::array<Size, 2> kSizes{{{.width = 1920, .height = 1080}, {.width = 3840, .height = 2160}}};
constexpr std::array<Precision, 2> kPrecisions{Precision::F16, Precision::F32};
constexpr gpu::vk::ColorSpec kSdr{
    .matrix = k::Matrix::Bt709, .range = k::Range::Limited, .siting = k::Siting::Left};
constexpr gpu::vk::ColorSpec kHdr{
    .matrix = k::Matrix::Bt2020Ncl, .range = k::Range::Limited, .siting = k::Siting::TopLeft};
constexpr std::uint32_t kGrey = 0x80808080U;  // gris moyen, en 8 comme en 16 bits

std::string size_text(Size s) {
  return std::format("{}x{}", s.width, s.height);
}

std::string precision_text(Precision p) {
  return p == Precision::F16 ? "f16" : "f32";
}

std::uint64_t tensor_size(Size s, std::uint32_t planes, Precision p) {
  return gpu::vk::tensor_bytes(s.width, s.height, planes, p);
}

// Mot de remplissage de 32 bits : un float, ou deux float16, valant value.
std::uint32_t fill_word(float value, Precision p) {
  if (p == Precision::F32) return std::bit_cast<std::uint32_t>(value);
  const std::uint32_t half = k::ref::float_to_half(value);
  return half | (half << 16U);
}

Size padded(Size s) {
  const k::RifeGeometry g = k::rife_geometry(static_cast<int>(s.width), static_cast<int>(s.height));
  return {.width = static_cast<std::uint32_t>(g.padded_width),
          .height = static_cast<std::uint32_t>(g.padded_height)};
}

// Tampons d'un cas, leurs remplissages initiaux, puis l'enregistrement mesuré. Le premier refus
// est gardé : les vues suivantes restent vides et finish le rend sans rien enregistrer.
class Builder {
 public:
  explicit Builder(Rig& rig) : rig_(rig) {}

  BufferView buffer(VkDeviceSize bytes) {
    if (error_) return {};
    auto made = gpu::vk::Buffer::create(rig_.context, rig_.memory, bytes, gpu::vk::MemoryUse::Device);
    if (!made) {
      error_ = made.error();
      return {};
    }
    out_.buffers.push_back(std::move(*made));
    return out_.buffers.back().view();
  }

  TensorView tensor(Size s, std::uint32_t planes, Precision p, std::optional<float> value = std::nullopt) {
    const TensorView t{.buffer = buffer(tensor_size(s, planes, p)),
                       .width = s.width,
                       .height = s.height,
                       .planes = planes,
                       .precision = p};
    if (value) fill(t.buffer, fill_word(*value, p));
    return t;
  }

  YuvView yuv(k::Format format, Size s, std::optional<std::uint32_t> word = std::nullopt) {
    if (error_) return {};
    auto layout = gpu::vk::tight_layout(format, s.width, s.height);
    if (!layout) {
      error_ = layout.error();
      return {};
    }
    const YuvView v{.buffer = buffer(layout->bytes),
                    .format = format,
                    .width = s.width,
                    .height = s.height,
                    .layout = *layout};
    if (word) fill(v.buffer, *word);
    return v;
  }

  const gpu::vk::StorageImage* image(Size s) {
    if (error_) return nullptr;
    auto made = gpu::vk::StorageImage::create(rig_.context, rig_.memory, s.width, s.height);
    if (!made) {
      error_ = made.error();
      return nullptr;
    }
    return &out_.image.emplace(std::move(*made));
  }

  // Remplissage initial : exécuté une fois, avant l'enregistrement mesuré.
  void fill(const BufferView& view, std::uint32_t word) { fills_.emplace_back(view, word); }

  std::expected<Prepared, Refusal> finish(const std::function<Done(Recorder&)>& body, std::uint64_t bytes) {
    if (error_) return std::unexpected(*error_);
    if (!fills_.empty()) {
      auto init = record([this](Recorder& r) -> Done {
        for (const auto& [view, word] : fills_) {
          if (auto ok = r.fill(view, word); !ok) return ok;
        }
        return {};
      });
      if (!init) return std::unexpected(init.error());
      if (auto ok = rig_.context.submit(**init); !ok) return std::unexpected(ok.error());
      if (auto ok = rig_.context.wait_on_host((*init)->last_run()); !ok) return std::unexpected(ok.error());
    }
    auto recording = record(body);
    if (!recording) return std::unexpected(recording.error());
    out_.recording = std::move(*recording);
    out_.bytes = bytes;
    return std::move(out_);
  }

 private:
  std::expected<std::unique_ptr<gpu::vk::Recording>, Refusal> record(
      const std::function<Done(Recorder&)>& body) {
    auto recorder = Recorder::begin(rig_.context, rig_.memory);
    if (!recorder) return std::unexpected(recorder.error());
    if (auto ok = body(*recorder); !ok) return std::unexpected(ok.error());
    return recorder->finish();
  }

  Rig& rig_;
  Prepared out_;
  std::vector<std::pair<BufferView, std::uint32_t>> fills_;
  std::optional<Refusal> error_;
};

using Prepare = std::function<std::expected<Prepared, Refusal>(Rig&)>;

Case make(std::string kernel, std::string size, std::string precision, Prepare prepare) {
  return {.kernel = std::move(kernel),
          .size = std::move(size),
          .precision = std::move(precision),
          .prepare = std::move(prepare)};
}

// Le coût fixe de la mesure : frontières de soumission, attente du sémaphore, barrière finale.
Case floor_case() {
  return make("plancher (remplissage de 4 o)", "-", "-", [](Rig& rig) {
    Builder b(rig);
    const BufferView word = b.buffer(4);
    return b.finish([&](Recorder& r) { return r.fill(word, 0); }, 4);
  });
}

Case ingest_case(k::Format format, const char* name, gpu::vk::ColorSpec color, Size s, Precision p) {
  return make(std::format("ingest {}", name), size_text(s), precision_text(p), [=](Rig& rig) {
    Builder b(rig);
    const YuvView src = b.yuv(format, s, kGrey);
    const TensorView dst = b.tensor(s, 3, p);
    return b.finish([&](Recorder& r) { return rig.kernels.ingest(r, src, color, dst); },
                    src.layout.bytes + tensor_size(s, 3, p));
  });
}

Case output_yuv_case(Size s, Precision p) {
  return make("output_yuv P010", size_text(s), precision_text(p), [=](Rig& rig) {
    Builder b(rig);
    const TensorView src = b.tensor(s, 3, p, 0.5F);
    const YuvView dst = b.yuv(k::Format::P010, s);
    return b.finish([&](Recorder& r) { return rig.kernels.output_yuv(r, src, kHdr, dst); },
                    tensor_size(s, 3, p) + dst.layout.bytes);
  });
}

Case output_rgba16_case(Size s, Precision p) {
  return make("output_rgba16", size_text(s), precision_text(p), [=](Rig& rig) {
    Builder b(rig);
    const TensorView src = b.tensor(s, 3, p, 0.5F);
    const gpu::vk::StorageImage* dst = b.image(s);
    return b.finish([&](Recorder& r) { return rig.kernels.output_rgba16(r, src, *dst); },
                    tensor_size(s, 3, p) + (std::uint64_t{s.width} * s.height * 8U));
  });
}

Case transfer_case(gpu::vk::TransferOp op, const char* name, Size s, Precision p) {
  return make(std::format("transfer {}", name), size_text(s), precision_text(p), [=](Rig& rig) {
    Builder b(rig);
    const TensorView src = b.tensor(s, 3, p, 0.5F);
    const TensorView dst = b.tensor(s, 3, p);
    return b.finish([&](Recorder& r) { return rig.kernels.transfer(r, op, src, dst); },
                    2 * tensor_size(s, 3, p));
  });
}

// Intermédiaire non compté : des lignes float de dst.width × src.height × 3, écrites puis relues.
Case resize_case(Size from, Size to, Precision p) {
  return make("resize Spline36", std::format("{} → {}", size_text(from), size_text(to)), precision_text(p),
              [=](Rig& rig) {
                Builder b(rig);
                const TensorView src = b.tensor(from, 3, p, 0.5F);
                const TensorView dst = b.tensor(to, 3, p);
                return b.finish(
                    [&](Recorder& r) { return rig.kernels.resize(r, src, dst, k::Filter::Spline36); },
                    tensor_size(from, 3, p) + tensor_size(to, 3, p));
              });
}

Case pad_case(Size s) {
  return make("rife_pad NV12", size_text(s), "-", [=](Rig& rig) {
    Builder b(rig);
    const YuvView src = b.yuv(k::Format::Nv12, s, kGrey);
    const YuvView dst = b.yuv(k::Format::Nv12, padded(s));
    return b.finish([&](Recorder& r) { return rig.kernels.rife_pad(r, src, dst); },
                    std::uint64_t{src.layout.bytes} + dst.layout.bytes);
  });
}

// Ne lit que la fenêtre de l'image dans le tenseur rempli.
Case crop_case(Size s, Precision p) {
  return make("rife_crop tenseur", size_text(s), precision_text(p), [=](Rig& rig) {
    Builder b(rig);
    const TensorView src = b.tensor(padded(s), 3, p, 0.5F);
    const TensorView dst = b.tensor(s, 3, p);
    return b.finish([&](Recorder& r) { return rig.kernels.rife_crop(r, src, dst); },
                    2 * tensor_size(s, 3, p));
  });
}

// Écrit les plans 6 à 10 de l'entrée remplie de RIFE, ne lit rien.
Case constants_case(Size s, Precision p) {
  return make("rife_constants", size_text(s), precision_text(p), [=](Rig& rig) {
    Builder b(rig);
    const TensorView input = b.tensor(padded(s), k::kRifeInputChannels, p);
    return b.finish([&](Recorder& r) { return rig.kernels.rife_constants(r, input, 0.5F); },
                    tensor_size(padded(s), 5, p));
  });
}

// Mouvement estimé à la taille motion (rempli au multiple de 64), images à la taille image.
Case warp_case(Size image, Size motion, Precision p) {
  const std::string size = image.width == motion.width
                               ? size_text(image)
                               : std::format("{} (mouvement {})", size_text(image), size_text(motion));
  return make("warp_blend t=0,5", size, precision_text(p), [=](Rig& rig) {
    Builder b(rig);
    const k::RifeGeometry g =
        k::rife_geometry(static_cast<int>(motion.width), static_cast<int>(motion.height));
    const TensorView a = b.tensor(image, 3, p, 0.5F);
    const TensorView c = b.tensor(image, 3, p, 0.25F);
    const TensorView flow = b.tensor(padded(motion), 4, p);
    // f0x, f0y, f1x, f1y : une translation de quelques pixels, en sens opposés vers a et b.
    constexpr std::array<float, 4> kFlow{2.5F, -1.5F, -2.5F, 1.5F};
    const VkDeviceSize plane = tensor_size(padded(motion), 1, p);
    for (std::size_t i = 0; i < kFlow.size(); ++i) {
      b.fill({.buffer = flow.buffer.buffer, .offset = flow.buffer.offset + (i * plane), .size = plane},
             fill_word(kFlow[i], p));
    }
    const TensorView mask = b.tensor(padded(motion), 1, p, 0.5F);
    const TensorView dst = b.tensor(image, 3, p);
    const gpu::vk::MotionView m{
        .flow = flow, .mask = mask, .area = {.x = g.left, .y = g.top, .width = g.width, .height = g.height}};
    // a, b et la sortie, plus les 4 plans de mouvement et le masque dans la fenêtre.
    return b.finish([&](Recorder& r) { return rig.kernels.warp_blend(r, a, c, m, 0.5F, dst); },
                    (3 * tensor_size(image, 3, p)) + tensor_size(motion, 5, p));
  });
}

Case scene_case(Size s) {
  return make("scene_diff NV12", size_text(s), "-", [=](Rig& rig) {
    Builder b(rig);
    const YuvView a = b.yuv(k::Format::Nv12, s, kGrey);
    const YuvView c = b.yuv(k::Format::Nv12, s, 0x7F7F7F7FU);
    const std::uint64_t partials =
        std::uint64_t{gpu::vk::KernelRunner::scene_partial_count(s.width, s.height)} * 4U;
    const BufferView out = b.buffer(partials);
    return b.finish([&](Recorder& r) { return rig.kernels.scene_diff(r, a, c, out); },
                    (2 * std::uint64_t{s.width} * s.height) + partials);
  });
}

Case block_case(Size s) {
  return make("block_diff NV12", size_text(s), "-", [=](Rig& rig) {
    Builder b(rig);
    const YuvView a = b.yuv(k::Format::Nv12, s, kGrey);
    const YuvView c = b.yuv(k::Format::Nv12, s, 0x7F7F7F7FU);
    const std::uint64_t result =
        (std::uint64_t{gpu::vk::KernelRunner::block_count(s.width, s.height)} + 1U) * 4U;
    const BufferView out = b.buffer(result);
    return b.finish([&](Recorder& r) { return rig.kernels.block_diff(r, a, c, out); },
                    (2 * std::uint64_t{s.width} * s.height) + result);
  });
}

}  // namespace

std::vector<Case> catalog() {
  std::vector<Case> cases{floor_case()};
  for (const Size s : kSizes) {
    for (const Precision p : kPrecisions) {
      cases.push_back(ingest_case(k::Format::Nv12, "NV12", kSdr, s, p));
      cases.push_back(ingest_case(k::Format::P010, "P010", kHdr, s, p));
      cases.push_back(output_yuv_case(s, p));
      cases.push_back(output_rgba16_case(s, p));
      cases.push_back(transfer_case(gpu::vk::TransferOp::PqEotf, "PQ EOTF", s, p));
      cases.push_back(transfer_case(gpu::vk::TransferOp::Bt709ToBt2020, "BT.709 → BT.2020", s, p));
      cases.push_back(crop_case(s, p));
      cases.push_back(constants_case(s, p));
      cases.push_back(warp_case(s, s, p));
    }
    cases.push_back(pad_case(s));
    cases.push_back(scene_case(s));
    cases.push_back(block_case(s));
  }
  for (const Precision p : kPrecisions) {
    cases.push_back(resize_case({.width = 1280, .height = 720}, {.width = 2560, .height = 1440}, p));
    cases.push_back(resize_case({.width = 1920, .height = 1080}, {.width = 3840, .height = 2160}, p));
    cases.push_back(warp_case({.width = 2560, .height = 1440}, {.width = 1280, .height = 720}, p));
  }
  return cases;
}

}  // namespace ajp::bench
