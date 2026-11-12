#include "gpu/vulkan/backend/rife_fused_engine.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {
namespace {

using kernels::kFusedEncodePlanes;
using kernels::kFusedFlowChannels;
using kernels::kFusedInputPlanes;

constexpr VkDeviceSize kHalf = 2;

// Les poussées des trois noyaux, dans l'ordre de leurs blocs GLSL.
struct EntryPush {
  std::uint32_t height, width, reduce, out_height, out_width;
};
struct GluePush {
  std::uint32_t height, width, reduce, out_height, out_width, flows;
  std::uint32_t pad[2];  // uvec4 aligné sur 16 octets (std430 des push_constant)
  std::array<std::uint32_t, 4> yh, yw, scale;
};
struct FinishPush {
  std::uint32_t height, width, y_height, y_width;
};

std::optional<int> number(std::string_view text) {
  int value = 0;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  return value;
}

std::vector<std::string_view> split(std::string_view text, char sep) {
  std::vector<std::string_view> parts;
  for (std::size_t at = 0;;) {
    const std::size_t next = text.find(sep, at);
    parts.push_back(text.substr(at, next == std::string_view::npos ? std::string_view::npos : next - at));
    if (next == std::string_view::npos) return parts;
    at = next + 1;
  }
}

std::uint32_t u32(int v) {
  return static_cast<std::uint32_t>(v);
}

}  // namespace

std::string rife_fused_artifact_key(const RifeFusedKey& key) {
  std::string reduce;
  for (int k = 0; k < key.plan.blocks; ++k)
    reduce += std::format("{}{}", k == 0 ? "" : ",", key.plan.reduce[static_cast<std::size_t>(k)]);
  std::string artifacts;
  for (std::size_t i = 0; i < key.artifacts.size(); ++i) artifacts += (i == 0 ? "" : "|") + key.artifacts[i];
  const std::string_view output =
      key.motion ? "motion" : (key.warp == kernels::RifeWarp::CatmullRom ? "image-cubic" : "image");
  return std::format("rife-fused-vk:{}x{}:{}:{}:{}", key.height, key.width, output, reduce, artifacts);
}

std::optional<RifeFusedKey> rife_fused_artifact_key_parse(std::string_view text) {
  constexpr std::string_view kPrefix = "rife-fused-vk:";
  if (!text.starts_with(kPrefix)) return std::nullopt;
  text.remove_prefix(kPrefix.size());
  // Les chemins des artefacts viennent en dernier : ils peuvent porter des « : » (Windows).
  std::vector<std::string_view> head;
  for (int field = 0; field < 3; ++field) {
    const std::size_t colon = text.find(':');
    if (colon == std::string_view::npos) return std::nullopt;
    head.push_back(text.substr(0, colon));
    text.remove_prefix(colon + 1);
  }
  RifeFusedKey key;
  const auto x = head[0].find('x');
  if (x == std::string_view::npos) return std::nullopt;
  const auto h = number(head[0].substr(0, x));
  const auto w = number(head[0].substr(x + 1));
  if (!h || !w) return std::nullopt;
  key.height = *h;
  key.width = *w;
  if (head[1] != "image" && head[1] != "motion" && head[1] != "image-cubic") return std::nullopt;
  key.motion = head[1] == "motion";
  key.warp = head[1] == "image-cubic" ? kernels::RifeWarp::CatmullRom : kernels::RifeWarp::Bilinear;
  const auto reduce = split(head[2], ',');
  if (reduce.size() > key.plan.reduce.size()) return std::nullopt;
  key.plan.blocks = static_cast<int>(reduce.size());
  for (std::size_t k = 0; k < reduce.size(); ++k) {
    const auto d = number(reduce[k]);
    if (!d) return std::nullopt;
    key.plan.reduce[k] = *d;
  }
  if (!kernels::rife_fused_size_valid(key.plan, key.height, key.width)) return std::nullopt;
  for (const std::string_view path : split(text, '|')) key.artifacts.emplace_back(path);
  if (key.artifacts.size() != static_cast<std::size_t>(key.plan.blocks) + 1) return std::nullopt;
  return key;
}

std::expected<std::unique_ptr<RifeFusedEngine>, Refusal> RifeFusedEngine::create(
    VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines, const RifeFusedKey& key,
    std::vector<std::unique_ptr<VulkanEngine>> pieces) {
  if (!kernels::rife_fused_size_valid(key.plan, key.height, key.width))
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("RIFE fusionné : {}×{} hors du découpage", key.width, key.height));
  if (pieces.size() != static_cast<std::size_t>(key.plan.blocks) + 1)
    return refuse(RefusalCode::InputMismatch, "RIFE fusionné : nombre de morceaux faux");
  const int h = key.height;
  const int w = key.width;
  // Chaque morceau : une entrée et une sortie FP16 de la forme du découpage.
  const auto expect = [](VulkanEngine& piece, std::string_view what, const ir::Dims& in,
                         const ir::Dims& out) -> std::expected<void, Refusal> {
    const ir::Graph& g = piece.net().graph();
    if (g.inputs.size() != 1 || g.outputs.size() != 1)
      return refuse(RefusalCode::InputMismatch,
                    std::format("RIFE fusionné : {} sans une entrée et une sortie", what));
    const ir::Tensor& x = g.tensors[g.inputs[0]];
    const ir::Tensor& y = g.tensors[g.outputs[0]];
    if (x.type != ir::DataType::F16 || y.type != ir::DataType::F16 || x.dims != in || y.dims != out) {
      return refuse(RefusalCode::InputMismatch,
                    std::format("RIFE fusionné : {} {} → {}, {} → {} attendu", what, ir::format_dims(x.dims),
                                ir::format_dims(y.dims), ir::format_dims(in), ir::format_dims(out)));
    }
    return {};
  };
  // Une image à la fois (fusion3) : un encodeur de l'ancienne découpe, en lot de 2, est refusé.
  if (auto ok = expect(*pieces[0], "encodeur", {1, 3, h, w}, {1, kFusedEncodePlanes / 2, h, w}); !ok)
    return std::unexpected(ok.error());
  for (int k = 0; k < key.plan.blocks; ++k) {
    const kernels::FusedDims x = kernels::rife_fused_x(key.plan, k, h, w);
    const kernels::FusedDims y = kernels::rife_fused_y(key.plan, k, h, w);
    if (auto ok = expect(*pieces[static_cast<std::size_t>(k) + 1], std::format("bloc {}", k),
                         {1, x.channels, x.height, x.width}, {1, y.channels, y.height, y.width});
        !ok) {
      return std::unexpected(ok.error());
    }
  }
  std::unique_ptr<RifeFusedEngine> engine(new RifeFusedEngine(context, memory, pipelines));
  engine->pieces_ = std::move(pieces);
  engine->plan_ = key.plan;
  engine->height_ = h;
  engine->width_ = w;
  engine->motion_ = key.motion;
  engine->warp_ = key.warp;
  while (engine->warm_blocks_ + 1 < key.plan.blocks &&
         key.plan.reduce[static_cast<std::size_t>(engine->warm_blocks_)] == key.plan.reduce[0]) {
    ++engine->warm_blocks_;
  }
  engine->layout_ = kernels::rife_fused_layout(key.plan, h, w);
  engine->outputs_ =
      key.motion ? std::vector<std::string>{"flow", "mask"} : std::vector<std::string>{"output"};
  auto scratch = Buffer::create(context, memory, engine->layout_.bytes, MemoryUse::Device);
  if (!scratch) return std::unexpected(std::move(scratch.error()));
  engine->scratch_.emplace(std::move(*scratch));
  return engine;
}

RifeFusedEngine::~RifeFusedEngine() = default;

BufferView RifeFusedEngine::scratch(std::size_t offset, std::int64_t elements) const {
  if (!scratch_) return {};  // posé par create(), toujours là
  return {
      .buffer = scratch_->handle(), .offset = offset, .size = static_cast<VkDeviceSize>(elements) * kHalf};
}

Done record_rife_fused_entry(PipelineCache& pipelines, Recorder& recorder, const kernels::RifeFusedPlan& plan,
                             int height, int width, const BufferView& input, const BufferView& encode,
                             const BufferView& x0) {
  const kernels::FusedDims x = kernels::rife_fused_x(plan, 0, height, width);
  const EntryPush push{u32(height), u32(width), u32(plan.reduce[0]), u32(x.height), u32(x.width)};
  const std::array buffers{input, encode, x0};
  return detail::run(pipelines, recorder, "rife_fused_entry", Precision::F16, {}, buffers, {},
                     detail::bytes_of(push),
                     {detail::groups_for(u32(x.width)), detail::groups_for(u32(x.height)), 1});
}

Done record_rife_fused_glue(PipelineCache& pipelines, Recorder& recorder, const kernels::RifeFusedPlan& plan,
                            int height, int width, int k, const BufferView& input, const BufferView& encode,
                            const BufferView& xk, std::span<const BufferView> y) {
  if (k < 1 || k >= plan.blocks || y.size() < static_cast<std::size_t>(k))
    return refuse(RefusalCode::InputMismatch,
                  std::format("RIFE fusionné : colle du bloc {} sans ses sorties", k));
  const kernels::FusedDims x = kernels::rife_fused_x(plan, k, height, width);
  GluePush push{};
  push.height = u32(height);
  push.width = u32(width);
  push.reduce = u32(plan.reduce[static_cast<std::size_t>(k)]);
  push.out_height = u32(x.height);
  push.out_width = u32(x.width);
  push.flows = u32(k);
  std::array<BufferView, 7> buffers{input, encode, xk};
  for (int s = 0; s < 4; ++s) {
    const auto source =
        static_cast<std::size_t>(std::min(s, k - 1));  // absente : liée à la dernière, non lue
    const kernels::FusedDims dims = kernels::rife_fused_y(plan, static_cast<int>(source), height, width);
    const auto i = static_cast<std::size_t>(s);
    buffers[3 + i] = y[source];
    push.yh[i] = u32(dims.height);
    push.yw[i] = u32(dims.width);
    push.scale[i] = u32(plan.reduce[source]);
  }
  return detail::run(pipelines, recorder, "rife_fused_glue", Precision::F16, {}, buffers, {},
                     detail::bytes_of(push),
                     {detail::groups_for(u32(x.width)), detail::groups_for(u32(x.height), 8), 1});
}

Done record_rife_fused_finish(PipelineCache& pipelines, Recorder& recorder,
                              const kernels::RifeFusedPlan& plan, int height, int width, bool motion,
                              const BufferView& input, const BufferView& x_last, const BufferView& y_last,
                              const BufferView& first, const BufferView& mask, kernels::RifeWarp warp) {
  const kernels::FusedDims y = kernels::rife_fused_y(plan, plan.blocks - 1, height, width);
  const FinishPush push{u32(height), u32(width), u32(y.height), u32(y.width)};
  const std::array buffers{input, x_last, y_last, first, motion ? mask : first};
  const std::array constants{motion ? 1U : 0U, !motion && warp == kernels::RifeWarp::CatmullRom ? 1U : 0U};
  return detail::run(pipelines, recorder, "rife_fused_finish", Precision::F16, constants, buffers, {},
                     detail::bytes_of(push),
                     {detail::groups_for(u32(width)), detail::groups_for(u32(height)), 1});
}

std::expected<Recording*, Refusal> RifeFusedEngine::shift_encoding() {
  if (shift_) return shift_.get();
  auto recorder = Recorder::begin(context_, memory_);
  if (!recorder) return std::unexpected(std::move(recorder.error()));
  const std::int64_t half_planes = std::int64_t{kFusedEncodePlanes / 2} * height_ * width_;
  const BufferView first = scratch(layout_.encode, half_planes);
  const BufferView second = scratch(layout_.encode + static_cast<std::size_t>(first.size), half_planes);
  const VkBuffer buffer = first.buffer;
  const DeviceFunctions& fn = context_.fn();
  const auto copy = [&fn, buffer, &first, &second](VkCommandBuffer cb) {
    const VkBufferCopy region{.srcOffset = second.offset, .dstOffset = first.offset, .size = first.size};
    fn.vkCmdCopyBuffer(cb, buffer, buffer, 1, &region);
  };
  if (auto ok = recorder->commands(copy, std::span(&first, 1), false); !ok)
    return std::unexpected(ok.error());
  auto recording = recorder->finish();
  if (!recording) return std::unexpected(std::move(recording.error()));
  shift_ = std::move(*recording);
  return shift_.get();
}

Done RifeFusedEngine::submit_net(BackendQueue& queue, VulkanEngine& piece, const BufferView& in,
                                 const BufferView& out) {
  const std::array ins{in};
  const std::array outs{out};
  auto recording = piece.net().recording(ins, outs);
  if (!recording) return std::unexpected(recording.error());
  return queue.submit(**recording);
}

Done RifeFusedEngine::encode_pair(BackendQueue& queue, const BufferView& input, Hint hint,
                                  const std::optional<BufferView>& encoded_from) {
  const auto same = [](const BufferView& a, const BufferView& b) {
    return a.buffer == b.buffer && a.offset == b.offset && a.size == b.size;
  };
  // L'encodage du dernier appel mené au bout sert tel quel au même instant près (même entrée), ou pour A
  // quand la paire suit ; sans appel précédent réussi, les deux images passent dans l'encodeur.
  if (hint == Hint::TimesOnly && encoded_from && same(*encoded_from, input)) return {};
  const std::int64_t plane = std::int64_t{height_} * width_;
  // L'encodeur lit une image (A aux plans 0 à 2 de l'entrée, B aux plans 3 à 5) et écrit ses 4 plans
  // d'encodage, A puis B.
  const auto image = [&](int first_plane) {
    return BufferView{.buffer = input.buffer,
                      .offset = input.offset + static_cast<VkDeviceSize>(first_plane * plane) * kHalf,
                      .size = static_cast<VkDeviceSize>(3 * plane) * kHalf};
  };
  const BufferView encode_a = scratch(layout_.encode, (kFusedEncodePlanes / 2) * plane);
  const BufferView encode_b =
      scratch(layout_.encode + static_cast<std::size_t>(encode_a.size), (kFusedEncodePlanes / 2) * plane);
  if (hint == Hint::Follows && encoded_from) {
    auto shift = shift_encoding();
    if (!shift) return std::unexpected(shift.error());
    if (auto ok = queue.submit(**shift); !ok) return ok;
  } else {
    if (auto ok = submit_net(queue, *pieces_[0], image(0), encode_a); !ok) return ok;
    ++encodes_;
  }
  if (auto ok = submit_net(queue, *pieces_[0], image(3), encode_b); !ok) return ok;
  ++encodes_;
  return {};
}

std::expected<std::unique_ptr<Recording>, Refusal> RifeFusedEngine::record_entry(const BufferView& input) {
  auto recorder = Recorder::begin(context_, memory_);
  if (!recorder) return std::unexpected(std::move(recorder.error()));
  const std::int64_t plane = std::int64_t{height_} * width_;
  if (auto ok = record_rife_fused_entry(
          pipelines_, *recorder, plan_, height_, width_, input,
          scratch(layout_.encode, kFusedEncodePlanes * plane),
          scratch(layout_.x[0], kernels::rife_fused_x(plan_, 0, height_, width_).elements()));
      !ok) {
    return std::unexpected(ok.error());
  }
  return recorder->finish();
}

std::expected<std::unique_ptr<Recording>, Refusal> RifeFusedEngine::record_glue(const BufferView& input,
                                                                                int k) {
  auto recorder = Recorder::begin(context_, memory_);
  if (!recorder) return std::unexpected(std::move(recorder.error()));
  const std::int64_t plane = std::int64_t{height_} * width_;
  std::vector<BufferView> y;
  y.reserve(static_cast<std::size_t>(k));
  for (int s = 0; s < k; ++s)
    y.push_back(scratch(layout_.y[static_cast<std::size_t>(s)],
                        kernels::rife_fused_y(plan_, s, height_, width_).elements()));
  const auto i = static_cast<std::size_t>(k);
  if (auto ok = record_rife_fused_glue(
          pipelines_, *recorder, plan_, height_, width_, k, input,
          scratch(layout_.encode, kFusedEncodePlanes * plane),
          scratch(layout_.x[i], kernels::rife_fused_x(plan_, k, height_, width_).elements()), y);
      !ok) {
    return std::unexpected(ok.error());
  }
  return recorder->finish();
}

std::expected<std::unique_ptr<Recording>, Refusal> RifeFusedEngine::record_finish(
    const BufferView& input, std::span<const BufferView> outputs) {
  auto recorder = Recorder::begin(context_, memory_);
  if (!recorder) return std::unexpected(std::move(recorder.error()));
  const int last = plan_.blocks - 1;
  const auto i = static_cast<std::size_t>(last);
  if (auto ok = record_rife_fused_finish(
          pipelines_, *recorder, plan_, height_, width_, motion_, input,
          scratch(layout_.x[i], kernels::rife_fused_x(plan_, last, height_, width_).elements()),
          scratch(layout_.y[i], kernels::rife_fused_y(plan_, last, height_, width_).elements()), outputs[0],
          motion_ ? outputs[1] : outputs[0], warp_);
      !ok) {
    return std::unexpected(ok.error());
  }
  return recorder->finish();
}

std::expected<RifeFusedEngine::Glue*, Refusal> RifeFusedEngine::glue_for(
    const BufferView& input, std::span<const BufferView> outputs) {
  retired_.collect();
  const std::array<BufferView, 3> io{input, outputs[0], outputs.size() > 1 ? outputs[1] : BufferView{}};
  const auto same = [](const BufferView& a, const BufferView& b) {
    return a.buffer == b.buffer && a.offset == b.offset && a.size == b.size;
  };
  for (auto found = glues_.begin(); found != glues_.end(); ++found) {
    const auto& glue = *found;
    if (same(glue->io[0], io[0]) && same(glue->io[1], io[1]) && same(glue->io[2], io[2])) {
      // Le moins récemment utilisé part le premier, pas le plus ancien créé (C45-08) : un jeu encore en
      // rotation passe en fin de liste à chaque emploi.
      std::rotate(found, found + 1, glues_.end());
      return glues_.back().get();
    }
  }
  auto glue = std::make_unique<Glue>();
  glue->io = io;
  auto entry = record_entry(input);
  if (!entry) return std::unexpected(std::move(entry.error()));
  glue->steps.push_back(std::move(*entry));
  for (int k = 1; k < plan_.blocks; ++k) {
    auto step = record_glue(input, k);
    if (!step) return std::unexpected(std::move(step.error()));
    glue->steps.push_back(std::move(*step));
  }
  auto finish = record_finish(input, outputs);
  if (!finish) return std::unexpected(std::move(finish.error()));
  glue->steps.push_back(std::move(*finish));
  if (glues_.size() >= kMaxGlues) {
    for (auto& step : glues_.front()->steps) retired_.retire(std::move(step));
    glues_.erase(glues_.begin());
  }
  glues_.push_back(std::move(glue));
  return glues_.back().get();
}

std::expected<void, Refusal> RifeFusedEngine::enqueue(BackendQueue& queue, std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
  const Hint hint = std::exchange(hint_, Hint::None);  // pour cet appel seulement, même refusé
  if (inputs.size() != 1 || outputs.size() != outputs_.size())
    return refuse(RefusalCode::InputMismatch,
                  std::format("RIFE fusionné : 1 entrée et {} sorties attendues", outputs_.size()));
  const std::int64_t plane = std::int64_t{height_} * width_;
  const auto view_of = [&](const Tensor& t, int planes,
                           std::string_view what) -> std::expected<BufferView, Refusal> {
    const Shape want{.dims = {1, planes, height_, width_}};
    if (t.type != TensorType::F16 || !(t.shape == want) || t.handle == 0)
      return refuse(RefusalCode::InputMismatch,
                    std::format("RIFE fusionné : {} FP16 1×{}×{}×{} attendu", what, planes, height_, width_));
    return BufferView{.buffer = detail::to_buffer(t.handle),
                      .offset = t.offset,
                      .size = static_cast<VkDeviceSize>(planes * plane) * kHalf};
  };
  auto input = view_of(inputs[0], kFusedInputPlanes, "entrée");
  if (!input) return std::unexpected(input.error());
  std::array<BufferView, 2> outs{};
  auto first = view_of(outputs[0], motion_ ? kFusedFlowChannels : 3, motion_ ? "flot" : "sortie");
  if (!first) return std::unexpected(first.error());
  outs[0] = *first;
  if (motion_) {
    auto mask = view_of(outputs[1], 1, "masque");
    if (!mask) return std::unexpected(mask.error());
    outs[1] = *mask;
  }
  const std::optional<BufferView> encoded_from =
      std::exchange(encoded_from_, std::nullopt);  // repris au bout
  auto glue = glue_for(*input, std::span<const BufferView>(outs).first(outputs_.size()));
  if (!glue) return std::unexpected(glue.error());
  if (auto ok = encode_pair(queue, *input, hint, encoded_from); !ok) return ok;
  const auto run_net = [&](VulkanEngine& piece, const BufferView& in, const BufferView& out) -> Done {
    return submit_net(queue, piece, in, out);
  };
  const int first_block = std::exchange(warm_next_, false) ? warm_blocks_ : 0;
  warm_ready_ = false;  // repris si la chaîne va au bout
  for (int k = first_block; k < plan_.blocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    if (auto ok = queue.submit(*(*glue)->steps[i]); !ok) return ok;  // entrée du bloc k
    const kernels::FusedDims x = kernels::rife_fused_x(plan_, k, height_, width_);
    const kernels::FusedDims y = kernels::rife_fused_y(plan_, k, height_, width_);
    if (auto ok = run_net(*pieces_[i + 1], scratch(layout_.x[i], x.elements()),
                          scratch(layout_.y[i], y.elements()));
        !ok) {
      return ok;
    }
  }
  auto done = queue.submit(*(*glue)->steps.back());
  warm_ready_ = done.has_value();
  if (done) encoded_from_ = *input;
  return done;
}

}  // namespace ajp::gpu::vk
