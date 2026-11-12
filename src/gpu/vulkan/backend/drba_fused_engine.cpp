#include "gpu/vulkan/backend/drba_fused_engine.h"

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

using kernels::kDrbaFusedBlocks;
using kernels::kDrbaFusedEncodeChannels;
using kernels::kDrbaFusedImagePlanes;
using kernels::kDrbaFusedInputPlanes;

constexpr VkDeviceSize kHalf = 2;
constexpr std::uint32_t kLinearGroup = 256;
constexpr std::uint32_t kMaxGroups = 65535;

// Les poussées des noyaux, dans l'ordre de leurs blocs GLSL.
struct PrePush {
  std::uint32_t height, width, reduce0, reduce1, reduce2, groups_x, times_only;
};
struct PostPush {
  std::uint32_t enc_height, enc_width, groups_x;
};
struct GluePush {
  std::uint32_t height, width, full_height, full_width;
};
struct FinishPush {
  std::uint32_t height, width, upscale;
};

std::optional<int> number(std::string_view text) {
  int value = 0;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  return value;
}

std::uint32_t u32(int v) {
  return static_cast<std::uint32_t>(v);
}

// Un dispatch linéaire de n fils en groupes de 256 : x borné, le reste en y.
std::array<std::uint32_t, 3> linear_groups(std::uint64_t n, std::uint32_t& groups_x) {
  const std::uint64_t groups = (n + kLinearGroup - 1) / kLinearGroup;
  groups_x = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, kMaxGroups));
  const auto rows = static_cast<std::uint32_t>((groups + groups_x - 1) / groups_x);
  return {groups_x, rows, 1};
}

}  // namespace

std::string drba_fused_artifact_key(const DrbaFusedKey& key) {
  std::string artifacts;
  for (std::size_t i = 0; i < key.artifacts.size(); ++i) artifacts += (i == 0 ? "" : "|") + key.artifacts[i];
  return std::format("drba-fused-vk:{}x{}:{}:{}", key.height, key.width, key.plan.tail, artifacts);
}

std::optional<DrbaFusedKey> drba_fused_artifact_key_parse(std::string_view text) {
  constexpr std::string_view kPrefix = "drba-fused-vk:";
  if (!text.starts_with(kPrefix)) return std::nullopt;
  text.remove_prefix(kPrefix.size());
  std::array<std::string_view, 2> head;
  for (std::string_view& field : head) {
    const std::size_t colon = text.find(':');
    if (colon == std::string_view::npos) return std::nullopt;
    field = text.substr(0, colon);
    text.remove_prefix(colon + 1);
  }
  DrbaFusedKey key;
  const auto x = head[0].find('x');
  if (x == std::string_view::npos) return std::nullopt;
  const auto h = number(head[0].substr(0, x));
  const auto w = number(head[0].substr(x + 1));
  const auto t = number(head[1]);
  if (!h || !w || !t) return std::nullopt;
  key.height = *h;
  key.width = *w;
  key.plan.tail = *t;
  if (!kernels::drba_fused_size_valid(key.plan, key.height, key.width)) return std::nullopt;
  // Les chemins viennent en dernier : ils peuvent porter des « : » (Windows).
  for (std::size_t at = 0;;) {
    const std::size_t bar = text.find('|', at);
    key.artifacts.emplace_back(
        text.substr(at, bar == std::string_view::npos ? std::string_view::npos : bar - at));
    if (bar == std::string_view::npos) break;
    at = bar + 1;
  }
  if (key.artifacts.size() != kDrbaFusedBlocks + 1) return std::nullopt;
  return key;
}

std::expected<std::unique_ptr<DrbaFusedEngine>, Refusal> DrbaFusedEngine::create(
    VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines, const DrbaFusedKey& key,
    std::vector<std::unique_ptr<VulkanEngine>> pieces) {
  if (!kernels::drba_fused_size_valid(key.plan, key.height, key.width))
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("DistilDRBA fusionné : {}×{} hors du découpage", key.width, key.height));
  if (pieces.size() != kDrbaFusedBlocks + 1)
    return refuse(RefusalCode::InputMismatch, "DistilDRBA fusionné : nombre de morceaux faux");
  const int h = key.height;
  const int w = key.width;
  const int t = key.plan.tail;
  const auto expect = [](VulkanEngine& piece, std::string_view what, const ir::Dims& in,
                         const ir::Dims& out) -> std::expected<void, Refusal> {
    const ir::Graph& g = piece.net().graph();
    if (g.inputs.size() != 1 || g.outputs.size() != 1)
      return refuse(RefusalCode::InputMismatch,
                    std::format("DistilDRBA fusionné : {} sans une entrée et une sortie", what));
    const ir::Tensor& x = g.tensors[g.inputs[0]];
    const ir::Tensor& y = g.tensors[g.outputs[0]];
    if (x.type != ir::DataType::F16 || y.type != ir::DataType::F16 || x.dims != in || y.dims != out) {
      return refuse(
          RefusalCode::InputMismatch,
          std::format("DistilDRBA fusionné : {} {} → {}, {} → {} attendu", what, ir::format_dims(x.dims),
                      ir::format_dims(y.dims), ir::format_dims(in), ir::format_dims(out)));
    }
    return {};
  };
  if (auto ok =
          expect(*pieces[0], "encodeur", {3, 3, h / t, w / t}, {3, kDrbaFusedEncodeChannels, h / t, w / t});
      !ok)
    return std::unexpected(ok.error());
  for (int k = 0; k < kDrbaFusedBlocks; ++k) {
    const kernels::DrbaFusedDims x = kernels::drba_fused_x(key.plan, k, h, w);
    const kernels::DrbaFusedDims y = kernels::drba_fused_y(key.plan, k, h, w);
    if (auto ok = expect(*pieces[static_cast<std::size_t>(k) + 1], std::format("bloc {}", k),
                         {1, x.channels, x.height, x.width}, {1, y.channels, y.height, y.width});
        !ok) {
      return std::unexpected(ok.error());
    }
  }
  std::unique_ptr<DrbaFusedEngine> engine(new DrbaFusedEngine(context, memory, pipelines));
  engine->pieces_ = std::move(pieces);
  engine->plan_ = key.plan;
  engine->height_ = h;
  engine->width_ = w;
  engine->layout_ = kernels::drba_fused_layout(key.plan, h, w);
  auto scratch = Buffer::create(context, memory, engine->layout_.bytes, MemoryUse::Device);
  if (!scratch) return std::unexpected(std::move(scratch.error()));
  engine->scratch_.emplace(std::move(*scratch));
  return engine;
}

DrbaFusedEngine::~DrbaFusedEngine() = default;

BufferView DrbaFusedEngine::scratch(std::size_t offset, std::int64_t elements) const {
  if (!scratch_) return {};  // posé par create(), toujours là
  return {
      .buffer = scratch_->handle(), .offset = offset, .size = static_cast<VkDeviceSize>(elements) * kHalf};
}

std::expected<DrbaFusedEngine::Glue*, Refusal> DrbaFusedEngine::glue_for(std::span<const BufferView, 4> io,
                                                                         bool motion) {
  retired_.collect();
  const auto same = [](const BufferView& a, const BufferView& b) {
    return a.buffer == b.buffer && a.offset == b.offset && a.size == b.size;
  };
  for (auto found = glues_.begin(); found != glues_.end(); ++found) {
    bool all = true;
    for (std::size_t i = 0; i < io.size(); ++i) all = all && same((*found)->io[i], io[i]);
    if (all) {
      // Le moins récemment utilisé part le premier, pas le plus ancien créé (C45-08).
      std::rotate(found, found + 1, glues_.end());
      return glues_.back().get();
    }
  }
  const int h = height_;
  const int w = width_;
  const int t = plan_.tail;
  const auto x_view = [&](int k) {
    return scratch(layout_.x[static_cast<std::size_t>(k)], kernels::drba_fused_x(plan_, k, h, w).elements());
  };
  const auto y_view = [&](int k) {
    return scratch(layout_.y[static_cast<std::size_t>(k)], kernels::drba_fused_y(plan_, k, h, w).elements());
  };
  auto glue = std::make_unique<Glue>();
  glue->io = {io[0], io[1], io[2], io[3]};
  const auto record = [&](auto&& body) -> std::expected<void, Refusal> {
    auto recorder = Recorder::begin(context_, memory_);
    if (!recorder) return std::unexpected(std::move(recorder.error()));
    if (auto ok = body(*recorder); !ok) return std::unexpected(ok.error());
    auto done = recorder->finish();
    if (!done) return std::unexpected(std::move(done.error()));
    glue->steps.push_back(std::move(*done));
    return {};
  };
  // pre : neuf plans d'images et un plan du temps par entrée de bloc ; times_only, le plan du temps seul.
  const auto pre = [&](bool times_only) {
    return [&, times_only](Recorder& r) -> Done {
      std::uint64_t elements = 0;
      for (int k = 0; k < kDrbaFusedBlocks; ++k) {
        const int s = kernels::drba_fused_reduce(plan_, k);
        elements += std::uint64_t{times_only ? 1U : 10U} * static_cast<std::uint64_t>(h / s) *
                    static_cast<std::uint64_t>(w / s);
      }
      PrePush push{u32(h),
                   u32(w),
                   u32(kernels::drba_fused_reduce(plan_, 0)),
                   u32(kernels::drba_fused_reduce(plan_, 1)),
                   u32(kernels::drba_fused_reduce(plan_, 2)),
                   0,
                   times_only ? 1U : 0U};
      const auto groups = linear_groups(elements, push.groups_x);
      const std::array buffers{io[0], io[1], x_view(0), x_view(1), x_view(2)};
      return detail::run(pipelines_, r, "drba_fused_pre", Precision::F16, {}, buffers, {},
                         detail::bytes_of(push), groups);
    };
  };
  if (auto ok = record(pre(false)); !ok) return std::unexpected(ok.error());
  if (auto ok = record(pre(true)); !ok) return std::unexpected(ok.error());
  glue->times = std::move(glue->steps.back());
  glue->steps.pop_back();
  // post
  if (auto ok = record([&](Recorder& r) -> Done {
        const std::uint64_t enc = static_cast<std::uint64_t>(h / t) * static_cast<std::uint64_t>(w / t);
        PostPush push{u32(h / t), u32(w / t), 0};
        const auto groups = linear_groups(std::uint64_t{48} * (enc / 4 + enc / 16), push.groups_x);
        const std::array buffers{x_view(2), x_view(1), x_view(0)};
        return detail::run(pipelines_, r, "drba_fused_post", Precision::F16, {}, buffers, {},
                           detail::bytes_of(push), groups);
      });
      !ok)
    return std::unexpected(ok.error());
  // colles des blocs 1 et 2
  for (int k = 1; k < kDrbaFusedBlocks; ++k) {
    if (auto ok = record([&](Recorder& r) -> Done {
          const kernels::DrbaFusedDims x = kernels::drba_fused_x(plan_, k, h, w);
          const GluePush push{u32(x.height), u32(x.width), u32(h), u32(w)};
          const std::array buffers{y_view(k - 1), io[0], x_view(k)};
          return detail::run(pipelines_, r, "drba_fused_glue", Precision::F16, {}, buffers, {},
                             detail::bytes_of(push),
                             {detail::groups_for(u32(x.width)), detail::groups_for(u32(x.height)), 1});
        });
        !ok)
      return std::unexpected(ok.error());
  }
  // fin
  if (auto ok = record([&](Recorder& r) -> Done {
        const FinishPush push{u32(h), u32(w), u32(t)};
        const std::array buffers{y_view(kDrbaFusedBlocks - 1), io[0], io[1], io[2], motion ? io[3] : io[2]};
        const std::array constants{motion ? 1U : 0U};
        return detail::run(pipelines_, r, "drba_fused_finish", Precision::F16, constants, buffers, {},
                           detail::bytes_of(push),
                           {detail::groups_for(u32(w)), detail::groups_for(u32(h)), 1});
      });
      !ok)
    return std::unexpected(ok.error());
  if (glues_.size() >= kMaxGlues) {
    for (auto& step : glues_.front()->steps) retired_.retire(std::move(step));
    retired_.retire(std::move(glues_.front()->times));
    glues_.erase(glues_.begin());
  }
  glues_.push_back(std::move(glue));
  return glues_.back().get();
}

std::expected<void, Refusal> DrbaFusedEngine::enqueue(BackendQueue& queue, std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
  if (inputs.size() != 2 || (outputs.size() != 1 && outputs.size() != 2))
    return refuse(RefusalCode::InputMismatch, "DistilDRBA fusionné : 2 entrées et 1 ou 2 sorties attendues");
  const bool motion = outputs.size() == 2;
  const std::int64_t plane = std::int64_t{height_} * width_;
  const auto view_of = [&](const Tensor& t, int planes,
                           std::string_view what) -> std::expected<BufferView, Refusal> {
    const Shape want{.dims = {1, planes, height_, width_}};
    if (t.type != TensorType::F16 || !(t.shape == want) || t.handle == 0)
      return refuse(
          RefusalCode::InputMismatch,
          std::format("DistilDRBA fusionné : {} FP16 1×{}×{}×{} attendu", what, planes, height_, width_));
    return BufferView{.buffer = detail::to_buffer(t.handle),
                      .offset = t.offset,
                      .size = static_cast<VkDeviceSize>(planes * plane) * kHalf};
  };
  std::array<BufferView, 4> io{};
  const std::array<std::expected<BufferView, Refusal>, 4> views{
      view_of(inputs[0], kDrbaFusedInputPlanes, "pair01"),
      view_of(inputs[1], kDrbaFusedInputPlanes, "pair12"),
      view_of(outputs[0], motion ? 4 : 3, motion ? "flot" : "sortie"),
      motion ? view_of(outputs[1], 1, "masque") : std::expected<BufferView, Refusal>(BufferView{})};
  for (std::size_t i = 0; i < views.size(); ++i) {
    if (!views[i]) return std::unexpected(views[i].error());
    io[i] = *views[i];
  }
  const auto same = [](const BufferView& a, const BufferView& b) {
    return a.buffer == b.buffer && a.offset == b.offset && a.size == b.size;
  };
  // D6 : mêmes entrées que l'appel précédent, mené jusqu'au bout, et l'appelant dit que seul le temps change.
  const bool times_only = std::exchange(times_only_next_, false) && last_inputs_ &&
                          same((*last_inputs_)[0], io[0]) && same((*last_inputs_)[1], io[1]);
  last_inputs_.reset();
  auto glue = glue_for(io, motion);
  if (!glue) return std::unexpected(glue.error());
  const auto& steps = (*glue)->steps;
  const auto run_net = [&](VulkanEngine& piece, const BufferView& in, const BufferView& out) -> Done {
    const std::array ins{in};
    const std::array outs{out};
    auto recording = piece.net().recording(ins, outs);
    if (!recording) return std::unexpected(recording.error());
    return queue.submit(**recording);
  };
  const int t = plan_.tail;
  const std::int64_t enc_plane = std::int64_t{height_ / t} * (width_ / t);
  const BufferView x2 = scratch(layout_.x[2], kernels::drba_fused_x(plan_, 2, height_, width_).elements());
  // L'encodeur lit les images réduites au début de X_2 et y écrit les encodages juste après.
  const BufferView images{.buffer = x2.buffer,
                          .offset = x2.offset,
                          .size = static_cast<VkDeviceSize>(kDrbaFusedImagePlanes * enc_plane) * kHalf};
  const BufferView encoded{.buffer = x2.buffer,
                           .offset = x2.offset + static_cast<VkDeviceSize>(kDrbaFusedImagePlanes) *
                                                     static_cast<VkDeviceSize>(enc_plane) * kHalf,
                           .size = static_cast<VkDeviceSize>(3 * kDrbaFusedEncodeChannels) *
                                   static_cast<VkDeviceSize>(enc_plane) * kHalf};
  if (times_only) {
    if (auto ok = queue.submit(*(*glue)->times); !ok) return ok;  // pre, plans du temps seuls
  } else {
    if (auto ok = queue.submit(*steps[0]); !ok) return ok;  // pre
    if (auto ok = run_net(*pieces_[0], images, encoded); !ok) return ok;
    if (auto ok = queue.submit(*steps[1]); !ok) return ok;  // post
  }
  for (int k = 0; k < kDrbaFusedBlocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    if (k > 0) {
      if (auto ok = queue.submit(*steps[1 + i]); !ok) return ok;  // colle du bloc k
    }
    if (auto ok = run_net(*pieces_[i + 1],
                          scratch(layout_.x[i], kernels::drba_fused_x(plan_, k, height_, width_).elements()),
                          scratch(layout_.y[i], kernels::drba_fused_y(plan_, k, height_, width_).elements()));
        !ok) {
      return ok;
    }
  }
  if (auto ok = queue.submit(*steps.back()); !ok) return ok;
  last_inputs_ = std::array{io[0], io[1]};
  return {};
}

}  // namespace ajp::gpu::vk
