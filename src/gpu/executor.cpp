#include "gpu/executor.h"

#include <algorithm>
#include <format>
#include <utility>

#include "core/cadence.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/resample.h"

namespace ajp::gpu {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Une mesure jamais relue : un dessin neuf, sans coupe, que la timeline interpole.
constexpr core::PairMeasure kUnknownMeasure{.worst_block_diff = core::drawings::kMaxDiff};
// Deux sources que rien ne relie (tailles différentes, mesure refusée) : une coupe.
constexpr core::PairMeasure kCutMeasure{.worst_block_diff = core::drawings::kMaxDiff, .scene_cut = true};

core::PairMeasure to_measure(const PairScores& scores) {
  const bool cut = kernels::is_scene_cut(scores.scene_score);
  return {.worst_block_diff = static_cast<double>(scores.worst_block) / 256.0,
          .scene_cut = cut,
          .identical = scores.identical && !cut};
}

}  // namespace

Executor::Executor(std::shared_ptr<const Program> program)
    : program_(std::move(program)), tensors_(program_->backend(), kMaxTensors), contents_(kMaxContents) {
  const std::vector<core::InterpLevel>& levels = program_->plan().levels;
  levels_.reserve(levels.size());
  for (std::size_t i = 0; i < levels.size(); ++i) {
    // Program::create a vérifié que chaque niveau se crée.
    levels_.push_back(*LevelScheduler::create(levels[i], static_cast<std::uint32_t>(i)));
  }
}

Executor::~Executor() = default;

std::expected<void, Refusal> Executor::push(const DecodedFrame& frame) {
  if (is_derived(frame.id)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("image {} : identité de décodage hors de [0, 2^31[", frame.id.value));
  }
  if (frame.epoch < epoch_) return {};
  if (frame.epoch > epoch_) flush(frame.epoch);
  ++counters_.sources;
  poll_scores();
  return accept(frame);
}

std::expected<void, Refusal> Executor::seed(const DecodedFrame& frame) {
  seeding_ = true;
  auto ok = push(frame);
  seeding_ = false;
  return ok;
}

std::expected<void, Refusal> Executor::finish() {
  poll_scores();
  if (auto ok = drain_waiting(true); !ok) return ok;
  for (std::size_t level = 0; level < levels_.size(); ++level) {
    auto ok = take(level, levels_[level].finish());
    prune(level);
    if (!ok) return ok;
  }
  return {};
}

void Executor::flush(std::uint32_t epoch) {
  epoch_ = std::max(epoch_, epoch);
  for (LevelScheduler& level : levels_) {
    const auto flushed = level.flush(epoch_);  // epoch_ est la plus récente : jamais refusé
    (void)flushed;
  }
  for (auto& store : stores_) store.fill({});
  waiting_.clear();
  finals_.clear();
  motion_ = {};
  sized_ = false;
  previous_.reset();
}

std::expected<void, Refusal> Executor::accept(const DecodedFrame& frame) {
  auto geometry = geometry_of(frame);
  if (!geometry) return std::unexpected(std::move(geometry.error()));
  const bool same_size = sized_ && geometry_.width == geometry->width && geometry_.height == geometry->height;
  geometry_ = *geometry;
  sized_ = true;
  auto content = ingest(frame);
  if (!content) return std::unexpected(std::move(content.error()));

  Waiting waiting{.input = {.id = frame.id, .pts_us = known_pts(frame.pts_us), .epoch = frame.epoch},
                  .content = std::move(*content)};
  if (!previous_) {
    waiting.measured = true;  // tête de séquence : la timeline ne lit pas sa mesure
  } else if (!same_size || previous_->format != frame.format) {
    waiting.input.measure = kCutMeasure;
    waiting.measured = true;
  } else if (auto launched = program_->kernels().measure(*previous_, frame); !launched) {
    waiting.input.measure = kCutMeasure;
    waiting.measured = true;
    ++counters_.lost_measures;
  }
  previous_ = frame;
  waiting_.push(std::move(waiting));  // drain_waiting en garde au plus kMaxWaiting
  poll_scores();
  return drain_waiting(false);
}

std::expected<Executor::Geometry, Refusal> Executor::geometry_of(const DecodedFrame& frame) const {
  const core::Plan& plan = program_->plan();
  const int factor = program_->upscale_factor();
  Geometry g{
      .width = frame.width, .height = frame.height, .work_width = frame.width, .work_height = frame.height};
  if (plan.downscale) {
    g.work_width = plan.output_width / factor;
    g.work_height = plan.output_height / factor;
  }
  if (frame.width <= 0 || frame.height <= 0 || g.work_width * factor != plan.output_width ||
      g.work_height * factor != plan.output_height || g.work_width > frame.width ||
      g.work_height > frame.height) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("image {}×{} : le Plan sort {}×{} par un agrandissement x{}{}", frame.width, frame.height,
                    plan.output_width, plan.output_height, factor, plan.downscale ? " après réduction" : ""));
  }
  if (!levels_.empty()) g.rife = kernels::rife_geometry(frame.width, frame.height);
  return g;
}

std::expected<ContentRef, Refusal> Executor::ingest(const DecodedFrame& frame) {
  auto content = new_content();
  auto low = image(frame.width, frame.height);
  if (!content || !low) return std::unexpected(!content ? content.error() : low.error());
  if (auto ok = program_->kernels().ingest(frame, IngestSpec{}, tensor_of(*low)); !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  (*content)->low = std::move(*low);
  if (!levels_.empty()) {
    auto rife = image(geometry_.rife.padded_width, geometry_.rife.padded_height);
    if (!rife) return std::unexpected(std::move(rife.error()));
    if (auto ok = program_->kernels().rife_pad(frame, tensor_of(*rife)); !ok)
      return std::unexpected(std::move(ok.error()));
    (*content)->rife = std::move(*rife);
  }
  return content;
}

void Executor::poll_scores() {
  const std::optional<PairScores> scores = program_->kernels().previous_scores();
  if (!scores) return;
  std::size_t match = waiting_.size();
  for (std::size_t i = 0; i < waiting_.size(); ++i) {
    if (!waiting_[i].measured && waiting_[i].input.id == scores->right) match = i;
  }
  if (match == waiting_.size()) return;  // la mesure d'une paire jetée depuis
  // Les mesures finissent dans l'ordre : celles d'avant la paire relue ne le seront plus.
  for (std::size_t i = 0; i < match; ++i) {
    if (waiting_[i].measured) continue;
    waiting_[i].input.measure = kUnknownMeasure;
    waiting_[i].measured = true;
    ++counters_.lost_measures;
  }
  Waiting& found = waiting_[match];
  found.input.measure = to_measure(*scores);
  found.measured = true;
  if (found.input.measure.scene_cut) ++counters_.scene_cuts;
}

std::expected<void, Refusal> Executor::drain_waiting(bool all) {
  while (!waiting_.empty()) {
    Waiting& front = waiting_.front();
    if (!front.measured) {
      if (!all && waiting_.size() <= kMaxWaiting) break;
      front.input.measure = kUnknownMeasure;
      front.measured = true;
      ++counters_.lost_measures;
    }
    Waiting next = waiting_.pop();
    if (auto ok = feed(0, next.input, std::move(next.content)); !ok) return ok;
  }
  return {};
}

std::expected<void, Refusal> Executor::feed(std::size_t level, const LevelInput& input, ContentRef content) {
  if (levels_.empty()) {  // S0 : chaque source telle quelle
    return emit(
        {.pts_us = pts_or_unknown(input.pts_us), .left = input.id, .right = input.id, .epoch = input.epoch},
        std::move(content));
  }
  auto& store = stores_[level];
  const auto free = std::ranges::find_if(store, [](const Picture& p) { return !p.content; });
  if (free == store.end()) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("niveau {} : plus de {} entrées gardées", level, kStore));
  }
  *free = {.id = input.id, .content = std::move(content)};
  auto outputs = levels_[level].push(input);
  std::expected<void, Refusal> ok;
  if (outputs) {
    ok = take(level, *outputs);
  } else {
    ok = std::unexpected(std::move(outputs.error()));
  }
  prune(level);
  return ok;
}

std::expected<void, Refusal> Executor::take(std::size_t level, std::span<const LevelOutput> outputs) {
  for (const LevelOutput& output : outputs) {
    auto content = materialize(level, output);
    if (!content) return std::unexpected(std::move(content.error()));
    std::expected<void, Refusal> ok;
    if (level + 1 < levels_.size()) {
      ok = feed(level + 1,
                {.id = output.id, .pts_us = output.pts_us, .epoch = output.epoch, .measure = output.measure},
                std::move(*content));
    } else {
      ok = emit(ring_frame(output), std::move(*content));
    }
    if (!ok) return ok;
  }
  return {};
}

RingFrame Executor::ring_frame(const LevelOutput& output) {
  const core::InstantDecision& decision = output.decision;
  RingFrame frame{.pts_us = pts_or_unknown(output.pts_us),
                  .left = decision.a,
                  .right = decision.a,
                  .scene_cut = output.scene_cut,
                  .epoch = output.epoch};
  switch (decision.show) {
    case core::Show::Source:
      break;
    case core::Show::Repeated: {
      const core::Ratio t = core::reduced(output.t);
      frame.left = output.pair_left;
      frame.right = output.pair_right;
      frame.t_num = t.num;
      frame.t_den = t.den;
      frame.repeated = true;
      ++counters_.repeated;
      break;
    }
    case core::Show::Interpolate:
      frame.right = decision.b;
      frame.t_num = decision.u.num;
      frame.t_den = decision.u.den;
      ++counters_.interpolated;
      break;
  }
  return frame;
}

std::expected<ContentRef, Refusal> Executor::materialize(std::size_t level, const LevelOutput& output) {
  const core::InstantDecision& decision = output.decision;
  const Picture* a = find(level, decision.a);
  const Picture* b = decision.show == core::Show::Interpolate ? find(level, decision.b) : a;
  if (a == nullptr || b == nullptr) {
    return refuse(RefusalCode::SourceUnknown, std::format("niveau {} : entrée {} ou {} déjà rendue", level,
                                                          decision.a.value, decision.b.value));
  }
  if (decision.show != core::Show::Interpolate) return a->content;
  if (levels_[level].level().schedule == core::Schedule::OutputRes) {
    return warp(level, output, a->content, b->content);
  }
  return interpolate(level, a->content, b->content, decision.u);
}

std::expected<ContentRef, Refusal> Executor::interpolate(std::size_t level, const ContentRef& a,
                                                         const ContentRef& b, core::Ratio u) {
  const kernels::RifeGeometry& rife = geometry_.rife;
  auto input = image(rife.padded_width, rife.padded_height, kernels::kRifeInputChannels);
  auto predicted = image(rife.padded_width, rife.padded_height);
  auto low = image(geometry_.width, geometry_.height);
  auto content = new_content();
  if (!input || !predicted || !low || !content) {
    return std::unexpected(!input       ? input.error()
                           : !predicted ? predicted.error()
                           : !low       ? low.error()
                                        : content.error());
  }
  const auto t = static_cast<float>(u.value());
  if (auto ok = program_->rife()->rife_input(tensor_of(a->rife), tensor_of(b->rife), t, tensor_of(*input));
      !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  const std::array inputs{tensor_of(*input)};
  std::array outputs{tensor_of(*predicted)};
  if (auto ok = program_->backend().enqueue(program_->level_engine(level), inputs, outputs); !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  if (auto ok = program_->kernels().rife_crop(tensor_of(*predicted), tensor_of(*low)); !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  (*content)->low = std::move(*low);
  (*content)->rife = std::move(*predicted);
  return content;
}

std::expected<ContentRef, Refusal> Executor::warp(std::size_t level, const LevelOutput& output,
                                                  const ContentRef& a, const ContentRef& b) {
  const kernels::RifeGeometry& rife = geometry_.rife;
  const core::InstantDecision& decision = output.decision;
  // Une passe de RIFE par paire : le mouvement au milieu sert à tous ses instants.
  if (!motion_.valid || motion_.a != decision.a || motion_.b != decision.b) {
    auto input = image(rife.padded_width, rife.padded_height, kernels::kRifeInputChannels);
    auto flow = image(rife.padded_width, rife.padded_height, 4);
    auto mask = image(rife.padded_width, rife.padded_height, 1);
    if (!input || !flow || !mask)
      return std::unexpected(!input ? input.error() : !flow ? flow.error() : mask.error());
    if (auto ok =
            program_->rife()->rife_input(tensor_of(a->rife), tensor_of(b->rife), 0.5F, tensor_of(*input));
        !ok) {
      return std::unexpected(std::move(ok.error()));
    }
    const std::array inputs{tensor_of(*input)};
    std::array outputs{tensor_of(*flow), tensor_of(*mask)};
    if (auto ok = program_->backend().enqueue(program_->level_engine(level), inputs, outputs); !ok) {
      return std::unexpected(std::move(ok.error()));
    }
    motion_ = {
        .valid = true, .a = decision.a, .b = decision.b, .flow = std::move(*flow), .mask = std::move(*mask)};
  }
  if (auto ok = finish_content(*a); !ok) return std::unexpected(std::move(ok.error()));
  if (auto ok = finish_content(*b); !ok) return std::unexpected(std::move(ok.error()));
  const core::Plan& plan = program_->plan();
  auto high = image(plan.output_width, plan.output_height);
  auto content = new_content();
  if (!high || !content) return std::unexpected(!high ? high.error() : content.error());
  const Motion motion{.flow = tensor_of(motion_.flow),
                      .mask = tensor_of(motion_.mask),
                      .area = {.x = rife.left, .y = rife.top, .width = rife.width, .height = rife.height}};
  const auto t = static_cast<float>(decision.u.value());
  if (auto ok =
          program_->kernels().warp_blend(tensor_of(a->high), tensor_of(b->high), motion, t, tensor_of(*high));
      !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  (*content)->high = std::move(*high);
  return content;
}

std::expected<void, Refusal> Executor::finish_content(Content& content) {
  if (content.high) return {};
  if (!content.low) return refuse(RefusalCode::SourceUnknown, "image sans RGB de la chaîne à agrandir");
  TensorRef current = content.low;
  int width = geometry_.work_width;
  int height = geometry_.work_height;
  const std::optional<core::DownscaleStep>& downscale = program_->plan().downscale;
  if (downscale && (width != geometry_.width || height != geometry_.height)) {
    auto down = image(width, height);
    if (!down) return std::unexpected(std::move(down.error()));
    const kernels::Filter filter =
        downscale->filter == core::DownscaleFilter::Area ? kernels::Filter::Area : kernels::Filter::Spline36;
    if (auto ok = program_->kernels().resample(tensor_of(current), filter, tensor_of(*down)); !ok) return ok;
    current = std::move(*down);
  }
  for (const UpscalePass& pass : program_->upscale()) {
    width *= pass.factor;
    height *= pass.factor;
    auto next = image(width, height);
    if (!next) return std::unexpected(std::move(next.error()));
    const std::array inputs{tensor_of(current)};
    std::array outputs{tensor_of(*next)};
    if (auto ok = program_->backend().enqueue(*pass.engine, inputs, outputs); !ok) return ok;
    current = std::move(*next);
  }
  content.high = std::move(current);
  return {};
}

std::expected<void, Refusal> Executor::emit(const RingFrame& frame, ContentRef content) {
  if (seeding_) return {};  // déjà montrée par le Program précédent
  if (auto ok = finish_content(*content); !ok) return ok;
  if (!finals_.push({.frame = frame, .content = std::move(content)})) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("plus de {} images finales en attente du ring", kMaxFinal));
  }
  ++counters_.outputs;
  return {};
}

std::expected<TensorRef, Refusal> Executor::image(int width, int height, int channels) {
  return tensors_.acquire(program_->precision(), Shape{{1, channels, height, width}});
}

std::expected<ContentRef, Refusal> Executor::new_content() {
  if (auto content = contents_.put(Content{})) return std::move(*content);
  return refuse(RefusalCode::SequenceOrder, std::format("plus de {} images vivantes", kMaxContents));
}

const Executor::Picture* Executor::find(std::size_t level, SourceId id) const {
  const auto& store = stores_[level];
  const auto it = std::ranges::find_if(store, [id](const Picture& p) { return p.content && p.id == id; });
  return it == store.end() ? nullptr : &*it;
}

void Executor::prune(std::size_t level) {
  const std::span<const SourceId> live = levels_[level].live();
  for (Picture& picture : stores_[level]) {
    if (picture.content && std::ranges::find(live, picture.id) == live.end()) picture = {};
  }
}

}  // namespace ajp::gpu
