#include "stage_rig.h"

#include <algorithm>
#include <array>
#include <format>
#include <thread>

namespace ajp::app::play::test {
namespace {

using namespace std::chrono_literals;

class EmptyFrame final : public FrameData {
 public:
  FrameMemory memory() const override { return FrameMemory::Host; }
  int plane_count() const override { return 2; }
  std::span<const std::byte> host_plane(int /*plane*/) const override { return {}; }
  std::size_t host_stride(int /*plane*/) const override { return 0; }
  std::uintptr_t native_plane(int /*plane*/) const override { return 0; }
  std::size_t native_pitch(int /*plane*/) const override { return 0; }
  std::uintptr_t ready_event() const override { return 0; }
  std::uint64_t ready_value() const override { return 0; }
};

class FakeTensorBuffer final : public gpu::TensorBuffer {
 public:
  FakeTensorBuffer(gpu::TensorType type, const gpu::Shape& shape, std::uintptr_t handle)
      : tensor_{.type = type, .shape = shape, .handle = handle, .offset = 0} {}
  const gpu::Tensor& tensor() const override { return tensor_; }

 private:
  gpu::Tensor tensor_;
};

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

void FakeBackend::score(SourceId left, SourceId right, double scene_score, std::uint32_t worst_block) {
  const std::scoped_lock lock(mutex_);
  scores_[{left.value, right.value}] = gpu::PairScores{.left = left,
                                                       .right = right,
                                                       .scene_score = scene_score,
                                                       .worst_block = worst_block,
                                                       .identical = false};
}

std::vector<std::string> FakeBackend::log() const {
  const std::scoped_lock lock(mutex_);
  return log_;
}

std::size_t FakeBackend::count(std::string_view entry) const {
  const std::scoped_lock lock(mutex_);
  return static_cast<std::size_t>(std::ranges::count(log_, entry));
}

std::vector<RingWritten> FakeBackend::written() const {
  const std::scoped_lock lock(mutex_);
  return written_;
}

std::vector<std::string> FakeBackend::laned() const {
  const std::scoped_lock lock(mutex_);
  return laned_;
}

std::vector<std::uintptr_t> FakeBackend::targets(std::string_view entry) const {
  const std::scoped_lock lock(mutex_);
  const auto found = targets_.find(entry);
  return found == targets_.end() ? std::vector<std::uintptr_t>{} : found->second;
}

std::expected<void, Refusal> FakeBackend::note(std::string entry, std::uintptr_t target) {
  const std::scoped_lock lock(mutex_);
  if (target != 0) targets_[entry].push_back(target);
  push(std::move(entry));
  return {};
}

void FakeBackend::push(std::string entry) {
  constexpr std::array<std::string_view, gpu::kLanes> kNames{"measure", "upscale", "interpolate"};
  laned_.push_back(std::format("{} {}", kNames[gpu::lane_index(lane_)], entry));
  log_.push_back(std::move(entry));
}

std::expected<gpu::Artifact, Refusal> FakeBackend::build(const gpu::ModelSpec& model, const gpu::ShapeSet&,
                                                         gpu::Progress&) {
  return refuse(RefusalCode::FeatureUnavailable,
                std::format("backend factice : {} ne se construit pas", model.id));
}

std::expected<std::unique_ptr<gpu::Engine>, Refusal> FakeBackend::load(const gpu::Artifact& artifact) {
  return refuse(RefusalCode::FeatureUnavailable,
                std::format("backend factice : {} ne se charge pas", artifact.path.string()));
}

std::expected<gpu::Tensor, Refusal> FakeBackend::import(const gpu::ExternalImage&) {
  return refuse(RefusalCode::FormatUnsupported, "backend factice : pas d'import");
}

std::expected<std::unique_ptr<gpu::TensorBuffer>, Refusal> FakeBackend::alloc_tensor(
    gpu::TensorType type, const gpu::Shape& shape) {
  for (const std::int64_t d : shape.dims) {
    if (d <= 0) return refuse(RefusalCode::ShapeInvalid, "backend factice : dimension nulle");
  }
  const std::scoped_lock lock(mutex_);
  push("alloc");
  return std::make_unique<FakeTensorBuffer>(type, shape, next_handle_++);
}

std::expected<void, Refusal> FakeBackend::enqueue(gpu::Engine& engine, std::span<const gpu::Tensor> inputs,
                                                  std::span<gpu::Tensor> outputs) {
  if (inputs.size() != engine.input_names().size() || outputs.size() != engine.output_names().size())
    return refuse(RefusalCode::InputMismatch, "backend factice : nombre de tenseurs du moteur");
  return note(std::format("engine {}", static_cast<const FakeEngine&>(engine).name()));
}

std::expected<void, Refusal> FakeBackend::ingest(const DecodedFrame& frame, const gpu::IngestSpec&,
                                                 const gpu::Tensor& rgb) {
  const std::scoped_lock lock(mutex_);
  if (refused_ == frame.id)
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("backend factice : image {} refusée", frame.id.value));
  ingested_[rgb.handle] = frame.id;
  push(std::format("ingest {}", frame.id.value));
  return {};
}

std::expected<void, Refusal> FakeBackend::write_ring(const gpu::Tensor&, const RingSlotDesc& slot,
                                                     const gpu::RingWrite& sync) {
  const std::scoped_lock lock(mutex_);
  push(std::format("write {}", slot.index));
  written_.push_back({.slot = slot.index, .wait_value = sync.wait_value, .signal_value = sync.signal_value});
  return {};
}

gpu::PairScores FakeBackend::scores_of(const gpu::Tensor& a, const gpu::Tensor& b) {
  const auto left = ingested_.find(a.handle);
  const auto right = ingested_.find(b.handle);
  gpu::PairScores scores{.scene_score = 0.0, .worst_block = 1000, .identical = false};
  if (left == ingested_.end() || right == ingested_.end()) return scores;
  scores.left = left->second;
  scores.right = right->second;
  if (const auto given = scores_.find({scores.left.value, scores.right.value}); given != scores_.end())
    scores = given->second;
  return scores;
}

std::expected<gpu::PairScores, Refusal> FakeBackend::scores_now(const gpu::Tensor& a, const gpu::Tensor& b) {
  const std::scoped_lock lock(mutex_);
  const gpu::PairScores scores = scores_of(a, b);
  push(std::format("scores_now {} {}", scores.left.value, scores.right.value));
  return scores;
}

std::expected<std::uint64_t, Refusal> FakeBackend::scores_start(const gpu::Tensor& a, const gpu::Tensor& b) {
  const std::scoped_lock lock(mutex_);
  if (!ahead_) return refuse(RefusalCode::FormatUnsupported, "scores_start : absent de ce backend");
  const gpu::PairScores scores = scores_of(a, b);
  push(std::format("scores_start {} {}", scores.left.value, scores.right.value));
  tickets_[next_ticket_] = scores;
  return next_ticket_++;
}

std::expected<gpu::PairScores, Refusal> FakeBackend::scores_take(std::uint64_t ticket) {
  const std::scoped_lock lock(mutex_);
  const auto found = tickets_.find(ticket);
  if (found == tickets_.end()) return refuse(RefusalCode::SequenceOrder, "scores_take : ticket inconnu");
  const gpu::PairScores scores = found->second;
  tickets_.erase(found);
  push("scores_take");
  return scores;
}

bool FrameRecord::try_push(const RingFrame& frame) {
  {
    const std::scoped_lock lock(mutex_);
    frames_.push_back(frame);
  }
  arrived_.notify_all();
  return true;
}

bool FrameRecord::wait_for(std::size_t n, std::chrono::milliseconds timeout) const {
  std::unique_lock lock(mutex_);
  return arrived_.wait_for(lock, timeout, [&] { return frames_.size() >= n; });
}

std::vector<RingFrame> FrameRecord::frames() const {
  const std::scoped_lock lock(mutex_);
  return frames_;
}

DecodedFrame host_frame(std::uint32_t id, int width, int height, std::int64_t pts_us, std::uint32_t epoch) {
  static const auto data = std::make_shared<EmptyFrame>();
  return DecodedFrame{.id = SourceId{id},
                      .width = width,
                      .height = height,
                      .format = PixelFormat::Nv12,
                      .color = {},
                      .pts_us = pts_us,
                      .duration_us = 40'000,
                      .epoch = epoch,
                      .data = data};
}

VideoStageInit StageRig::init(const MasterClock* clock) {
  return VideoStageInit{.track = nullptr,
                        .ring = &sink,
                        .credit = [this] { ++credits; },
                        .report = ThreadReport([this](const Refusal& refusal) {
                          const std::scoped_lock lock(refusals_mutex);
                          refusals.push_back(refusal);
                        }),
                        .clock = clock};
}

std::vector<RingSlotDesc> StageRig::slots(std::uint32_t count, int width, int height) {
  std::vector<RingSlotDesc> out;
  for (std::uint32_t i = 0; i < count; ++i)
    out.push_back(RingSlotDesc{.index = i, .generation = 1, .width = width, .height = height});
  return out;
}

bool StageRig::push(DecodedFrame frame, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!stage->frames().try_push(DecodedFrame{frame})) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

bool StageRig::collect(std::size_t n, bool release, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const std::vector<RingFrame> got = sink.frames();
    if (release) release_all();
    if (got.size() >= n) return true;
    const auto left = deadline - std::chrono::steady_clock::now();
    if (left <= 0ms) return false;
    sink.wait_for(got.size() + 1, std::chrono::duration_cast<std::chrono::milliseconds>(left));
  }
}

std::size_t StageRig::release_all() {
  const std::vector<RingFrame> got = sink.frames();
  std::size_t released = 0;
  for (; released_ < got.size(); ++released_, ++released) stage->post(release_of(got[released_]));
  return released;
}

std::vector<Refusal> StageRig::reported() {
  const std::scoped_lock lock(refusals_mutex);
  return refusals;
}

}  // namespace ajp::app::play::test
