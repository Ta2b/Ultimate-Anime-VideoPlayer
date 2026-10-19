#include "gpu_test_support.h"

#include <chrono>
#include <format>
#include <system_error>

namespace ajp::gpu::test {

using ir::test::attr_int;
using ir::test::attr_ints;
using ir::test::attr_string;
using ir::test::ModelBuilder;

TempDir::TempDir(const std::string& name) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  path_ = std::filesystem::temp_directory_path() / std::format("ajp-gpu-{}-{}", name, stamp);
  std::filesystem::create_directories(path_);
}

TempDir::~TempDir() {
  std::error_code ignored;
  std::filesystem::remove_all(path_, ignored);
}

std::vector<std::byte> upscale_x2_model() {
  // Le canal k = c·4 + i·2 + j de la convolution recopie le canal c ; DepthToSpace (CRD) le range
  // au sous-pixel (i, j) : chaque pixel devient un bloc 2 × 2.
  std::vector<float> weights(12 * 3, 0.0F);
  for (std::size_t k = 0; k < 12; ++k) weights[k * 3 + k / 4] = 1.0F;
  ModelBuilder model(17);
  model.input("input", {1, 3, -1, -1})
      .output("output")
      .weight("w", {12, 3, 1, 1}, weights)
      .node("Conv", {"input", "w"}, {"c"}, {attr_ints("kernel_shape", {1, 1})})
      .node("DepthToSpace", {"c"}, {"output"}, {attr_int("blocksize", 2), attr_string("mode", "CRD")});
  return model.bytes();
}

std::vector<std::byte> rife_blend_model() {
  ModelBuilder model(17);
  model.input("input", {1, 11, -1, -1})
      .output("output")
      .ints("s0", {0})
      .ints("s3", {3})
      .ints("s6", {6})
      .ints("s7", {7})
      .ints("axis", {1})
      .node("Slice", {"input", "s0", "s3", "axis"}, {"a"})
      .node("Slice", {"input", "s3", "s6", "axis"}, {"b"})
      .node("Slice", {"input", "s6", "s7", "axis"}, {"t"})
      .node("Sub", {"b", "a"}, {"d"})
      .node("Mul", {"d", "t"}, {"dt"})
      .node("Add", {"a", "dt"}, {"output"});
  return model.bytes();
}

std::vector<std::byte> motion_model() {
  ModelBuilder model(17);
  model.input("input", {1, 11, -1, -1})
      .output("flow")
      .output("mask")
      .ints("s0", {0})
      .ints("s4", {4})
      .ints("s6", {6})
      .ints("s7", {7})
      .ints("axis", {1})
      .weight("zero", {1}, {0.0F})
      .weight("half", {1}, {0.5F})
      .node("Slice", {"input", "s0", "s4", "axis"}, {"x"})
      .node("Mul", {"x", "zero"}, {"flow"})
      .node("Slice", {"input", "s6", "s7", "axis"}, {"t"})
      .node("Mul", {"t", "zero"}, {"tz"})
      .node("Add", {"tz", "half"}, {"mask"});
  return model.bytes();
}

std::filesystem::path write_model(const std::filesystem::path& dir, const std::string& name,
                                  const std::vector<std::byte>& bytes) {
  const std::filesystem::path path = dir / (name + ".onnx");
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return path;
}

Artifact build_artifact(Backend& backend, const std::filesystem::path& onnx, const std::string& id) {
  Progress progress;
  auto artifact =
      backend.build(ModelSpec{.id = id, .onnx_path = onnx, .sha256 = "test"}, ShapeSet{}, progress);
  return artifact ? *artifact : Artifact{};
}

std::shared_ptr<Engine> load_engine(Backend& backend, const std::filesystem::path& onnx,
                                    const std::string& id) {
  auto engine = backend.load(build_artifact(backend, onnx, id));
  if (!engine) return nullptr;
  return std::shared_ptr<Engine>(std::move(*engine));
}

HostRing::HostRing(std::size_t slots, int width, int height, std::uint32_t generation) {
  const std::size_t samples = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;
  memory_.assign(slots, std::vector<std::uint16_t>(samples, 0));
  for (std::size_t i = 0; i < slots; ++i) {
    slots_.push_back({.index = static_cast<std::uint32_t>(i),
                      .generation = generation,
                      .width = width,
                      .height = height,
                      .image = reinterpret_cast<std::uintptr_t>(memory_[i].data()),
                      .memory_size = samples * sizeof(std::uint16_t)});
  }
}

bool CollectingSink::try_push(RingFrame&& frame) {
  if (unread >= bound_) return false;
  frames.push_back(frame);
  ++unread;
  if (on_frame) on_frame(frames.back());
  return true;
}

std::vector<SlotFree> CollectingSink::free_all() {
  std::vector<SlotFree> freed;
  for (; freed_ < frames.size(); ++freed_) {
    const RingFrame& frame = frames[freed_];
    freed.push_back(
        {.slot = frame.slot, .semaphore_value = frame.ready_value + 1, .generation = frame.generation});
  }
  unread = 0;
  return freed;
}

core::Plan upscale_plan() {
  core::Plan plan;
  plan.upscale = core::UpscaleStep{.passes = {core::ModelPass{.model = "up2", .factor = 2}}};
  plan.output_width = 2 * kWidth;
  plan.output_height = 2 * kHeight;
  plan.output_fps = 25.0;
  plan.hash = core::compute_hash(plan);
  return plan;
}

core::Plan with_level(core::Plan plan, core::Ratio factor, core::Policy policy, core::Schedule schedule,
                      core::Pacing pacing, const std::string& model) {
  plan.levels.push_back(
      {.factor = factor, .model = model, .policy = policy, .pacing = pacing, .schedule = schedule});
  plan.output_fps *= factor.value();
  plan.hash = core::compute_hash(plan);
  return plan;
}

}  // namespace ajp::gpu::test
