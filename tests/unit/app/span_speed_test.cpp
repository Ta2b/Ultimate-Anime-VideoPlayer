// Le mode Performance de SPAN ne perd en qualité que contre un gain mesuré (B02-01) : la préparation compare
// ses noyaux FP8 au SPAN FP16 de TensorRT, le dit, et refuse sous kMinSpeedup. Un paquet d'une porte plus
// ancienne, qui ne l'avait pas mesuré, se refait. Doublures : rien ici ne touche au GPU.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <expected>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/span_derive.h"
#include "app/play/span_int8.h"
#include "french_ui.h"
#include "gpu/span_fp8/derive.h"
#include "span_derive_rig.h"

namespace ajp::app::play {

using namespace span_test;
using namespace std::chrono_literals;

namespace {

// Un temps du SPAN FP16 donné, et la taille à laquelle on l'a demandé.
struct Fp16Clock {
  double ms = 0.0;
  int calls = 0;
  int width = 0;
  int height = 0;
  SpanDerivePlan with(SpanDerivePlan plan) {
    plan.time_fp16 = [this](int w, int h, gpu::Progress&) {
      ++calls;
      width = w;
      height = h;
      return std::expected<double, Refusal>(ms);
    };
    return plan;
  }
};

// Un backend qui note la forme demandée et compte les passes enfilées.
class TimedBackend final : public gpu::Backend {
 public:
  gpu::ShapeSet built;
  int runs = 0;
  int drains = 0;
  gpu::BackendCaps capabilities() const override { return {.name = "factice"}; }
  gpu::DeviceMatch match(const DeviceUuid& /*render_device*/) const override {
    return gpu::DeviceMatch::Same;
  }
  std::expected<gpu::Artifact, Refusal> build(const gpu::ModelSpec& model, const gpu::ShapeSet& shapes,
                                              gpu::Progress& /*progress*/) override {
    built = shapes;
    return gpu::Artifact{.cache_key = model.id, .path = model.onnx_path};
  }
  std::expected<std::unique_ptr<gpu::Engine>, Refusal> load(const gpu::Artifact& /*artifact*/) override {
    return std::make_unique<Span>();
  }
  std::expected<std::unique_ptr<gpu::TensorBuffer>, Refusal> alloc_tensor(gpu::TensorType type,
                                                                          const gpu::Shape& shape) override {
    return std::make_unique<Buffer>(gpu::Tensor{.type = type, .shape = shape, .handle = 1, .offset = 0});
  }
  gpu::KernelOps* kernels() override { return nullptr; }
  std::expected<void, Refusal> enqueue(gpu::Engine& /*engine*/, std::span<const gpu::Tensor> inputs,
                                       std::span<gpu::Tensor> outputs) override {
    if (inputs.size() != 1 || outputs.size() != 1 || outputs[0].shape.dims[2] != 2 * inputs[0].shape.dims[2])
      return std::unexpected(Refusal{RefusalCode::InputMismatch, "tenseurs du SPAN"});
    ++runs;
    return {};
  }
  std::expected<gpu::Fence, Refusal> signal() override { return gpu::Fence{}; }
  std::expected<void, Refusal> wait(const gpu::Fence& /*fence*/) override { return {}; }
  void drain() override { ++drains; }

 private:
  struct Span final : gpu::Engine {
    std::vector<std::string> in{"input"};
    std::vector<std::string> out{"output"};
    std::span<const std::string> input_names() const override { return in; }
    std::span<const std::string> output_names() const override { return out; }
  };
  struct Buffer final : gpu::TensorBuffer {
    explicit Buffer(gpu::Tensor t) : held(t) {}
    gpu::Tensor held;
    const gpu::Tensor& tensor() const override { return held; }
  };
};

std::string receipt_of(const Folder& folder) {
  std::ifstream in(folder.receipt());
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

}  // namespace

TEST_CASE("la préparation mesure le SPAN FP16 de TensorRT et annonce son gain (B02-01)",
          "[app][span_derive][b02]") {
  FrenchUi french;
  const Folder folder;
  const FakeBackend backend;  // noyaux FP8 à 9,2 ms
  Fp16Clock clock{.ms = 11.5};
  SpanDeriveProgress progress;
  const auto report = derive_span_fp8(clock.with(folder.plan()), source_of(twelve()), backend.deriver(),
                                      progress, std::stop_token{});
  REQUIRE(report);
  CHECK(report->passed);
  CHECK(clock.calls == 1);
  CHECK(clock.width == 128);  // à la taille jugée
  CHECK(clock.height == 4);
  CHECK(report->speedup == Catch::Approx(11.5 / 9.2));
  INFO(report->text);
  CHECK(report->text.find("TensorRT FP16 prend ici 11,5 ms") != std::string::npos);
  CHECK(report->text.find("1,25 fois plus vite") != std::string::npos);
  const std::string receipt = receipt_of(folder);
  CHECK(receipt.find("trt_fp16_ms = 11.5000") != std::string::npos);
  CHECK(receipt.find("speedup = 1.2500") != std::string::npos);
  CHECK(receipt.find("timed_during_playback") == std::string::npos);  // préparée à l'ouverture
  const SpanFp8Context here{.gpu_arch = "sm_89", .kernels_version = 1, .width = 128, .height = 4};
  CHECK(find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models(), &here).variant);
}

TEST_CASE("un FP8 qui ne va pas plus vite que le FP16 est refusé, pour de bon (B02-01)",
          "[app][span_derive][b02]") {
  FrenchUi french;
  const Folder folder;
  const FakeBackend backend;
  Fp16Clock clock{.ms = 9.3};  // 1,01 fois : la perte de qualité ne s'achète par rien
  SpanDeriveProgress progress;
  const auto report = derive_span_fp8(clock.with(folder.plan()), source_of(twelve()), backend.deriver(),
                                      progress, std::stop_token{});
  REQUIRE(report);
  CHECK_FALSE(report->passed);
  CHECK(report->too_slow);
  CHECK(report->package.empty());
  INFO(report->text);
  CHECK(report->text.find("Mode Performance refusé : noyaux FP8 9,2 ms contre 9,3 ms") != std::string::npos);
  CHECK(receipt_of(folder).find("passed = false") != std::string::npos);
  const SpanFp8Context here{.gpu_arch = "sm_89", .kernels_version = 1, .width = 128, .height = 4};
  const SpanInt8Lookup found =
      find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models(), &here);
  CHECK_FALSE(found.variant);
  CHECK_FALSE(found.stale);  // la préparation automatique ne le refait pas à chaque vidéo
  CHECK(found.note.find("pas plus rapide que TensorRT FP16") != std::string::npos);
}

TEST_CASE("une qualité refusée ne mesure pas le FP16 ; une taille nouvelle le mesure (B02-01)",
          "[app][span_derive][b02]") {
  FrenchUi french;
  const Folder folder;
  FakeBackend bad;
  bad.psnr = 36.0;
  Fp16Clock clock{.ms = 20.0};
  SpanDeriveProgress progress;
  REQUIRE(derive_span_fp8(clock.with(folder.plan()), source_of(twelve()), bad.deriver(), progress,
                          std::stop_token{}));
  CHECK(clock.calls == 0);

  // Un paquet admis, puis une taille nouvelle trop lente : refusée à cette taille seulement.
  const FakeBackend good;
  REQUIRE(derive_span_fp8(clock.with(folder.plan()), source_of(twelve()), good.deriver(), progress,
                          std::stop_token{}));
  CHECK(clock.calls == 1);
  std::vector<media::Still> bigger = twelve();
  for (media::Still& still : bigger) {
    still.width = 256;
    still.height = 8;
    still.rgb.assign(std::size_t{3} * 256 * 8, still.rgb.front());
  }
  clock.ms = 9.0;
  const auto validated =
      validate_span_size(clock.with(folder.plan()), source_of(bigger), good.deriver(), progress, {});
  REQUIRE(validated);
  CHECK(clock.calls == 2);
  CHECK(clock.width == 256);
  CHECK_FALSE(validated->passed);
  CHECK(validated->too_slow);
  CHECK(receipt_of(folder).find("refused_sizes = [\"256x8\"]") != std::string::npos);
}

TEST_CASE("un paquet admis par une porte plus ancienne se refait (B02-01, B02-02)",
          "[app][span_derive][b02]") {
  FrenchUi french;
  const Folder folder;
  const FakeBackend backend;
  SpanDeriveProgress progress;
  REQUIRE(
      derive_span_fp8(folder.plan(), source_of(twelve()), backend.deriver(), progress, std::stop_token{}));
  std::string text = receipt_of(folder);
  const std::string now = std::format("gate_version = {}", gpu::span_fp8::kGateVersion);
  const auto at = text.find(now);
  REQUIRE(at != std::string::npos);
  text.replace(at, now.size(), "gate_version = 3");
  std::ofstream(folder.receipt(), std::ios::trunc) << text;
  const SpanFp8Context here{.gpu_arch = "sm_89", .kernels_version = 1, .width = 128, .height = 4};
  const SpanInt8Lookup old =
      find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models(), &here);
  CHECK_FALSE(old.variant);
  CHECK(old.stale);
  CHECK(old.note.find("sans mesure du gain") != std::string::npos);
  // Sans contexte (outil, test), le reçu se lit comme avant.
  CHECK(find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models()).variant);
}

TEST_CASE("le temps du SPAN FP16 : le moteur de la lecture à cette taille, des passes enchaînées (B02-01)",
          "[app][span_derive][b02]") {
  const Folder folder;
  TimedBackend backend;
  gpu::Progress progress;
  const auto ms = time_span_fp16(backend, folder.model, 1280, 720, progress);
  REQUIRE(ms);
  CHECK(*ms >= 0.0);
  const gpu::Shape wanted{.dims = {1, 3, 720, 1280}};
  CHECK(backend.built.min == wanted);  // la forme que chain_setup.cpp construit : le moteur est en cache
  CHECK(backend.built.opt == wanted);
  CHECK(backend.built.max == wanted);
  CHECK(backend.runs == kSpanTimingWarmup + kSpanTimingBatches * kSpanTimingBatchRuns);
  CHECK(backend.drains == 1 + kSpanTimingBatches);
  std::stop_source stop;
  stop.request_stop();
  gpu::Progress stopped{.report = {}, .stop = stop.get_token()};
  const auto cancelled = time_span_fp16(backend, folder.model, 1280, 720, stopped);
  REQUIRE_FALSE(cancelled);
  CHECK(cancelled.error().code == RefusalCode::Cancelled);
}

TEST_CASE("pendant la lecture, la préparation laisse la carte entre deux images (B02-08)",
          "[app][span_derive][b02]") {
  const auto spent = [](std::string_view stage, const std::stop_token& stop) {
    const auto start = std::chrono::steady_clock::now();
    pause_for_playback(stage, stop);
    return std::chrono::steady_clock::now() - start;
  };
  CHECK(spent("calibration", {}) >= kBackgroundPause);
  CHECK(spent("validation", {}) >= kBackgroundPause);
  CHECK(spent("mesure du temps", {}) < kBackgroundPause);  // la mesure enchaîne ses passes
  std::stop_source stop;
  stop.request_stop();
  CHECK(spent("calibration", stop.get_token()) < kBackgroundPause);  // un arrêt n'attend pas la pause
}

}  // namespace ajp::app::play
