// Le mode Performance de SPAN ne perd en qualité que contre un gain mesuré (B02-01) : la préparation fait
// chronométrer au backend ses noyaux FP8 avec le SPAN FP16 de TensorRT (C02-01), à la taille de travail
// (C02-05), le dit, et refuse sous kMinSpeedup ; la variante f16 ne se garde que si elle gagne (C02-03), un
// écrêtage sur la validation refuse (C02-04). Un paquet d'une porte plus ancienne se refait. Doublures : rien
// ici ne touche au GPU.
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

// Le moteur du SPAN FP16 à chronométrer, et la taille à laquelle on l'a demandé.
struct Fp16Engine {
  int calls = 0;
  int width = 0;
  int height = 0;
  SpanDerivePlan with(SpanDerivePlan plan) {
    plan.fp16_engine = [this](int w, int h, gpu::Progress&) {
      ++calls;
      width = w;
      height = h;
      return std::expected<gpu::Artifact, Refusal>(
          gpu::Artifact{.cache_key = "span-fp16", .path = "span.engine"});
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
  FakeBackend backend;  // noyaux FP8 à 9,2 ms
  backend.trt_ms = 11.5;
  Fp16Engine engine;
  SpanDeriveProgress progress;
  const auto report = derive_span_fp8(engine.with(folder.plan()), source_of(twelve()), backend.deriver(),
                                      progress, std::stop_token{});
  REQUIRE(report);
  CHECK(report->passed);
  CHECK(engine.calls == 1);
  CHECK(engine.width == 128);  // sans taille de travail : celle des images
  CHECK(engine.height == 4);
  REQUIRE(backend.seen.fp16);  // le backend le chronomètre avec les noyaux FP8 (C02-01)
  CHECK(backend.seen.fp16->cache_key == "span-fp16");
  CHECK(backend.seen.timing_width == 128);
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
  FakeBackend backend;
  backend.trt_ms = 9.3;  // 1,01 fois : la perte de qualité ne s'achète par rien
  Fp16Engine engine;
  SpanDeriveProgress progress;
  const auto report = derive_span_fp8(engine.with(folder.plan()), source_of(twelve()), backend.deriver(),
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

TEST_CASE("une qualité refusée n'exige pas de gain ; une taille nouvelle le mesure (B02-01)",
          "[app][span_derive][b02]") {
  FrenchUi french;
  const Folder folder;
  FakeBackend bad;
  bad.psnr = 36.0;
  bad.trt_ms = 20.0;
  Fp16Engine engine;
  SpanDeriveProgress progress;
  const auto refused = derive_span_fp8(engine.with(folder.plan()), source_of(twelve()), bad.deriver(),
                                       progress, std::stop_token{});
  REQUIRE(refused);
  CHECK_FALSE(refused->passed);
  CHECK_FALSE(refused->too_slow);  // refusé pour sa qualité, pas pour sa vitesse
  CHECK(refused->speedup == 0.0);

  // Un paquet admis, puis une taille nouvelle trop lente : refusée à cette taille seulement.
  FakeBackend good;
  good.trt_ms = 20.0;
  REQUIRE(derive_span_fp8(engine.with(folder.plan()), source_of(twelve()), good.deriver(), progress,
                          std::stop_token{}));
  CHECK(engine.calls == 2);
  std::vector<media::Still> bigger = twelve();
  for (media::Still& still : bigger) {
    still.width = 256;
    still.height = 8;
    still.rgb.assign(std::size_t{3} * 256 * 8, still.rgb.front());
  }
  good.trt_ms = 9.0;
  const auto validated =
      validate_span_size(engine.with(folder.plan()), source_of(bigger), good.deriver(), progress, {});
  REQUIRE(validated);
  CHECK(engine.calls == 3);
  CHECK(engine.width == 256);
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
  CHECK(old.note.find("porte de qualité et de vitesse plus ancienne") != std::string::npos);
  // Sans contexte (outil, test), le reçu se lit comme avant.
  CHECK(find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models()).variant);
}

TEST_CASE("le moteur du SPAN FP16 chronométré est celui de la lecture à cette taille (B02-01)",
          "[app][span_derive][b02]") {
  const Folder folder;
  TimedBackend backend;
  gpu::Progress progress;
  const auto engine = build_span_fp16(backend, folder.model, 1280, 720, progress);
  REQUIRE(engine);
  const gpu::Shape wanted{.dims = {1, 3, 720, 1280}};
  CHECK(backend.built.min == wanted);  // la forme que chain_setup.cpp construit : le moteur est en cache
  CHECK(backend.built.opt == wanted);
  CHECK(backend.built.max == wanted);
  const auto none = build_span_fp16(backend, folder.model, 0, 720, progress);
  REQUIRE_FALSE(none);
  CHECK(none.error().code == RefusalCode::ShapeInvalid);
}

TEST_CASE("les temps se prennent à la taille de travail, que le reçu range (C02-05)",
          "[app][span_derive][c02]") {
  FrenchUi french;
  const Folder folder;
  FakeBackend backend;
  backend.trt_ms = 11.5;
  Fp16Engine engine;
  SpanDerivePlan plan = engine.with(folder.plan());
  plan.work_width = 2561;  // impaire : ramenée à 2560, ce que les noyaux prennent
  plan.work_height = 1440;
  SpanDeriveProgress progress;
  const auto report =
      derive_span_fp8(plan, source_of(twelve()), backend.deriver(), progress, std::stop_token{});
  REQUIRE(report);
  CHECK(report->passed);
  CHECK(engine.width == 2560);  // le moteur FP16 de la lecture, pas celui de la découpe
  CHECK(engine.height == 1440);
  CHECK(backend.seen.timing_width == 2560);
  CHECK(backend.seen.timing_height == 1440);
  CHECK(backend.seen.validation.front().width == 128);  // la qualité, sur la découpe
  CHECK(report->size == "2560x1440");
  const std::string receipt = receipt_of(folder);
  CHECK(receipt.find("validation_size = \"128x4\"") != std::string::npos);
  CHECK(receipt.find("timed_size = \"2560x1440\"") != std::string::npos);
  CHECK(receipt.find("validated_sizes = [\"2560x1440\"]") != std::string::npos);
  const SpanFp8Context here{.gpu_arch = "sm_89", .kernels_version = 1, .width = 2560, .height = 1440};
  CHECK(find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models(), &here).variant);
  // Une autre taille au-delà de la découpe n'est pas jugée par celle-ci.
  const SpanFp8Context other{.gpu_arch = "sm_89", .kernels_version = 1, .width = 3840, .height = 2160};
  const SpanInt8Lookup found =
      find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models(), &other);
  CHECK_FALSE(found.variant);
  CHECK(found.unvalidated_size == "3840x2160");
}

TEST_CASE("la variante f16 n'est gardée que si elle va plus vite (C02-03)", "[app][span_derive][c02]") {
  FrenchUi french;
  const Folder folder;
  FakeBackend backend;
  backend.trt_ms = 11.5;
  backend.psnr16 = 49.8;  // −0,2 dB : dans la tolérance de 0,3 dB
  backend.ms16 = 9.6;     // plus lente que les 9,2 ms de la variante f32
  Fp16Engine engine;
  SpanDeriveProgress progress;
  const auto slower = derive_span_fp8(engine.with(folder.plan()), source_of(twelve()), backend.deriver(),
                                      progress, std::stop_token{});
  REQUIRE(slower);
  CHECK(slower->passed);
  CHECK_FALSE(slower->acc16);
  CHECK(receipt_of(folder).find("accumulator = \"f32\"") != std::string::npos);

  backend.ms16 = 8.28;  // 10 % plus rapide : gardée, et annoncée avec sa perte
  const auto faster = derive_span_fp8(engine.with(folder.plan()), source_of(twelve()), backend.deriver(),
                                      progress, std::stop_token{});
  REQUIRE(faster);
  CHECK(faster->acc16);
  CHECK(faster->acc16_loss_db == Catch::Approx(0.2));
  INFO(faster->text);
  CHECK(faster->text.find("elle coûte 0,2 dB de PSNR moyen et fait gagner 10,0 % du temps") !=
        std::string::npos);
  const std::string receipt = receipt_of(folder);
  CHECK(receipt.find("accumulator = \"f16\"") != std::string::npos);
  CHECK(receipt.find("acc16_psnr_loss_db = 0.2000") != std::string::npos);
  CHECK(receipt.find("kernels_ms_f32 = 9.2000") != std::string::npos);
}

TEST_CASE("un tenseur écrêté sur les images de validation refuse le paquet (C02-04)",
          "[app][span_derive][c02]") {
  FrenchUi french;
  const Folder folder;
  FakeBackend backend;  // échelles à 0,05 : 22,4 de plage
  backend.trt_ms = 11.5;
  backend.validation_peak = 30.0F;
  Fp16Engine engine;
  SpanDeriveProgress progress;
  const auto report = derive_span_fp8(engine.with(folder.plan()), source_of(twelve()), backend.deriver(),
                                      progress, std::stop_token{});
  REQUIRE(report);
  CHECK_FALSE(report->passed);
  CHECK(report->package.empty());
  CHECK(report->verdict.clipped_tensors == 25);
  INFO(report->text);
  CHECK(report->text.find("25 tenseurs FP8 seraient écrêtés") != std::string::npos);
  const std::string receipt = receipt_of(folder);
  CHECK(receipt.find("passed = false") != std::string::npos);
  CHECK(receipt.find("clipped_tensors = 25") != std::string::npos);
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
  CHECK(spent("mesure du temps", {}) >= kBackgroundPause);  // entre deux lots de la mesure aussi (C02-02)
  CHECK(spent("référence FP16", {}) < kBackgroundPause);
  std::stop_source stop;
  stop.request_stop();
  CHECK(spent("calibration", stop.get_token()) < kBackgroundPause);  // un arrêt n'attend pas la pause
}

}  // namespace ajp::app::play
