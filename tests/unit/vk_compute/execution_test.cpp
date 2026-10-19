// Exécution : enregistrement rejoué, sémaphores timeline, barrières, refus du Recorder et
// cache de pipelines.
#include <array>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/compute/sync.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;

namespace {

// La constante de spécialisation 0 de transfer.comp : PQ EOTF.
constexpr std::array kPqOp{static_cast<std::uint32_t>(TransferOp::PqEotf)};

// Deux dispatchs dépendants : PQ inverse vers un tenseur intermédiaire, puis PQ EOTF.
std::unique_ptr<Recording> round_trip(Rig& rig, const TensorView& src, const TensorView& middle,
                                      const TensorView& dst) {
  return record(rig, [&](Recorder& r) -> Done {
    if (auto ok = rig.kernels().transfer(r, TransferOp::PqInverseEotf, src, middle); !ok) return ok;
    return rig.kernels().transfer(r, TransferOp::PqEotf, middle, dst);
  });
}

}  // namespace

TEST_CASE("exécution : un enregistrement rejoué donne le même résultat", "[vk_compute][execution]") {
  Rig rig = make_rig();
  const kernels::RgbImage image = synthetic_rgb(33, 17, 5, 0.0F, 1000.0F);
  TensorBuffer src = upload_rgb(rig, image, Precision::F32);
  TensorBuffer middle = tensor_target(rig, 33, 17, 3, Precision::F32);
  TensorBuffer dst = tensor_target(rig, 33, 17, 3, Precision::F32);
  const std::unique_ptr<Recording> recording = round_trip(rig, src.view, middle.view, dst.view);
  CHECK(recording->dispatches() == 2);
  run(rig, *recording);
  const kernels::RgbImage first = download_rgb(rig, dst);
  // La seconde passe lit ce que la première écrit : sans barrière, l'aller-retour échouerait.
  CHECK(max_abs_diff(first, image) <= 1000.0F * 1e-3F);
  for (int replay = 0; replay < 3; ++replay) {
    run(rig, *recording);
    CHECK(download_rgb(rig, dst) == first);
  }
  CHECK(recording->submitted() == 4);
  // Le rejeu relit ses entrées : une nouvelle image donne un nouveau résultat.
  const kernels::RgbImage other = synthetic_rgb(33, 17, 6, 0.0F, 1000.0F);
  TensorBuffer replacement = upload_rgb(rig, other, Precision::F32);
  write_bytes(rig, src.buffer, read_bytes(rig, replacement.buffer, src.view.buffer.size));
  run(rig, *recording);
  const kernels::RgbImage second = download_rgb(rig, dst);
  CHECK(max_abs_diff(second, other) <= 1000.0F * 1e-3F);
  CHECK(max_abs_diff(second, first) > 1.0F);
}

TEST_CASE("exécution : soumissions enchaînées côté GPU, une seule attente hôte", "[vk_compute][execution]") {
  Rig rig = make_rig();
  const kernels::RgbImage image = synthetic_rgb(64, 64, 9, 0.0F, 1.0F);
  TensorBuffer src = upload_rgb(rig, image, Precision::F32);
  TensorBuffer middle = tensor_target(rig, 64, 64, 3, Precision::F32);
  TensorBuffer dst = tensor_target(rig, 64, 64, 3, Precision::F32);
  const std::unique_ptr<Recording> recording = round_trip(rig, src.view, middle.view, dst.view);
  const std::uint64_t waits = rig.context->host_waits();
  for (int i = 0; i < 5; ++i) require_ok(rig.context->submit(*recording));
  // Aucune soumission n'a attendu l'hôte : chacune attend la précédente sur le sémaphore runs.
  CHECK(rig.context->host_waits() == waits);
  CHECK(recording->last_run().value == 5);
  require_ok(rig.context->wait_on_host(recording->last_run()));
  CHECK(rig.context->host_waits() == waits + 1);
  CHECK(require_value(recording->runs().value()) == 5);
}

TEST_CASE("exécution : attente et signal de sémaphores timeline externes", "[vk_compute][execution]") {
  Rig rig = make_rig();
  const kernels::RgbImage image = synthetic_rgb(16, 16, 2, 0.0F, 1.0F);
  TensorBuffer src = upload_rgb(rig, image, Precision::F32);
  TensorBuffer middle = tensor_target(rig, 16, 16, 3, Precision::F32);
  TensorBuffer dst = tensor_target(rig, 16, 16, 3, Precision::F32);
  const std::unique_ptr<Recording> recording = round_trip(rig, src.view, middle.view, dst.view);
  TimelineSemaphore gate = require_value(TimelineSemaphore::create(*rig.context, 0));
  TimelineSemaphore done = require_value(TimelineSemaphore::create(*rig.context, 0));
  const std::array waits{gate.at(1)};
  const std::array signals{done.at(5)};
  require_ok(rig.context->submit(*recording, waits, signals));
  // Le GPU attend la porte : rien n'est signalé tant que l'hôte ne l'ouvre pas.
  CHECK(require_value(done.value()) == 0);
  CHECK(require_value(recording->runs().value()) == 0);
  require_ok(gate.signal(1));
  require_ok(rig.context->wait_on_host(done.at(5)));
  CHECK(require_value(done.value()) == 5);
  CHECK(require_value(recording->runs().value()) == 1);
  CHECK(max_abs_diff(download_rgb(rig, dst), image) <= 1e-3F);
  // Une attente hôte qui expire est un refus qui cite l'appel.
  const Done late = rig.context->wait_on_host(done.at(6), std::chrono::milliseconds(1));
  require_refusal(late, RefusalCode::VulkanCallFailed);
  CHECK(late.error().detail.find("vkWaitSemaphores") != std::string::npos);
  CHECK(late.error().detail.find("VK_TIMEOUT") != std::string::npos);
}

TEST_CASE("exécution : refus du Recorder et des soumissions", "[vk_compute][execution]") {
  Rig rig = make_rig();
  TensorBuffer a = tensor_target(rig, 16, 16, 3, Precision::F32);
  TensorBuffer b = tensor_target(rig, 16, 16, 3, Precision::F32);
  const Pipeline& pipeline = *require_value(rig.pipelines->get(
      {.shader = "transfer", .constants = kPqOp, .buffers = 2, .images = 0, .push_bytes = 32}));
  const std::array<std::byte, 32> push{};
  const std::array buffers{a.view.buffer, b.view.buffer};

  Recorder recorder =
      require_value(Recorder::begin(*rig.context, *rig.memory, {.sets = 1, .buffers = 8, .images = 1}));
  require_refusal(recorder.dispatch(pipeline, std::span(buffers).first(1), {}, push, {1, 1, 1}),
                  RefusalCode::InputMismatch);
  require_refusal(recorder.dispatch(pipeline, buffers, {}, std::span(push).first(16), {1, 1, 1}),
                  RefusalCode::InputMismatch);
  require_refusal(recorder.dispatch(pipeline, buffers, {}, push, {0, 1, 1}), RefusalCode::FeatureUnavailable);
  const std::uint32_t too_many = rig.context->caps().max_workgroup_count[0] + 1U;
  require_refusal(recorder.dispatch(pipeline, buffers, {}, push, {too_many, 1, 1}),
                  RefusalCode::FeatureUnavailable);
  const std::array empty{BufferView{}, b.view.buffer};
  require_refusal(recorder.dispatch(pipeline, empty, {}, push, {1, 1, 1}), RefusalCode::InputMismatch);
  require_refusal(recorder.fill({.buffer = a.view.buffer.buffer, .offset = 2, .size = 8}, 0),
                  RefusalCode::InputMismatch);
  // Le pool ne tient qu'un ensemble : le second dispatch est refusé avant le pilote.
  require_ok(recorder.dispatch(pipeline, buffers, {}, push, {1, 1, 1}));
  require_refusal(recorder.dispatch(pipeline, buffers, {}, push, {1, 1, 1}), RefusalCode::GpuMemoryExhausted);
  const std::unique_ptr<Recording> recording = require_value(recorder.finish());
  require_refusal(recorder.finish(), RefusalCode::InputMismatch);
  require_refusal(recorder.dispatch(pipeline, buffers, {}, push, {1, 1, 1}), RefusalCode::InputMismatch);

  // Un enregistrement ne se soumet que sur son contexte.
  Rig other = make_rig();
  require_refusal(other.context->submit(*recording), RefusalCode::InputMismatch);
  const std::array null_wait{TimelinePoint{}};
  require_refusal(rig.context->submit(*recording, null_wait), RefusalCode::InputMismatch);
  require_refusal(rig.context->wait_on_host({}), RefusalCode::InputMismatch);
  // Le noyau ne touche qu'aux limites de ses arguments.
  require_refusal(Recorder::begin(*rig.context, *rig.memory, {.sets = 0}), RefusalCode::FieldOutOfRange);
}

TEST_CASE("exécution : une vue mal alignée est refusée", "[vk_compute][execution]") {
  Rig rig = make_rig();
  const VkDeviceSize alignment = rig.context->caps().min_storage_buffer_alignment;
  if (alignment <= 4)
    SKIP("minStorageBufferOffsetAlignment de " << alignment << " : tout offset de mot convient");
  Buffer buffer = make_buffer(rig, 4096);
  const Pipeline& pipeline = *require_value(rig.pipelines->get(
      {.shader = "transfer", .constants = kPqOp, .buffers = 2, .images = 0, .push_bytes = 32}));
  const std::array<std::byte, 32> push{};
  const std::array buffers{BufferView{.buffer = buffer.handle(), .offset = 4, .size = 1024}, buffer.view()};
  Recorder recorder = require_value(Recorder::begin(*rig.context, *rig.memory));
  require_refusal(recorder.dispatch(pipeline, buffers, {}, push, {1, 1, 1}), RefusalCode::InputMismatch);
}

TEST_CASE("pipelines : cache en mémoire par shader et constantes", "[vk_compute][execution]") {
  Rig rig = make_rig();
  PipelineCache& cache = *rig.pipelines;
  const std::array pq{0U};
  const std::array srgb{5U};
  const PipelineDesc desc{.shader = "transfer", .constants = pq, .buffers = 2, .images = 0, .push_bytes = 32};
  const Pipeline* first = require_value(cache.get(desc));
  CHECK(require_value(cache.get(desc)) == first);
  CHECK(cache.size() == 1);
  PipelineDesc other = desc;
  other.constants = srgb;
  CHECK(require_value(cache.get(other)) != first);
  CHECK(cache.size() == 2);
  CHECK(cache.shader_modules() == 1);
  PipelineDesc wrong = desc;
  wrong.buffers = 3;
  require_refusal(cache.get(wrong), RefusalCode::InputMismatch);
  require_refusal(cache.get({.shader = "absent", .buffers = 1}), RefusalCode::FeatureUnavailable);
  require_refusal(cache.get({.shader = "transfer_f16", .buffers = 2, .push_bytes = 30}),
                  RefusalCode::InputMismatch);
  const std::uint32_t over = rig.context->caps().max_push_constants + 4U;
  require_refusal(cache.get({.shader = "transfer_f16", .buffers = 2, .push_bytes = over}),
                  RefusalCode::FeatureUnavailable);
}
