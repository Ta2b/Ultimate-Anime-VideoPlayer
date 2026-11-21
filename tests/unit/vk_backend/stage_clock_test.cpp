// Le profil GPU de la lecture sous Vulkan (stage_clock.h, KernelOps::gpu_profile) : chaque étape de la chaîne
// et chaque moteur mesurés aux noms de TensorRT, sans rien changer à ce que la file signale ni à l'image.
#include "gpu/vulkan/backend/stage_clock.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/image.h"
#include "gpu/vulkan/compute/sync.h"
#include "kernels/ref/color.h"
#include "kernels/ref/resample.h"
#include "kernels/spec/tolerance.h"
#include "net_rig.h"
#include "test_models.h"

using namespace kirari;
using namespace kirari::gpu;
using namespace kirari::gpu::vk::test;
namespace k = kirari::kernels;

namespace {

const GpuStageSummary* find_stage(const std::vector<GpuStageSummary>& stages, std::string_view name) {
  const auto found = std::ranges::find(stages, name, &GpuStageSummary::name);
  return found == stages.end() ? nullptr : &*found;
}

// Les signaux d'un même lot Vulkan ne sont pas ordonnés. Une jointure sur la même file crée un lot distinct,
// dont le signal vient après les signaux des estampilles, sans attendre l'appareil entier.
void settle_stamps(vk::VulkanBackend& backend) {
  auto& context = backend.context();
  auto done = require_value(vk::TimelineSemaphore::create(context, 0));
  const std::array signals{done.at(1)};
  require_ok(context.enqueue_join({}, signals, backend.queue().current_file()));
  require_ok(context.wait_on_host(signals.front()));
}

void check_measured(const GpuStageSummary& s, std::size_t count) {
  INFO(s.name);
  CHECK(s.count == count);
  CHECK(std::isfinite(s.total_ms));
  CHECK(s.total_ms >= 0.0);
  CHECK(s.median_ms >= 0.0);
  CHECK(s.p95_ms >= s.median_ms);
}

}  // namespace

TEST_CASE("Profil GPU : les noyaux de la chaîne, aux noms de TensorRT", "[vk_backend][stage_clock]") {
  TempDir dir("profil");
  auto backend = make_backend(dir.path());
  auto& ops = static_cast<vk::VulkanKernelOps&>(*backend->kernels());
  if (ops.clock() == nullptr) SKIP("pas d'horodatage sur la file de calcul : profil vide");
  CHECK(ops.gpu_profile().empty());  // rien de mesuré avant la première étape

  const k::RgbImage image = synthetic_rgb(40, 24, 5);
  auto in = rgb_tensor(*backend, 40, 24, TensorType::F32);
  write_tensor(*backend, in->tensor(), rgb_bytes(image, TensorType::F32));
  auto big = rgb_tensor(*backend, 80, 48, TensorType::F32);
  auto small = rgb_tensor(*backend, 20, 12, TensorType::F32);
  for (int i = 0; i < 3; ++i) {
    require_ok(ops.resample(in->tensor(), k::Filter::Spline36, big->tensor()));
    require_ok(ops.resample(in->tensor(), k::Filter::Area, small->tensor()));
  }
  settle_stamps(*backend);
  const std::vector<GpuStageSummary> stages = ops.gpu_profile();
  const GpuStageSummary* up = find_stage(stages, "agrandissement Spline36");
  const GpuStageSummary* down = find_stage(stages, "réduction Area");
  REQUIRE(up != nullptr);
  REQUIRE(down != nullptr);
  check_measured(*up, 3);
  check_measured(*down, 3);
  // Mesurer ne change pas l'image.
  const auto want = k::ref::resize(image, 80, 48, k::Filter::Spline36);
  REQUIRE(want);
  const k::RgbImage got =
      rgb_from(read_tensor(*backend, big->tensor(), 4U * 3 * 80 * 48), 80, 48, TensorType::F32);
  CHECK(max_abs_diff(got, *want) <= k::tolerance::kRgbF16);
}

TEST_CASE("Profil GPU : une étape qui signale un slot part avec son estampille de fin",
          "[vk_backend][stage_clock]") {
  TempDir dir("profil-ring");
  auto backend = make_backend(dir.path());
  auto& ops = static_cast<vk::VulkanKernelOps&>(*backend->kernels());
  if (ops.clock() == nullptr) SKIP("pas d'horodatage sur la file de calcul : profil vide");
  auto image = vk::StorageImage::create(backend->context(), backend->memory(), 20, 12);
  if (!image && image.error().code == RefusalCode::FeatureUnavailable) SKIP(describe(image.error()));
  REQUIRE(image);
  auto semaphore = require_value(vk::TimelineSemaphore::create(backend->context(), 0));
  const auto bits = [](auto handle) { return vk::detail::to_bits(handle); };
  const RingSlotDesc slot{.index = 0,
                          .generation = 1,
                          .width = 20,
                          .height = 12,
                          .image = bits(image->image()),
                          .semaphore = bits(semaphore.handle())};
  const k::RgbImage rgb = synthetic_rgb(20, 12, 21);
  auto t = rgb_tensor(*backend, 20, 12, TensorType::F32);
  write_tensor(*backend, t->tensor(), rgb_bytes(rgb, TensorType::F32));
  const std::uint64_t before = backend->context().queue_submits();
  require_ok(ops.write_ring(t->tensor(), slot, {.wait_value = 0, .signal_value = 1}));
  // Le slot est signalé sans autre appel : le lot est parti, estampille de fin comprise.
  CHECK(backend->context().queue_submits() > before);
  require_ok(backend->context().wait_on_host(semaphore.at(1)));
  settle_stamps(*backend);
  const std::vector<GpuStageSummary> stages = ops.gpu_profile();
  const GpuStageSummary* ring = find_stage(stages, "écriture du ring");
  REQUIRE(ring != nullptr);
  check_measured(*ring, 1);
  const auto codes = require_value(vk::read_rgba16(backend->context(), backend->memory(), *image));
  const auto want = k::ref::to_rgba16(rgb);
  REQUIRE(want);
  CHECK(codes == *want);
}

TEST_CASE("Profil GPU : un moteur agrandit, son rôle et sa forme", "[vk_backend][stage_clock]") {
  TempDir dir("profil-moteur");
  auto backend = make_backend(dir.path() / "cache", {.autotune = false});
  auto& ops = static_cast<vk::VulkanKernelOps&>(*backend->kernels());
  if (ops.clock() == nullptr) SKIP("pas d'horodatage sur la file de calcul : profil vide");
  const bool half = backend->profile().target.f16_storage;
  const TensorType type = half ? TensorType::F16 : TensorType::F32;
  const ModelSpec model = write_model(dir.path(), "span_profil", span_like(half).bytes(), type);
  const Shape shape{{1, 3, 24, 32}};
  Progress progress;
  const Artifact artifact =
      require_value(backend->build(model, ShapeSet{.min = shape, .opt = shape, .max = shape}, progress));
  auto engine = require_value(backend->load(artifact));
  auto in = require_value(backend->alloc_tensor(type, shape));
  auto out = require_value(backend->alloc_tensor(type, Shape{{1, 3, 48, 64}}));
  std::array inputs{in->tensor()};
  std::array outputs{out->tensor()};
  require_ok(backend->enqueue(*engine, inputs, outputs));
  require_ok(backend->enqueue(*engine, inputs, outputs));
  // Les estampilles ne comptent pas dans la file : deux moteurs, deux valeurs, comme sans profil.
  const Fence fence = require_value(backend->signal());
  CHECK(fence.value == 2);
  require_ok(backend->wait(fence));
  settle_stamps(*backend);
  const std::vector<GpuStageSummary> stages = ops.gpu_profile();
  const GpuStageSummary* net = find_stage(stages, "moteur 3×24×32");
  const GpuStageSummary* role = find_stage(stages, "rôle agrandissement");
  REQUIRE(net != nullptr);
  REQUIRE(role != nullptr);
  check_measured(*net, 2);
  check_measured(*role, 2);
  CHECK(find_stage(stages, "rôle interpolation") == nullptr);
}
