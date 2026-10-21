// Le réseau d'un runtime selon la file du backend : différé au fil du runtime quand la file n'est
// qu'au backend, synchrone sur une file partagée avec l'affichage (sous son verrou), où une
// soumission qui attendrait le signal de l'hôte retiendrait chaque image affichée.
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "backend_rig.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;

namespace {

struct Calls {
  std::atomic<int> run{0};
  std::atomic<int> deferred{0};
};
struct Seen {
  int run = 0;
  int deferred = 0;
};

// Recopie son entrée dans sa sortie, sur ce fil (run) ou sur le sien (run_deferred).
class CopyNet final : public vk::net::RuntimeNet {
 public:
  explicit CopyNet(std::shared_ptr<Calls> calls) : calls_(std::move(calls)) {}
  CopyNet(const CopyNet&) = delete;
  CopyNet& operator=(const CopyNet&) = delete;
  CopyNet(CopyNet&&) = delete;
  CopyNet& operator=(CopyNet&&) = delete;
  ~CopyNet() override {
    if (worker_.joinable()) worker_.join();
  }
  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }
  std::expected<void, Refusal> run(std::span<const vk::net::HostBinding> inputs,
                                   std::span<const vk::net::HostBinding> outputs) override {
    ++calls_->run;
    return copy(inputs, outputs);
  }
  std::expected<void, Refusal> run_deferred(std::span<const vk::net::HostBinding> inputs,
                                            std::span<const vk::net::HostBinding> outputs,
                                            std::function<std::expected<void, Refusal>()> ready,
                                            std::function<void(std::expected<void, Refusal>)> done) override {
    ++calls_->deferred;
    if (worker_.joinable()) worker_.join();
    worker_ = std::thread([in = std::vector(inputs.begin(), inputs.end()),
                           out = std::vector(outputs.begin(), outputs.end()), ready = std::move(ready),
                           done = std::move(done)] {
      auto ok = ready();
      done(ok ? copy(in, out) : ok);
    });
    return {};
  }

 private:
  static std::expected<void, Refusal> copy(std::span<const vk::net::HostBinding> inputs,
                                           std::span<const vk::net::HostBinding> outputs) {
    std::memcpy(outputs[0].data, inputs[0].data, outputs[0].bytes);
    return {};
  }

  std::shared_ptr<Calls> calls_;
  std::vector<std::string> inputs_{"x"};
  std::vector<std::string> outputs_{"y"};
  std::thread worker_;
};

class CopyRuntime final : public vk::net::Runtime {
 public:
  explicit CopyRuntime(std::shared_ptr<Calls> calls) : calls_(std::move(calls)) {}
  std::string_view name() const override { return "copie"; }
  std::string device_name() const override { return "test"; }
  std::string version() const override { return "1"; }
  std::string_view extension() const override { return ".copie"; }
  std::expected<void, Refusal> build(const ModelSpec& /*model*/,
                                     std::span<const vk::net::InputShape> /*inputs*/,
                                     const std::filesystem::path& path, Progress& /*progress*/) override {
    const std::vector<std::byte> bytes{std::byte{1}};
    return vk::net::write_file(path, bytes);
  }
  std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& /*path*/) override {
    return std::make_unique<CopyNet>(calls_);
  }

 private:
  std::shared_ptr<Calls> calls_;
};

struct Locks {
  int lock = 0;
  int unlock = 0;
};
void lock_queue(void* context, std::uint32_t /*family*/, std::uint32_t /*index*/) {
  ++static_cast<Locks*>(context)->lock;
}
void unlock_queue(void* context, std::uint32_t /*family*/, std::uint32_t /*index*/) {
  ++static_cast<Locks*>(context)->unlock;
}

// Un réseau du runtime sur l'appareil d'owner, adopté avec (shared) ou sans verrou de file :
// l'entrée recopiée dans la sortie, et ce que le runtime a reçu.
Seen run_adopted(vk::VulkanBackend& owner, const std::filesystem::path& dir, bool shared, Locks& locks) {
  std::filesystem::create_directories(dir);
  const vk::VkContext& c = owner.context();
  GpuDevice render{.api = GpuApi::Vulkan,
                   .pci_vendor = c.caps().vendor_id,
                   .pci_device = c.caps().device_id,
                   .name = c.caps().device_name,
                   .uuid = c.uuid()};
  VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                       .timelineSemaphore = VK_TRUE};
  const VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                           .pNext = &v12};
  render.vulkan =
      VulkanDevice{.instance = c.instance(),
                   .physical_device = c.physical_device(),
                   .device = c.device(),
                   .get_instance_proc_addr = reinterpret_cast<VoidFunction>(c.get_instance_proc_addr()),
                   .api_version = VK_API_VERSION_1_2,
                   .device_extensions = c.caps().extensions,
                   .enabled_features = &features,
                   .compute = {.family = c.queue_family(), .index = c.queue_index()}};
  if (shared) {
    render.vulkan->lock_queue = lock_queue;
    render.vulkan->unlock_queue = unlock_queue;
    render.vulkan->queue_context = &locks;
  }
  auto calls = std::make_shared<Calls>();
  vk::BackendOptions options{.runtime = [calls](const vk::DeviceCaps&) {
    return std::expected<std::shared_ptr<vk::net::Runtime>, Refusal>(std::make_shared<CopyRuntime>(calls));
  }};
  auto backend = require_value(
      vk::VulkanBackend::create(BackendInit{.render = &render, .cache_dir = dir}, std::move(options)));
  CHECK(backend->context().shares_queue() == shared);

  ir::test::ModelBuilder m;
  m.input("x", {1, 3, 8, 8}).node("Relu", {"x"}, {"y"}).output("y");
  const ModelSpec model = write_model(dir, shared ? "partagee" : "propre", m.bytes(), TensorType::F32);
  Progress progress;
  const Shape shape{{1, 3, 8, 8}};
  const Artifact artifact =
      require_value(backend->build(model, {.min = shape, .opt = shape, .max = shape}, progress));
  auto engine = require_value(backend->load(artifact));
  auto in = require_value(backend->alloc_tensor(TensorType::F32, shape));
  auto out = require_value(backend->alloc_tensor(TensorType::F32, shape));
  constexpr std::size_t kValues = 192;  // 3 × 8 × 8
  std::vector<std::byte> bytes(kValues * sizeof(float));
  for (std::size_t k = 0; k < bytes.size(); ++k) bytes[k] = static_cast<std::byte>(k * 7 + 1);
  write_tensor(*backend, in->tensor(), bytes);
  const std::array<Tensor, 1> inputs{in->tensor()};
  std::array<Tensor, 1> outputs{out->tensor()};
  require_ok(backend->enqueue(*engine, inputs, outputs));
  require_ok(backend->context().wait_on_host(backend->queue().last()));
  CHECK(read_tensor(*backend, out->tensor(), bytes.size()) == bytes);
  return Seen{.run = calls->run.load(), .deferred = calls->deferred.load()};
}

}  // namespace

TEST_CASE("Runtime : différé sur la file du backend, synchrone sur une file partagée",
          "[vk_backend][runtime]") {
  TempDir dir("file-runtime");
  auto owner = make_backend(dir.path() / "owner");
  {
    Locks locks;
    const Seen seen = run_adopted(*owner, dir.path() / "propre", false, locks);
    CHECK(seen.deferred == 1);
    CHECK(seen.run == 0);
    CHECK(locks.lock == 0);
  }
  {
    Locks locks;
    const Seen seen = run_adopted(*owner, dir.path() / "partagee", true, locks);
    // Aucune soumission n'attend sur la file partagée un signal que l'hôte donnera plus tard.
    CHECK(seen.deferred == 0);
    CHECK(seen.run == 1);
    CHECK(locks.lock > 0);
    CHECK(locks.lock == locks.unlock);
  }
}
