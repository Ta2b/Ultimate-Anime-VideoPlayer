// Contexte : chargeur introuvable, création et rapport des capacités, choix par UUID, et
// adoption d'un appareil créé par le test, qui reste à lui.
#include <array>
#include <filesystem>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/compute/functions.h"
#include "gpu/vulkan/compute/library.h"
#include "gpu/vulkan/compute/sync.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;

namespace {

// Un appareil que le test crée et détruit lui-même, comme libplacebo pour le lecteur.
struct OwnDevice {
  VulkanLibrary library;
  GlobalFunctions global;
  InstanceFunctions instance_fn;
  DeviceFunctions device_fn;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t family = 0;
  std::uint32_t api = VK_API_VERSION_1_2;
  VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                       .timelineSemaphore = VK_TRUE};
  VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12};

  explicit OwnDevice(VulkanLibrary lib) : library(std::move(lib)) {}
  OwnDevice(const OwnDevice&) = delete;
  OwnDevice& operator=(const OwnDevice&) = delete;
  OwnDevice(OwnDevice&&) = delete;
  OwnDevice& operator=(OwnDevice&&) = delete;
  ~OwnDevice() {
    if (device != VK_NULL_HANDLE) device_fn.vkDestroyDevice(device, nullptr);
    if (instance != VK_NULL_HANDLE) instance_fn.vkDestroyInstance(instance, nullptr);
  }

  AdoptedDevice adopted() const {
    return {.get_instance_proc_addr = library.get_instance_proc_addr(),
            .instance = instance,
            .physical_device = physical,
            .device = device,
            .api_version = api,
            .queue_family = family,
            .queue_index = 0,
            .features = &features,
            .extensions = {}};
  }
};

// Instance Vulkan 1.2, premier appareil, première famille compute, timeline activé.
std::unique_ptr<OwnDevice> own_device() {
  auto library = VulkanLibrary::open_system();
  if (!library) SKIP("aucun Vulkan : " << describe(library.error()));
  auto own = std::make_unique<OwnDevice>(std::move(*library));
  own->global = require_value(load_global(own->library.get_instance_proc_addr()));
  const VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = own->api};
  const VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
  REQUIRE(own->global.vkCreateInstance(&info, nullptr, &own->instance) == VK_SUCCESS);
  own->instance_fn = require_value(load_instance(own->library.get_instance_proc_addr(), own->instance));
  std::uint32_t count = 1;
  const VkResult listed = own->instance_fn.vkEnumeratePhysicalDevices(own->instance, &count, &own->physical);
  if (count == 0 || own->physical == VK_NULL_HANDLE) SKIP("aucun appareil Vulkan");
  REQUIRE((listed == VK_SUCCESS || listed == VK_INCOMPLETE));
  std::uint32_t families = 0;
  own->instance_fn.vkGetPhysicalDeviceQueueFamilyProperties(own->physical, &families, nullptr);
  std::vector<VkQueueFamilyProperties> props(families);
  own->instance_fn.vkGetPhysicalDeviceQueueFamilyProperties(own->physical, &families, props.data());
  const auto compute = std::ranges::find_if(
      props, [](const VkQueueFamilyProperties& p) { return (p.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; });
  REQUIRE(compute != props.end());
  own->family = static_cast<std::uint32_t>(compute - props.begin());
  const float priority = 1.0F;
  const VkDeviceQueueCreateInfo queue{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                      .queueFamilyIndex = own->family,
                                      .queueCount = 1,
                                      .pQueuePriorities = &priority};
  const VkDeviceCreateInfo device{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                  .pNext = &own->features,
                                  .queueCreateInfoCount = 1,
                                  .pQueueCreateInfos = &queue};
  REQUIRE(own->instance_fn.vkCreateDevice(own->physical, &device, nullptr, &own->device) == VK_SUCCESS);
  own->device_fn = require_value(load_device(own->instance_fn, own->device));
  return own;
}

}  // namespace

TEST_CASE("chargeur : bibliothèque Vulkan introuvable ou sans vkGetInstanceProcAddr",
          "[vk_compute][context]") {
  const std::filesystem::path missing = "/nonexistent/ajp/libvulkan-factice.so.1";
  require_refusal(VulkanLibrary::open(missing), RefusalCode::LibraryUnavailable);
  require_refusal(VkContext::create({.library = missing}), RefusalCode::LibraryUnavailable);
  // Une vraie bibliothèque partagée, construite pour le test, qui n'exporte pas le point d'entrée.
  const std::filesystem::path fake = AJP_TEST_FAKE_VULKAN;
  REQUIRE(std::filesystem::exists(fake));
  require_refusal(VulkanLibrary::open(fake), RefusalCode::LibraryUnavailable);
  require_refusal(VkContext::create({.library = fake}), RefusalCode::LibraryUnavailable);
  require_refusal(load_global(nullptr), RefusalCode::LibraryUnavailable);
}

TEST_CASE("contexte : appareil créé, file compute, capacités rapportées", "[vk_compute][context]") {
  Rig rig = make_rig();
  const VkContext& ctx = *rig.context;
  const DeviceCaps& caps = ctx.caps();
  CHECK(ctx.owns_device());
  CHECK(ctx.queue() != VK_NULL_HANDLE);
  CHECK(caps.api_version >= kMinApiVersion);
  CHECK(caps.enabled.timeline_semaphore);
  CHECK(caps.uuid != DeviceUuid{});
  CHECK(caps.max_push_constants >= 128);
  CHECK(caps.max_workgroup_invocations >= 256);
  const std::string report = describe(caps);
  INFO(report);
  CHECK(report.find(caps.device_name) != std::string::npos);
  CHECK(report.find(uuid_text(caps.uuid)) != std::string::npos);
  CHECK(report.find("timeline") != std::string::npos);
  CHECK(ctx.host_waits() == 0);
}

TEST_CASE("contexte : appareil choisi par UUID, UUID inconnu refusé", "[vk_compute][context]") {
  Rig rig = make_rig();
  const DeviceUuid uuid = rig.context->uuid();
  auto same = require_value(VkContext::create({.device = uuid}));
  CHECK(same->uuid() == uuid);
  CHECK(same->caps().device_name == rig.context->caps().device_name);

  DeviceUuid unknown{};
  unknown.fill(0xEE);
  const auto refused = VkContext::create({.device = unknown});
  require_refusal(refused, RefusalCode::DeviceUnavailable);
  CHECK(refused.error().detail.find(uuid_text(uuid)) != std::string::npos);
}

TEST_CASE("contexte : couche de validation demandée", "[vk_compute][context]") {
  Rig rig = make_rig();
  auto validated = VkContext::create({.validation = true});
  if (!validated) {
    // La couche n'est pas installée partout : son absence est un refus nommé, pas un succès.
    require_refusal(validated, RefusalCode::FeatureUnavailable);
    CHECK(validated.error().detail.find("VK_LAYER_KHRONOS_validation") != std::string::npos);
  } else {
    CHECK((*validated)->caps().enabled.timeline_semaphore);
  }
}

TEST_CASE("contexte : extension d'appareil absente refusée", "[vk_compute][context]") {
  Rig rig = make_rig();
  require_refusal(VkContext::create({.extensions = {"VK_AJP_extension_inexistante"}}),
                  RefusalCode::FeatureUnavailable);
}

TEST_CASE("adoption : appareil du test utilisé puis laissé intact", "[vk_compute][context]") {
  const std::unique_ptr<OwnDevice> own = own_device();
  int locks = 0;
  int unlocks = 0;
  AdoptedDevice adopted = own->adopted();
  adopted.lock_queue = [&locks] { ++locks; };
  adopted.unlock_queue = [&unlocks] { ++unlocks; };
  {
    Rig rig;
    rig.context = require_value(VkContext::adopt(std::move(adopted)));
    CHECK_FALSE(rig.context->owns_device());
    CHECK(rig.context->device() == own->device);
    CHECK(rig.context->caps().enabled.timeline_semaphore);
    rig.memory = require_value(MemoryAllocator::create(*rig.context));
    rig.pipelines = require_value(PipelineCache::create(*rig.context));
    rig.runner.emplace(*rig.context, *rig.pipelines);

    // Un noyau complet sur l'appareil adopté : la conversion de primaires, exacte en float.
    const kernels::RgbImage image = synthetic_rgb(8, 6, 3);
    TensorBuffer src = upload_rgb(rig, image, Precision::F32);
    TensorBuffer dst = tensor_target(rig, 8, 6, 3, Precision::F32);
    record_and_run(rig, [&](Recorder& r) {
      return rig.kernels().transfer(r, TransferOp::Bt709ToBt2020, src.view, dst.view);
    });
    CHECK(max_abs_diff(download_rgb(rig, dst), image) > 0.0F);
    CHECK(locks > 0);
    CHECK(locks == unlocks);
  }
  // Le contexte détruit, l'appareil appartient toujours au test.
  CHECK(own->device_fn.vkDeviceWaitIdle(own->device) == VK_SUCCESS);
  auto semaphore = create_timeline(own->device_fn, own->device, 0);
  REQUIRE(semaphore);
  own->device_fn.vkDestroySemaphore(own->device, *semaphore, nullptr);
}

TEST_CASE("adoption : appareil incomplet, version, timeline ou file refusés", "[vk_compute][context]") {
  const std::unique_ptr<OwnDevice> own = own_device();
  AdoptedDevice no_device = own->adopted();
  no_device.device = VK_NULL_HANDLE;
  require_refusal(VkContext::adopt(no_device), RefusalCode::DeviceUnavailable);

  AdoptedDevice old = own->adopted();
  old.api_version = VK_API_VERSION_1_1;
  require_refusal(VkContext::adopt(old), RefusalCode::FeatureUnavailable);

  AdoptedDevice no_timeline = own->adopted();
  no_timeline.features = nullptr;
  require_refusal(VkContext::adopt(no_timeline), RefusalCode::FeatureUnavailable);

  AdoptedDevice bad_family = own->adopted();
  bad_family.queue_family = 99;
  require_refusal(VkContext::adopt(bad_family), RefusalCode::DeviceUnavailable);

  AdoptedDevice bad_queue = own->adopted();
  bad_queue.queue_index = 7;
  require_refusal(VkContext::adopt(bad_queue), RefusalCode::DeviceUnavailable);
}
