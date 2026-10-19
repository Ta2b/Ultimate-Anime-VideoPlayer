// Contexte : chargeur introuvable, création et rapport des capacités, choix par UUID, et
// adoption d'un appareil créé par le test, qui reste à lui.
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/compute/functions.h"
#include "gpu/vulkan/compute/library.h"
#include "gpu/vulkan/compute/sync.h"
#include "kernels/ref/transfer.h"
#include "kernels/spec.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;

namespace {

// Erreurs reçues par le messager de l'instance du test (couche de validation présente).
VKAPI_ATTR VkBool32 VKAPI_CALL count_errors(VkDebugUtilsMessageSeverityFlagBitsEXT /*severity*/,
                                            VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                            const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
  static_cast<std::atomic<std::uint64_t>*>(user)->fetch_add(1);
  // Appelé depuis le thread de l'appel Vulkan, pas forcément celui du test : stderr, pas Catch2.
  std::fprintf(stderr, "validation : %s\n",
               data != nullptr && data->pMessage != nullptr ? data->pMessage : "?");
  return VK_FALSE;
}

// Un appareil que le test crée et détruit lui-même, comme libplacebo pour le lecteur. Avec
// VK_EXT_debug_utils, un messager compte les erreurs de validation : un test qui se termine
// normalement échoue s'il en a reçu, destruction de l'appareil comprise.
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
  std::atomic<std::uint64_t> errors{0};
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger = nullptr;

  explicit OwnDevice(VulkanLibrary lib) : library(std::move(lib)) {}
  OwnDevice(const OwnDevice&) = delete;
  OwnDevice& operator=(const OwnDevice&) = delete;
  OwnDevice(OwnDevice&&) = delete;
  OwnDevice& operator=(OwnDevice&&) = delete;
  ~OwnDevice() {
    if (device != VK_NULL_HANDLE) device_fn.vkDestroyDevice(device, nullptr);
    if (messenger != VK_NULL_HANDLE) destroy_messenger(instance, messenger, nullptr);
    if (instance != VK_NULL_HANDLE) instance_fn.vkDestroyInstance(instance, nullptr);
    if (std::uncaught_exceptions() == 0) CHECK(errors.load() == 0);
  }

  VkDebugUtilsMessengerCreateInfoEXT messenger_info() {
    return {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
            .pfnUserCallback = &count_errors,
            .pUserData = &errors};
  }

  AdoptedDevice adopted() const {
    return {.get_instance_proc_addr = library.get_instance_proc_addr(),
            .instance = instance,
            .physical_device = physical,
            .device = device,
            .api_version = api,
            .queue_family = family,
            .queue_index = 0,
            .queue_count = 1,
            .features = &features,
            .extensions = {}};
  }
};

// Instance Vulkan 1.2, premier appareil, première famille compute, timeline activé.
std::unique_ptr<OwnDevice> own_device() {
  auto library = VulkanLibrary::open_system();
  if (!library) SKIP("aucun Vulkan : " << describe(library.error()));
  auto own = std::make_unique<OwnDevice>(std::move(*library));
  const PFN_vkGetInstanceProcAddr gipa = own->library.get_instance_proc_addr();
  own->global = require_value(load_global(gipa));
  std::uint32_t count = 0;
  REQUIRE(own->global.vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) == VK_SUCCESS);
  std::vector<VkExtensionProperties> available(count);
  REQUIRE(own->global.vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data()) ==
          VK_SUCCESS);
  const bool debug = has_extension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  const char* const debug_name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
  const VkDebugUtilsMessengerCreateInfoEXT messenger = own->messenger_info();
  const VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = own->api};
  const VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                  .pNext = debug ? &messenger : nullptr,
                                  .pApplicationInfo = &app,
                                  .enabledExtensionCount = debug ? 1U : 0U,
                                  .ppEnabledExtensionNames = &debug_name};
  REQUIRE(own->global.vkCreateInstance(&info, nullptr, &own->instance) == VK_SUCCESS);
  own->instance_fn = require_value(load_instance(gipa, own->instance));
  if (debug) {
    const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        gipa(own->instance, "vkCreateDebugUtilsMessengerEXT"));
    own->destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        gipa(own->instance, "vkDestroyDebugUtilsMessengerEXT"));
    REQUIRE(create != nullptr);
    REQUIRE(own->destroy_messenger != nullptr);
    REQUIRE(create(own->instance, &messenger, nullptr, &own->messenger) == VK_SUCCESS);
  }
  count = 1;
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
    // La validation va avec son messager : ses erreurs sont comptées.
    CHECK((*validated)->debug_messages());
    CHECK((*validated)->validation_errors() == 0);
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
    rig.runner.emplace(*rig.pipelines);

    // Un noyau complet sur l'appareil adopté : la conversion de primaires, identique au bit près
    // à la référence en float (produits et sommes precise, dans l'ordre de ref::convert_primaries,
    // arrondis correctement par toute implémentation Vulkan).
    const kernels::RgbImage image = synthetic_rgb(8, 6, 3);
    TensorBuffer src = upload_rgb(rig, image, Precision::F32);
    TensorBuffer dst = tensor_target(rig, 8, 6, 3, Precision::F32);
    record_and_run(rig, [&](Recorder& r) {
      return rig.kernels().transfer(r, TransferOp::Bt709ToBt2020, src.view, dst.view);
    });
    const kernels::RgbImage got = download_rgb(rig, dst);
    for (int y = 0; y < image.height(); ++y) {
      for (int x = 0; x < image.width(); ++x) {
        const kernels::Rgb in{
            .r = image.planes[0].at(x, y), .g = image.planes[1].at(x, y), .b = image.planes[2].at(x, y)};
        const kernels::Rgb want = kernels::ref::convert_primaries(in, kernels::kBt709ToBt2020);
        CHECK(std::bit_cast<std::uint32_t>(got.planes[0].at(x, y)) == std::bit_cast<std::uint32_t>(want.r));
        CHECK(std::bit_cast<std::uint32_t>(got.planes[1].at(x, y)) == std::bit_cast<std::uint32_t>(want.g));
        CHECK(std::bit_cast<std::uint32_t>(got.planes[2].at(x, y)) == std::bit_cast<std::uint32_t>(want.b));
      }
    }
    CHECK(max_abs_diff(got, image) > 1e-3F);  // la conversion a bien eu lieu
    CHECK(locks > 0);
    CHECK(locks == unlocks);

    // Le test n'a pas activé le stockage 16 bits : les variantes float16 sont refusées par leur
    // nom, avant le pilote, au niveau du noyau comme du cache de pipelines.
    REQUIRE_FALSE(rig.context->caps().enabled.storage_buffer_16bit);
    TensorBuffer half_src = upload_rgb(rig, image, Precision::F16);
    TensorBuffer half_dst = tensor_target(rig, 8, 6, 3, Precision::F16);
    Recorder r = require_value(Recorder::begin(*rig.context, *rig.memory));
    require_refusal(rig.kernels().transfer(r, TransferOp::Bt709ToBt2020, half_src.view, half_dst.view),
                    RefusalCode::FeatureUnavailable);
    // Refusé avant vkCreateShaderModule : un module qui déclare StorageBuffer16BitAccess sans la
    // fonction activée est déjà un usage invalide (VUID-VkShaderModuleCreateInfo-pCode-08740).
    const std::size_t modules = rig.pipelines->shader_modules();
    const std::array op{static_cast<std::uint32_t>(TransferOp::Bt709ToBt2020)};
    require_refusal(
        rig.pipelines->get({.shader = "transfer_f16", .constants = op, .buffers = 2, .push_bytes = 32}),
        RefusalCode::FeatureUnavailable);
    CHECK(rig.pipelines->shader_modules() == modules);
    CHECK(require_value(r.finish())->dispatches() == 0);
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

  // Une seule file créée : la file 1 n'existe pas, même si la famille en offre seize (RTX, famille
  // 0). vkGetDeviceQueue ne doit pas la demander (VUID-vkGetDeviceQueue-queueIndex-00385).
  AdoptedDevice bad_queue = own->adopted();
  bad_queue.queue_index = 1;
  require_refusal(VkContext::adopt(bad_queue), RefusalCode::DeviceUnavailable);
  AdoptedDevice far_queue = own->adopted();
  far_queue.queue_index = 7;
  require_refusal(VkContext::adopt(far_queue), RefusalCode::DeviceUnavailable);
  AdoptedDevice no_queue = own->adopted();
  no_queue.queue_count = 0;
  require_refusal(VkContext::adopt(no_queue), RefusalCode::DeviceUnavailable);
  AdoptedDevice too_many = own->adopted();
  too_many.queue_count = 1000;
  require_refusal(VkContext::adopt(too_many), RefusalCode::DeviceUnavailable);

  // Un verrou de file sans son déverrou (ou l'inverse) laisserait la file de libplacebo bloquée.
  AdoptedDevice lock_only = own->adopted();
  lock_only.lock_queue = [] {};
  require_refusal(VkContext::adopt(lock_only), RefusalCode::InputMismatch);
  AdoptedDevice unlock_only = own->adopted();
  unlock_only.unlock_queue = [] {};
  require_refusal(VkContext::adopt(unlock_only), RefusalCode::InputMismatch);
}

TEST_CASE("contexte : messages VK_EXT_debug_utils comptés et journalisés", "[vk_compute][context]") {
  Rig rig = make_rig();
  auto context = VkContext::create({.debug_messages = true});
  if (!context && context.error().code == RefusalCode::FeatureUnavailable) {
    SKIP("VK_EXT_debug_utils absente du chargeur : " << describe(context.error()));
  }
  REQUIRE(context);
  VkContext& ctx = **context;
  REQUIRE(ctx.debug_messages());
  CHECK(ctx.validation_errors() == 0);
  // Le chargeur peut déjà avoir averti (VK_INSTANCE_LAYERS sous with-validation) : on compte
  // à partir d'ici.
  const std::uint64_t warnings = ctx.validation_warnings();
  const std::shared_ptr<const DebugCounts> counts = ctx.debug_counts();
  REQUIRE(counts != nullptr);
  // Le chargeur relaie un message soumis par l'application à chaque messager de l'instance.
  const auto submit = reinterpret_cast<PFN_vkSubmitDebugUtilsMessageEXT>(
      ctx.get_instance_proc_addr()(ctx.instance(), "vkSubmitDebugUtilsMessageEXT"));
  REQUIRE(submit != nullptr);
  const VkDebugUtilsMessengerCallbackDataEXT data{
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
      .pMessageIdName = "AJP-test",
      .pMessage = "message de test de vk_compute"};
  submit(ctx.instance(), VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, &data);
  submit(ctx.instance(), VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, &data);
  // Un message d'information n'est pas demandé au chargeur : il n'est pas compté.
  submit(ctx.instance(), VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
         VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, &data);
  CHECK(ctx.validation_errors() == 1);
  CHECK(ctx.validation_warnings() == warnings + 1);
  context->reset();
  // Les compteurs survivent au contexte.
  CHECK(counts->errors.load() == 1);
  // Sans demande, aucun messager et aucun compte.
  const auto plain = require_value(VkContext::create());
  CHECK_FALSE(plain->debug_messages());
  CHECK(plain->debug_counts() == nullptr);
  CHECK(plain->validation_errors() == 0);
  CHECK(plain->validation_warnings() == 0);
}
