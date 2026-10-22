// Les tenseurs d'un runtime dans sa propre mémoire hôte, importée par Vulkan (shared_tensor.h,
// host_imported_tensor), sur l'appareil Vulkan du système (lavapipe en CI) : la vérification du
// premier import passe sur une mémoire que le GPU voit vraiment, puis ce que le GPU écrit dans le
// tenseur se lit à l'adresse du runtime.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../../support/vulkan_validation.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/backend/shared_tensor.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/context.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk;

namespace {

// Une mémoire hôte alignée, comme un runtime la donnerait (USM d'OpenVINO, IOSurface de Core ML).
class AlignedMemory final : public net::HostMemory {
 public:
  AlignedMemory(std::uint64_t bytes, std::uint64_t alignment)
      : alignment_(alignment),
        data_(static_cast<std::byte*>(::operator new(bytes, std::align_val_t{alignment}))) {
    std::memset(data_, 0, bytes);
  }
  AlignedMemory(const AlignedMemory&) = delete;
  AlignedMemory& operator=(const AlignedMemory&) = delete;
  AlignedMemory(AlignedMemory&&) = delete;
  AlignedMemory& operator=(AlignedMemory&&) = delete;
  ~AlignedMemory() override { ::operator delete(data_, std::align_val_t{alignment_}); }
  std::byte* data() const override { return data_; }

 private:
  std::uint64_t alignment_;
  std::byte* data_;
};

// Un runtime qui ne fait qu'allouer sa mémoire hôte ; l'adresse de la dernière.
class HostRuntime final : public net::Runtime {
 public:
  std::string_view name() const override { return "hote"; }
  std::string device_name() const override { return "hôte"; }
  std::string version() const override { return "1"; }
  std::string_view extension() const override { return ".hote"; }
  std::expected<void, Refusal> build(const ModelSpec& /*model*/, std::span<const net::InputShape> /*inputs*/,
                                     const std::filesystem::path& /*path*/, Progress& /*progress*/) override {
    return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "pas de réseau"});
  }
  std::expected<std::unique_ptr<net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& /*path*/) override {
    return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "pas de réseau"});
  }
  std::expected<std::unique_ptr<net::HostMemory>, Refusal> allocate_host(std::uint64_t bytes,
                                                                         std::uint64_t alignment,
                                                                         TensorType /*type*/,
                                                                         const Shape& /*shape*/) override {
    auto memory = std::make_unique<AlignedMemory>(bytes, alignment);
    last = memory->data();
    return memory;
  }
  std::byte* last = nullptr;
};

}  // namespace

TEST_CASE("host_imported_tensor : le GPU voit la mémoire hôte du runtime elle-même",
          "[vk_backend][runtime]") {
  auto context = ajp::test::with_validation([](bool validation, ajp::test::ValidationSink sink) {
    return VkContext::create({.extensions = {VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME},
                              .validation = validation,
                              .validation_sink = std::move(sink)});
  });
  if (!context) SKIP("pas d'appareil Vulkan avec VK_EXT_external_memory_host : " << context.error().detail);
  const std::uint64_t alignment = host_import_alignment(**context);
  REQUIRE(alignment != 0);

  HostRuntime runtime;
  auto views = std::make_shared<HostViews>();
  constexpr std::uint64_t kBytes = 4 * 1024 * 2;  // 1 × 1 × 4 × 1024 en FP16
  const Shape shape{.dims = {1, 1, 4, 1024}};
  auto imported =
      host_imported_tensor(**context, runtime, TensorType::F16, shape, kBytes, alignment, views, true);
  if (!imported) FAIL(vk::test::describe(imported.error()));
  const Tensor& tensor = (*imported)->tensor();
  {
    const auto view = views->views.find(tensor.handle);
    REQUIRE(view != views->views.end());
    CHECK(view->second.data == runtime.last);
    CHECK(view->second.bytes == kBytes);
  }
  // La vérification a remis ses mots à zéro.
  std::vector<std::byte> zeros(16, std::byte{0});
  CHECK(std::memcmp(runtime.last, zeros.data(), zeros.size()) == 0);

  // Le GPU remplit tout le tenseur de 1,0 en FP16 : l'hôte le lit à l'adresse du runtime.
  const DeviceFunctions& fn = (*context)->fn();
  const VkBuffer buffer = detail::to_buffer(tensor.handle);
  auto filled = (*context)->run_once([&](VkCommandBuffer cb) {
    fn.vkCmdFillBuffer(cb, buffer, 0, kBytes, 0x3C00'3C00U);
    const VkMemoryBarrier to_host{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                            nullptr, 0, nullptr);
  });
  if (!filled) FAIL(vk::test::describe(filled.error()));
  std::vector<std::uint16_t> halves(kBytes / 2);
  std::memcpy(halves.data(), runtime.last, kBytes);
  CHECK(std::ranges::all_of(halves, [](std::uint16_t h) { return h == 0x3C00; }));
}
