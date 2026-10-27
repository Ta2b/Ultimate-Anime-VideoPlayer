// Les tenseurs partagés avec un runtime (shared_tensor.h), sur l'appareil Vulkan du système
// (lavapipe en CI). Le runtime est simulé : il mappe le descripteur exporté, comme le fait un
// runtime qui importe la mémoire. Quand il la voit vraiment, la vérification passe et ce que le GPU
// écrit se lit à son adresse ; quand il n'en voit qu'une copie, elle refuse ; un runtime qui ne sait
// pas échanger de mots laisse le tenseur non vérifié ; un import refusé rend le descripteur au
// backend, sans rien laisser dans les vues, pour que la mémoire hôte prenne le relais.
// Linux seulement : le simulateur mappe des descripteurs de fichier.
#if defined(__linux__)

#include "gpu/vulkan/backend/shared_tensor.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../../support/vulkan_validation.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk;

namespace {

// Ce que le runtime fait de la mémoire qu'on lui donne.
enum class Seen {
  Shared,      // il mappe la mémoire exportée : il voit ce que la chaîne écrit, et réciproquement
  Copy,        // il en prend une copie à l'import : il ne voit plus ce que la chaîne écrit ensuite
  NoExchange,  // il voit la mémoire, mais ne sait pas échanger de mots (ImportedMemory par défaut)
  Refuse,      // il refuse l'import, sans toucher au descripteur
};

// La mémoire d'un import : le mappage du descripteur, ou la copie prise à l'import.
class FakeImport final : public net::ImportedMemory {
 public:
  FakeImport(std::byte* mapped, std::uint64_t bytes, std::vector<std::byte> copy, bool exchanges)
      : mapped_(mapped), bytes_(bytes), copy_(std::move(copy)), exchanges_(exchanges) {
    base_ = mapped_ != nullptr ? mapped_ : copy_.data();
  }
  FakeImport(const FakeImport&) = delete;
  FakeImport& operator=(const FakeImport&) = delete;
  FakeImport(FakeImport&&) = delete;
  FakeImport& operator=(FakeImport&&) = delete;
  ~FakeImport() override {
    if (mapped_ != nullptr) munmap(mapped_, bytes_);
  }

  std::byte* base() const { return base_; }
  void* device() const override { return base_; }
  std::expected<std::uint32_t, Refusal> exchange(std::uint32_t value) override {
    if (!exchanges_)
      return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "sans échange"});
    std::uint32_t seen = 0;
    std::memcpy(&seen, base_, sizeof(seen));
    std::memcpy(base_ + sizeof(seen), &value, sizeof(value));
    return seen;
  }

 private:
  std::byte* mapped_;
  std::uint64_t bytes_;
  std::vector<std::byte> copy_;
  bool exchanges_;
  std::byte* base_ = nullptr;
};

class FakeRuntime final : public net::Runtime {
 public:
  FakeRuntime(net::MemoryHandle handle, Seen seen) : handle_(handle), seen_(seen) {}

  std::string_view name() const override { return "faux"; }
  std::string device_name() const override { return "faux"; }
  std::string version() const override { return "1"; }
  std::string_view extension() const override { return ".faux"; }
  std::expected<void, Refusal> build(const ModelSpec& /*model*/, std::span<const net::InputShape> /*inputs*/,
                                     const std::filesystem::path& /*path*/, Progress& /*progress*/) override {
    return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "pas de réseau"});
  }
  std::expected<std::unique_ptr<net::RuntimeNet>, Refusal> load(
      const std::filesystem::path& /*path*/) override {
    return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "pas de réseau"});
  }

  bool imports_memory() const override { return true; }
  net::MemoryHandle memory_handle() const override { return handle_; }

  std::expected<std::unique_ptr<net::ImportedMemory>, Refusal> import_memory(net::HandleValue handle,
                                                                             std::uint64_t bytes, TensorType,
                                                                             const Shape&) override {
    ++imports;
    last_handle = handle;
    if (seen_ == Seen::Refuse)
      return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "import refusé"});
    const int fd = static_cast<int>(handle);
    void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED)
      return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "mmap"});
    close(fd);  // le runtime prend le descripteur ; sa mémoire reste mappée
    auto* bytes_ptr = static_cast<std::byte*>(mapped);
    std::vector<std::byte> copy;
    if (seen_ == Seen::Copy) {
      copy.assign(bytes_ptr, bytes_ptr + bytes);
      munmap(mapped, bytes);
      mapped = nullptr;
    }
    auto made = std::make_unique<FakeImport>(static_cast<std::byte*>(mapped), bytes, std::move(copy),
                                             seen_ != Seen::NoExchange);
    last = made->base();
    return std::unique_ptr<net::ImportedMemory>(std::move(made));
  }

  int imports = 0;
  net::HandleValue last_handle = -1;
  std::byte* last = nullptr;  // l'adresse que le runtime voit

 private:
  net::MemoryHandle handle_;
  Seen seen_;
};

constexpr std::uint64_t kBytes = 4 * 1024 * 2;  // 1 × 1 × 4 × 1024 en FP16
const Shape kShape{.dims = {1, 1, 4, 1024}};

// Les extensions qui exportent le descripteur demandé, et le contexte qui les a.
std::expected<std::unique_ptr<VkContext>, Refusal> context_for(net::MemoryHandle handle) {
  std::vector<std::string> extensions{VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                                      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
  if (handle == net::MemoryHandle::DmaBuf)
    extensions.emplace_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
  return ajp::test::with_validation([&](bool validation, ajp::test::ValidationSink sink) {
    return VkContext::create(
        {.extensions = extensions, .validation = validation, .validation_sink = std::move(sink)});
  });
}

// Le GPU remplit tout le tenseur de 1,0 en FP16, puis l'hôte attend la fin de la soumission.
void fill_ones(VkContext& context, VkBuffer buffer) {
  const DeviceFunctions& fn = context.fn();
  auto filled = context.run_once([&](VkCommandBuffer cb) {
    fn.vkCmdFillBuffer(cb, buffer, 0, kBytes, 0x3C00'3C00U);
    const VkMemoryBarrier to_host{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                            nullptr, 0, nullptr);
  });
  if (!filled) FAIL(vk::test::describe(filled.error()));
}

}  // namespace

TEST_CASE("shared_tensor : le runtime voit la mémoire de la chaîne, la vérification passe",
          "[vk_backend][runtime]") {
  for (const net::MemoryHandle handle : {net::MemoryHandle::OpaqueFd, net::MemoryHandle::DmaBuf}) {
    DYNAMIC_SECTION((handle == net::MemoryHandle::DmaBuf ? "dma-buf" : "descripteur opaque")) {
      auto context = context_for(handle);
      if (!context) SKIP("pas d'export de ce descripteur : " << context.error().detail);
      if (!exports_memory(**context, handle)) SKIP("tampon non exportable sous ce descripteur");
      auto memory = MemoryAllocator::create(**context);
      if (!memory) FAIL(vk::test::describe(memory.error()));

      FakeRuntime runtime(handle, Seen::Shared);
      auto views = std::make_shared<HostViews>();
      auto shared = shared_tensor(**context, **memory, runtime, TensorType::F16, kShape, kBytes, views, true);
      if (!shared) {
        // La vérification a refusé : le runtime ne voit pas ce que la chaîne écrit. Ce refus est la
        // protection voulue ; c'est l'appareil (lavapipe exporte une copie) qui ne permet pas le test.
        if (shared.error().detail.find("mmap") != std::string::npos)
          SKIP("le descripteur exporté par le pilote ne se mappe pas côté hôte : " << shared.error().detail);
        if (shared.error().detail.find("ne voit pas") != std::string::npos)
          SKIP("l'appareil n'expose pas sa mémoire exportée au mappage, la vérification refuse : "
               << shared.error().detail);
        FAIL(vk::test::describe(shared.error()));
      }
      REQUIRE(runtime.imports == 1);

      // La vue du runtime : son adresse, et la taille du tenseur.
      const Tensor& tensor = (*shared)->tensor();
      const auto view = views->views.find(tensor.handle);
      REQUIRE(view != views->views.end());
      CHECK(view->second.device == runtime.last);
      CHECK(view->second.bytes == kBytes);

      // Ce que le GPU écrit se lit à l'adresse du runtime : c'est la même mémoire.
      fill_ones(**context, detail::to_buffer(tensor.handle));
      std::vector<std::uint16_t> halves(kBytes / 2);
      std::memcpy(halves.data(), runtime.last, kBytes);
      CHECK(std::ranges::all_of(halves, [](std::uint16_t h) { return h == 0x3C00; }));
    }
  }
}

TEST_CASE("shared_tensor : un runtime qui ne voit qu'une copie est refusé", "[vk_backend][runtime]") {
  for (const net::MemoryHandle handle : {net::MemoryHandle::OpaqueFd, net::MemoryHandle::DmaBuf}) {
    DYNAMIC_SECTION((handle == net::MemoryHandle::DmaBuf ? "dma-buf" : "descripteur opaque")) {
      auto context = context_for(handle);
      if (!context) SKIP("pas d'export de ce descripteur : " << context.error().detail);
      if (!exports_memory(**context, handle)) SKIP("tampon non exportable sous ce descripteur");
      auto memory = MemoryAllocator::create(**context);
      if (!memory) FAIL(vk::test::describe(memory.error()));

      FakeRuntime runtime(handle, Seen::Copy);
      auto views = std::make_shared<HostViews>();
      auto shared = shared_tensor(**context, **memory, runtime, TensorType::F16, kShape, kBytes, views, true);
      REQUIRE_FALSE(shared);
      if (shared.error().detail.find("mmap") != std::string::npos)
        SKIP("le descripteur exporté par le pilote ne se mappe pas côté hôte : " << shared.error().detail);
      CHECK(shared.error().code == RefusalCode::FeatureUnavailable);
      CHECK(shared.error().detail.find("ne voit pas") != std::string::npos);
      CHECK(views->views.empty());
    }
  }
}

TEST_CASE("shared_tensor : sans échange de mots, le tenseur reste non vérifié", "[vk_backend][runtime]") {
  auto context = context_for(net::MemoryHandle::OpaqueFd);
  if (!context) SKIP("pas d'export de ce descripteur : " << context.error().detail);
  auto memory = MemoryAllocator::create(**context);
  if (!memory) FAIL(vk::test::describe(memory.error()));

  FakeRuntime runtime(net::MemoryHandle::OpaqueFd, Seen::NoExchange);
  auto views = std::make_shared<HostViews>();
  auto shared = shared_tensor(**context, **memory, runtime, TensorType::F16, kShape, kBytes, views, true);
  if (!shared && shared.error().detail.find("mmap") != std::string::npos)
    SKIP("le descripteur exporté par le pilote ne se mappe pas côté hôte : " << shared.error().detail);
  if (!shared) FAIL(vk::test::describe(shared.error()));
  CHECK(views->views.contains((*shared)->tensor().handle));
}

TEST_CASE("shared_tensor : un import refusé rend le descripteur et ne laisse aucune vue",
          "[vk_backend][runtime]") {
  auto context = context_for(net::MemoryHandle::OpaqueFd);
  if (!context) SKIP("pas d'export de ce descripteur : " << context.error().detail);
  auto memory = MemoryAllocator::create(**context);
  if (!memory) FAIL(vk::test::describe(memory.error()));

  FakeRuntime runtime(net::MemoryHandle::OpaqueFd, Seen::Refuse);
  auto views = std::make_shared<HostViews>();
  auto shared = shared_tensor(**context, **memory, runtime, TensorType::F16, kShape, kBytes, views, true);
  REQUIRE_FALSE(shared);
  CHECK(shared.error().code == RefusalCode::FeatureUnavailable);
  // Le descripteur, que le runtime n'a pas pris, est fermé par le backend.
  const int status = fcntl(static_cast<int>(runtime.last_handle), F_GETFD);
  const int error = errno;
  CHECK(status == -1);
  CHECK(error == EBADF);
  // Sans vue, le backend passe à la mémoire hôte pour ce tenseur et les suivants.
  CHECK(views->views.empty());
}

#endif  // __linux__
