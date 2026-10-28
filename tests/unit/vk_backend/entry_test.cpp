// ajp_backend_vulkan chargée comme le lecteur la charge : par son chemin, points d'entrée lus
// par nom, version et empreinte vérifiées par check_backend_abi avant ajp_backend_create. Une
// bibliothèque d'une autre ABI est refusée ; une mauvaise version passée à create aussi.
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "gpu/backend.h"
#include "net_rig.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace ajp;
using namespace ajp::gpu;

namespace {

// Une bibliothèque ouverte, fermée à la destruction.
class Library {
 public:
  explicit Library(const char* path) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(path);
#else
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
  }
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  Library(Library&&) = delete;
  Library& operator=(Library&&) = delete;
  ~Library() {
    if (handle_ == nullptr) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
  }
  bool open() const { return handle_ != nullptr; }
  template <class Fn>
  Fn symbol(const char* name) const {
#if defined(_WIN32)
    return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    return reinterpret_cast<Fn>(dlsym(handle_, name));
#endif
  }

 private:
  void* handle_ = nullptr;
};

}  // namespace

TEST_CASE("Entrée : ajp_backend_vulkan se charge et vérifie son ABI", "[vk_backend][entry]") {
  const Library library(AJP_TEST_BACKEND);
  REQUIRE(library.open());
  const auto abi = library.symbol<BackendAbiFn>("ajp_backend_abi");
  const auto layout = library.symbol<BackendLayoutFn>("ajp_backend_layout");
  const auto create = library.symbol<BackendCreateFn>("ajp_backend_create");
  const auto destroy = library.symbol<BackendDestroyFn>("ajp_backend_destroy");
  REQUIRE(abi != nullptr);
  REQUIRE(layout != nullptr);
  REQUIRE(create != nullptr);
  REQUIRE(destroy != nullptr);
  CHECK(check_backend_abi(abi(), layout()).has_value());

  // Une autre version passée à create : nullptr et AbiMismatch, sans toucher à l'appareil.
  Refusal refusal;
  BackendInit init{.refusal = &refusal};
  CHECK(create(kBackendAbi + 1, &init) == nullptr);
  CHECK(refusal.code == RefusalCode::AbiMismatch);
  CHECK(create(kBackendAbi, nullptr) == nullptr);

  // Un appareil de rendu qui n'est pas Vulkan : DeviceMismatch.
  const GpuDevice cuda{.api = GpuApi::Cuda};
  init.render = &cuda;
  CHECK(create(kBackendAbi, &init) == nullptr);
  CHECK(refusal.code == RefusalCode::DeviceMismatch);

  // Sans appareil fourni : le backend crée le sien, et journalise par le rappel.
  int messages = 0;
  init = BackendInit{.log = [&](BackendLogLevel, std::string_view) { ++messages; }, .refusal = &refusal};
  Backend* backend = create(kBackendAbi, &init);
  if (backend == nullptr) {
    if (refusal.code == RefusalCode::LibraryUnavailable || refusal.code == RefusalCode::DeviceUnavailable) {
      SKIP("aucun Vulkan : " << vk::test::describe(refusal));
    }
    FAIL(vk::test::describe(refusal));
  }
  CHECK(backend->capabilities().name == "vulkan");
  CHECK(messages > 0);
  CHECK(backend->kernels() != nullptr);
  destroy(backend);
}

TEST_CASE("Entrée : une bibliothèque d'une autre ABI est refusée", "[vk_backend][entry]") {
  const Library library(AJP_TEST_FAKE_BACKEND);
  REQUIRE(library.open());
  const auto abi = library.symbol<BackendAbiFn>("ajp_backend_abi");
  const auto layout = library.symbol<BackendLayoutFn>("ajp_backend_layout");
  REQUIRE(abi != nullptr);
  REQUIRE(layout != nullptr);
  const auto checked = check_backend_abi(abi(), layout());
  REQUIRE_FALSE(checked);
  CHECK(checked.error().code == RefusalCode::AbiMismatch);
  CHECK(library.symbol<BackendCreateFn>("ajp_backend_create") == nullptr);
}
