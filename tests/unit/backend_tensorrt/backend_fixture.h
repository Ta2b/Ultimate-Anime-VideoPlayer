// Le backend TensorRT chargé par son chemin (dlopen, LoadLibrary), comme le lecteur le charge, et
// l'aller-retour des tenseurs FP16 avec l'hôte : partagés par les tests du backend.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/backend.h"
#include "kernels/ref/half.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp::test {

struct Library {
#if defined(_WIN32)
  void* handle = LoadLibraryA(AJP_BACKEND_TENSORRT);
  ~Library() {
    if (handle != nullptr) FreeLibrary(static_cast<HMODULE>(handle));
  }
  template <class F>
  F* symbol(const char* name) const {
    return reinterpret_cast<F*>(reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name)));
  }
#else
  void* handle = dlopen(AJP_BACKEND_TENSORRT, RTLD_NOW | RTLD_LOCAL);
  ~Library() {
    if (handle != nullptr) dlclose(handle);
  }
  template <class F>
  F* symbol(const char* name) const {
    return reinterpret_cast<F*>(dlsym(handle, name));
  }
#endif
};

inline bool has_cuda() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct Loaded {
  Library library;
  gpu::Backend* backend = nullptr;
  Refusal refusal{};
  std::filesystem::path cache = std::filesystem::path(AJP_TEST_CACHE_DIR);

  Loaded() {
    REQUIRE(library.handle != nullptr);
    std::filesystem::create_directories(cache);
    const gpu::BackendInit init{.cache_dir = cache, .refusal = &refusal};
    backend = library.symbol<std::remove_pointer_t<gpu::BackendCreateFn>>("ajp_backend_create")(
        gpu::kBackendAbi, &init);
  }
  ~Loaded() {
    if (backend != nullptr) {
      library.symbol<std::remove_pointer_t<gpu::BackendDestroyFn>>("ajp_backend_destroy")(backend);
    }
  }
};

// Plans RGB FP16 du tenseur, relus sur l'hôte.
inline std::vector<float> download(const gpu::Tensor& tensor) {
  std::size_t count = 1;
  for (const auto d : tensor.shape.dims) count *= static_cast<std::size_t>(d);
  std::vector<std::uint16_t> half(count);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaMemcpy(half.data(), reinterpret_cast<const void*>(tensor.handle), count * 2,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<float> out(count);
  for (std::size_t i = 0; i < count; ++i) out[i] = kernels::ref::half_to_float(half[i]);
  return out;
}

// Un tenseur FP16 rempli depuis l'hôte, plan après plan.
inline void upload(const gpu::Tensor& tensor, const std::vector<float>& values) {
  std::vector<std::uint16_t> half(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) half[i] = kernels::ref::float_to_half(values[i]);
  REQUIRE(cudaMemcpy(reinterpret_cast<void*>(tensor.handle), half.data(), half.size() * 2,
                     cudaMemcpyHostToDevice) == cudaSuccess);
}

}  // namespace ajp::test
