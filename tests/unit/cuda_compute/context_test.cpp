// Appareil choisi par UUID, pool borné, flux non bloquants et événements.
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/compute/device.h"
#include "gpu/cuda/compute/memory.h"
#include "gpu/cuda/compute/refusal.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;

TEST_CASE("l'appareil s'ouvre par son UUID et seulement par lui", "[cuda][context]") {
  AJP_CUDA_RIG(rig);
  auto devices = list_devices();
  REQUIRE(devices);
  REQUIRE_FALSE(devices->empty());
  const DeviceInfo& first = devices->front();
  CHECK(rig.device.info().uuid == first.uuid);
  CHECK(rig.device.info().major * 10 + rig.device.info().minor >= 75);  // Turing au moins
  CHECK_FALSE(rig.device.info().name.empty());
  int current = -1;
  REQUIRE(cudaGetDevice(&current) == cudaSuccess);
  CHECK(current == first.ordinal);

  Uuid unknown = first.uuid;
  unknown[0] ^= 0xFFU;
  auto missing = Device::open(unknown);
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::CudaFailed);
  CHECK(missing.error().detail.find("UUID") != std::string::npos);
}

TEST_CASE("une erreur CUDA devient un refus nommé qui cite l'appel", "[cuda][context]") {
  const Refusal memory = cuda_refusal(cudaErrorMemoryAllocation, "cudaMalloc");
  CHECK(memory.code == RefusalCode::GpuMemoryExhausted);
  CHECK(memory.detail.starts_with("cudaMalloc : cudaErrorMemoryAllocation"));
  const Refusal other = cuda_refusal(cudaErrorInvalidValue, "cudaMemcpyAsync");
  CHECK(other.code == RefusalCode::CudaFailed);
  CHECK(other.detail.starts_with("cudaMemcpyAsync : cudaErrorInvalidValue"));
  CHECK(cuda_check(cudaSuccess, "rien"));
}

TEST_CASE("le pool borné refuse ce qui ne tient pas, et se recompose", "[cuda][context]") {
  AJP_CUDA_RIG(rig, std::size_t{1} << 20);
  CHECK(rig.pool->capacity() == std::size_t{1} << 20);
  auto pool = DevicePool::create(4096);
  REQUIRE(pool);
  CHECK((*pool)->capacity() == 4096);
  CHECK((*pool)->allocate(0, "vide").error().code == RefusalCode::FieldOutOfRange);
  {
    auto a = (*pool)->allocate(1000, "a");  // arrondi à 1024
    auto b = (*pool)->allocate(1024, "b");
    auto c = (*pool)->allocate(2048, "c");
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(c);
    CHECK((*pool)->used() == 4096);
    CHECK(a->size() == 1024);
    CHECK(b->address() == a->address() + 1024);
    CHECK(b->address() % DevicePool::kAlignment == 0);
    auto full = (*pool)->allocate(1, "de trop");
    REQUIRE_FALSE(full);
    CHECK(full.error().code == RefusalCode::GpuMemoryExhausted);
    CHECK(full.error().detail.starts_with("de trop"));

    // a et b libérés : leurs zones se fusionnent en 2048 octets contigus, au début.
    const std::uintptr_t start = a->address();
    *a = DeviceBuffer();
    *b = DeviceBuffer();
    CHECK((*pool)->used() == 2048);
    auto merged = (*pool)->allocate(2048, "fusion");
    REQUIRE(merged);
    CHECK(merged->address() == start);
    CHECK((*pool)->allocate(256, "plein").error().code == RefusalCode::GpuMemoryExhausted);
    // Un tampon déplacé ne se rend qu'une fois.
    DeviceBuffer moved = std::move(*merged);
    CHECK((*pool)->used() == 4096);
  }
  CHECK((*pool)->used() == 0);
  auto whole = (*pool)->allocate(4096, "tout");
  CHECK(whole);

  // Plus que la mémoire de l'appareil : refus nommé de cudaMalloc, pas un plantage.
  auto huge = DevicePool::create(std::size_t{1} << 46);
  REQUIRE_FALSE(huge);
  CHECK(huge.error().code == RefusalCode::GpuMemoryExhausted);
  CHECK(huge.error().detail.starts_with("cudaMalloc"));
  // L'échec n'empêche pas le lancement suivant de réussir.
  CHECK(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("les flux sont non bloquants et les événements ne font pas attendre", "[cuda][context]") {
  AJP_CUDA_RIG(rig);
  unsigned flags = 0;
  REQUIRE(cudaStreamGetFlags(rig.s(), &flags) == cudaSuccess);
  CHECK(flags == cudaStreamNonBlocking);

  auto start = Event::create(true);
  auto end = Event::create(true);
  REQUIRE(start);
  REQUIRE(end);
  REQUIRE(start->record(rig.s()));
  auto buffer = rig.pool->allocate(std::size_t{64} << 20, "test : remplissage");
  REQUIRE(buffer);
  REQUIRE(cudaMemsetAsync(buffer->data(), 0, buffer->size(), rig.s()) == cudaSuccess);
  REQUIRE(end->record(rig.s()));
  auto done = end->done();  // la réponse vient tout de suite, finie ou non
  REQUIRE(done);
  REQUIRE(end->synchronize());
  auto finished = end->done();
  REQUIRE(finished);
  CHECK(*finished);
  auto ms = Event::elapsed_ms(*start, *end);
  REQUIRE(ms);
  CHECK(*ms >= 0.0F);

  // Un autre flux attend l'événement côté GPU (comme le CUevent de NVDEC).
  auto other = Stream::create();
  REQUIRE(other);
  REQUIRE(other->wait(end->get()));
  REQUIRE(other->synchronize());
}
