// Remplacement d'operator new et delete pour le binaire de test de vk_compute (voir
// host_allocations.h). Toutes les variantes non alignées passent par malloc et free, pour que
// les sanitizers voient des paires cohérentes ; les variantes alignées gardent celles de la
// bibliothèque standard.
#include "host_allocations.h"

#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

thread_local bool probing = false;
thread_local std::uint64_t allocations = 0;

void* allocate(std::size_t size) noexcept {
  if (probing) ++allocations;
  return std::malloc(size == 0 ? 1 : size);
}

void* allocate_or_throw(std::size_t size) {
  void* memory = allocate(size);
  if (memory == nullptr) throw std::bad_alloc();
  return memory;
}

}  // namespace

void* operator new(std::size_t size) {
  return allocate_or_throw(size);
}
void* operator new[](std::size_t size) {
  return allocate_or_throw(size);
}
void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  return allocate(size);
}
void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  return allocate(size);
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete[](void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t /*size*/) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, std::size_t /*size*/) noexcept {
  std::free(memory);
}
void operator delete(void* memory, const std::nothrow_t& /*tag*/) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, const std::nothrow_t& /*tag*/) noexcept {
  std::free(memory);
}

namespace ajp::gpu::vk::test {

HostAllocationProbe::HostAllocationProbe() {
  allocations = 0;
  probing = true;
}

HostAllocationProbe::~HostAllocationProbe() {
  probing = false;
}

std::uint64_t HostAllocationProbe::count() const {
  return allocations;
}

}  // namespace ajp::gpu::vk::test
