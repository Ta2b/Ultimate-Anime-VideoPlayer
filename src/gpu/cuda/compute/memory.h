// Mémoire de l'appareil bornée : un pool alloué une fois, découpé en tampons à la préparation des
// plans. Rien ne s'alloue par image : quand le pool manque, la préparation refuse
// (GpuMemoryExhausted) au lieu de déborder sur la mémoire des moteurs.
//
// Pool simple : première zone libre assez grande, alignée sur 256 octets, fusion des zones
// voisines à la libération. Thread : celui qui possède le pool, comme ses tampons.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "gpu/cuda/compute/refusal.h"

namespace ajp::gpu::cuda {

class DevicePool;

// Un tampon du pool, rendu à sa destruction. Le pool doit lui survivre.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer();

  void* data() const { return data_; }
  std::uintptr_t address() const { return reinterpret_cast<std::uintptr_t>(data_); }
  std::size_t size() const { return size_; }
  template <class T>
  T* as() const {
    return static_cast<T*>(data_);
  }

 private:
  friend class DevicePool;
  DeviceBuffer(DevicePool* pool, std::size_t offset, std::size_t size, void* data)
      : pool_(pool), offset_(offset), size_(size), data_(data) {}
  void reset();

  DevicePool* pool_ = nullptr;
  std::size_t offset_ = 0;
  std::size_t size_ = 0;
  void* data_ = nullptr;
};

class DevicePool {
 public:
  static constexpr std::size_t kAlignment = 256;

  // capacity octets de l'appareil courant (Device::open avant). Refus : capacité nulle, ou
  // cudaMalloc refusé (GpuMemoryExhausted s'il manque de la mémoire).
  static Result<std::unique_ptr<DevicePool>> create(std::size_t capacity);

  DevicePool(const DevicePool&) = delete;
  DevicePool& operator=(const DevicePool&) = delete;
  DevicePool(DevicePool&&) = delete;
  DevicePool& operator=(DevicePool&&) = delete;
  ~DevicePool();

  // bytes octets, arrondis à kAlignment. Refus : bytes nul, ou aucune zone libre assez grande
  // (GpuMemoryExhausted, avec ce qui reste) ; what nomme le tampon dans le refus.
  Result<DeviceBuffer> allocate(std::size_t bytes, std::string_view what);

  std::size_t capacity() const { return capacity_; }
  std::size_t used() const { return used_; }

 private:
  struct Range {
    std::size_t offset = 0;
    std::size_t size = 0;
  };

  DevicePool(void* base, std::size_t capacity);
  friend class DeviceBuffer;
  void release(std::size_t offset, std::size_t size);

  void* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
  std::vector<Range> free_;  // triées par adresse, jamais voisines
};

// Mémoire hôte verrouillée (cudaHostAlloc), pour les relectures asynchrones de quelques octets.
class PinnedBuffer {
 public:
  static Result<PinnedBuffer> create(std::size_t bytes);

  PinnedBuffer() = default;

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& other) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;
  ~PinnedBuffer();

  void* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  PinnedBuffer(void* data, std::size_t size) : data_(data), size_(size) {}
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace ajp::gpu::cuda
