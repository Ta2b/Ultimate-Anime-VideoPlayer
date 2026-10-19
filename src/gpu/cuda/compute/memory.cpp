#include "gpu/cuda/compute/memory.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <format>
#include <iterator>
#include <utility>

namespace ajp::gpu::cuda {
namespace {

std::size_t align_up(std::size_t bytes) {
  return (bytes + DevicePool::kAlignment - 1) / DevicePool::kAlignment * DevicePool::kAlignment;
}

}  // namespace

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      offset_(std::exchange(other.offset_, 0)),
      size_(std::exchange(other.size_, 0)),
      data_(std::exchange(other.data_, nullptr)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    pool_ = std::exchange(other.pool_, nullptr);
    offset_ = std::exchange(other.offset_, 0);
    size_ = std::exchange(other.size_, 0);
    data_ = std::exchange(other.data_, nullptr);
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() {
  reset();
}

void DeviceBuffer::reset() {
  if (pool_ != nullptr) pool_->release(offset_, size_);
  pool_ = nullptr;
  data_ = nullptr;
  size_ = 0;
}

Result<std::unique_ptr<DevicePool>> DevicePool::create(std::size_t capacity) {
  if (capacity == 0) return refuse(RefusalCode::FieldOutOfRange, "pool CUDA de capacité nulle");
  capacity = align_up(capacity);
  void* base = nullptr;
  if (auto ok =
          cuda_check(cudaMalloc(&base, capacity), std::format("cudaMalloc ({} octets du pool)", capacity));
      !ok) {
    return std::unexpected(ok.error());
  }
  return std::unique_ptr<DevicePool>(new DevicePool(base, capacity));
}

DevicePool::DevicePool(void* base, std::size_t capacity)
    : base_(base), capacity_(capacity), free_{{.offset = 0, .size = capacity}} {}

DevicePool::~DevicePool() {
  cudaFree(base_);
}

Result<DeviceBuffer> DevicePool::allocate(std::size_t bytes, std::string_view what) {
  if (bytes == 0) return refuse(RefusalCode::FieldOutOfRange, std::format("{} : tampon de 0 octet", what));
  const std::size_t size = align_up(bytes);
  const auto zone = std::ranges::find_if(free_, [size](const Range& r) { return r.size >= size; });
  if (zone == free_.end()) {
    std::size_t largest = 0;
    for (const Range& r : free_) largest = std::max(largest, r.size);
    return refuse(RefusalCode::GpuMemoryExhausted,
                  std::format("{} : {} octets demandés, pool de {} octets dont {} utilisés, plus grande zone "
                              "libre {} octets",
                              what, bytes, capacity_, used_, largest));
  }
  const std::size_t offset = zone->offset;
  zone->offset += size;
  zone->size -= size;
  if (zone->size == 0) free_.erase(zone);
  used_ += size;
  return DeviceBuffer(this, offset, size, static_cast<std::byte*>(base_) + offset);
}

void DevicePool::release(std::size_t offset, std::size_t size) {
  used_ -= size;
  auto next = std::ranges::upper_bound(free_, offset, {}, &Range::offset);
  next = free_.insert(next, {.offset = offset, .size = size});
  // Fusion avec la zone suivante, puis avec la précédente.
  if (const auto after = std::next(next);
      after != free_.end() && next->offset + next->size == after->offset) {
    next->size += after->size;
    free_.erase(after);
  }
  if (next != free_.begin()) {
    const auto before = std::prev(next);
    if (before->offset + before->size == next->offset) {
      before->size += next->size;
      free_.erase(next);
    }
  }
}

Result<PinnedBuffer> PinnedBuffer::create(std::size_t bytes) {
  void* data = nullptr;
  if (auto ok = cuda_check(cudaHostAlloc(&data, bytes, cudaHostAllocDefault), "cudaHostAlloc"); !ok) {
    return std::unexpected(ok.error());
  }
  return PinnedBuffer(data, bytes);
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) cudaFreeHost(data_);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

PinnedBuffer::~PinnedBuffer() {
  if (data_ != nullptr) cudaFreeHost(data_);
}

}  // namespace ajp::gpu::cuda
