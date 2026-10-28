#include "gpu/cuda/backend/gpu_timing.h"

#include <algorithm>
#include <format>
#include <utility>

namespace ajp::gpu::cuda {

GpuTiming::GpuTiming(std::string label, Report report)
    : label_(std::move(label)), report_(std::move(report)) {
  for (std::size_t i = 0; i < kPairs; ++i) {
    cudaEventCreate(&start_[i]);
    cudaEventCreate(&stop_[i]);
  }
  samples_.reserve(kReport);
}

GpuTiming::~GpuTiming() {
  for (std::size_t i = 0; i < kPairs; ++i) {
    if (start_[i] != nullptr) cudaEventDestroy(start_[i]);
    if (stop_[i] != nullptr) cudaEventDestroy(stop_[i]);
  }
}

void GpuTiming::begin(cudaStream_t stream) {
  collect();
  if (begun_ != ended_ || begun_ - read_ >= kPairs) return;  // paire ouverte, ou toutes en vol : on saute
  cudaEventRecord(start_[begun_ % kPairs], stream);
  ++begun_;
}

void GpuTiming::end(cudaStream_t stream) {
  if (ended_ == begun_) return;
  cudaEventRecord(stop_[ended_ % kPairs], stream);
  ++ended_;
}

void GpuTiming::collect() {
  while (read_ < ended_ && cudaEventQuery(stop_[read_ % kPairs]) == cudaSuccess) {
    float ms = 0.0F;
    if (cudaEventElapsedTime(&ms, start_[read_ % kPairs], stop_[read_ % kPairs]) == cudaSuccess) {
      if (report_) samples_.push_back(ms);
      if (all_.size() < kMaxAll) all_.push_back(ms);
    }
    ++read_;
  }
  if (!report_ || samples_.size() < kReport) return;
  std::ranges::sort(samples_);
  const float median = samples_[samples_.size() / 2];
  const float p95 = samples_[samples_.size() * 95 / 100];
  report_(std::format("{} : médiane {:.2f} ms, p95 {:.2f} ms, max {:.2f} ms sur {} images", label_, median,
                      p95, samples_.back(), samples_.size()));
  samples_.clear();
}

std::string GpuTiming::summary() {
  collect();
  if (all_.empty()) return {};
  std::vector<float> sorted = all_;
  std::ranges::sort(sorted);
  return std::format("{} : médiane {:.2f} ms, p95 {:.2f} ms, min {:.2f} ms ({} mesures)", label_,
                     sorted[sorted.size() / 2], sorted[sorted.size() * 95 / 100], sorted.front(),
                     sorted.size());
}

}  // namespace ajp::gpu::cuda
