#include "gpu/cuda/backend/gpu_timing.h"

#include <algorithm>
#include <format>
#include <utility>

#include "gpu/cuda/backend/cuda_warn.h"

namespace ajp::gpu::cuda {

GpuTiming::GpuTiming(std::string label, Report report, std::chrono::seconds every)
    : label_(std::move(label)), report_(std::move(report)), every_(every) {
  for (std::size_t i = 0; i < kPairs; ++i) {
    cuda_warn(cudaEventCreate(&start_[i]), "cudaEventCreate");
    cuda_warn(cudaEventCreate(&stop_[i]), "cudaEventCreate");
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
  cuda_warn(cudaEventRecord(start_[begun_ % kPairs], stream), "cudaEventRecord");
  open_stream_ = stream;
  ++begun_;
}

void GpuTiming::end(cudaStream_t stream) {
  if (ended_ == begun_) return;
  if (stream != open_stream_) {
    // Fin sur une autre file que le début (autre voie, sans dépendance garantie) : la paire ne
    // mesurerait rien de juste, elle est abandonnée.
    --begun_;
    return;
  }
  cuda_warn(cudaEventRecord(stop_[ended_ % kPairs], stream), "cudaEventRecord");
  ++ended_;
}

void GpuTiming::collect() {
  while (read_ < ended_ && cudaEventQuery(stop_[read_ % kPairs]) == cudaSuccess) {
    float ms = 0.0F;
    if (cudaEventElapsedTime(&ms, start_[read_ % kPairs], stop_[read_ % kPairs]) == cudaSuccess) {
      if (report_ && samples_.size() < kMaxSamples) samples_.push_back(ms);
      all_.add(ms);
    }
    ++read_;
  }
  if (!report_ || samples_.empty()) return;
  if (every_.count() > 0) {
    const auto now = std::chrono::steady_clock::now();
    if (now - reported_ < every_) return;
    reported_ = now;
  } else if (samples_.size() < kReport) {
    return;
  }
  std::ranges::sort(samples_);
  const float median = samples_[samples_.size() / 2];
  const float p95 = samples_[samples_.size() * 95 / 100];
  report_(std::format("{} : médiane {:.2f} ms, p95 {:.2f} ms, max {:.2f} ms sur {} images", label_, median,
                      p95, samples_.back(), samples_.size()));
  samples_.clear();
}

namespace {

std::string summary_of(const std::string& label, const StageSamples& all) {
  if (all.count() == 0) return {};
  std::vector<float> sorted = all.samples();
  std::ranges::sort(sorted);
  // A03-22 : nombre, total et minimum de toute la séance ; passé le plafond, les centiles viennent d'un
  // échantillon tiré sur toute la séance, et la ligne le dit.
  return std::format("{} : médiane {:.2f} ms, p95 {:.2f} ms, min {:.2f} ms, total {:.0f} ms ({} mesures{})",
                     label, sorted[sorted.size() / 2], sorted[sorted.size() * 95 / 100], all.min_ms(),
                     all.total_ms(), all.count(),
                     all.sampled() ? std::format(", centiles sur {} tirées au sort", sorted.size()) : "");
}

}  // namespace

std::string GpuTiming::summary() {
  collect();
  return summary_of(label_, all_);
}

GpuStageSummary GpuTiming::profile() {
  collect();
  return all_.summary(label_);
}

GpuSpan::GpuSpan(std::string label) : label_(std::move(label)) {
  for (Pair& pair : pairs_) {
    cuda_warn(cudaEventCreate(&pair.start), "cudaEventCreate");
    cuda_warn(cudaEventCreate(&pair.stop), "cudaEventCreate");
  }
}

GpuSpan::~GpuSpan() {
  for (const Pair& pair : pairs_) {
    if (pair.start != nullptr) cudaEventDestroy(pair.start);
    if (pair.stop != nullptr) cudaEventDestroy(pair.stop);
  }
}

void GpuSpan::start(std::uintptr_t key, cudaStream_t stream) {
  collect();
  if (key == 0) return;
  // Une source déjà en vol sous cette clé (tenseur réutilisé avant la fin de son SPAN) : celle-ci la
  // remplace.
  for (Pair& pair : pairs_)
    if (pair.state == Pair::State::Started && pair.key == key) pair.state = Pair::State::Free;
  Pair* chosen = nullptr;
  for (std::size_t i = 0; i < pairs_.size() && chosen == nullptr; ++i) {
    Pair& pair = pairs_.at((next_ + i) % pairs_.size());
    if (pair.state == Pair::State::Free) chosen = &pair;
  }
  if (chosen == nullptr) {  // tout est en vol : la plus ancienne pas encore finie est oubliée
    Pair& oldest = pairs_.at(next_ % pairs_.size());
    if (oldest.state == Pair::State::Finished) return;  // une paire finie pas encore relue : on saute
    chosen = &oldest;
  }
  if (chosen->start == nullptr || cudaEventRecord(chosen->start, stream) != cudaSuccess) return;
  chosen->key = key;
  chosen->state = Pair::State::Started;
  next_ = static_cast<std::size_t>(chosen - pairs_.data()) + 1;
}

void GpuSpan::finish(std::uintptr_t key, cudaStream_t stream) {
  for (Pair& pair : pairs_) {
    if (pair.state != Pair::State::Started || pair.key != key) continue;
    if (cudaEventRecord(pair.stop, stream) == cudaSuccess) pair.state = Pair::State::Finished;
    return;
  }
}

void GpuSpan::collect() {
  for (Pair& pair : pairs_) {
    if (pair.state != Pair::State::Finished || cudaEventQuery(pair.stop) != cudaSuccess) continue;
    float ms = 0.0F;
    if (cudaEventElapsedTime(&ms, pair.start, pair.stop) == cudaSuccess) all_.add(ms);
    pair.state = Pair::State::Free;
  }
}

std::string GpuSpan::summary() {
  collect();
  return summary_of(label_, all_);
}

GpuStageSummary GpuSpan::profile() {
  collect();
  return all_.summary(label_);
}

}  // namespace ajp::gpu::cuda
