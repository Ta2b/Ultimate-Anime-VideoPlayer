#include "gpu/cuda/backend/gpu_timing.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

#include "gpu/cuda/backend/cuda_warn.h"

namespace ajp::gpu::cuda {

GpuTiming::GpuTiming(std::string label, Report report)
    : label_(std::move(label)), report_(std::move(report)) {
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
  // Une paire restée ouverte (l'étape d'avant est sortie en erreur entre begin et end) est abandonnée : la
  // refermer ici mesurerait du début de l'image précédente à la fin de celle-ci, une fausse pointe au profil.
  if (begun_ != ended_) --begun_;
  if (begun_ - read_ >= kPairs) return;  // toutes en vol : on saute
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
      if (report_) samples_.push_back(ms);
      all_.add(ms);
    }
    ++read_;
  }
  if (!report_ || samples_.size() < kReport) return;
  std::erase_if(samples_, [](float ms) { return !std::isfinite(ms); });
  std::ranges::sort(samples_);
  if (!samples_.empty()) {
    report_(std::format("{} : médiane {:.2f} ms, p95 {:.2f} ms, max {:.2f} ms sur {} images", label_,
                        sorted_median(samples_), sorted_p95(samples_), samples_.back(), samples_.size()));
  }
  samples_.clear();
}

namespace {

std::string summary_of(const std::string& label, const core::SampleReservoir<float>& all) {
  if (all.empty()) return {};
  std::vector<float> sorted = all.kept();
  std::erase_if(sorted, [](float ms) { return !std::isfinite(ms); });
  if (sorted.empty()) return {};
  std::ranges::sort(sorted);
  const std::string kept =
      sorted.size() < all.seen() ? std::format(", centiles sur {} gardées", sorted.size()) : std::string();
  return std::format("{} : médiane {:.2f} ms, p95 {:.2f} ms, min {:.2f} ms, total {:.0f} ms ({} mesures{})",
                     label, sorted_median(sorted), sorted_p95(sorted), sorted.front(), all.total(),
                     all.seen(), kept);
}

// Le résumé pour le .gpu.tsv : centiles de l'échantillon, nombre et total de toutes les mesures.
GpuStageSummary profile_of(const std::string& label, const core::SampleReservoir<float>& all) {
  GpuStageSummary out = summarize(label, all.kept());
  out.count = static_cast<std::size_t>(all.seen());
  out.total_ms = all.total();
  return out;
}

}  // namespace

std::string GpuTiming::summary() {
  collect();
  return summary_of(label_, all_);
}

GpuStageSummary GpuTiming::profile() {
  collect();
  return profile_of(label_, all_);
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
  return profile_of(label_, all_);
}

}  // namespace ajp::gpu::cuda
