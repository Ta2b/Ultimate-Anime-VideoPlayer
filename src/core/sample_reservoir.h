// Mesures d'une lecture entière en mémoire bornée (A33-05) : toutes comptées et additionnées, et, au-delà de
// la capacité, un échantillon réservoir (algorithme R) uniforme sur toute la lecture. Les centiles d'un banc
// de soirée couvrent alors la chauffe et la fin, pas seulement la première demi-heure.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ajp::core {

template <class T>
class SampleReservoir {
 public:
  explicit SampleReservoir(std::size_t capacity) : capacity_(capacity) {}

  void add(T value) {
    ++seen_;
    total_ += static_cast<double>(value);
    if (kept_.size() < capacity_) {
      kept_.push_back(value);
      return;
    }
    // Chaque mesure a la même chance capacity / seen d'être gardée.
    const std::uint64_t at = next() % seen_;
    if (at < capacity_) kept_[static_cast<std::size_t>(at)] = value;
  }

  const std::vector<T>& kept() const { return kept_; }
  std::vector<T>& kept() { return kept_; }
  std::uint64_t seen() const { return seen_; }  // toutes les mesures, gardées ou non
  double total() const { return total_; }       // leur somme
  bool empty() const { return seen_ == 0; }

 private:
  // splitmix64 : déterministe, sans état global.
  std::uint64_t next() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  }

  std::size_t capacity_;
  std::vector<T> kept_;
  std::uint64_t seen_ = 0;
  double total_ = 0.0;
  std::uint64_t state_ = 0;
};

}  // namespace ajp::core
