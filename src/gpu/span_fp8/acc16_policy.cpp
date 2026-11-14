#include "gpu/span_fp8/acc16_policy.h"

#include <algorithm>

namespace ajp::gpu::span_fp8 {

Acc16Change Acc16Policy::poll(std::int64_t overflowed) {
  ++launch_;
  if (!f16_) {
    seen_ =
        std::max(seen_, overflowed);  // les débordements d'avant le passage, vus en retard, ne comptent pas
    if (launch_ < f32_until_) return Acc16Change::None;
    f16_ = true;
    return Acc16Change::ToF16;
  }
  // Plusieurs images débordées vues d'un coup : chacune compte, toutes à ce lancement.
  for (; seen_ < overflowed; ++seen_) {
    recent_[static_cast<std::size_t>(next_)] = launch_;
    next_ = (next_ + 1) % kFrames;
    count_ = std::min(count_ + 1, kFrames);
  }
  if (count_ < kFrames) return Acc16Change::None;
  const std::int64_t oldest =
      recent_[static_cast<std::size_t>(next_)];  // l'anneau plein : next_ est la plus ancienne
  if (launch_ - oldest >= kWindow) return Acc16Change::None;
  f16_ = false;
  f32_until_ = launch_ + kCooldown;
  count_ = 0;
  next_ = 0;
  return Acc16Change::ToF32;
}

}  // namespace ajp::gpu::span_fp8
