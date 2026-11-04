#include "core/present_ledger.h"

#include <algorithm>

namespace ajp::core {

PresentLedger::Issue PresentLedger::issue(std::int64_t submit_us, std::int64_t target_us, bool wanted) {
  const Issue out{.id = next_id_++, .timing = wanted && !suspended_};
  if (!out.timing) ++untimed_;
  if (!out.timing) return out;
  pending_.push_back(Pending{.id = out.id, .submit_us = submit_us, .target_us = target_us});
  while (pending_.size() > kLost) {
    pending_.pop_front();
    ++lost_;
  }
  return out;
}

void PresentLedger::queue_full() {
  suspended_ = true;
  if (!pending_.empty() && pending_.back().id == next_id_ - 1) {
    pending_.pop_back();
    ++untimed_;
  }
}

std::optional<PresentSample> PresentLedger::report(std::uint64_t id, std::int64_t visible_us) {
  const auto it = std::ranges::find(pending_, id, &Pending::id);
  if (it == pending_.end()) return std::nullopt;
  const PresentSample sample{
      .id = id, .submit_us = it->submit_us, .target_us = it->target_us, .visible_us = visible_us};
  pending_.erase(it);
  ++returned_;
  suspended_ = false;  // la file de retour a fait de la place
  return sample;
}

void PresentLedger::forget() {
  lost_ += pending_.size();
  pending_.clear();
  suspended_ = false;
}

void TimeDomainMap::calibrate(std::uint64_t domain_ns, std::uint64_t monotonic_ns, std::int64_t now_us) {
  if (monotonic_) return;
  offset_ns_ = static_cast<std::int64_t>(monotonic_ns) - static_cast<std::int64_t>(domain_ns);
  taken_us_ = now_us;
  known_ = true;
}

std::int64_t TimeDomainMap::to_host_us(std::uint64_t domain_ns) const {
  return (static_cast<std::int64_t>(domain_ns) + offset_ns_) / 1000;
}

std::uint64_t TimeDomainMap::to_domain_ns(std::int64_t host_us) const {
  if (host_us == 0) return 0;
  return static_cast<std::uint64_t>(std::max<std::int64_t>(host_us * 1000 - offset_ns_, 1));
}

}  // namespace ajp::core
