#include "app/play/present_stats.h"

#include <array>
#include <cstdint>
#include <format>
#include <ostream>
#include <string_view>
#include <utility>

namespace ajp::app::play {
namespace {

struct Column {
  std::string_view name;
  std::string (*text)(const PresentRecord&);
};

std::string number(std::int64_t value) {
  return std::to_string(value);
}

constexpr std::array<Column, kStatsColumns> kColumns{{
    {"seq", [](const PresentRecord& r) { return number(r.seq); }},
    {"host_us", [](const PresentRecord& r) { return number(r.host_us); }},
    {"pts_us", [](const PresentRecord& r) { return number(r.pts_us); }},
    {"fresh", [](const PresentRecord& r) { return std::string(r.fresh ? "1" : "0"); }},
    {"acquire_us", [](const PresentRecord& r) { return number(r.acquire_us); }},
    {"render_us", [](const PresentRecord& r) { return number(r.render_us); }},
    {"queue_wait_us", [](const PresentRecord& r) { return number(r.queue_wait_us); }},
    {"interval_us", [](const PresentRecord& r) { return number(r.interval_us); }},
    {"dropped", [](const PresentRecord& r) { return number(r.dropped); }},
    {"why", [](const PresentRecord& r) { return r.why; }},
    {"missing", [](const PresentRecord& r) { return number(r.missing); }},
    {"lag_us", [](const PresentRecord& r) { return number(r.lag_us); }},
    {"uninterpolated", [](const PresentRecord& r) { return number(r.uninterpolated); }},
    {"display_gpu_us", [](const PresentRecord& r) { return number(r.display_gpu_us); }},
    {"present_id", [](const PresentRecord& r) { return std::to_string(r.present_id); }},
    {"feedback_id", [](const PresentRecord& r) { return std::to_string(r.feedback_id); }},
    {"target_us", [](const PresentRecord& r) { return number(r.target_us); }},
    {"visible_us", [](const PresentRecord& r) { return number(r.visible_us); }},
    {"vsync_k", [](const PresentRecord& r) { return number(r.vsync_k); }},
    {"level", [](const PresentRecord& r) { return r.level; }},
}};

}  // namespace

std::string stats_header() {
  std::string line;
  for (const Column& column : kColumns) {
    if (!line.empty()) line += '\t';
    line += column.name;
  }
  line += '\n';
  return line;
}

std::string stats_line(const PresentRecord& record) {
  std::string line;
  bool first = true;
  for (const Column& column : kColumns) {
    if (!first) line += '\t';
    first = false;
    line += column.text(record);
  }
  line += '\n';
  return line;
}

void StatsSink::add(const PresentRecord& record, std::int64_t now_us) {
  pending_ += stats_line(record);
  if (posted_at_us_ == 0) posted_at_us_ = now_us;
  if (now_us - posted_at_us_ >= kEveryUs || pending_.size() >= kMaxBytes) {
    flush();
    posted_at_us_ = now_us;
  }
}

void StatsSink::flush() {
  if (pending_.empty()) return;
  // Une clé par paquet, dans l'ordre (le fil d'écriture prend les clés en attente dans l'ordre) : aucun
  // paquet n'en remplace un autre.
  writer_.post(std::format("stats.{:012}", ++chunks_),
               [out = &out_, text = std::exchange(pending_, {})]() -> std::expected<void, Refusal> {
                 out->write(text.data(), static_cast<std::streamsize>(text.size()));
                 out->flush();
                 return {};
               });
}

}  // namespace ajp::app::play
