#include "gpu/vulkan/backend/stage_clock.h"

#include <array>
#include <utility>

namespace ajp::gpu::vk {

std::unique_ptr<StageClock> StageClock::create(VkContext& context, MemoryAllocator& memory,
                                               BackendQueue& queue) {
  const net::TimestampSupport support = net::timestamp_support(context);
  if (!(support.period_ns > 0.0)) return nullptr;
  auto pool = std::make_unique<net::QueryPool>(context, net::query_functions(context),
                                               static_cast<std::uint32_t>(2 * kPairs));
  if (!pool->create()) return nullptr;
  return std::unique_ptr<StageClock>(new StageClock(context, memory, queue, support, std::move(pool)));
}

StageClock::StageClock(VkContext& context, MemoryAllocator& memory, BackendQueue& queue,
                       net::TimestampSupport support, std::unique_ptr<net::QueryPool> pool)
    : context_(context),
      memory_(memory),
      queue_(queue),
      support_(support),
      pool_(std::move(pool)),
      pairs_(kPairs) {}

StageClock::~StageClock() {
  // Les estampilles d'abord (chacune attend sa dernière exécution), le pool ensuite (ordre des membres).
  std::scoped_lock lock(mutex_);
  pairs_.clear();
  // Un GPU bloqué garde les pools des estampilles (~Recording) : le pool de requêtes reste aussi, comme les
  // autres objets du socle (C49-04).
  if (context_.stalled()) {
    [[maybe_unused]] const net::QueryPool* kept = pool_.release();
  }
}

std::expected<std::unique_ptr<Recording>, Refusal> StageClock::stamp(std::uint32_t query) {
  auto recorder = Recorder::begin(context_, memory_, {.sets = 1, .buffers = 1, .images = 1});
  if (!recorder) return std::unexpected(recorder.error());
  const net::QueryPool& pool = *pool_;
  // Aucun tampon ni image écrit : l'ordre entre voies n'en attend rien.
  if (auto ok = recorder->commands(
          [&](VkCommandBuffer cb) {
            pool.fn().reset(cb, pool.handle(), query, 1);
            pool.fn().write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool.handle(), query);
          },
          {}, false);
      !ok) {
    return std::unexpected(ok.error());
  }
  return recorder->finish();
}

Done StageClock::ready(std::size_t i) {
  Pair& pair = pairs_[i];
  if (pair.begin && pair.end) return {};
  auto begin = stamp(static_cast<std::uint32_t>(2 * i));
  if (!begin) return std::unexpected(begin.error());
  auto end = stamp(static_cast<std::uint32_t>((2 * i) + 1));
  if (!end) return std::unexpected(end.error());
  pair.begin = std::move(*begin);
  pair.end = std::move(*end);
  return {};
}

std::size_t StageClock::stage_index(std::string_view name) {
  if (const auto found = by_name_.find(name); found != by_name_.end()) return found->second;
  stages_.push_back(Stage{.name = std::string(name)});
  by_name_.emplace(std::string(name), stages_.size() - 1);
  return stages_.size() - 1;
}

void StageClock::collect() {
  for (std::size_t i = 0; i < pairs_.size(); ++i) {
    Pair& pair = pairs_[i];
    if (pair.state == State::Abandoned && pair.begin->idle()) pair.state = State::Free;
    if (pair.state != State::Closed || !pair.begin->idle() || !pair.end->idle()) continue;
    std::array<std::uint64_t, 2> ticks{};
    const net::QueryPool& pool = *pool_;
    // Sans VK_QUERY_RESULT_WAIT_BIT : les deux estampilles sont finies, leurs valeurs sont là.
    const VkResult read =
        pool.fn().results(context_.device(), pool.handle(), static_cast<std::uint32_t>(2 * i), 2,
                          sizeof(ticks), ticks.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
    if (read == VK_SUCCESS) {
      stages_[pair.stage].all.add(static_cast<float>(net::ticks_to_ns(ticks[0], ticks[1], support_) / 1.0e6));
    }
    pair.state = State::Free;  // une lecture refusée perd la mesure, pas la paire
  }
}

std::size_t StageClock::open(std::string_view name) {
  std::scoped_lock lock(mutex_);
  collect();
  std::size_t i = 0;
  while (i < pairs_.size() && pairs_[i].state != State::Free) ++i;
  if (i == pairs_.size() || !ready(i)) return kNone;
  Pair& pair = pairs_[i];
  pair.file = queue_.current_file();
  if (!context_.enqueue(*pair.begin, {}, {}, pair.file)) return kNone;
  pair.stage = stage_index(name);
  pair.state = State::Open;
  return i;
}

void StageClock::close(std::size_t i) {
  if (i == kNone) return;
  std::scoped_lock lock(mutex_);
  Pair& pair = pairs_[i];
  if (pair.state != State::Open) return;
  // Une autre file que celle de l'ouverture, ou une estampille refusée : la paire ne mesure rien de juste.
  const bool same = queue_.current_file() == pair.file;
  pair.state = same && context_.enqueue(*pair.end, {}, {}, pair.file) ? State::Closed : State::Abandoned;
}

std::vector<GpuStageSummary> StageClock::profile() {
  std::scoped_lock lock(mutex_);
  collect();
  std::vector<GpuStageSummary> out;
  out.reserve(stages_.size());
  for (const Stage& stage : stages_) {
    if (stage.all.empty()) continue;
    GpuStageSummary s = summarize(stage.name, stage.all.kept());
    s.count = static_cast<std::size_t>(stage.all.seen());
    s.total_ms = stage.all.total();
    out.push_back(std::move(s));
  }
  return out;
}

}  // namespace ajp::gpu::vk
