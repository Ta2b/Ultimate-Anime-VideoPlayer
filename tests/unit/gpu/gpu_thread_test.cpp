// Thread gpu sur la vraie bibliothèque du backend CPU : boîte bornée, Credit au producteur,
// ring rendu par un thread main de test. Passe sous TSan (preset linux-prefix-tsan-player).
#include "gpu/gpu_thread.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/loader.h"
#include "gpu_test_support.h"

using ajp::RefusalCode;
using ajp::RingFrame;
using namespace ajp::gpu;
using namespace std::chrono_literals;

namespace {

const std::filesystem::path kCpuDir = AJP_TEST_CPU_BACKEND_DIR;

// La vraie bibliothèque du backend CPU, que le thread prendra.
LoadedBackend cpu_backend() {
  auto loaded = LoadedBackend::load(kCpuDir, "cpu", BackendInit{});
  REQUIRE(loaded);
  return std::move(*loaded);
}

// La boîte du thread main : les images reçues, une à une.
class MainSink final : public RingFrameSink {
 public:
  bool try_push(RingFrame&& frame) override {
    {
      const std::scoped_lock lock(mutex_);
      if (pending_.size() >= 8) return false;
      pending_.push_back(frame);
      all_.push_back(frame);
    }
    ready_.notify_all();
    return true;
  }
  std::optional<RingFrame> wait(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (!ready_.wait_for(lock, timeout, [this] { return !pending_.empty(); })) return std::nullopt;
    RingFrame frame = pending_.front();
    pending_.pop_front();
    return frame;
  }
  std::vector<RingFrame> all() const {
    const std::scoped_lock lock(mutex_);
    return all_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<RingFrame> pending_;
  std::vector<RingFrame> all_;
};

// Les Credit que le thread gpu rend au producteur.
class Credits {
 public:
  void give() {
    {
      const std::scoped_lock lock(mutex_);
      ++count_;
    }
    ready_.notify_all();
  }
  bool take(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (!ready_.wait_for(lock, timeout, [this] { return count_ > 0; })) return false;
    --count_;
    ++taken_;
    return true;
  }
  int taken() const {
    const std::scoped_lock lock(mutex_);
    return taken_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  int count_ = 0;
  int taken_ = 0;
};

}  // namespace

TEST_CASE("Thread gpu : boîte bornée, Credit au producteur, ring rendu par main", "[thread]") {
  LoadedBackend backend = cpu_backend();
  test::TempDir dir("thread");
  // prepare construit l'artefact avec sa propre instance ; gpu le charge dans la sienne.
  const std::vector<ModelArtifact> artifacts{
      {.id = "up2",
       .artifact = test::build_artifact(
           backend.backend(), test::write_model(dir.path(), "up2", test::upscale_x2_model()), "up2")}};
  const test::HostRing ring(3, 2 * test::kWidth, 2 * test::kHeight);

  MainSink sink;
  Credits credits;
  std::mutex refusals_mutex;
  std::vector<ajp::Refusal> refusals;
  {
    auto thread =
        GpuThread::start(std::move(backend), {.frames = &sink,
                                              .credit = [&credits] { credits.give(); },
                                              .report = ajp::ThreadReport([&](const ajp::Refusal& refusal) {
                                                const std::scoped_lock lock(refusals_mutex);
                                                refusals.push_back(refusal);
                                              })});
    REQUIRE(thread);
    GpuThread& gpu = **thread;

    // Sans Program, le thread garde ses 4 images : la cinquième est refusée.
    constexpr std::uint32_t kFrames = 16;
    std::uint32_t next = 0;
    while (gpu.try_push(test::make_frame(next, static_cast<std::uint8_t>(40 + next)))) ++next;
    CHECK(next == 4);

    // main : chaque image reçue est rendue après son « rendu ». Catch2 n'assertit que depuis
    // le thread du test : main ne fait que compter.
    std::atomic<int> refused_frees = 0;
    std::jthread main([&](const std::stop_token& stop) {
      std::size_t received = 0;
      while (!stop.stop_requested() && received < kFrames) {
        const std::optional<RingFrame> frame = sink.wait(100ms);
        if (!frame) continue;
        ++received;
        if (!gpu.slot_free({.slot = frame->slot,
                            .semaphore_value = frame->ready_value + 1,
                            .generation = frame->generation})) {
          ++refused_frees;
        }
      }
    });
    REQUIRE(gpu.ring(RingGeneration{ring.slots()}));
    REQUIRE(gpu.install(ajp::Install{std::make_shared<ProgramRequest>(test::upscale_plan(), artifacts)}));
    while (next < kFrames) {
      if (gpu.try_push(test::make_frame(next, static_cast<std::uint8_t>(40 + next)))) {
        ++next;
      } else {
        REQUIRE(credits.take(10s));  // le thread gpu rend une place après une image consommée
      }
    }
    for (int i = 0; i < 200 && sink.all().size() < kFrames; ++i) std::this_thread::sleep_for(10ms);
    main.request_stop();
    main.join();
    CHECK(refused_frees == 0);

    const std::vector<RingFrame> frames = sink.all();
    REQUIRE(frames.size() == kFrames);
    for (std::size_t i = 0; i < frames.size(); ++i)
      CHECK(frames[i].pts_us == std::int64_t(i) * test::kFrameUs);
    CHECK(credits.taken() >= 1);
    const GpuStats stats = gpu.stats();
    CHECK(stats.frames == kFrames);
    CHECK(stats.written == kFrames);
    CHECK(stats.host_waits == 0);
    CHECK(gpu.flush(ajp::Flush{.epoch = 1}));
    CHECK(gpu.command(ajp::cmd::Pause{}));
  }  // arrêt par stop_token, puis attente du thread
  const std::scoped_lock lock(refusals_mutex);
  CHECK(refusals.empty());
}

TEST_CASE("Thread gpu : refus au démarrage", "[thread]") {
  MainSink sink;
  const ajp::ThreadReport report([](const ajp::Refusal&) {});
  CHECK(GpuThread::start(cpu_backend(), {.report = report}).error().code == RefusalCode::FieldMissing);
  CHECK(GpuThread::start(cpu_backend(), {.frames = &sink}).error().code == RefusalCode::FieldMissing);
  CHECK(GpuThread::start(cpu_backend(), {.frames = &sink, .report = report}, GpuBounds{.frames = 0})
            .error()
            .code == RefusalCode::FieldOutOfRange);
  CHECK(GpuThread::start(cpu_backend(), {.frames = &sink, .report = report},
                         GpuBounds{.frames = GpuWorker::kMaxBacklog + 1})
            .error()
            .code == RefusalCode::FieldOutOfRange);
  auto idle = GpuThread::start(cpu_backend(), {.frames = &sink, .report = report});
  REQUIRE(idle);  // détruit aussitôt : le thread s'arrête sans avoir rien reçu, puis son backend
}
