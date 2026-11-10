// Le relevé de l'OSD (ChainStats) et la durée de vie du backend dont il lit les noyaux (A08-02).
#include "app/play/chain_stats.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

// Noyaux factices : seule la mémoire vidéo répond, et compte ses lectures.
class CountingKernels final : public gpu::KernelOps {
 public:
  std::atomic<int> reads{0};
  std::optional<DeviceMemory> device_memory() override {
    ++reads;
    return DeviceMemory{.used = 1, .total = 2};
  }
  std::expected<void, Refusal> ingest(const DecodedFrame&, const gpu::IngestSpec&,
                                      const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> resample(const gpu::Tensor&, kernels::Filter, const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> rife_pad(const DecodedFrame&, const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> rife_crop(const gpu::Tensor&, const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> warp_blend(const gpu::Tensor&, const gpu::Tensor&, const gpu::Motion&, float,
                                          const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> write_ring(const gpu::Tensor&, const RingSlotDesc&,
                                          const gpu::RingWrite&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> measure(const DecodedFrame&, const DecodedFrame&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::optional<gpu::PairScores> previous_scores() override { return std::nullopt; }
};

TEST_CASE("ChainStats : après stop(), plus aucun relevé ne lit les noyaux, même si l'OSD redemande (A08-02)",
          "[app][stats]") {
  const LiveSettings live;
  CountingKernels kernels;
  ChainStats stats(live, kernels, false, "RIFE", 24.0);
  static_cast<void>(stats.memory());  // le fil naît au premier appel
  for (int i = 0; i < 200 && kernels.reads.load() == 0; ++i) std::this_thread::sleep_for(5ms);
  REQUIRE(kernels.reads.load() > 0);
  stats.stop();
  const int seen = kernels.reads.load();
  static_cast<void>(stats.memory());   // l'OSD encore ouvert : le fil ne renaît pas
  std::this_thread::sleep_for(700ms);  // plus d'un tour du relevé (500 ms)
  CHECK(kernels.reads.load() == seen);
}

TEST_CASE("ChainStats : stop() sans relevé lancé ne démarre rien", "[app][stats]") {
  const LiveSettings live;
  CountingKernels kernels;
  ChainStats stats(live, kernels, false, "RIFE", 24.0);  // une lecture à l'ouverture (A32-10)
  stats.stop();                                          // une à la fermeture, pour le journal (A33-08)
  const int seen = kernels.reads.load();
  CHECK(seen == 2);
  static_cast<void>(stats.memory());
  std::this_thread::sleep_for(50ms);
  CHECK(kernels.reads.load() == seen);  // aucun fil de relevé
}

}  // namespace
}  // namespace ajp::app::play
