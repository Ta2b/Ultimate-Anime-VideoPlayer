// Le relevé de l'OSD (ChainStats) et la durée de vie du backend dont il lit les noyaux (A08-02).
#include "app/play/chain_stats.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

// Noyaux factices : seule la mémoire vidéo répond, et compte ses lectures.
class CountingKernels final : public gpu::KernelOps {
 public:
  std::atomic<int> reads{0};
  std::atomic<bool> reading{false};   // un relevé en cours
  std::chrono::milliseconds slow{0};  // durée d'un relevé (cudaMemGetInfo qui attend le verrou du contexte)
  std::optional<DeviceMemory> device_memory() override {
    reading = true;
    std::this_thread::sleep_for(slow);
    ++reads;
    reading = false;
    return DeviceMemory{.used = 1, .total = 2};
  }
  std::expected<void, Refusal> ingest(const DecodedFrame&, const gpu::IngestSpec&,
                                      const gpu::Tensor&) override {
    return refuse(RefusalCode::FormatUnsupported, "factice");
  }
  std::expected<void, Refusal> resample(const gpu::Tensor&, kernels::Filter, const gpu::Tensor&) override {
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

// A30-07 : un relevé lent en vol quand la chaîne s'arrête. stop() doit l'attendre : le backend est détruit
// juste après, et le relevé lirait des noyaux rendus.
TEST_CASE("ChainStats : stop() attend le relevé en vol avant de rendre les noyaux (A30-07)", "[app][stats]") {
  const LiveSettings live;
  CountingKernels kernels;
  kernels.slow = 300ms;
  ChainStats stats(live, kernels, false, "RIFE", 24.0);
  static_cast<void>(stats.memory());
  for (int i = 0; i < 200 && !kernels.reading.load(); ++i) std::this_thread::sleep_for(2ms);
  REQUIRE(kernels.reading.load());
  stats.stop();
  CHECK_FALSE(kernels.reading.load());
  CHECK(kernels.reads.load() >= 1);
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

// Une paire de 4 images, chacune `each_us` de calcul, vue écrite par main.
void feed_pair(LiveSettings& live, std::uint32_t pair, std::int64_t each_us, bool heavy, std::int64_t& now) {
  live.auto_expect(pair, 4, 0, heavy);
  for (int image = 0; image < 4; ++image) {
    const std::int64_t submitted = now;
    now += each_us;
    live.auto_written(image == 3 ? pair : pair - 1, pair, submitted, now);
  }
}

// L'écart attendu du Rapide que lisent l'OSD et le compte des manques sans écart planifié : celui d'une paire
// calme. Une paire lourde planifie le sien, plus long, et le porte sur ses images (RingFrame::planned_gap_us)
// : sans cela, sur un écran de 165 Hz, chacune de ses images comptait une manquée et le Rapide se repliait.
TEST_CASE("ChainStats : Rapide à 165 Hz, la période de sortie est celle d'une paire calme",
          "[app][stats][free]") {
  LiveSettings live;
  live.set_free(true, 6'061, 0);
  live.set_vsync(6'061, 3'000);
  std::int64_t now = 20'000'000;
  for (std::uint32_t pair = 1; pair <= 8; ++pair) feed_pair(live, pair, 4'000, false, now);
  for (std::uint32_t pair = 9; pair <= 16; ++pair) feed_pair(live, pair, 9'000, true, now);
  CountingKernels kernels;
  ChainStats stats(live, kernels, true, "RIFE", 23.976);
  CHECK(stats.output_period_us() == 6'061);  // 4,4 ms, arrondis au vsync au-dessus
  const auto [heavy_step, gap] = live.free_step(now, true);
  CHECK(heavy_step == 9'900);  // 9 ms × 1,10 : deux vsyncs sur la grille
  CHECK(gap == 6'061);
  CHECK(heavy_step > stats.output_period_us() + stats.output_period_us() / 4);
  stats.stop();
}

TEST_CASE("Onglet Moteurs : DistilDRBA n'est pas étiqueté SPAN (C08-09)", "[app][stats]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-engine-badges-c0809";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto write = [&](const char* log, const char* onnx) {
    std::ofstream(dir / log)
        << "&&&& RUNNING # trtexec --onnx=/m/" << onnx
        << " --optShapes=input:1x3x1088x1920\n[I] Engine built in 42.0 sec.\n&&&& PASSED\n";
  };
  write("a.build.log", "distil_drba_v2_lite.onnx");
  write("b.build.log", "2x_AnimeJaNai_V3_SharpBeta_Compact.onnx");
  const std::vector<EngineBadge> badges = engine_badges(dir, {});
  REQUIRE(badges.size() == 2);
  const auto named = [&](std::string_view start) {
    return std::ranges::count_if(badges, [&](const EngineBadge& b) { return b.name.starts_with(start); });
  };
  CHECK(named("DistilDRBA · distil_drba_v2_lite") == 1);
  CHECK(named("SPAN") == 1);
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace ajp::app::play
