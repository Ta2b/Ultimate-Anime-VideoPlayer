// Temps de TrueHDR Maxine par image sur la RTX, caché ([.bench]) : à lancer à la main sous le
// verrou GPU, la machine au repos :
//   ajp-gpu build/linux-prefix/tests/unit/vendor_hdr/ajp_vendor_hdr_tests "[bench]"
// Temps hôte de process() : NvVFX_Run sur le flux 0 rend la main une fois l'image écrite, c'est
// donc aussi le temps où le thread gpu est bloqué. 20 images d'échauffement, puis 5 passes de
// 100 images ; médiane et écart type sur les 500, et médiane de chaque passe.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "real_sdk.h"
#include "vendor_hdr/maxine_vfx.h"

using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;

namespace {

using Clock = std::chrono::steady_clock;

double median(std::vector<double> values) {
  std::ranges::sort(values);
  return values[values.size() / 2];
}

double deviation(const std::vector<double>& values) {
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  double sum = 0.0;
  for (const double value : values) sum += (value - mean) * (value - mean);
  return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

void measure(const std::filesystem::path& sdk, const Cuda& cuda, int width, int height) {
  constexpr int kWarmup = 20;
  constexpr int kPasses = 5;
  constexpr int kFrames = 100;
  const DeviceImage in(cuda, width, height, TrueHdrPixel::Rgba8);
  const DeviceImage out(cuda, width, height, TrueHdrPixel::Rgb10A2);
  CudaGpuStream stream(cuda);
  in.upload(source_rgba8(width, height, true));
  cuda.synchronize();

  const auto opening = Clock::now();
  auto opened = detail::open_maxine_at(sdk, sdr_config(width, height, cuda.context()));
  REQUIRE(opened);
  TrueHdr& hdr = **opened;
  const TrueHdrInput input{.image = in.view(), .prepared = {.dithered = true}};
  REQUIRE(hdr.process(input, out.view(), stream));  // premier appel : chargement du modèle
  cuda.synchronize();
  const double cold = milliseconds(Clock::now() - opening);

  for (int i = 0; i < kWarmup; ++i) REQUIRE(hdr.process(input, out.view(), stream));
  cuda.synchronize();
  std::vector<double> all;
  std::string passes;
  for (int pass = 0; pass < kPasses; ++pass) {
    std::vector<double> times;
    for (int i = 0; i < kFrames; ++i) {
      const auto start = Clock::now();
      REQUIRE(hdr.process(input, out.view(), stream));
      times.push_back(milliseconds(Clock::now() - start));
    }
    cuda.synchronize();
    passes += std::format(" {:.3f}", median(times));
    all.insert(all.end(), times.begin(), times.end());
  }
  std::fputs(std::format("TrueHDR {}x{} : médiane {:.3f} ms, écart type {:.3f} ms ({} images), médianes par "
                         "passe{} ; ouverture et premier appel {:.1f} ms\n",
                         width, height, median(all), deviation(all), all.size(), passes, cold)
                 .c_str(),
             stdout);
}

}  // namespace

TEST_CASE("Maxine sur la RTX : temps de TrueHDR par image, 1080p, 1440p et 4K", "[.][bench][nvidia][cuda]") {
  const std::filesystem::path sdk = require_maxine_sdk();
  const std::unique_ptr<Cuda> cuda = require_cuda();
  std::fputs(std::format("Appareil : {}\n", cuda->device_name()).c_str(), stdout);
  measure(sdk, *cuda, 1920, 1080);
  measure(sdk, *cuda, 2560, 1440);  // sortie de la chaîne de référence du lecteur actuel (baseline.md)
  measure(sdk, *cuda, 3840, 2160);
}
