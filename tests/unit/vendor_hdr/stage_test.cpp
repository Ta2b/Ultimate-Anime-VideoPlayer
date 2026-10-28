// Repli de TrueHDR sur la route native, et l'événement annoncé à l'OSD pendant 3 s.
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "support.h"
#include "vendor_hdr/true_hdr_stage.h"

using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;
using namespace std::chrono_literals;

namespace {

// TrueHdr scripté : réussit jusqu'à ce qu'on lui demande d'échouer.
class ScriptedHdr final : public TrueHdr {
 public:
  explicit ScriptedHdr(bool& fail) : fail_(fail) {}
  const TrueHdrConfig& config() const override { return config_; }
  TrueHdrPixel input_pixel() const override { return TrueHdrPixel::Rgba8; }
  Done set_params(const TrueHdrParams& params) override {
    if (fail_) return refuse(RefusalCode::VendorCallFailed, "réglage refusé par le SDK");
    config_.params = params;
    return {};
  }
  Done process(const TrueHdrInput&, const TrueHdrImage&, GpuStream&) override {
    time_.host += 5ms;
    ++time_.calls;
    if (fail_) return refuse(RefusalCode::VendorCallFailed, "TrueHDR Maxine, exécution : erreur (-1)");
    return {};
  }
  VendorTime vendor_time() const override { return time_; }

 private:
  bool& fail_;
  TrueHdrConfig config_{.width = 4, .height = 4};
  VendorTime time_;
};

struct Frame {
  HostImage in{4, 4, TrueHdrPixel::Rgba8};
  HostImage out{4, 4, TrueHdrPixel::Rgb10A2};
  RecordingStream stream;
  HdrPath run(TrueHdrStage& stage) {
    return stage.process(TrueHdrInput{.image = in.view(), .prepared = {.dithered = true}}, out.view(),
                         stream);
  }
};

}  // namespace

TEST_CASE("étage : l'annonce dure au moins 3 s", "[stage]") {
  CHECK(kFallbackNotice == 3s);
}

TEST_CASE("étage : TrueHDR qui ne s'ouvre pas bascule tout de suite", "[stage]") {
  TrueHdrStage stage(
      std::unexpected(ajp::Refusal{RefusalCode::LibraryUnavailable, "libVideoFX.so ne se charge pas"}));
  CHECK(stage.path() == HdrPath::Native);
  const auto fallback = stage.take_fallback();
  REQUIRE(fallback);
  CHECK(fallback->cause.code == RefusalCode::LibraryUnavailable);
  CHECK(fallback->cause.detail == "libVideoFX.so ne se charge pas");
  CHECK(fallback->notice == 3000ms);
  CHECK_FALSE(stage.take_fallback());

  Frame frame;
  CHECK(frame.run(stage) == HdrPath::Native);
  CHECK(frame.stream.events.empty());
  CHECK_FALSE(stage.take_fallback());
}

TEST_CASE("étage : un pointeur nul compte comme un échec d'ouverture", "[stage]") {
  TrueHdrStage stage(std::unique_ptr<TrueHdr>{});
  CHECK(stage.path() == HdrPath::Native);
  CHECK(stage.take_fallback());
}

TEST_CASE("étage : une erreur en lecture bascule une fois, pour de bon", "[stage]") {
  bool fail = false;
  TrueHdrStage stage(std::make_unique<ScriptedHdr>(fail));
  Frame frame;
  CHECK(stage.path() == HdrPath::TrueHdr);
  CHECK(frame.run(stage) == HdrPath::TrueHdr);
  CHECK_FALSE(stage.take_fallback());

  fail = true;
  CHECK(frame.run(stage) == HdrPath::Native);
  CHECK(stage.path() == HdrPath::Native);
  const auto fallback = stage.take_fallback();
  REQUIRE(fallback);
  CHECK(fallback->cause.code == RefusalCode::VendorCallFailed);
  CHECK(fallback->notice == kFallbackNotice);

  // Le SDK est fermé : même rétabli, il n'est plus appelé, et rien n'est réannoncé.
  fail = false;
  CHECK(frame.run(stage) == HdrPath::Native);
  CHECK_FALSE(stage.take_fallback());
}

TEST_CASE("étage : le temps du SDK reste compté après la bascule", "[stage]") {
  bool fail = false;
  TrueHdrStage stage(std::make_unique<ScriptedHdr>(fail));
  Frame frame;
  CHECK(frame.run(stage) == HdrPath::TrueHdr);
  fail = true;
  CHECK(frame.run(stage) == HdrPath::Native);
  CHECK(frame.run(stage) == HdrPath::Native);  // route native : plus aucun appel du SDK
  CHECK(stage.vendor_time().calls == 2);
  CHECK(stage.vendor_time().host == 10ms);

  const TrueHdrStage never(std::unexpected(ajp::Refusal{RefusalCode::LibraryUnavailable, "absent"}));
  CHECK(never.vendor_time().calls == 0);
}

TEST_CASE("étage : set_params refuse hors plage sans basculer", "[stage]") {
  bool fail = false;
  TrueHdrStage stage(std::make_unique<ScriptedHdr>(fail));
  auto refused = stage.set_params({.contrast = 300});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(stage.path() == HdrPath::TrueHdr);
  CHECK_FALSE(stage.take_fallback());

  CHECK(stage.set_params({.contrast = 150}) == HdrPath::TrueHdr);
}

TEST_CASE("étage : une erreur du SDK sur un réglage bascule", "[stage]") {
  bool fail = true;
  TrueHdrStage stage(std::make_unique<ScriptedHdr>(fail));
  CHECK(stage.set_params({.contrast = 150}) == HdrPath::Native);
  CHECK(stage.take_fallback()->cause.detail == "réglage refusé par le SDK");
  // Route native : un réglage valide est accepté, la route native le lit ailleurs.
  CHECK(stage.set_params({.contrast = 120}) == HdrPath::Native);
  CHECK_FALSE(stage.take_fallback());
}
