// Fige l'ordre des bloqueurs repris de manager-v2
// AnimeJaNaiConfEditor/ViewModels/OptionalSdkViewModel.cs:39-48.
#include "vendor_hdr/blockers.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/capabilities.h"
#include "core/resolve.h"

using ajp::RefusalCode;
using namespace ajp::vendor_hdr;

namespace {

// Tout est réuni : TrueHDR peut être tenté.
TrueHdrEnvironment ready(HostOs os = HostOs::Linux) {
  return TrueHdrEnvironment{.os = os,
                            .nvidia_gpu = true,
                            .tensorrt_engine = true,
                            .sdk_ready = true,
                            .sdk_reason = {},
                            .module_present = true,
                            .adapter_present = true,
                            .output_pq = true};
}

}  // namespace

TEST_CASE("bloqueurs : rien ne bloque quand tout est réuni", "[blockers]") {
  CHECK_FALSE(first_blocker(ready(HostOs::Linux)));
  CHECK_FALSE(first_blocker(ready(HostOs::Windows)));
}

TEST_CASE("bloqueurs : chaque cause seule", "[blockers]") {
  struct Case {
    void (*spoil)(TrueHdrEnvironment&);
    TrueHdrBlocker expected;
  };
  const Case cases[] = {
      {[](TrueHdrEnvironment& e) { e.os = HostOs::MacOs; }, TrueHdrBlocker::UnsupportedOs},
      {[](TrueHdrEnvironment& e) { e.nvidia_gpu = false; }, TrueHdrBlocker::NotNvidiaGpu},
      {[](TrueHdrEnvironment& e) { e.tensorrt_engine = false; }, TrueHdrBlocker::EngineNotTensorRt},
      {[](TrueHdrEnvironment& e) { e.sdk_ready = false; }, TrueHdrBlocker::SdkMissing},
      {[](TrueHdrEnvironment& e) { e.module_present = false; }, TrueHdrBlocker::ModuleMissing},
      {[](TrueHdrEnvironment& e) { e.adapter_present = false; }, TrueHdrBlocker::AdapterMissing},
      {[](TrueHdrEnvironment& e) { e.output_pq = false; }, TrueHdrBlocker::OutputNotPq},
  };
  for (const Case& item : cases) {
    TrueHdrEnvironment environment = ready();
    item.spoil(environment);
    const auto blocked = first_blocker(environment);
    REQUIRE(blocked);
    CHECK(blocked->blocker == item.expected);
    CHECK_FALSE(blocked->reason.empty());
  }
}

TEST_CASE("bloqueurs : la première cause dans l'ordre l'emporte", "[blockers]") {
  // Tout manque : on retire les causes une à une, la suivante prend la place.
  TrueHdrEnvironment environment{.os = HostOs::MacOs};
  const TrueHdrBlocker order[] = {TrueHdrBlocker::UnsupportedOs,     TrueHdrBlocker::NotNvidiaGpu,
                                  TrueHdrBlocker::EngineNotTensorRt, TrueHdrBlocker::SdkMissing,
                                  TrueHdrBlocker::ModuleMissing,     TrueHdrBlocker::AdapterMissing,
                                  TrueHdrBlocker::OutputNotPq};
  void (*const fix[])(TrueHdrEnvironment&) = {
      [](TrueHdrEnvironment& e) { e.os = HostOs::Linux; },
      [](TrueHdrEnvironment& e) { e.nvidia_gpu = true; },
      [](TrueHdrEnvironment& e) { e.tensorrt_engine = true; },
      [](TrueHdrEnvironment& e) { e.sdk_ready = true; },
      [](TrueHdrEnvironment& e) { e.module_present = true; },
      [](TrueHdrEnvironment& e) { e.adapter_present = true; },
      [](TrueHdrEnvironment& e) { e.output_pq = true; },
  };
  for (std::size_t i = 0; i < std::size(order); ++i) {
    const auto blocked = first_blocker(environment);
    REQUIRE(blocked);
    CHECK(blocked->blocker == order[i]);
    fix[i](environment);
  }
  CHECK_FALSE(first_blocker(environment));
}

TEST_CASE("bloqueurs : le SDK manquant porte la raison du magasin", "[blockers]") {
  TrueHdrEnvironment environment = ready(HostOs::Windows);
  environment.sdk_ready = false;
  CHECK(first_blocker(environment)->reason == "Importer le RTX Video SDK 1.1 (Paramètres > Moteurs).");

  environment.os = HostOs::Linux;
  CHECK(first_blocker(environment)->reason == "Importer le Maxine VFX 1.3 (TrueHDR) (Paramètres > Moteurs).");

  environment.sdk_reason = "lib/libVideoFX.so a changé depuis l'import : réimporter le SDK.";
  CHECK(first_blocker(environment)->reason == environment.sdk_reason);
}

TEST_CASE("bloqueurs : un SDK par système", "[blockers]") {
  CHECK(sdk_for(HostOs::Windows) == SdkKind::RtxVideo);
  CHECK(sdk_for(HostOs::Linux) == SdkKind::MaxineVfx);
  CHECK_FALSE(sdk_for(HostOs::MacOs));
}

TEST_CASE("bloqueurs : le refus de resolve() reprend la raison", "[blockers]") {
  TrueHdrEnvironment environment = ready();
  environment.output_pq = false;
  const ajp::Refusal refusal = to_refusal(*first_blocker(environment));
  CHECK(refusal.code == RefusalCode::HdrUnavailable);
  CHECK(refusal.detail == "TrueHDR n'agit qu'en sortie HDR PQ : activer le HDR de l'écran.");
}

// Le câblage que app fait au démarrage : la raison de vendor_hdr entre dans les capacités de
// core, et resolve() la rend telle quelle pour un profil TrueHDR.
TEST_CASE("bloqueurs : la raison arrive jusqu'au refus de resolve()", "[blockers]") {
  using namespace ajp::core;
  TrueHdrEnvironment environment = ready();
  CHECK(first_blocker_reason(environment).empty());
  environment.nvidia_gpu = false;
  const std::string reason = first_blocker_reason(environment);
  CHECK(reason == first_blocker(environment)->reason);

  const std::vector<CapabilityFacts> facts{
      CapabilityFacts{.id = {.kind = CapabilityKind::Route, .name = std::string(kRouteTrueHdr)},
                      .compiled = true,
                      .delivered = true,
                      .installed = true}};
  const Capabilities caps = to_resolve_capabilities(facts, reason);
  const Profile profile{.hdr = HdrMode::TrueHdr,
                        .rules = {Rule{.max_pixels = limits::kPixels.hi, .max_fps = limits::kFps.hi}}};
  const DisplayInfo display{.width = 3840, .height = 2160, .refresh_hz = 120.0, .hdr = true};
  auto plan = resolve(profile, caps, SourceInfo{.width = 1920, .height = 1080, .fps = 23.976}, display);
  REQUIRE_FALSE(plan);
  CHECK(plan.error().code == RefusalCode::HdrUnavailable);
  CHECK(plan.error().detail == reason);
}
