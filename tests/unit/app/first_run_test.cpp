#include "app/play/first_run.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "core/settings.h"

using namespace ajp;
using namespace ajp::app::play;

namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
const GpuFacts kRtx{.name = "NVIDIA GeForce RTX 4070 Ti",
                    .vendor = GpuVendor::Nvidia,
                    .vram_bytes = 12 * kGiB,
                    .tensor_cores = true};
const EnginesHere kHere{.tensorrt = true, .maxine = true, .maxine_status = "importé"};

const EngineRow* row_of(const FirstRunView& view, core::EngineId id) {
  const auto found = std::ranges::find(view.engines, id, [](const EngineRow& row) { return row.offer.id; });
  return found == view.engines.end() ? nullptr : &*found;
}

}  // namespace

TEST_CASE("Une RTX de 12 Go : TensorRT et Maxine cochés, profil 1", "[app][first_run]") {
  CHECK(gpu_line(kRtx) == "NVIDIA GeForce RTX 4070 Ti · 12 Go · cœurs Tensor");
  CHECK(start_profile_for(kRtx) == 1);
  const FirstRunView view = first_run_view(kRtx, kHere, ProfilesView{});
  const EngineRow* vulkan = row_of(view, core::EngineId::Vulkan);
  const EngineRow* tensorrt = row_of(view, core::EngineId::TensorRt);
  const EngineRow* maxine = row_of(view, core::EngineId::MaxineVfx);
  const EngineRow* amd = row_of(view, core::EngineId::MiGraphX);
  REQUIRE(vulkan != nullptr);
  REQUIRE(tensorrt != nullptr);
  REQUIRE(maxine != nullptr);
  REQUIRE(amd != nullptr);
  CHECK(vulkan->chosen);
  CHECK(tensorrt->chosen);
  CHECK(std::ranges::contains(tensorrt->offer.badges, core::Badge::Recommended));
  CHECK(maxine->chosen);
  CHECK_FALSE(amd->chosen);
  CHECK_FALSE(amd->usable);
  CHECK(row_of(view, core::EngineId::DirectMl) == nullptr);  // Windows seulement
}

TEST_CASE("Profil conseillé selon le GPU", "[app][first_run]") {
  GpuFacts small = kRtx;
  small.vram_bytes = 8 * kGiB;
  CHECK(start_profile_for(small) == 7);
  GpuFacts gtx = kRtx;
  gtx.tensor_cores = false;
  CHECK(start_profile_for(gtx) == 9);
  const GpuFacts amd{.name = "AMD Radeon", .vendor = GpuVendor::Amd, .vram_bytes = 16 * kGiB};
  CHECK(start_profile_for(amd) == 9);
}

TEST_CASE("Commencer écrit des moteurs valides pour settings.toml", "[app][first_run]") {
  FirstRunView view = first_run_view(kRtx, kHere, ProfilesView{});
  const core::Settings all = settings_after(view, core::Settings{});
  CHECK(all.first_run_done);
  CHECK(core::validate_settings(all).has_value());
  CHECK(std::ranges::contains(all.chosen_engines, core::EngineId::TensorRt));
  CHECK(all.preferred[static_cast<std::size_t>(core::Usage::Upscale)] == core::EngineId::TensorRt);
  CHECK(all.preferred[static_cast<std::size_t>(core::Usage::TrueHdr)] == core::EngineId::MaxineVfx);
  CHECK(chosen_line(all) == "Vulkan, TensorRT, NVIDIA Maxine (RTX HDR)");

  // Décocher TensorRT et Maxine : Vulkan seul, toujours valide.
  for (EngineRow& row : view.engines) {
    if (row.offer.id != core::EngineId::Vulkan) row.chosen = false;
  }
  const core::Settings vulkan = settings_after(view, core::Settings{});
  CHECK(core::validate_settings(vulkan).has_value());
  CHECK(vulkan.chosen_engines == std::vector{core::EngineId::Vulkan});
  CHECK_FALSE(vulkan.preferred[static_cast<std::size_t>(core::Usage::TrueHdr)].has_value());

  // --first-run ensuite : l'écran repart du choix enregistré, pas de la présélection.
  const FirstRunView again = first_run_view(kRtx, kHere, ProfilesView{}, vulkan);
  const EngineRow* tensorrt = row_of(again, core::EngineId::TensorRt);
  REQUIRE(tensorrt != nullptr);
  CHECK_FALSE(tensorrt->chosen);
}

TEST_CASE("Sans TensorRT ni Maxine, la chaîne garde l'image source et le dit", "[app][first_run]") {
  PlaySettings s;
  s.upscale = true;
  s.rife_factor = 3;
  s.true_hdr = true;
  core::Settings vulkan;
  vulkan.first_run_done = true;
  std::vector<std::string> notes;
  const PlaySettings plain = within_engines(s, vulkan, notes);
  CHECK_FALSE(plain.upscale);
  CHECK(plain.rife_factor == 0);
  CHECK_FALSE(plain.true_hdr);
  CHECK(plain.hdr_native);
  CHECK(notes.size() == 2);

  notes.clear();
  const core::Settings all = settings_after(first_run_view(kRtx, kHere, ProfilesView{}), core::Settings{});
  const PlaySettings kept = within_engines(s, all, notes);
  CHECK(kept.upscale);
  CHECK(kept.rife_factor == 3);
  CHECK(kept.true_hdr);
  CHECK(notes.empty());
}
