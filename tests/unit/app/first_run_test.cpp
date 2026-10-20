#include "app/play/first_run.h"

#include <algorithm>
#include <array>
#include <filesystem>

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

TEST_CASE("Sans moteur d'inférence ni Maxine, la chaîne garde l'image source et le dit", "[app][first_run]") {
  PlaySettings s;
  s.upscale = true;
  s.rife_factor = 3;
  s.true_hdr = true;
  core::Settings vulkan;
  vulkan.first_run_done = true;
  core::Settings none = vulkan;
  none.chosen_engines.clear();
  std::vector<std::string> notes;
  const PlaySettings plain = within_engines(s, none, notes);
  CHECK_FALSE(plain.upscale);
  CHECK(plain.rife_factor == 0);
  CHECK_FALSE(plain.true_hdr);
  CHECK(plain.hdr_native);
  CHECK(notes.size() == 2);

  // Vulkan seul (AMD, Intel) : SPAN et RIFE passent par le backend Vulkan.
  notes.clear();
  core::Settings vulkan_only = vulkan;
  vulkan_only.chosen_engines = {core::EngineId::Vulkan};
  const PlaySettings on_vulkan = within_engines(s, vulkan_only, notes);
  CHECK(on_vulkan.upscale);
  CHECK(on_vulkan.rife_factor == 3);
  CHECK_FALSE(on_vulkan.true_hdr);
  CHECK(notes.size() == 1);

  notes.clear();
  const core::Settings all = settings_after(first_run_view(kRtx, kHere, ProfilesView{}), core::Settings{});
  const PlaySettings kept = within_engines(s, all, notes);
  CHECK(kept.upscale);
  CHECK(kept.rife_factor == 3);
  CHECK(kept.true_hdr);
  CHECK(notes.empty());
}

TEST_CASE("TensorRT n'est essayé que s'il a été choisi", "[app][first_run]") {
  const std::array<std::filesystem::path, 2> candidates{"base/ajp_backend_tensorrt.so",
                                                        "base/ajp_backend_vulkan.so"};
  core::Settings vulkan_only;
  vulkan_only.chosen_engines = {core::EngineId::Vulkan};
  const auto without = backends_within(candidates, vulkan_only);
  REQUIRE(without.size() == 1);
  CHECK(without.front().filename() == "ajp_backend_vulkan.so");

  core::Settings both = vulkan_only;
  both.chosen_engines.push_back(core::EngineId::TensorRt);
  CHECK(backends_within(candidates, both).size() == 2);
}

TEST_CASE("MIGraphX et OpenVINO ne sont essayés que choisis", "[app][first_run]") {
  const std::array<std::filesystem::path, 3> candidates{
      "base/ajp_backend_migraphx.so", "base/ajp_backend_openvino.so", "base/ajp_backend_vulkan.so"};
  core::Settings settings;
  settings.chosen_engines = {core::EngineId::Vulkan, core::EngineId::OpenVino};
  const auto kept = backends_within(candidates, settings);
  REQUIRE(kept.size() == 2);
  CHECK(kept.front().filename() == "ajp_backend_openvino.so");
  CHECK(kept.back().filename() == "ajp_backend_vulkan.so");

  // Core ML est le moteur Metal de l'écran (Mac).
  const std::array<std::filesystem::path, 2> mac{"base/ajp_backend_coreml.so", "base/ajp_backend_vulkan.so"};
  CHECK(backends_within(mac, settings).size() == 1);
  settings.chosen_engines.push_back(core::EngineId::Metal);
  CHECK(backends_within(mac, settings).size() == 2);
}

TEST_CASE("Moteur à télécharger : licence acceptée avant tout choix", "[app][first_run]") {
  EnginesHere here;
  here.downloads.push_back(EngineDownload{
      .id = core::EngineId::TensorRt, .bytes = 900ULL << 20, .license = "https://example.org/sla"});
  FirstRunView view = first_run_view(kRtx, here, ProfilesView{});
  EngineRow* tensorrt = nullptr;
  for (EngineRow& row : view.engines) {
    if (row.offer.id == core::EngineId::TensorRt) tensorrt = &row;
  }
  REQUIRE(tensorrt != nullptr);
  CHECK(tensorrt->usable);
  CHECK(tensorrt->download);
  CHECK_FALSE(tensorrt->chosen);  // jamais coché d'office avec une licence
  CHECK(tensorrt->status == "à télécharger (900 Mo)");

  tensorrt->chosen = true;
  const core::Settings refused = settings_after(view, {});
  CHECK_FALSE(chosen(refused, core::EngineId::TensorRt));
  CHECK_FALSE(std::ranges::contains(refused.installed_engines, core::EngineId::TensorRt));

  tensorrt->license_accepted = true;
  const core::Settings accepted = settings_after(view, {});
  CHECK(chosen(accepted, core::EngineId::TensorRt));
  CHECK(accepted.preferred[static_cast<std::size_t>(core::Usage::Upscale)] == core::EngineId::TensorRt);

  const core::Settings failed = without_engine(accepted, core::EngineId::TensorRt);
  CHECK_FALSE(chosen(failed, core::EngineId::TensorRt));
  CHECK(failed.preferred[static_cast<std::size_t>(core::Usage::Upscale)] == core::EngineId::Vulkan);
}
