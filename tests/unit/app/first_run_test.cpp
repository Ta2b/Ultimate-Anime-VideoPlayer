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
// Les moteurs proposés dépendent du système : ces cas sont ceux de Linux, sur tout hôte.
const EnginesHere kHere{.os = core::Os::Linux, .tensorrt = true, .maxine = true, .maxine_status = "importé"};

const EngineRow* row_of(const FirstRunView& view, core::EngineId id) {
  const auto found = std::ranges::find(view.engines, id, [](const EngineRow& row) { return row.offer.id; });
  return found == view.engines.end() ? nullptr : &*found;
}
// La ligne pointe dans la vue : une vue temporaire meurt avant qu'on la lise (ASan).
const EngineRow* row_of(FirstRunView&& view, core::EngineId id) = delete;

}  // namespace

TEST_CASE("Resizable BAR : actif si le tas visible approche la VRAM, inactif à 256 Mo", "[app][first_run]") {
  GpuFacts gpu = kRtx;
  CHECK(rebar_text(gpu).empty());  // le pilote n'expose aucun tas visible
  gpu.host_visible_vram_bytes = 12ULL << 30U;
  CHECK(rebar_text(gpu) == "actif, 12 Go");
  CHECK(gpu_line(gpu) == "NVIDIA GeForce RTX 4070 Ti · 12 Go · Resizable BAR actif, 12 Go · cœurs Tensor");
  gpu.host_visible_vram_bytes = 256ULL << 20U;
  CHECK(rebar_text(gpu) == "inactif, 256 Mo");
  gpu.vendor = GpuVendor::Intel;
  CHECK(rebar_text(gpu).empty());
}

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

TEST_CASE("Sous Windows, RTX HDR passe par le RTX Video SDK, sans offre DirectML", "[app][first_run]") {
  EnginesHere here = kHere;
  here.os = core::Os::Windows;
  const FirstRunView view = first_run_view(kRtx, here, ProfilesView{});
  const EngineRow* tensorrt = row_of(view, core::EngineId::TensorRt);
  REQUIRE(tensorrt != nullptr);
  CHECK(tensorrt->chosen);
  CHECK(row_of(view, core::EngineId::DirectMl) == nullptr);
  CHECK(row_of(view, core::EngineId::RtxVideoSdk) != nullptr);
  CHECK(row_of(view, core::EngineId::MaxineVfx) == nullptr);  // Linux seulement
}

TEST_CASE("Sur Mac, Core ML est livré et coché, sans moteur NVIDIA ni AMD", "[app][first_run]") {
  const GpuFacts m2{.name = "Apple M2", .vendor = GpuVendor::Apple, .vram_bytes = 16 * kGiB};
  const FirstRunView view =
      first_run_view(m2, EnginesHere{.os = core::Os::MacOs, .coreml = true}, ProfilesView{});
  const EngineRow* coreml = row_of(view, core::EngineId::Metal);
  REQUIRE(coreml != nullptr);
  CHECK(coreml->usable);
  CHECK(coreml->chosen);
  CHECK(row_of(view, core::EngineId::TensorRt) == nullptr);
  CHECK(row_of(view, core::EngineId::MaxineVfx) == nullptr);
  CHECK(row_of(view, core::EngineId::MiGraphX) == nullptr);

  // Un paquet sans le backend Core ML ne le coche pas.
  const FirstRunView bare = first_run_view(m2, EnginesHere{.os = core::Os::MacOs}, ProfilesView{});
  const EngineRow* missing = row_of(bare, core::EngineId::Metal);
  REQUIRE(missing != nullptr);
  CHECK_FALSE(missing->chosen);
  CHECK(missing->status == "absent de ce paquet");
}

TEST_CASE("Profil conseillé selon le GPU", "[app][first_run]") {
  GpuFacts small = kRtx;
  small.vram_bytes = 8 * kGiB;
  CHECK(start_profile_for(small) == 7);
  GpuFacts gtx = kRtx;
  gtx.tensor_cores = false;
  CHECK(start_profile_for(gtx) == 7);
  gtx.vram_bytes = 2 * kGiB;
  CHECK(start_profile_for(gtx) == 9);
  const GpuFacts amd{.name = "AMD Radeon", .vendor = GpuVendor::Amd, .vram_bytes = 16 * kGiB};
  CHECK(start_profile_for(amd) == 7);
  const GpuFacts rx6500{.name = "AMD Radeon RX 6500 XT", .vendor = GpuVendor::Amd, .vram_bytes = 4 * kGiB};
  CHECK(start_profile_for(rx6500) == 7);
  const GpuFacts rx9070xt{
      .name = "AMD Radeon RX 9070 XT", .vendor = GpuVendor::Amd, .vram_bytes = 16 * kGiB, .wmma = true};
  CHECK(start_profile_for(rx9070xt) == 1);
  const GpuFacts rtx4050m{.name = "NVIDIA GeForce RTX 4050 Laptop GPU",
                          .vendor = GpuVendor::Nvidia,
                          .vram_bytes = 6 * kGiB,
                          .tensor_cores = true};
  CHECK(start_profile_for(rtx4050m) == 7);
  const GpuFacts uhd{.name = "Intel(R) UHD Graphics 770", .vendor = GpuVendor::Intel, .vram_bytes = 8 * kGiB};
  CHECK(start_profile_for(uhd) == 9);
}

TEST_CASE("Unités matricielles AMD : RDNA3 et RDNA4 seulement", "[app][first_run]") {
  CHECK(amd_matrix_cores("AMD Radeon RX 9070 XT"));
  CHECK(amd_matrix_cores("AMD Radeon RX 9070 XT (RADV GFX1201)"));
  CHECK(amd_matrix_cores("AMD Radeon RX 7900 XTX (RADV NAVI31)"));
  CHECK(amd_matrix_cores("AMD Radeon RX 7600S"));
  CHECK_FALSE(amd_matrix_cores("AMD Radeon RX 6800 XT"));
  CHECK_FALSE(amd_matrix_cores("AMD Radeon RX 6500M (RADV NAVI24)"));
}

TEST_CASE("MIGraphX proposé aux seuls GPU que ROCm sert", "[app][first_run]") {
  CHECK(rocm_listed("AMD Radeon RX 9070 XT (RADV GFX1201)"));
  CHECK(rocm_listed("AMD Radeon RX 7900 XTX (RADV NAVI31)"));
  CHECK(rocm_listed("AMD Radeon RX 6800 XT"));
  CHECK(rocm_listed("AMD Radeon RX 9070"));
  CHECK_FALSE(rocm_listed("AMD Radeon RX 6500M (RADV NAVI24)"));
  CHECK_FALSE(rocm_listed("AMD Radeon RX 6500M"));
  CHECK_FALSE(rocm_listed("AMD Radeon RX 6800M"));
  CHECK_FALSE(rocm_listed("AMD Radeon 680M (RADV REMBRANDT)"));
  // Ryzen AI 9 HX et Ryzen AI Max, pas le Ryzen AI 7 350.
  CHECK(rocm_listed("AMD Radeon 890M (RADV GFX1150)"));
  CHECK(rocm_listed("AMD Radeon 8060S Graphics (RADV GFX1151)"));
  CHECK(rocm_listed("AMD Radeon(TM) 890M Graphics"));
  CHECK(rocm_listed("AMD Radeon(TM) 8050S Graphics"));
  CHECK_FALSE(rocm_listed("AMD Radeon 860M (RADV GFX1152)"));
  CHECK_FALSE(rocm_listed("AMD Radeon(TM) 860M Graphics"));
  const GpuFacts rx6500m{
      .name = "AMD Radeon RX 6500M (RADV NAVI24)", .vendor = GpuVendor::Amd, .vram_bytes = 4 * kGiB};
  const FirstRunView view = first_run_view(rx6500m, kHere, ProfilesView{});
  const EngineRow* migraphx = row_of(view, core::EngineId::MiGraphX);
  REQUIRE(migraphx != nullptr);
  CHECK_FALSE(migraphx->chosen);
}

TEST_CASE("MIGraphX de Windows ML proposé à l'essai aux AMD RDNA 3 et après", "[app][first_run]") {
  CHECK(winml_listed("AMD Radeon(TM) 860M Graphics"));
  CHECK(winml_listed("AMD Radeon 780M Graphics"));
  CHECK(winml_listed("AMD Radeon(TM) 8060S Graphics"));
  CHECK(winml_listed("AMD Radeon RX 7600M XT"));
  CHECK(winml_listed("AMD Radeon RX 9070 XT"));
  CHECK(winml_listed("AMD Radeon PRO W7800"));
  CHECK_FALSE(winml_listed("AMD Radeon RX 6800 XT"));
  CHECK_FALSE(winml_listed("AMD Radeon(TM) 680M"));
  CHECK_FALSE(winml_listed("AMD Radeon RX 580 Series"));

  const GpuFacts apu{
      .name = "AMD Radeon(TM) 860M Graphics", .vendor = GpuVendor::Amd, .vram_bytes = 4 * kGiB};
  EnginesHere here{.os = core::Os::Windows};
  here.downloads.push_back(EngineDownload{.id = core::EngineId::MiGraphX,
                                          .bytes = 40U << 20U,
                                          .license = "https://www.nuget.org/packages/x/License"});
  const FirstRunView view = first_run_view(apu, here, ProfilesView{});
  const EngineRow* migraphx = row_of(view, core::EngineId::MiGraphX);
  REQUIRE(migraphx != nullptr);
  CHECK(migraphx->offer.relevant);
  CHECK(migraphx->offer.reason == core::Reason::CompareWithVulkan);
  CHECK(migraphx->usable);
  CHECK(migraphx->download);
  CHECK_FALSE(migraphx->chosen);  // à l'essai, et sa licence attend l'accord
  CHECK(migraphx->status == "à télécharger (40 Mo)");
  CHECK(row_of(view, core::EngineId::DirectMl) == nullptr);
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
  EnginesHere here{.os = core::Os::Linux};
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

TEST_CASE("RTX HDR par notre paquet fermé : licence NVIDIA, ni clé ni NGC", "[app][first_run]") {
  EnginesHere here{.os = core::Os::Linux,
                   .tensorrt = true,
                   .maxine_status = "SDK absent (/usr/local/VideoFX)",
                   .maxine_license = "https://example.org/ngc"};
  here.downloads.push_back(EngineDownload{
      .id = core::EngineId::MaxineVfx, .bytes = 70ULL << 20, .license = "https://example.org/nvidia-sla"});
  FirstRunView view = first_run_view(kRtx, here, ProfilesView{});
  EngineRow* maxine = nullptr;
  for (EngineRow& row : view.engines) {
    if (row.offer.id == core::EngineId::MaxineVfx) maxine = &row;
  }
  REQUIRE(maxine != nullptr);
  CHECK(maxine->usable);
  CHECK(maxine->download);
  CHECK_FALSE(maxine->needs_key);  // le paquet passe avant NGC
  CHECK(maxine->license == "https://example.org/nvidia-sla");
  CHECK(maxine->status == "à télécharger (70 Mo), paquet fermé sous licence NVIDIA");
  CHECK_FALSE(maxine->chosen);

  maxine->chosen = true;
  maxine->license_accepted = true;
  CHECK(chosen(settings_after(view, {}), core::EngineId::MaxineVfx));
}
