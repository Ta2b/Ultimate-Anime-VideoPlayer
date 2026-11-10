#include "app/play/first_run.h"

#include <algorithm>
#include <array>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "app/play/engine_install.h"
#include "core/settings.h"
#include "french_ui.h"

using namespace ajp;
using namespace ajp::app::play;

namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
const GpuFacts kRtx{.name = "NVIDIA GeForce RTX 4070 Ti",
                    .vendor = GpuVendor::Nvidia,
                    .vram_bytes = 12 * kGiB,
                    .tensor_cores = true};
// Les moteurs proposés dépendent du système : ces cas sont ceux de Linux, sur tout hôte.
const EnginesHere kHere{.os = core::Os::Linux, .tensorrt = true};

const EngineRow* row_of(const FirstRunView& view, core::EngineId id) {
  const auto found = std::ranges::find(view.engines, id, [](const EngineRow& row) { return row.offer.id; });
  return found == view.engines.end() ? nullptr : &*found;
}
// La ligne pointe dans la vue : une vue temporaire meurt avant qu'on la lise (ASan).
const EngineRow* row_of(FirstRunView&& view, core::EngineId id) = delete;

}  // namespace

TEST_CASE("Resizable BAR : actif si le tas visible approche la VRAM, inactif à 256 Mo", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  GpuFacts gpu = kRtx;
  CHECK(rebar_text(gpu).empty());  // le pilote n'expose aucun tas visible
  gpu.host_visible_vram_bytes = 12ULL << 30U;
  CHECK(rebar_text(gpu) == "actif, 12 Go");
  CHECK(gpu_line(gpu) == "NVIDIA GeForce RTX 4070 Ti · 12 Go · Resizable BAR actif, 12 Go · cœurs Tensor");
  gpu.host_visible_vram_bytes = 256ULL << 20U;
  CHECK(rebar_text(gpu) == "inactif, 256 Mo");
  CHECK(shown_size(1536ULL << 20U) == "1,5 Go");
  gpu.vendor = GpuVendor::Intel;
  CHECK(rebar_text(gpu).empty());
}

TEST_CASE("Bienvenue : la ligne du GPU et les tailles suivent la langue de l'interface (A25-12)",
          "[app][first_run]") {
  GpuFacts gpu = kRtx;
  gpu.host_visible_vram_bytes = 12ULL << 30U;
  CHECK(gpu_line(gpu) == "NVIDIA GeForce RTX 4070 Ti · 12 GB · Resizable BAR active, 12 GB · Tensor cores");
  gpu.host_visible_vram_bytes = 256ULL << 20U;
  CHECK(rebar_text(gpu) == "inactive, 256 MB");
  CHECK(shown_size(900ULL << 20U) == "900 MB");
  CHECK(shown_size(1536ULL << 20U) == "1.5 GB");
}

TEST_CASE("Une RTX de 12 Go : TensorRT coché, profil 1", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  CHECK(gpu_line(kRtx) == "NVIDIA GeForce RTX 4070 Ti · 12 Go · cœurs Tensor");
  const FirstRunView view = first_run_view(kRtx, kHere);
  const EngineRow* vulkan = row_of(view, core::EngineId::Vulkan);
  const EngineRow* tensorrt = row_of(view, core::EngineId::TensorRt);
  REQUIRE(vulkan != nullptr);
  REQUIRE(tensorrt != nullptr);
  CHECK(vulkan->chosen);
  CHECK(tensorrt->chosen);
  CHECK(std::ranges::contains(tensorrt->offer.badges, core::Badge::Recommended));
  CHECK(view.engines.size() == 2);  // Vulkan et TensorRT seulement
}

TEST_CASE("Sous Windows, TensorRT coché, sans offre DirectML", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  EnginesHere here = kHere;
  here.os = core::Os::Windows;
  const FirstRunView view = first_run_view(kRtx, here);
  const EngineRow* tensorrt = row_of(view, core::EngineId::TensorRt);
  REQUIRE(tensorrt != nullptr);
  CHECK(tensorrt->chosen);
  CHECK(view.engines.size() == 2);
}

TEST_CASE("Sur Mac, Core ML est livré et coché, sans moteur NVIDIA ni AMD", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  const GpuFacts m2{.name = "Apple M2", .vendor = GpuVendor::Apple, .vram_bytes = 16 * kGiB};
  const FirstRunView view = first_run_view(m2, EnginesHere{.os = core::Os::MacOs, .coreml = true});
  const EngineRow* coreml = row_of(view, core::EngineId::Metal);
  REQUIRE(coreml != nullptr);
  CHECK(coreml->usable);
  CHECK(coreml->chosen);
  CHECK(row_of(view, core::EngineId::TensorRt) == nullptr);
  CHECK(view.engines.size() == 2);  // Vulkan et Core ML

  // Un paquet sans le backend Core ML ne le coche pas.
  const FirstRunView bare = first_run_view(m2, EnginesHere{.os = core::Os::MacOs});
  const EngineRow* missing = row_of(bare, core::EngineId::Metal);
  REQUIRE(missing != nullptr);
  CHECK_FALSE(missing->chosen);
  CHECK(missing->status == "absent de ce paquet");
}

TEST_CASE("Unités matricielles AMD : RDNA3 et RDNA4 seulement", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  CHECK(amd_matrix_cores("AMD Radeon RX 9070 XT"));
  CHECK(amd_matrix_cores("AMD Radeon RX 9070 XT (RADV GFX1201)"));
  CHECK(amd_matrix_cores("AMD Radeon RX 7900 XTX (RADV NAVI31)"));
  CHECK(amd_matrix_cores("AMD Radeon RX 7600S"));
  CHECK_FALSE(amd_matrix_cores("AMD Radeon RX 6800 XT"));
  CHECK_FALSE(amd_matrix_cores("AMD Radeon RX 6500M (RADV NAVI24)"));
}

TEST_CASE("Commencer écrit des moteurs valides pour settings.toml", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  FirstRunView view = first_run_view(kRtx, kHere);
  const core::Settings all = settings_after(view, core::Settings{});
  CHECK(all.first_run_done);
  CHECK(core::validate_settings(all).has_value());
  CHECK(std::ranges::contains(all.chosen_engines, core::EngineId::TensorRt));
  CHECK(all.preferred[static_cast<std::size_t>(core::Usage::Upscale)] == core::EngineId::TensorRt);
  CHECK(chosen_line(all) == "Vulkan, TensorRT");

  // Décocher TensorRT : Vulkan seul, toujours valide.
  for (EngineRow& row : view.engines) {
    if (row.offer.id != core::EngineId::Vulkan) row.chosen = false;
  }
  const core::Settings vulkan = settings_after(view, core::Settings{});
  CHECK(core::validate_settings(vulkan).has_value());
  CHECK(vulkan.chosen_engines == std::vector{core::EngineId::Vulkan});

  // --first-run ensuite : l'écran repart du choix enregistré, pas de la présélection.
  const FirstRunView again = first_run_view(kRtx, kHere, vulkan);
  const EngineRow* tensorrt = row_of(again, core::EngineId::TensorRt);
  REQUIRE(tensorrt != nullptr);
  CHECK_FALSE(tensorrt->chosen);
}

TEST_CASE("Sans moteur d'inférence, la chaîne garde l'image source et le dit", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  PlaySettings s;
  s.upscale = true;
  s.rife_factor = 3;
  s.hdr_native = true;
  core::Settings vulkan;
  vulkan.first_run_done = true;
  core::Settings none = vulkan;
  none.chosen_engines.clear();
  std::vector<std::string> notes;
  const PlaySettings plain = within_engines(s, none, notes);
  CHECK_FALSE(plain.upscale);
  CHECK(plain.rife_factor == 0);
  CHECK(plain.hdr_native);
  CHECK(notes.size() == 1);

  // Vulkan seul (AMD, Intel) : SPAN et RIFE passent par le backend Vulkan.
  notes.clear();
  core::Settings vulkan_only = vulkan;
  vulkan_only.chosen_engines = {core::EngineId::Vulkan};
  const PlaySettings on_vulkan = within_engines(s, vulkan_only, notes);
  CHECK(on_vulkan.upscale);
  CHECK(on_vulkan.rife_factor == 3);
  CHECK(notes.empty());

  notes.clear();
  const core::Settings all = settings_after(first_run_view(kRtx, kHere), core::Settings{});
  const PlaySettings kept = within_engines(s, all, notes);
  CHECK(kept.upscale);
  CHECK(kept.rife_factor == 3);
  CHECK(kept.hdr_native);
  CHECK(notes.empty());
}

TEST_CASE("TensorRT n'est essayé que s'il a été choisi", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
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

TEST_CASE("Core ML n'est essayé que choisi", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  core::Settings settings;
  settings.chosen_engines = {core::EngineId::Vulkan};
  // Core ML est le moteur Metal de l'écran (Mac).
  const std::array<std::filesystem::path, 2> mac{"base/ajp_backend_coreml.so", "base/ajp_backend_vulkan.so"};
  CHECK(backends_within(mac, settings).size() == 1);
  settings.chosen_engines.push_back(core::EngineId::Metal);
  CHECK(backends_within(mac, settings).size() == 2);
}

TEST_CASE("Moteur à télécharger : licence acceptée avant tout choix", "[app][first_run]") {
  ajp::app::play::FrenchUi french;
  EnginesHere here{.os = core::Os::Linux};
  here.downloads.push_back(EngineDownload{
      .id = core::EngineId::TensorRt, .bytes = 900ULL << 20, .license = "https://example.org/sla"});
  FirstRunView view = first_run_view(kRtx, here);
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

TEST_CASE("Assistant fermé pendant le téléchargement : la case et la licence reviennent (A25-10)",
          "[app][first_run]") {
  EnginesHere here{.os = core::Os::Linux};
  here.downloads.push_back(EngineDownload{
      .id = core::EngineId::TensorRt, .bytes = 900ULL << 20, .license = "https://example.org/sla"});
  FirstRunView view = first_run_view(kRtx, here);
  for (EngineRow& row : view.engines) {
    if (row.offer.id != core::EngineId::TensorRt) continue;
    row.chosen = true;
    row.license_accepted = true;
  }
  // Ce que l'écran écrit avant le téléchargement : le choix, l'assistant pas encore fini.
  core::Settings written = settings_after(view, {});
  written.first_run_done = false;
  const auto tensorrt_of = [](const FirstRunView& v) {
    const auto row = std::ranges::find_if(
        v.engines, [](const EngineRow& r) { return r.offer.id == core::EngineId::TensorRt; });
    REQUIRE(row != v.engines.end());
    return *row;
  };
  const EngineRow resumed = tensorrt_of(first_run_view(kRtx, here, written));
  CHECK(resumed.chosen);
  CHECK(resumed.license_accepted);
  CHECK(chosen(settings_after(first_run_view(kRtx, here, written), {}), core::EngineId::TensorRt));
  // Sans ce choix écrit, la licence attend toujours l'accord.
  const EngineRow fresh = tensorrt_of(first_run_view(kRtx, here));
  CHECK_FALSE(fresh.chosen);
  CHECK_FALSE(fresh.license_accepted);
}

TEST_CASE("Résumé : la ligne Moteurs donne la taille, une licence refusée est dite (P3.4)",
          "[app][first_run]") {
  EngineRow vulkan{.offer = {.id = ajp::core::EngineId::Vulkan}, .usable = true, .chosen = true};
  EngineRow trt{.offer = {.id = ajp::core::EngineId::TensorRt},
                .usable = true,
                .chosen = true,
                .download = true,
                .license = "https://exemple/licence",
                .license_accepted = true,
                .bytes = 650'000'000};
  const auto accepted = engines_summary({vulkan, trt});
  REQUIRE(accepted.size() == 1);
  CHECK(accepted[0].find("TensorRT") != std::string::npos);
  trt.license_accepted = false;
  const auto refused = engines_summary({vulkan, trt});
  REQUIRE(refused.size() == 2);
  CHECK(refused[0].find("TensorRT") == std::string::npos);
  // Le moteur recommandé pour la carte laissé décoché : dit, la calibration mesurerait l'autre (A25-02).
  trt.chosen = false;
  CHECK(engines_summary({vulkan, trt}).size() == 1);  // sans badge Recommandé : rien à dire
  trt.offer.badges = {ajp::core::Badge::Recommended};
  const auto unchecked = engines_summary({vulkan, trt});
  REQUIRE(unchecked.size() == 2);
  CHECK(unchecked[0].find("TensorRT") == std::string::npos);
  CHECK(unchecked[1].find("TensorRT") != std::string::npos);
  trt.usable = false;  // pas installable ici : rien à cocher, mais la raison se lit (A25-13)
  CHECK(engines_summary({vulkan, trt}).size() == 1);
  trt.status = "NVIDIA driver 580 or newer required, you have 575.02";
  const auto locked = engines_summary({vulkan, trt});
  REQUIRE(locked.size() == 2);
  CHECK(locked[1].find("TensorRT") != std::string::npos);
  CHECK(locked[1].find("575.02") != std::string::npos);
}

TEST_CASE("Pilote NVIDIA lu avant le téléchargement de TensorRT (P3.6)", "[app][first_run]") {
  GpuFacts gpu{.name = "NVIDIA GeForce RTX 4070 Ti", .vendor = ajp::GpuVendor::Nvidia};
  gpu.driver = std::to_string((575U << 22U) | (64U << 14U));
  const auto old = nvidia_driver(gpu);
  REQUIRE(old);
  CHECK(old->first == 575);
  CHECK(old->second == 64);
  gpu.driver = std::to_string((580U << 22U) | (82U << 14U));
  CHECK(nvidia_driver(gpu)->first >= kTensorRtDriverMajor);
  gpu.vendor = ajp::GpuVendor::Amd;
  CHECK_FALSE(nvidia_driver(gpu));
}
