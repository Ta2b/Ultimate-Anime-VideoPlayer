// Modèle de l'écran « Moteurs » : présélection, badges, mode avancé, tailles, place
// libre, validation en un clic, réouverture depuis les réglages.
#include "app/engine_chooser.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "app/engine_text.h"

using namespace ajp;
using namespace ajp::app;
using core::EngineId;

namespace {

constexpr std::uint64_t kMib = 1ULL << 20;
constexpr std::uint64_t kGib = 1ULL << 30;

core::SystemInfo system_with(core::GpuInfo gpu, core::Os os = core::Os::Linux) {
  return core::SystemInfo{.os = os, .gpus = {std::move(gpu)}};
}

core::SystemInfo rtx() {
  return system_with(core::GpuInfo{.vendor = core::GpuVendor::Nvidia,
                                   .name = "NVIDIA GeForce RTX 4070 Ti",
                                   .vram_bytes = 12 * kGib,
                                   .tensor_cores = true});
}

core::EnginePackage package(EngineId engine, core::Os os, std::uint64_t size) {
  return core::EnginePackage{.engine = engine,
                             .platform = os,
                             .url = "https://engines.example/p",
                             .sha256 = std::string(64, 'c'),
                             .size = size,
                             .version = "1.0.0"};
}

ChooserInput input_for(core::SystemInfo system, std::uint64_t free = 100 * kGib) {
  return ChooserInput{.os = system.os,
                      .system = std::move(system),
                      .packages = {package(EngineId::TensorRt, core::Os::Linux, kGib),
                                   package(EngineId::TensorRt, core::Os::Windows, 2 * kGib),
                                   package(EngineId::MiGraphX, core::Os::Linux, 2 * kGib),
                                   package(EngineId::OpenVino, core::Os::Linux, 500 * kMib)},
                      .missing_models = {core::ModelEntry{.id = "m", .size = 100 * kMib, .profiles = {1}}},
                      .free_bytes = free};
}

std::vector<EngineId> checked(const EngineChooser& chooser) {
  std::vector<EngineId> ids;
  for (const EngineRow& row : chooser.rows()) {
    if (row.checked) ids.push_back(row.offer.id);
  }
  return ids;
}

}  // namespace

TEST_CASE("Moteurs, premier lancement sur une RTX : la présélection se valide d'un clic", "[app][chooser]") {
  const EngineChooser chooser(input_for(rtx()));
  CHECK(chooser.first_run());
  REQUIRE(chooser.rows().size() == 5);
  CHECK(chooser.rows()[1].offer.id == EngineId::TensorRt);
  CHECK(chooser.rows()[1].offer.badges == std::vector{core::Badge::Recommended, core::Badge::NvidiaOnly});
  CHECK(checked(chooser) == std::vector{EngineId::Vulkan, EngineId::TensorRt, EngineId::MaxineVfx});
  CHECK(chooser.download_bytes() == kGib + 100 * kMib);
  CHECK(chooser.fits());

  const auto plan = chooser.confirm();
  REQUIRE(plan);
  CHECK(plan->settings.first_run_done);
  CHECK(plan->settings.chosen_engines ==
        std::vector{EngineId::Vulkan, EngineId::TensorRt, EngineId::MaxineVfx});
  CHECK(plan->settings.installed_engines == std::vector{EngineId::Vulkan});  // l'installation les ajoutera
  REQUIRE(plan->downloads.size() == 1);
  CHECK(plan->downloads.front().engine == EngineId::TensorRt);
  CHECK(plan->downloads.front().platform == core::Os::Linux);
  CHECK(plan->imports == std::vector{EngineId::MaxineVfx});
  CHECK(plan->models.size() == 1);
  CHECK(plan->removals.empty());
  CHECK(plan->download_bytes == kGib + 100 * kMib);
}

TEST_CASE("Moteurs : lignes verrouillées, mode avancé et raisons en français", "[app][chooser]") {
  EngineChooser chooser(input_for(rtx()));
  CHECK(chooser.lock_of(EngineId::Vulkan) == RowLock::Bundled);
  const auto vulkan = chooser.set_checked(EngineId::Vulkan, false);
  REQUIRE_FALSE(vulkan);
  CHECK(vulkan.error().code == RefusalCode::FieldOutOfRange);
  CHECK(vulkan.error().detail == "Vulkan est livré avec le lecteur et reste inclus");

  CHECK(chooser.lock_of(EngineId::MiGraphX) == RowLock::Advanced);
  const auto migraphx = chooser.toggle(EngineId::MiGraphX);
  REQUIRE_FALSE(migraphx);
  CHECK(migraphx.error().detail.find("Mode avancé") != std::string::npos);

  chooser.set_advanced(true);
  CHECK(chooser.lock_of(EngineId::MiGraphX) == RowLock::Free);
  REQUIRE(chooser.toggle(EngineId::MiGraphX));
  CHECK(chooser.download_bytes() == 3 * kGib + 100 * kMib);
  chooser.set_advanced(false);  // un moteur grisé non installé se décoche en quittant le mode avancé
  CHECK_FALSE(chooser.row(EngineId::MiGraphX)->checked);

  CHECK(chooser.set_checked(EngineId::DirectMl, true).error().detail == "DirectML n'existe pas sur linux");
  REQUIRE(chooser.toggle(EngineId::TensorRt));  // une ligne libre se décoche
  CHECK(checked(chooser) == std::vector{EngineId::Vulkan, EngineId::MaxineVfx});
}

TEST_CASE("Moteurs : sans paquet dans l'index, rien ne se coche ni ne se télécharge", "[app][chooser]") {
  ChooserInput input = input_for(rtx());
  input.packages.clear();
  EngineChooser chooser(std::move(input));
  CHECK(chooser.lock_of(EngineId::TensorRt) == RowLock::NoPackage);
  CHECK_FALSE(chooser.row(EngineId::TensorRt)->checked);
  CHECK(chooser.set_checked(EngineId::TensorRt, true).error().detail.find("aucun paquet") !=
        std::string::npos);
  CHECK(chooser.download_bytes() == 100 * kMib);  // les modèles seulement
}

TEST_CASE("Moteurs : un pilote trop ancien bloque TensorRT, qui reste installable en mode avancé",
          "[app][chooser]") {
  ChooserInput input = input_for(rtx());
  input.blockers = {EngineBlocker{.engine = EngineId::TensorRt, .reason = "pilote CUDA 12.8 trop ancien"}};
  EngineChooser chooser(std::move(input));
  CHECK(chooser.row(EngineId::TensorRt)->blocker == "pilote CUDA 12.8 trop ancien");
  CHECK(chooser.lock_of(EngineId::TensorRt) == RowLock::Blocked);
  CHECK(checked(chooser) == std::vector{EngineId::Vulkan, EngineId::MaxineVfx});  // pas présélectionné
  CHECK(chooser.download_bytes() == 100 * kMib);

  const auto refused = chooser.set_checked(EngineId::TensorRt, true);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(refused.error().detail ==
        "TensorRT : pilote CUDA 12.8 trop ancien ; cochez « Mode avancé » pour l'installer quand même");

  chooser.set_advanced(true);
  CHECK(chooser.lock_of(EngineId::TensorRt) == RowLock::Free);
  REQUIRE(chooser.set_checked(EngineId::TensorRt, true));
  CHECK(chooser.confirm()->downloads.size() == 1);
  chooser.set_advanced(false);
  CHECK_FALSE(chooser.row(EngineId::TensorRt)->checked);
  chooser.reset_to_recommended();
  CHECK_FALSE(chooser.row(EngineId::TensorRt)->checked);

  // Installé quand même en mode avancé : noté, mais Vulkan reste préféré.
  const std::vector<EngineBlocker> blockers{{.engine = EngineId::TensorRt, .reason = "pilote"}};
  const core::Settings settings = record_installed(core::Settings{}, rtx(), EngineId::TensorRt, blockers);
  CHECK(settings.installed_engines == std::vector{EngineId::Vulkan, EngineId::TensorRt});
  CHECK(settings.preferred[0] == EngineId::Vulkan);
}

TEST_CASE("Moteurs : la place libre se vérifie avant de valider", "[app][chooser]") {
  EngineChooser chooser(input_for(rtx(), 500 * kMib));
  CHECK_FALSE(chooser.fits());
  const auto refused = chooser.confirm();
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::DiskFull);
  CHECK(refused.error().detail == "1,1 Gio à télécharger, 500,0 Mio libres dans le dossier des données");

  REQUIRE(chooser.set_checked(EngineId::TensorRt, false));
  CHECK(chooser.fits());
  CHECK(chooser.confirm());
}

TEST_CASE("Moteurs : réouverture depuis les réglages, retrait d'un moteur et retour à la présélection",
          "[app][chooser]") {
  ChooserInput input = input_for(rtx());
  input.settings.first_run_done = true;
  input.settings.chosen_engines = {EngineId::Vulkan, EngineId::TensorRt};
  input.settings.installed_engines = {EngineId::Vulkan, EngineId::TensorRt};
  input.settings.preferred = {EngineId::TensorRt, EngineId::TensorRt, std::nullopt};
  input.missing_models.clear();
  EngineChooser chooser(std::move(input));

  CHECK_FALSE(chooser.first_run());
  CHECK(checked(chooser) == std::vector{EngineId::Vulkan, EngineId::TensorRt});  // le choix enregistré
  CHECK(chooser.row(EngineId::TensorRt)->installed);
  CHECK(chooser.download_bytes() == 0);

  REQUIRE(chooser.set_checked(EngineId::TensorRt, false));
  const auto plan = chooser.confirm();
  REQUIRE(plan);
  CHECK(plan->removals == std::vector{EngineId::TensorRt});
  CHECK(plan->settings.installed_engines == std::vector{EngineId::Vulkan});
  CHECK(plan->settings.preferred[0] == EngineId::Vulkan);  // le moteur préféré retiré revient à Vulkan
  CHECK(plan->settings.preferred[1] == EngineId::Vulkan);

  chooser.reset_to_recommended();
  CHECK(checked(chooser) == std::vector{EngineId::Vulkan, EngineId::TensorRt, EngineId::MaxineVfx});
  CHECK(chooser.confirm()->removals.empty());
}

TEST_CASE("Moteurs : présélection par constructeur, de la détection au plan", "[app][chooser]") {
  const EngineChooser amd(
      input_for(system_with(core::GpuInfo{.vendor = core::GpuVendor::Amd, .rocm_supported = true})));
  CHECK(checked(amd) == std::vector{EngineId::Vulkan, EngineId::MiGraphX});
  CHECK(amd.row(EngineId::Vulkan)->offer.badges == std::vector{core::Badge::Compatibility});

  const EngineChooser intel(
      input_for(system_with(core::GpuInfo{.vendor = core::GpuVendor::Intel, .openvino_gpu = true})));
  CHECK(checked(intel) == std::vector{EngineId::Vulkan, EngineId::OpenVino});

  // Ancienne Radeon sous Windows : DirectML proposé, jamais coché d'office.
  const EngineChooser old_amd(
      input_for(system_with(core::GpuInfo{.vendor = core::GpuVendor::Amd}, core::Os::Windows)));
  CHECK(old_amd.row(EngineId::DirectMl)->offer.relevant);
  CHECK(checked(old_amd) == std::vector{EngineId::Vulkan});
  CHECK(old_amd.row(EngineId::Vulkan)->offer.badges == std::vector{core::Badge::Recommended});
}

TEST_CASE("Moteurs : un moteur installé est noté et préféré s'il est le plus rapide", "[app][chooser]") {
  core::Settings settings = record_installed(core::Settings{}, rtx(), EngineId::TensorRt);
  CHECK(settings.installed_engines == std::vector{EngineId::Vulkan, EngineId::TensorRt});
  CHECK(settings.preferred[0] == EngineId::TensorRt);
  CHECK(settings.preferred[1] == EngineId::TensorRt);
  settings = record_installed(std::move(settings), rtx(), EngineId::MaxineVfx);
  CHECK(settings.preferred[2] == EngineId::MaxineVfx);
  CHECK(core::validate_settings(settings));

  // Sur une GTX sans unités matricielles, TensorRT installé en mode avancé n'est pas préféré.
  const core::Settings gtx = record_installed(
      core::Settings{}, system_with(core::GpuInfo{.vendor = core::GpuVendor::Nvidia}), EngineId::TensorRt);
  CHECK(gtx.preferred[0] == EngineId::Vulkan);
  CHECK(core::validate_settings(gtx));
}

TEST_CASE("Textes de l'écran Moteurs", "[app][chooser]") {
  CHECK(size_text(512) == "512 octets");
  CHECK(size_text(1536) == "1,5 Kio");
  CHECK(size_text(1288490189) == "1,2 Gio");
  CHECK(size_text((1ULL << 20) - 1) == "1,0 Mio");  // pas « 1024,0 Kio »
  CHECK(size_text(1023 * 1024) == "1023,0 Kio");
  CHECK(badge_label(core::Badge::AmdOnly) == "Seulement AMD");
  CHECK(badge_label(core::Badge::Recommended) == "Recommandé");
  CHECK(engine_title(EngineId::MaxineVfx) == "Maxine VFX (TrueHDR)");
  CHECK(reason_text(core::Reason::NoMatchingGpu) == "aucun GPU compatible détecté");
}
