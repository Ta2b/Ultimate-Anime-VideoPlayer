// Un moteur dont l'installation a échoué sur une panne passagère reste choisi et se réessaie (C06-02).
#include "app/play/engine_retry.h"

#include <algorithm>
#include <chrono>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

using namespace ajp;
using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

core::Settings with_tensorrt() {
  core::Settings engines;
  engines.chosen_engines = {core::EngineId::Vulkan, core::EngineId::TensorRt};
  engines.installed_engines = {core::EngineId::Vulkan, core::EngineId::TensorRt};
  engines.preferred = {core::EngineId::TensorRt, core::EngineId::TensorRt};
  return engines;
}

}  // namespace

TEST_CASE("Installation : une panne passagère garde le choix, un refus définitif non", "[app][install]") {
  for (const RefusalCode transient :
       {RefusalCode::IoFailed, RefusalCode::DecodeFailed, RefusalCode::CacheIo, RefusalCode::BuilderFailed})
    CHECK(install_failure_is_transient(transient));
  for (const RefusalCode definitive :
       {RefusalCode::HashMismatch, RefusalCode::InsufficientSpace, RefusalCode::FileUnwritable,
        RefusalCode::SdkRejected, RefusalCode::Cancelled})
    CHECK_FALSE(install_failure_is_transient(definitive));
}

TEST_CASE("Installation : un moteur reporté reste choisi, non installé, et les préférés retombent sur Vulkan",
          "[app][install]") {
  const core::Settings after = postponed_engine(with_tensorrt(), core::EngineId::TensorRt);
  CHECK(std::ranges::contains(after.chosen_engines, core::EngineId::TensorRt));
  CHECK_FALSE(std::ranges::contains(after.installed_engines, core::EngineId::TensorRt));
  CHECK(is_postponed(after, core::EngineId::TensorRt));
  CHECK(after.preferred[static_cast<std::size_t>(core::Usage::Upscale)] == core::EngineId::Vulkan);
  CHECK(after.preferred[static_cast<std::size_t>(core::Usage::Interpolate)] == core::EngineId::Vulkan);
  // Le réglage reste écrivable : un préféré non installé serait refusé par validate_settings.
  CHECK(core::validate_settings(after));
  CHECK_FALSE(is_postponed(with_tensorrt(), core::EngineId::TensorRt));
}

TEST_CASE("Installation : le moteur réinstallé redevient installé et préféré", "[app][install]") {
  const core::Settings postponed = postponed_engine(with_tensorrt(), core::EngineId::TensorRt);
  const core::Settings after = installed_engine(postponed, core::EngineId::TensorRt);
  CHECK(std::ranges::contains(after.installed_engines, core::EngineId::TensorRt));
  CHECK(after.preferred[static_cast<std::size_t>(core::Usage::Upscale)] == core::EngineId::TensorRt);
  CHECK(core::validate_settings(after));
  CHECK(std::ranges::count(after.installed_engines, core::EngineId::TensorRt) == 1);
}

TEST_CASE("Installation : un essai raté ne se refait pas avant six heures", "[app][install]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-retry-marker";
  fs::remove_all(dir);
  const fs::path marker = retry_marker(dir, core::EngineId::TensorRt);
  CHECK(marker.filename() == "tensorrt.retry");
  CHECK(retry_install_due(marker));  // jamais essayé
  note_install_attempt(marker);
  CHECK_FALSE(retry_install_due(marker));
  fs::last_write_time(marker, fs::file_time_type::clock::now() - std::chrono::hours(7));
  CHECK(retry_install_due(marker));
  fs::remove_all(dir);
}
