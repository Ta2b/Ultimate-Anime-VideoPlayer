// Une construction refusée sur un disque plein se dit « disque plein » (A31-11).
#include <filesystem>
#include <limits>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

#include "app/play/engine_build.h"
#include "app/play/visible_error.h"

namespace ajp::app::play {
namespace {

TEST_CASE("Construction refusée à l'écriture, disque plein : InsufficientSpace", "[app][engines]") {
  std::error_code error;
  const std::filesystem::path cache = std::filesystem::temp_directory_path(error) / "ajp-engines";
  const std::uintmax_t huge = std::numeric_limits<std::uintmax_t>::max() / 2;
  const std::expected<gpu::Artifact, Refusal> refused =
      std::unexpected(Refusal{RefusalCode::BuilderFailed, "trtexec : écriture du plan refusée"});
  const auto judged = with_space_reason(refused, cache, huge);
  REQUIRE_FALSE(judged);
  CHECK(judged.error().code == RefusalCode::InsufficientSpace);
  CHECK(judged.error().detail.find("trtexec") != std::string::npos);  // la cause d'origine reste au journal
  CHECK(visible_error(judged.error()) == "not enough disk space");
  // De la place : le refus tel quel. Un autre refus (le modèle, l'appareil) n'est pas une affaire de place.
  CHECK(with_space_reason(refused, cache, 0).error().code == RefusalCode::BuilderFailed);
  const std::expected<gpu::Artifact, Refusal> other =
      std::unexpected(Refusal{RefusalCode::DeviceMismatch, "pas de GPU NVIDIA"});
  CHECK(with_space_reason(other, cache, huge).error().code == RefusalCode::DeviceMismatch);
}

}  // namespace
}  // namespace ajp::app::play
