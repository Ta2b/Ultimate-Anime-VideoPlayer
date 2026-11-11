// Une écriture de réglages refusée est dite et reprise, pas oubliée (B31-08, B36-01).
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

#include "app/play/i18n.h"
#include "app/play/latest_writer.h"
#include "app/play/settings.h"
#include "app/play/ui.h"
#include "app/play/visible_error.h"
#include "core/log.h"

namespace ajp::app::play {

// Relève l'échec d'écriture que le fil d'écriture a rapporté : un avis, une fois par séance (`said`), et vrai
// quand c'est play.toml qui a échoué (l'appelant le remet à écrire, plus tard).
inline bool settings_write_failed(LatestWriter& writer, PlayerUi& ui, bool& said, std::int64_t now_us) {
  const auto failed = writer.take_failure();
  if (!failed) return false;
  if (!std::exchange(said, true))
    ui.controls().warn(i18n::ui_tr_format("Settings cannot be saved: {}", visible_error(failed->refusal)),
                       now_us);
  return failed->key == "play.toml";
}

// Un dernier essai, sur place, à la fin de la chaîne : après sa fermeture, plus personne ne reprend ce
// réglage.
inline void retry_settings_save(LatestWriter& writer, const std::filesystem::path& path,
                                const PlaySettings& saved) {
  const auto failed = writer.take_failure();
  if (!failed || failed->key != "play.toml") return;
  if (auto again = save_settings(path, saved); !again) log_channel("app")->warn("{}", again.error().detail);
}

}  // namespace ajp::app::play
