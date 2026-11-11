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

// Faut-il écrire play.toml maintenant ? Relève aussi l'échec d'écriture que le fil d'écriture a rapporté : un
// avis, une fois par écrivain ; si c'est play.toml, il est remis à écrire (`dirty`), après une pause plus
// longue (`saved_at` avance d'autant).
inline bool settings_save_due(LatestWriter& writer, PlayerUi& ui, bool& dirty, std::int64_t& saved_at,
                              std::int64_t now_us) {
  if (const auto failed = writer.take_failure()) {
    if (!std::exchange(writer.failure_said, true))
      ui.controls().warn(i18n::ui_tr_format("Settings cannot be saved: {}", visible_error(failed->refusal)),
                         now_us);
    if (failed->key == "play.toml") {
      dirty = true;
      saved_at = now_us + kSettingsRetryUs - kSettingsSaveEveryUs;
    }
  }
  return dirty && now_us - saved_at >= kSettingsSaveEveryUs;
}

// Vide l'écrivain (attente bornée) puis, si play.toml a échoué, un dernier essai sur place : après la
// fermeture de la chaîne, plus personne ne reprend ce réglage.
inline void finish_settings_writes(LatestWriter& writer, const std::filesystem::path& path,
                                   const PlaySettings& saved) {
  writer.flush();
  const auto failed = writer.take_failure();
  if (!failed || failed->key != "play.toml") return;
  if (auto again = save_settings(path, saved); !again) log_channel("app")->warn("{}", again.error().detail);
}

}  // namespace ajp::app::play
