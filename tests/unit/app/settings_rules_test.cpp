// Règles du panneau de réglages : appareil audio, copie 720p de RIFE, agrandissement éteint, dossier saisi.
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"
#include "app/play/settings_engines.h"
#include "app/play/settings_sound.h"

namespace ajp::app::play {

// C23-01 : le délai d'un autre appareil ne se pose pas en direct sur l'appareil en cours.
TEST_CASE("Appareil audio : le délai du nouvel appareil attend la réouverture (C23-01)", "[app][settings]") {
  PlaySettings running;
  running.audio_device = "A";
  PlaySettings draft = running;
  draft.audio_device_delays_ms["B"] = 200;
  draft.audio_device = "B";
  switch_device_delay(draft, "A", {});
  REQUIRE(draft.audio_delay_ms == 200);
  const PlaySettings live = live_part(running, draft, false, nullptr);
  CHECK(live.audio_device == "A");
  CHECK(live.audio_delay_ms == running.audio_delay_ms);
  CHECK(live.audio_device_delays_ms == running.audio_device_delays_ms);
  // Une fois l'appareil appliqué, le délai du brouillon vaut.
  PlaySettings applied_running = draft;
  draft.audio_delay_ms = 250;
  CHECK(live_part(applied_running, draft, false, nullptr).audio_delay_ms == 250);
}

// C23-04 : la note de la copie 720p lit la condition de la chaîne.
TEST_CASE("RIFE : la copie 720p suit la taille de travail jouée (C23-04)", "[app][settings]") {
  PlaySettings s;
  s.upscale = true;
  s.work_height = 0;
  CHECK(rife_motion_reduced(chain_work_size(s, 1920, 1080)));
  s.work_height = 720;
  CHECK_FALSE(rife_motion_reduced(chain_work_size(s, 1920, 1080)));
  s.work_height = 1080;  // une source 4K réduite à 1080p : la copie 720p est faite aussi
  CHECK(rife_motion_reduced(chain_work_size(s, 3840, 2160)));
}

// C22-05 : sans agrandissement, ni le modèle ni les traits ne rechargent rien ; la hauteur de travail le dit
// sans parler de SPAN.
TEST_CASE("Réglages : sans agrandissement, « SPAN reloads » ne s'annonce pas (C22-05)", "[app][settings]") {
  PlaySettings running;
  running.upscale = false;
  PlaySettings wanted = running;
  wanted.restore_lines = !running.restore_lines;
  CHECK(change_between(live_part(running, wanted, false, nullptr), wanted, false).cost != ChangeCost::Reopen);
  CHECK(live_part(running, wanted, false, nullptr).restore_lines == wanted.restore_lines);
  PlaySettings height = running;
  height.work_height = running.work_height == 540 ? 720 : 540;
  const Change reopen = change_between(running, height, false);
  CHECK(reopen.cost == ChangeCost::Reopen);
  CHECK(reopen.reason.find("SPAN") == std::string::npos);
}

// C23-08 : « Copier comme chemin » de l'Explorateur donne un chemin entre guillemets.
TEST_CASE("Dossier d'installation : guillemets et blancs de bord retirés (C23-08)", "[app][settings]") {
  CHECK(clean_folder_text("\"D:\\Moteurs\"") == "D:\\Moteurs");
  CHECK(clean_folder_text("  D:\\Moteurs \t") == "D:\\Moteurs");
  CHECK(clean_folder_text(" \"D:\\Mes moteurs\" ") == "D:\\Mes moteurs");
  CHECK(clean_folder_text("/home/a b/moteurs") == "/home/a b/moteurs");
  CHECK(clean_folder_text("\"").empty() == false);  // un guillemet seul n'est pas une paire
  CHECK(clean_folder_text("").empty());
}

}  // namespace ajp::app::play
