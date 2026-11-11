// Rapport de plantage (P2.2). Sous Windows, une exception non gérée ou un abort (std::terminate, assert)
// écrit un minidump ajp-<date>-<pid>.dmp, à joindre à un signalement ; ailleurs, rien (le core du système).
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace ajp::app::play {

// Les cinq derniers vidages gardés dans dir ; à appeler une fois, tôt, depuis main.
inline constexpr int kKeptCrashDumps = 5;
void install_crash_dumps(const std::filesystem::path& dir);

// L'étape en cours (« construction du moteur span », « lecture de ep.mkv »), tenue dans un tampon fixe : un
// plantage dur l'écrit au bout d'ajp.log, alors que les dernières lignes du journal, encore dans sa file
// asynchrone, sont perdues (B39-10). Tronquée à 200 octets. Utilisable de tout thread.
void note_crash_stage(std::string_view stage);
// Ce que note_crash_stage a retenu (les tests).
std::string crash_stage();

// Un minidump du processus en cours dans file, sans exception : faux s'il n'est pas écrit, ou hors de
// Windows.
bool write_crash_dump(const std::filesystem::path& file);

}  // namespace ajp::app::play
