// Rapport de plantage (P2.2). Sous Windows, une exception non gérée ou un abort (std::terminate, assert)
// écrit un minidump ajp-<date>-<pid>.dmp, à joindre à un signalement ; ailleurs, rien (le core du système).
#pragma once

#include <filesystem>

namespace ajp::app::play {

// Les cinq derniers vidages gardés dans dir ; à appeler une fois, tôt, depuis main.
inline constexpr int kKeptCrashDumps = 5;
void install_crash_dumps(const std::filesystem::path& dir);

// Un minidump du processus en cours dans file, sans exception : faux s'il n'est pas écrit, ou hors de
// Windows.
bool write_crash_dump(const std::filesystem::path& file);

}  // namespace ajp::app::play
