// Lancement d'un processus enfant (le constructeur trtexec) : sortie lue ligne
// par ligne, annulation, délai. Un plantage de l'enfant devient un résultat,
// jamais un plantage du lecteur.
//
// posix_spawn sous Linux et macOS (process_posix.cpp), CreateProcessW sous
// Windows (process_win.cpp). Aucune recherche dans PATH et aucun shell :
// l'environnement hérité passe tel quel, sans être lu.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "prepare/utf8.h"

namespace ajp::prepare {

struct ProcessSpec {
  std::filesystem::path program;        // chemin complet de l'exécutable
  std::vector<std::string> args;        // en UTF-8, sans argv[0]
  std::chrono::milliseconds timeout{};  // strictement positif
  std::filesystem::path working_dir;    // vide : celui du lecteur
  // « NOM=valeur » ajoutés à l'environnement hérité (une clé passée à un script sans paraître
  // dans sa ligne de commande) ; un nom déjà présent est remplacé.
  std::vector<std::string> extra_env;
};

enum class ExitKind : std::uint8_t {
  Exited,     // fin normale, code de sortie dans code
  Crashed,    // tué par un signal (POSIX) ou une exception NTSTATUS (Windows), dans code
  TimedOut,   // tué par nous après le délai
  Cancelled,  // tué par nous sur demande d'arrêt
};

struct ProcessExit {
  ExitKind kind = ExitKind::Exited;
  std::int64_t code = 0;
  std::chrono::milliseconds elapsed{};
};

using LineSink = std::function<void(std::string_view line)>;

// Lance program, transmet chaque ligne non vide de stdout et stderr mêlés, et
// rend la fin de l'enfant. stdin vient du périphérique nul. Refus BuilderFailed
// si l'exécutable ne se lance pas, FieldOutOfRange si le délai n'est pas positif.
std::expected<ProcessExit, Refusal> run_process(const ProcessSpec& spec, const LineSink& on_line,
                                                const std::stop_token& stop);

// « code 3 », « signal 6 », « délai dépassé »… pour les journaux et les refus.
std::string describe(const ProcessExit& exit);

// Découpe un flux d'octets en lignes, sur \n comme sur \r (barres de progression).
// Une ligne trop longue est coupée en morceaux de max_line octets.
class LineSplitter {
 public:
  explicit LineSplitter(LineSink sink, std::size_t max_line = std::size_t{64} * 1024);
  void feed(std::string_view bytes);
  void finish();  // rend la dernière ligne sans fin de ligne

 private:
  void emit();

  LineSink sink_;
  std::size_t max_line_;
  std::string pending_;
};

// Ligne de commande Windows qui redonne exactement argv par CommandLineToArgvW.
std::string windows_command_line(const std::vector<std::string>& argv);

}  // namespace ajp::prepare
