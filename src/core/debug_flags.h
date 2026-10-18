// DebugFlags : la seule structure d'options de débogage du lecteur.
//
// Elle se remplit par la ligne de commande et par rien d'autre : aucune
// variable d'environnement n'est lue. Ce qui était une bascule cachée dans le
// fork (AJI_NO_GRAPH, AJI_PIPELINE_TRACE…) devient ici une option déclarée,
// visible dans --help et dans le journal.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/engines.h"
#include "core/plan.h"
#include "core/resolve.h"
#include "core/types.h"

namespace ajp::core {

enum class LogLevel : std::uint8_t { Trace, Debug, Info, Warn, Error, Off };

struct DebugFlags {
  LogLevel log_level = LogLevel::Info;    // --log-level=debug
  bool vulkan_validation = false;         // --vulkan-validation : couches de validation Vulkan
  bool no_cuda_graphs = false;            // --no-cuda-graphs : chaque paire lancée sans graphe
  bool trace_pairs = false;               // --trace-pairs : temps et attentes de chaque paire au journal
  bool no_exact_reuse = false;            // --no-exact-reuse : recalcule chaque paire, sans cache
  bool dump_plan = false;                 // --dump-plan : écrit chaque Plan résolu au journal
  std::optional<EngineId> force_engine;   // --engine=tensorrt : ignore le moteur préféré
  std::optional<RingFormat> ring_format;  // --ring-format=yuv444p16 : qualifie le repli du ring
  std::optional<DisplayInfo> display;     // --display=2560x1440@165[:hdr] : remplace l'écran détecté
  std::int64_t max_frames = 0;            // --max-frames=N : s'arrête après N images ; 0 sans limite
};

struct CommandLine {
  DebugFlags debug;
  std::vector<std::string> inputs;  // fichiers à ouvrir, dans l'ordre ; tout ce qui suit « -- » en est
  bool help = false;                // --help
};

namespace limits {
inline constexpr Range<std::int64_t> kMaxFrames{0, 100'000'000};
inline constexpr Range<std::int64_t> kDisplaySide{1, 16384};
inline constexpr Range<double> kRefreshHz{1.0, 1000.0};
}  // namespace limits

// Arguments sans argv[0]. Une option inconnue, ou une valeur hors plage, est un refus nommé.
std::expected<CommandLine, Refusal> parse_command_line(std::span<const std::string_view> args);
// Commodité pour main() : argv tel que le système le passe, argv[0] compris.
std::expected<CommandLine, Refusal> parse_command_line(int argc, const char* const* argv);

// Texte de --help, une ligne par option, tiré de la même table que l'analyse.
std::string command_line_help();

}  // namespace ajp::core
