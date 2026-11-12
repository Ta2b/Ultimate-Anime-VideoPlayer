// Petits pas de play() (player.cpp) : la machine vue de l'écran, la relance après perte du GPU, la reprise
// lue hors de la fenêtre, la chaîne refaite pour un nouvel écran.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

#include <SDL3/SDL_video.h>

#include "app/play/auto_bench.h"
#include "app/play/auto_config.h"
#include "app/play/backend_library.h"
#include "app/play/calibrated_chain.h"
#include "app/play/chain.h"
#include "app/play/gpu_tier.h"
#include "app/play/options.h"
#include "app/play/settings.h"
#include "core/engines.h"
#include "core/settings.h"

namespace ajp::app::play {

struct EngineCatalog;
struct PlayerPaths;

// La machine vue de l'écran où est la fenêtre : à l'ouverture, et quand elle change d'écran (B29-01).
// explored : l'exploration du banc quand elle vaut pour cette carte ; calibrated : la calibration de l'écran.
// host : la carte et son moteur, d'où Machine::span_fp8 (span_fp8_default).
Machine machine_of(SDL_Window* window, Tier tier, const BenchResult* explored, const CalibrationHost& host,
                   std::optional<CalibratedChain> calibrated);
// A31-02 : la relance après une perte du GPU, sauf une seconde dans la minute.
bool relaunch_allowed(bool lost_recently, std::int64_t launched_s);
// B29-08 : weakly_canonical et file_size sur un partage lent, hors du thread de la fenêtre.
std::optional<std::int64_t> resume_from(const std::filesystem::path& store,
                                        const std::filesystem::path& file);
// B29-01 : la chaîne du préréglage pour le nouvel écran ; Personnalisé reste tel quel (apply_preset). Un
// raccourci de séance (Ctrl+I) garde son interpolation.
// La machine à laquelle une calibration se compare (A25-02, A07-05), écran compris (hauteur et fréquence,
// C37-08).
CalibrationHost calibration_host(const GpuFacts& here, SDL_Window* window, const BackendLibrary* library,
                                 const std::optional<core::EngineId>& opened_for,
                                 const core::Settings& engines, const EngineCatalog& catalog,
                                 const PlayerPaths& paths);
PlaySettings rederived(const PlaySettings& settings, const Machine& machine, const core::SourceInfo& source,
                       const PlayOptions& file_options, const KeptChoices& kept);

}  // namespace ajp::app::play
