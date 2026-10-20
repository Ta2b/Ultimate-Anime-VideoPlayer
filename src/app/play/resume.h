// Mémoire de lecture : où chaque fichier s'est arrêté, pour reprendre au même endroit à la
// prochaine ouverture. Un fichier texte (resume.tsv, à côté de play.toml), une ligne par vidéo :
// position, taille du fichier et chemin absolu. Une vidéo remplacée (autre taille) repart du
// début ; les plus anciennes au-delà de kResumeEntries sont oubliées.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>

#include "core/types.h"

namespace ajp::app::play {

inline constexpr std::size_t kResumeEntries = 500;
// Pas de reprise si l'on s'est arrêté avant ce temps, ou à moins de kResumeTailUs (ou des
// derniers 5 %) de la fin : l'épisode est vu, la prochaine ouverture repart du début.
inline constexpr std::int64_t kResumeMinUs = 30'000'000;
inline constexpr std::int64_t kResumeTailUs = 30'000'000;
// Pendant la lecture, la position est retenue à ce rythme (et à la fermeture).
inline constexpr std::int64_t kResumeEveryUs = 15'000'000;

// La position à reprendre pour ce fichier, ou rien.
std::optional<std::int64_t> resume_position(const std::filesystem::path& store,
                                            const std::filesystem::path& file);

// Retient la position, ou oublie le fichier si elle est trop près du début ou de la fin (la
// lecture allée au bout s'arrête là). duration_us : 0 si inconnue.
std::expected<void, Refusal> remember_position(const std::filesystem::path& store,
                                               const std::filesystem::path& file, std::int64_t position_us,
                                               std::int64_t duration_us);

}  // namespace ajp::app::play
