// Mémoire de lecture : où chaque fichier s'est arrêté, pour reprendre au même endroit à la
// prochaine ouverture. Un fichier texte (resume.tsv, à côté de play.toml), une ligne par vidéo :
// position, taille du fichier et chemin absolu. Une vidéo remplacée (autre taille) repart du
// début ; les plus anciennes au-delà de kResumeEntries sont oubliées. P3.83 : une vidéo déplacée se retrouve
// à son nom et à sa taille ; sous Windows et macOS, le chemin se compare sans égard à la casse.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>

#include "core/types.h"

namespace ajp::app::play {

inline constexpr std::size_t kResumeEntries = 500;
// Pas de reprise si l'on s'est arrêté avant ce temps, ou dans la fin : le dixième de la durée, entre
// kResumeTailMinUs et kResumeTailUs (P3.83 : l'ending et l'aperçu d'un épisode, 2 min 30) ; l'épisode est vu,
// la prochaine ouverture repart du début.
inline constexpr std::int64_t kResumeMinUs = 30'000'000;
inline constexpr std::int64_t kResumeTailMinUs = 30'000'000;
inline constexpr std::int64_t kResumeTailUs = 150'000'000;
// Pendant la lecture, la position est retenue à ce rythme (et à la fermeture).
inline constexpr std::int64_t kResumeEveryUs = 15'000'000;

// Deux chemins désignent-ils la même vidéo ? Chemins absolus sans . ni .., comparés comme les positions
// retenues (casse ignorée sous Windows). Un chemin qui ne se résout pas ne vaut jamais le même.
bool same_video_file(const std::filesystem::path& a, const std::filesystem::path& b);

// La position à reprendre pour ce fichier, ou rien.
std::optional<std::int64_t> resume_position(const std::filesystem::path& store,
                                            const std::filesystem::path& file);

// Retient la position, ou oublie le fichier si elle est trop près du début ou de la fin (la
// lecture allée au bout s'arrête là). duration_us : 0 si inconnue. Refus : écriture impossible, fichier
// présent mais illisible, ou verrou tenu plus de 2 s par un autre lecteur ; le fichier reste alors tel quel.
std::expected<void, Refusal> remember_position(const std::filesystem::path& store,
                                               const std::filesystem::path& file, std::int64_t position_us,
                                               std::int64_t duration_us);

}  // namespace ajp::app::play
