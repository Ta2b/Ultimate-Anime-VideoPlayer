// L'écriture sur disque d'un fichier, et du dossier qui le nomme, autour d'un renommage : sans
// elle, une coupure juste après le rename peut laisser un fichier vide (btrfs, ext4 en
// data=writeback, NTFS). Sous Windows, FlushFileBuffers ; le dossier n'y a pas d'équivalent, NTFS
// journalise le renommage lui-même.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>

#include "core/types.h"

namespace ajp::core {

// Le contenu du fichier sur le disque ; l'erreur de l'ouverture ou de l'écriture sinon.
std::error_code sync_file(const std::filesystem::path& file);
// Les entrées du dossier (un renommage) sur le disque, au mieux : rien sous Windows.
void sync_directory(const std::filesystem::path& folder);

// rename(from, to), réessayé environ 1,5 s sous Windows tant que le refus est passager (antivirus,
// indexeur qui tient l'un des deux fichiers ouvert sans partage de suppression) ; ailleurs, un seul
// essai : rename(2) remplace même un fichier ouvert (A20-01, A05-04). patience : le temps total des essais ;
// un dossier de plusieurs Go que Defender analyse peut le tenir 5 à 10 s (B31-06).
inline constexpr std::chrono::milliseconds kRenamePatience{1500};
std::error_code rename_retrying(const std::filesystem::path& from, const std::filesystem::path& to,
                                std::chrono::milliseconds patience = kRenamePatience);

// La seule écriture atomique du projet (P5.11) : un temporaire au nom unique dans le même dossier
// (« .<nom>.<instant>-<fil>.tmp »), écrit sur le disque, puis renommé sur path, et le dossier écrit à son
// tour. Un lecteur voit l'ancien fichier entier ou le nouveau entier ; une coupure ne laisse pas un
// fichier vide. Le dossier est créé au besoin ; le temporaire est retiré en cas d'échec. Sous Windows, un
// renommage refusé parce qu'un autre processus tient le fichier ouvert se réessaie environ 1,5 s (A20-01).
// Un lien symbolique reste un lien : le fichier qu'il nomme est remplacé. Après l'écriture, les temporaires
// du même nom laissés depuis plus d'une heure par un arrêt brutal sont retirés (A20-08).
std::expected<void, Refusal> write_atomic(const std::filesystem::path& path,
                                          std::span<const std::byte> bytes);
// Moins de needed octets libres sous dir (ou son premier parent qui existe) : le refus InsufficientSpace qui
// dit combien il reste ; rien s'il y a la place ou si la place ne se lit pas (A31-11). Une écriture refusée
// le demande avant de dire « fichier illisible ou inscriptible » : un disque plein se dit tel quel.
std::optional<Refusal> lacking_space(const std::filesystem::path& dir, std::uintmax_t needed);

std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text);

}  // namespace ajp::core
