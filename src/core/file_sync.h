// L'écriture sur disque d'un fichier, et du dossier qui le nomme, autour d'un renommage : sans
// elle, une coupure juste après le rename peut laisser un fichier vide (btrfs, ext4 en
// data=writeback, NTFS). Sous Windows, FlushFileBuffers ; le dossier n'y a pas d'équivalent, NTFS
// journalise le renommage lui-même.
#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
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
// essai : rename(2) remplace même un fichier ouvert (A20-01, A05-04).
std::error_code rename_retrying(const std::filesystem::path& from, const std::filesystem::path& to);

// La seule écriture atomique du projet (P5.11) : un temporaire au nom unique dans le même dossier
// (« .<nom>.<instant>-<fil>.tmp »), écrit sur le disque, puis renommé sur path, et le dossier écrit à son
// tour. Un lecteur voit l'ancien fichier entier ou le nouveau entier ; une coupure ne laisse pas un
// fichier vide. Le dossier est créé au besoin ; le temporaire est retiré en cas d'échec. Sous Windows, un
// renommage refusé parce qu'un autre processus tient le fichier ouvert se réessaie environ 1,5 s (A20-01).
std::expected<void, Refusal> write_atomic(const std::filesystem::path& path,
                                          std::span<const std::byte> bytes);
std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text);

}  // namespace ajp::core
