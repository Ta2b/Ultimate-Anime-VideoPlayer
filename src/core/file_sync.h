// L'écriture sur disque d'un fichier, et du dossier qui le nomme, autour d'un renommage : sans
// elle, une coupure juste après le rename peut laisser un fichier vide (btrfs, ext4 en
// data=writeback, NTFS). Sous Windows, FlushFileBuffers ; le dossier n'y a pas d'équivalent, NTFS
// journalise le renommage lui-même.
#pragma once

#include <filesystem>
#include <system_error>

namespace ajp::core {

// Le contenu du fichier sur le disque ; l'erreur de l'ouverture ou de l'écriture sinon.
std::error_code sync_file(const std::filesystem::path& file);
// Les entrées du dossier (un renommage) sur le disque, au mieux : rien sous Windows.
void sync_directory(const std::filesystem::path& folder);

}  // namespace ajp::core
