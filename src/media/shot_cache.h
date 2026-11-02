// Le cache des plans analysés (media/shot_probe.h) : une vidéo déjà analysée, rouverte ou revue, ne se
// réanalyse pas. Un fichier par vidéo dans un dossier du lecteur, nommé par une empreinte de la vidéo
// (taille, date de modification, premier mégaoctet) : une vidéo remplacée ou réencodée change d'empreinte.
#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "media/shot_cutter.h"

namespace ajp::media {

// L'empreinte de la vidéo, en hexadécimal ; vide si elle ne se lit pas.
std::string shot_cache_key(const std::filesystem::path& video);

// Les plans gardés d'une vidéo : toute la vidéo (complete), ou ce que l'analyse avait fait avant d'être
// arrêtée (une vidéo quittée, des parties sautées), qu'elle reprendra à la prochaine ouverture.
struct CachedShots {
  std::vector<Shot> shots;
  bool complete = false;
};

// Écrit les plans sous dir (créé au besoin), par un fichier temporaire renommé ; faux si l'écriture
// échoue. Au-delà de keep fichiers, les plus anciens partent.
bool save_shot_cache(const std::filesystem::path& dir, const std::string& key, std::span<const Shot> shots,
                     bool complete, std::size_t keep = 400);
// Les plans gardés pour cette empreinte ; rien si le fichier manque ou ne se lit pas entier.
std::optional<CachedShots> load_shot_cache(const std::filesystem::path& dir, const std::string& key);

}  // namespace ajp::media
