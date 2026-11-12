// Le cache des plans analysés (media/shot_probe.h) : une vidéo déjà analysée, rouverte ou revue, ne se
// réanalyse pas. Un fichier par vidéo dans un dossier du lecteur, nommé par une empreinte de la vidéo
// (taille, date de modification, premier mégaoctet) : une vidéo remplacée ou réencodée change d'empreinte.
#pragma once

#include <cstddef>
#include <cstdint>
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
  std::vector<FlashSpan> flashes;  // les séquences de flashs dangereux de la part analysée
  bool complete = false;
};

// La version de l'analyse (A42-10) : les seuils de la découpe, la grille et le limiteur de flashs. Elle monte
// à chaque changement qui donnerait d'autres plans ; un fichier d'une autre version se refait, comme celui
// d'un autre chemin de décodage (`variant` : 0 logiciel, 1 NVDEC réduit, qui ne mesurent pas la même image ;
// app/play/chain_shots.cpp y ajoute le gain des flashs et la recherche des visages).
// 2 : animation tenue, éclairs de deux ou trois images, flashs par zones. 3 : flashs jugés au blanc du grade
// d'aujourd'hui (flash_gain_of), plus clair : les flashs marqués sous la version 2 en manquaient.
inline constexpr std::uint32_t kShotAnalysis = 3;
inline constexpr std::size_t kShotCacheKeep = 400;  // fichiers gardés

// Écrit les plans sous dir (créé au besoin), par un fichier temporaire renommé ; faux si l'écriture
// échoue. Au-delà de keep fichiers, les plus anciens partent. Chaque variante a son fichier (C42-05).
bool save_shot_cache(const std::filesystem::path& dir, const std::string& key, std::span<const Shot> shots,
                     bool complete, std::size_t keep = kShotCacheKeep,
                     std::span<const FlashSpan> flashes = {}, std::uint32_t variant = 0);
// Les plans gardés pour cette empreinte ; rien si le fichier manque, ne se lit pas entier, ou vient d'une
// autre version de l'analyse ou d'un autre chemin de décodage. Un fichier lu reprend la date du jour : le
// tri du cache garde ce qui se regarde, pas seulement ce qui s'est écrit en dernier.
std::optional<CachedShots> load_shot_cache(const std::filesystem::path& dir, const std::string& key,
                                           std::uint32_t variant = 0);

}  // namespace ajp::media
