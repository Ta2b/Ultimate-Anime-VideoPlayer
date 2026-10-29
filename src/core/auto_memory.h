// Mémoire du mode Auto : le dernier palier tenu pour une vidéo, un moteur, une hauteur de travail et un
// GPU, pour repartir de là à la lecture suivante (un cran en dessous la première seconde, puis la montée
// rapide). Fichier à part (auto_memory.toml, à côté de resume.tsv), jamais play.toml : il change à chaque
// lecture et ne se partage pas. Borné à kAutoMemoryEntries (les plus anciennes sont oubliées), tolérant :
// absent ou abîmé, il se lit comme vide et les lignes illisibles sont ignorées.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"

namespace ajp::core {

inline constexpr std::size_t kAutoMemoryEntries = 200;

// La clé d'une lecture : fichier (chemin et taille), moteur, hauteur de travail, GPU. Les caractères qu'un
// fichier TOML ne porte pas proprement sont remplacés.
std::string auto_memory_key(std::string_view file, std::uintmax_t size, std::string_view engine,
                            int work_height, std::string_view gpu);

class AutoMemory {
 public:
  // Lu depuis le texte d'un fichier ; les entrées au-delà de la borne et les lignes illisibles sont écartées.
  static AutoMemory parse(std::string_view text);
  std::string dump() const;

  // Le palier retenu en centièmes de multiplicateur (250 = ×2,5) : indépendant de la granularité.
  std::optional<int> find(std::string_view key) const;
  // Retient ce palier : l'entrée devient la plus récente ; la plus ancienne sort au-delà de la borne.
  void put(std::string_view key, int centi);
  std::size_t size() const { return entries_.size(); }

 private:
  std::vector<std::pair<std::string, int>> entries_;  // de la plus ancienne à la plus récente
};

std::optional<int> load_auto_step(const std::filesystem::path& store, std::string_view key);
std::expected<void, Refusal> store_auto_step(const std::filesystem::path& store, std::string_view key,
                                             int centi);

}  // namespace ajp::core
