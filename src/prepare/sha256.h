// SHA-256 (FIPS 180-4) en C++ pur, avec l'encadrement des clés du moteur.
//
// Repris de animejanai-inference-rtx@118c272 src/content_identity.h:16-80 (classe
// Sha256 : update, integer, text, hex). Les clés de cache en dépendent : un
// changement d'encadrement invaliderait tous les moteurs déjà construits.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "core/types.h"

namespace ajp::prepare {

class Sha256 {
 public:
  void update(std::span<const std::byte> bytes);
  void update(std::string_view bytes);
  // Entier sur 8 octets gros-boutistes, comme identity::Sha256::integer.
  void integer(std::uint64_t value);
  // Longueur puis octets : deux textes voisins ne se confondent jamais.
  void text(std::string_view value);
  // 64 chiffres hexadécimaux minuscules ; l'état courant reste utilisable.
  std::string hex() const;

 private:
  void block(const std::uint8_t* data);

  std::array<std::uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t used_ = 0;
  std::uint64_t bytes_ = 0;
};

std::string sha256_hex(std::string_view bytes);

// Hachage d'un fichier entier, lu par blocs. Refus CacheIo s'il est illisible.
std::expected<std::string, Refusal> sha256_file(const std::filesystem::path& file);
// Comme sha256_file, gardé pour le processus tant que le fichier garde sa taille et sa date de
// modification : chaque ouverture de chaîne rehachait ses modèles (GMFSS : ~60 Mo, des
// centaines de ms). Pour reconnaître un modèle, pas pour vérifier un téléchargement.
std::expected<std::string, Refusal> sha256_file_known(const std::filesystem::path& file);

// 64 chiffres hexadécimaux minuscules.
bool is_sha256_hex(std::string_view text);

}  // namespace ajp::prepare
