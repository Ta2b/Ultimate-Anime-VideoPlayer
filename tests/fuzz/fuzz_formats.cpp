// Deux formats lus sur disque : la section [keys] de user.toml (la seule chose que ce fichier garde) et le
// paquet de poids .span8 du SPAN FP8. Chacun lit des octets quelconques sans plantage ni lecture hors du
// tampon, et vérifie une propriété.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "../support/span_fp8_synthetic.h"
#include "core/keys.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"
#include "gpu/span_fp8/package.h"

int ajp::fuzz::keys(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  // La lecture stricte, puis la tolérante : la seconde ne refuse jamais, quoi que le fichier contienne.
  (void)ajp::core::parse_keys(text);
  const ajp::core::KeysRead read = ajp::core::read_keys(text);
  // Un raccourci de la carte retrouve son action : une touche n'ouvre pas deux actions.
  for (std::size_t index = 0; index < ajp::core::kActionCount; ++index) {
    const auto action = static_cast<ajp::core::Action>(index);
    if (const auto& chord = read.map.chord(action)) {
      ajp::fuzz::require(read.map.action_for(*chord) == action, "un raccourci retrouve son action");
    }
  }
  return 0;
}

// Les octets reçus d'abord, tels quels (l'en-tête, les refus) ; puis un vrai petit paquet (synthetic) qu'ils
// abîment, car un paquet valide pèse bien plus que la borne des entrées du pilote de rejeu : le premier octet
// dit le nombre de blocs (bits 2 à 4), s'il faut couper le paquet (bit 0) et recalculer le CRC après
// l'abîmage (bit 1, pour atteindre les contrôles qui suivent le CRC : valeurs finies, échelles positives) ;
// les deux suivants disent où (le bit 7 du premier : dans l'en-tête), le reste s'y mélange par ou exclusif.
int ajp::fuzz::span_package(const std::uint8_t* data, std::size_t size) {
  namespace pkg = ajp::gpu::span_fp8;
  const auto raw = pkg::parse({reinterpret_cast<const std::byte*>(data), size});
  if (raw) {
    // Un paquet lu se réécrit et se relit à l'identique (la lecture a vérifié tailles, CRC et valeurs
    // finies).
    const auto again = pkg::parse(pkg::serialize(*raw));
    ajp::fuzz::require(again.has_value(), "un paquet écrit se relit");
  }
  if (size < 3) return 0;
  const int blocks =
      pkg::kMinBlocks +
      static_cast<int>(((data[0] >> 2U) & 7U) % static_cast<unsigned>(pkg::kBlocks - pkg::kMinBlocks + 1));
  std::vector<std::byte> bytes = pkg::serialize(pkg::testing::synthetic(0.01F, blocks));
  const std::size_t at =
      (data[1] & 0x80U) != 0U
          ? data[2] % pkg::kHeaderBytes
          : pkg::kHeaderBytes + ((std::size_t{data[1]} << 8U) | data[2]) % (bytes.size() - pkg::kHeaderBytes);
  for (std::size_t i = 3; i < size && i < 67; ++i)
    bytes[(at + i - 3) % bytes.size()] ^= static_cast<std::byte>(data[i]);
  if ((data[0] & 2U) != 0U) {
    const std::uint32_t crc = pkg::crc32(std::span<const std::byte>(bytes).subspan(pkg::kHeaderBytes));
    for (unsigned i = 0; i < 4U; ++i) bytes[32 + i] = static_cast<std::byte>((crc >> (8U * i)) & 0xFFU);
  }
  if ((data[0] & 1U) != 0U) bytes.resize(at);
  (void)pkg::parse(bytes);
  return 0;
}
