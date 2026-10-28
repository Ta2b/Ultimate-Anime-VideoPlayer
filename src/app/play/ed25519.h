// Vérification d'une signature Ed25519 (RFC 8032), pour le manifeste des mises à jour
// (docs/ux-parametres.md, 4.5). Seule la vérification existe ici : la clé privée ne vit jamais dans le
// lecteur. L'arithmétique est celle de TweetNaCl (domaine public), réécrite ; les constantes de la courbe
// sont recalculées au démarrage plutôt que recopiées, et le SHA-512 vient de libavutil.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ajp::app::play {

// true si signature (64 octets) est celle de message sous la clé publique (32 octets).
bool ed25519_verify(std::span<const std::uint8_t, 32> public_key, std::span<const std::uint8_t> message,
                    std::span<const std::uint8_t, 64> signature);

// « 7d78… » → octets ; rien si la longueur ou un chiffre est faux.
std::optional<std::vector<std::uint8_t>> from_hex(std::string_view hex);

// Les mêmes, depuis du texte hexadécimal ; false si les tailles ne sont pas 32 et 64 octets.
bool ed25519_verify_hex(std::string_view public_key_hex, std::string_view message,
                        std::string_view signature_hex);

}  // namespace ajp::app::play
