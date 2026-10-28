// Décompression deflate brute (RFC 1951) en flux, sans zlib : le contenu d'un membre de roue ou de
// zip que le lecteur télécharge par plages (engine_install.h). Une fenêtre de 32 Kio en mémoire,
// jamais le fichier entier : libnvinfer pèse plus de 600 Mo décompressée.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>

#include "core/types.h"

namespace ajp::core {

// Lit jusqu'à buffer.size() octets compressés ; rend 0 à la fin des données.
using InflateRead = std::function<std::size_t(std::span<std::uint8_t> buffer)>;
// Reçoit les octets décompressés, dans l'ordre ; false interrompt (Cancelled).
using InflateWrite = std::function<bool(std::span<const std::uint8_t> bytes)>;

// Rend le nombre d'octets décompressés. Refus : DecodeFailed (flux corrompu ou tronqué),
// Cancelled (write a rendu false).
std::expected<std::uint64_t, Refusal> inflate_raw(const InflateRead& read, const InflateWrite& write);

}  // namespace ajp::core
