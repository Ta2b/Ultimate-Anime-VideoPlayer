// Statistiques d'une série de durées GPU du banc : médiane, p95, écart type, extrêmes, et le
// débit effectif qui en découle. Fonctions pures, testées dans tests/unit/vk_bench.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "core/types.h"

namespace ajp::bench {

struct Summary {
  std::size_t count = 0;
  double median = 0.0;  // moyenne des deux valeurs centrales pour un nombre pair
  double p95 = 0.0;     // rang le plus proche : la ⌈0,95 n⌉-ième plus petite valeur
  double stddev = 0.0;  // écart type de l'échantillon (division par n − 1) ; 0 pour une valeur
  double min = 0.0;
  double max = 0.0;
  // Durées au-delà de 1,5 × la médiane : passes interrompues (partage du GPU avec un autre
  // contexte, horloges), qui gonflent p95 et écart type sans toucher la médiane.
  std::size_t slow = 0;
};

// Refus FieldOutOfRange : série vide, ou valeur négative ou non finie.
std::expected<Summary, Refusal> summarize(std::span<const double> samples);

// Débit effectif en Go/s (10⁹ octets par seconde) : bytes en milliseconds ms. Refus
// FieldOutOfRange : durée nulle, négative ou non finie.
std::expected<double, Refusal> gigabytes_per_second(std::uint64_t bytes, double milliseconds);

}  // namespace ajp::bench
