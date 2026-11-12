// Attendre une échéance à la microseconde (étape 6.7) : une attente grossière du système jusqu'à
// 1,5 ms (2,5 ms sous Windows) avant l'échéance, puis une attente absolue précise. Fonction pure ; l'appelant
// fait les attentes (SDL_WaitEventTimeout, puis sleep_until_precise de loop_wait.cpp).
#pragma once

#include <algorithm>
#include <cstdint>

namespace ajp::core {

inline constexpr std::int64_t kWaitMarginUs = 1500;
// Sous Windows, l'attente grossière (SDL_WaitEventTimeout) dort comme Sleep, qui glisse de 1 à 2 ms même avec
// timeBeginPeriod(1) : une marge de 1,5 ms laissait le réveil tomber après l'échéance (le tour qui choisit
// l'image en retard d'une fraction de période à 165 Hz). Le minuteur haute résolution fait le reste.
inline constexpr std::int64_t kWaitMarginWindowsUs = 2500;

struct WaitPlan {
  std::int64_t coarse_ms = 0;         // attente grossière (0 : passer directement à la précise)
  std::int64_t precise_until_us = 0;  // échéance de l'attente absolue
};

inline WaitPlan wait_plan(std::int64_t now_us, std::int64_t deadline_us,
                          std::int64_t margin_us = kWaitMarginUs) {
  const std::int64_t room = deadline_us - now_us - margin_us;
  return {.coarse_ms = std::max<std::int64_t>(room / 1000, 0), .precise_until_us = deadline_us};
}

}  // namespace ajp::core
