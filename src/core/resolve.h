// resolve() : profil, capacités, source et écran deviennent un Plan immuable,
// ou un refus nommé. Fonction pure, sans GPU ni état global.
#pragma once

#include <expected>
#include <string>
#include <vector>

#include "core/plan.h"
#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

// Ce que le système sait faire maintenant : modèles installés, routes HDR.
struct Capabilities {
  std::vector<ModelId> models;
  bool true_hdr = false;
};

struct SourceInfo {
  int width = 0;
  int height = 0;
  double fps = 0.0;
};

struct DisplayInfo {
  int width = 0;
  int height = 0;
  double refresh_hz = 0.0;  // fixe : le rafraîchissement ; VRR : le plafond du panneau
  bool hdr = false;
  bool vrr = false;  // rafraîchissement variable actif
};

// Première règle dont les bornes, incluses, contiennent la source ; nullptr sinon.
const Rule* match_rule(const Profile& profile, const SourceInfo& source);

std::expected<Plan, Refusal> resolve(const Profile& profile, const Capabilities& caps,
                                     const SourceInfo& source, const DisplayInfo& display);

}  // namespace ajp::core
