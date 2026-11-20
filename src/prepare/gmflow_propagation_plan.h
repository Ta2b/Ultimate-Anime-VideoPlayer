// Les deux projections 128 → 128 de FeatureFlowAttention, partagées par les deux échelles de GMFlow.
#pragma once

#include <vector>

namespace kirari::prepare {

struct GmflowProjection {
  std::vector<float> weights;  // [sortie][entrée], biais en float
  std::vector<float> bias;

  bool operator==(const GmflowProjection&) const = default;
};

struct GmflowPropagation {
  GmflowProjection query;
  GmflowProjection keys;
};

}  // namespace kirari::prepare
