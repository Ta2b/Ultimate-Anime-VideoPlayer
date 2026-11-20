// Les convolutions qui produisent les 144 logits de l'agrandissement convexe de GMFlow.
#pragma once

#include "prepare/gmflow_backbone.h"

namespace kirari::prepare {

struct GmflowUpsampler {
  GmflowConv first;  // 3 × 3, 130 → 256, suivie d'une Relu
  GmflowConv mask;   // 1 × 1, 256 → 144
};

}  // namespace kirari::prepare
