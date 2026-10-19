// Le noyau qui manque à KernelOps (ABI 3) pour nourrir RIFE : l'entrée à 11 plans de
// kernels/spec/rife.h. rife_pad() prépare une image, mais rien n'écrit l'instant t ni les
// grilles constantes. À verser dans gpu/kernel_ops.h à la prochaine version de l'ABI ;
// d'ici là, un backend l'expose par le point d'entrée facultatif ajp_backend_rife_input,
// que le chargeur cherche sans l'exiger.
#pragma once

#include <expected>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/tensor.h"

namespace ajp::gpu {

class RifeInputOps {
 public:
  virtual ~RifeInputOps() = default;

  // a et b : RGB préparés de 1 × 3 × ph × pw (rife_pad, ou la sortie d'un RIFE précédent) ;
  // input : 1 × 11 × ph × pw, dans l'ordre de kernels/spec/rife.h (ref::rife_input). Refus :
  // tailles différentes ou non multiples de 64, t hors de [0, 1], type que le backend ne tient
  // pas. S'ajoute à la file du backend comme les noyaux de KernelOps, sans attente hôte.
  virtual std::expected<void, Refusal> rife_input(const Tensor& a, const Tensor& b, float t,
                                                  const Tensor& input) = 0;
};

}  // namespace ajp::gpu

// Facultatif : les noyaux d'entrée de RIFE du backend, vivants autant que lui ; nullptr s'il
// n'en a pas. Le chargeur le cherche après ajp_backend_create. noexcept comme les autres
// points d'entrée (gpu/backend.h).
extern "C" ajp::gpu::RifeInputOps* ajp_backend_rife_input(ajp::gpu::Backend* backend) noexcept;

namespace ajp::gpu {

using BackendRifeInputFn = decltype(&ajp_backend_rife_input);

}  // namespace ajp::gpu
