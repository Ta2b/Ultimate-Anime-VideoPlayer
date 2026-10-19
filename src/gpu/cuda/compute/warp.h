// Warp arrière par flot et mélange de deux images à l'instant t (kernels/spec/warp.h), et plus
// grand déplacement de la paire, relu sans attente (k_flow_max du moteur de référence).
#pragma once

#include <cuda_runtime_api.h>
#include <optional>

#include "gpu/cuda/compute/device.h"
#include "gpu/cuda/compute/image.h"
#include "gpu/cuda/compute/memory.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::cuda {

// Mouvement estimé par RIFE au milieu de la paire : 4 plans de déplacements (f0x, f0y, f1x, f1y)
// en pixels de la fenêtre, un masque (poids de a) de même taille, et la fenêtre de l'image dans
// ces plans remplis.
struct MotionView {
  Tensor flow;  // 4 canaux
  Tensor mask;  // 1 canal
  kernels::Window area;
};

// out = a déformé × masque + b déformé × (1 − masque) à l'instant t de ]0, 1[ ; t = 1/2 : le
// mouvement tel quel (kernels::ref::warp_blend_at). a, b et out : 3 plans de même taille, de même
// type ; le mouvement en F16 ou F32. Refus : t hors de ]0, 1[, tailles, fenêtre hors des plans.
Result<> warp_blend(const Tensor& a, const Tensor& b, const MotionView& motion, float t, const Tensor& out,
                    cudaStream_t stream);

// Plus grand |déplacement| de la fenêtre (kernels::ref::flow_max), relu en 4 octets sans attente :
// launch() le calcule pour une paire, poll() le rend quand le GPU l'a fini, une seule fois. Le
// lecteur s'en sert pour la paire suivante (une seule passe RIFE sous single_run_motion).
// Un launch() avant que poll() ait rendu la mesure précédente la remplace.
class FlowMax {
 public:
  static Result<FlowMax> create(DevicePool& pool);

  Result<> launch(const MotionView& motion, cudaStream_t stream);
  // La dernière mesure finie et pas encore rendue ; rien sinon. N'attend jamais.
  Result<std::optional<float>> poll();

 private:
  DeviceBuffer device_;
  PinnedBuffer host_;
  Event ready_;
  bool pending_ = false;
};

// La fenêtre du mouvement, au contrat de kernels::ref::check_motion.
Result<> check_motion(const MotionView& motion);

}  // namespace ajp::gpu::cuda
