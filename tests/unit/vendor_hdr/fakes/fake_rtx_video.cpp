// Faux RTX Video SDK, chemin CUDA : les trois fonctions de rtx_video_abi.h. L'API du vrai SDK
// n'a pas de poignée et garde son état dans la DLL ; le faux l'imite par un état de module,
// que les tests lisent par fake_rtx_video_*. Les « pointeurs CUDA » sont de la mémoire hôte.
// Sortie, au format RGB10A2 de l'entrée : chaque composante × réglage / 100 (gris moyen / 50),
// bornée à 1023, alpha 3.
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "vendor_hdr/rtx_video_abi.h"

using namespace ajp::vendor_hdr::rtx_video;

namespace {

struct State {
  bool created = false;
  bool refuse_create = false;
  CUcontext context = nullptr;
  int evaluations = 0;
};

State& state() {
  static State instance;
  return instance;
}

std::uint32_t scale(std::uint32_t value10, int numerator, int denominator) {
  const auto scaled =
      static_cast<std::uint32_t>(numerator) * value10 / static_cast<std::uint32_t>(denominator);
  return std::min<std::uint32_t>(1023, scaled);
}

}  // namespace

extern "C" {

bool rtx_video_api_cuda_create(CUcontext* context, CUstream stream, bool thdr, bool vsr) {
  State& current = state();
  // Une seule instance ; TrueHDR seul, sur le flux 0 hérité.
  if (current.refuse_create || current.created || context == nullptr || stream != nullptr || !thdr || vsr) {
    return false;
  }
  current.created = true;
  current.context = *context;
  return true;
}

bool rtx_video_api_cuda_evaluate(CUdeviceptr input, CUdeviceptr output, ApiRect input_rect,
                                 ApiRect output_rect, VsrSetting* vsr, ThdrSetting* thdr) {
  State& current = state();
  if (!current.created || thdr == nullptr || vsr == nullptr || input == 0 || output == 0) return false;
  if (input_rect.left != 0 || input_rect.top != 0 || input_rect.right != output_rect.right ||
      input_rect.bottom != output_rect.bottom || input_rect.right <= 0 || input_rect.bottom <= 0) {
    return false;
  }
  const auto count = static_cast<std::size_t>(input_rect.right) * static_cast<std::size_t>(input_rect.bottom);
  const auto* source = reinterpret_cast<const unsigned char*>(input);  // NOLINT(performance-no-int-to-ptr)
  auto* target = reinterpret_cast<unsigned char*>(output);             // NOLINT(performance-no-int-to-ptr)
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t pixel = 0;
    std::memcpy(&pixel, source + i * 4, sizeof pixel);
    const std::uint32_t packed = scale(pixel & 0x3FFU, thdr->contrast, 100) |
                                 (scale((pixel >> 10U) & 0x3FFU, thdr->saturation, 100) << 10U) |
                                 (scale((pixel >> 20U) & 0x3FFU, thdr->middle_gray, 50) << 20U) | (3U << 30U);
    std::memcpy(target + i * 4, &packed, sizeof packed);
  }
  ++current.evaluations;
  return true;
}

void rtx_video_api_cuda_shutdown() {
  State& current = state();
  current.created = false;
  current.context = nullptr;
}

// Accès des tests à l'état du faux.
bool fake_rtx_video_is_created() {
  return state().created;
}
CUcontext fake_rtx_video_context() {
  return state().context;
}
int fake_rtx_video_evaluations() {
  return state().evaluations;
}
void fake_rtx_video_refuse_create(bool refuse) {
  state().refuse_create = refuse;
}

}  // extern "C"
