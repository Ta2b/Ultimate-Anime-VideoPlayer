// Faux pont aji_rtx, chemin CUDA : les trois fonctions de rtx_video_abi.h. Le vrai pont
// n'a pas de poignée et garde son état dans la DLL ; le faux l'imite par un état de module,
// que les tests lisent par fake_rtx_video_*. Les « pointeurs CUDA » sont de la mémoire hôte.
// Sortie, au format RGB10A2 de l'entrée : chaque composante × réglage / 100 (gris moyen / 50),
// bornée à 1023, alpha 3.
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

struct State {
  bool created = false;
  bool refuse_create = false;
  void* context = nullptr;
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

int aji_rtx_create(void* context, void* stream, int thdr, int vsr) {
  State& current = state();
  // Une seule instance ; TrueHDR seul, sur le flux 0 hérité.
  if (current.refuse_create || current.created || stream != nullptr || thdr != 1 || vsr != 0) return 0;
  current.created = true;
  current.context = context;
  return 1;
}

int aji_rtx_eval(unsigned long long input, unsigned long long output, int input_width, int input_height,
                 int output_width, int output_height, int /*vsr_quality*/, int contrast, int saturation,
                 int middle_gray, int /*max_luminance*/) {
  State& current = state();
  if (!current.created || input == 0 || output == 0) return 0;
  if (input_width != output_width || input_height != output_height || input_width <= 0 || input_height <= 0) {
    return 0;
  }
  const auto count = static_cast<std::size_t>(input_width) * static_cast<std::size_t>(input_height);
  const auto* source = reinterpret_cast<const unsigned char*>(input);  // NOLINT(performance-no-int-to-ptr)
  auto* target = reinterpret_cast<unsigned char*>(output);             // NOLINT(performance-no-int-to-ptr)
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t pixel = 0;
    std::memcpy(&pixel, source + i * 4, sizeof pixel);
    const std::uint32_t packed = scale(pixel & 0x3FFU, contrast, 100) |
                                 (scale((pixel >> 10U) & 0x3FFU, saturation, 100) << 10U) |
                                 (scale((pixel >> 20U) & 0x3FFU, middle_gray, 50) << 20U) | (3U << 30U);
    std::memcpy(target + i * 4, &packed, sizeof packed);
  }
  ++current.evaluations;
  return 1;
}

void aji_rtx_shutdown() {
  State& current = state();
  current.created = false;
  current.context = nullptr;
}

// Accès des tests à l'état du faux.
bool fake_rtx_video_is_created() {
  return state().created;
}
void* fake_rtx_video_context() {
  return state().context;
}
int fake_rtx_video_evaluations() {
  return state().evaluations;
}
void fake_rtx_video_refuse_create(bool refuse) {
  state().refuse_create = refuse;
}

}  // extern "C"
