// L'accumulateur f16 du SPAN FP8 face à ses débordements (docs/span-fp8-cutlass.md, garde-fou). Une image
// dont une somme a débordé sort avec cette somme bornée au plus grand f16, ce qui ne se voit pas. Tant que
// ces images restent rares, le réseau garde l'accumulateur f16 ; si kFrames d'entre elles tombent dans
// kWindow lancements (un passage très lumineux qui dure), il passe au f32 pendant kCooldown lancements, puis
// réessaie le f16. C++ pur : le corps CUDA (span_fp8_body.cuh) lui donne le compte des images débordées, lu
// sans synchronisation, donc avec un lancement ou deux de retard.
#pragma once

#include <array>
#include <cstdint>

namespace ajp::gpu::span_fp8 {

enum class Acc16Change : std::uint8_t { None, ToF32, ToF16 };

class Acc16Policy {
 public:
  static constexpr int kFrames = 3;             // images débordées...
  static constexpr std::int64_t kWindow = 240;  // ... dans autant de lancements (environ 2 s à 120 i/s)
  static constexpr std::int64_t kCooldown =
      2400;  // lancements en f32 avant de réessayer le f16 (environ 20 s)

  // Un lancement de plus ; overflowed : le compte, croissant, des images débordées depuis la création.
  Acc16Change poll(std::int64_t overflowed);
  bool f16() const { return f16_; }

 private:
  std::int64_t launch_ = 0;
  std::int64_t seen_ = 0;
  std::int64_t f32_until_ = 0;
  std::array<std::int64_t, kFrames> recent_{};  // lancements des dernières images débordées, en anneau
  int count_ = 0;                               // places remplies de recent_
  int next_ = 0;                                // prochaine place
  bool f16_ = true;
};

}  // namespace ajp::gpu::span_fp8
