// Débogage des paires (menu Statistiques, « Paires tenues et coupées ») : chaque paire (a, b) que l'étage gpu
// juge tenue (dessin identique, pas de RIFE) ou coupée (coupe de scène) est notée ici avec son intervalle de
// pts, puis teinte à l'écran tant qu'une image de cet intervalle est montrée (b seule, d'ordinaire : une
// fraction de seconde), et écrite dans ajp.log avec ses mesures. Rien ne s'écrit ni ne se teint tant que le
// réglage est éteint : la lecture normale n'en voit rien. Le réglage ne vit que le temps de la session.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace ajp::gpu {
struct PairScores;
}  // namespace ajp::gpu

namespace ajp::app::play {

enum class PairVerdict : std::uint8_t { Held, Cut };

struct PairMark {
  std::uint32_t epoch = 0;
  std::int64_t from_us = 0;  // pts de a
  std::int64_t to_us = 0;    // pts de b
  PairVerdict verdict = PairVerdict::Held;
};

// Les dernières paires notées, écrites par gpu et lues par main à chaque présentation (verrou court).
class PairMarks {
 public:
  // Assez pour l'avance de gpu sur l'écran (anneau et images en vol) à ×1 comme à ×8.
  static constexpr std::size_t kKept = 64;
  void add(const PairMark& mark);
  // Le verdict de la paire qui finit sur ou après pts_us et commence avant : l'image montrée en fait partie.
  std::optional<PairVerdict> at(std::uint32_t epoch, std::int64_t pts_us) const;
  void clear();

 private:
  mutable std::mutex mutex_;
  std::array<PairMark, kKept> marks_{};
  std::size_t count_ = 0;
  std::size_t next_ = 0;
};

// La ligne d'ajp.log d'une paire notée : « paire tenue 12.345 s → 12.387 s, pire bloc 0.41 (tenue ≤ 0.70),
// score 0.031 (coupe > 0.120), plans : même plan ». known_cut : la réponse des plans de la pré-analyse, rien
// hors de leur chronologie. useful_area : la part de l'image hors bandes noires (le score y est rapporté).
std::string pair_mark_line(const PairMark& mark, const gpu::PairScores& scores, std::optional<bool> known_cut,
                           double useful_area);

// La teinte d'un verdict, en sRGB : un voile léger sur toute l'image (wash) et un cadre plus franc (frame).
// Bleu pour un dessin tenu, rouge pour une coupe.
struct PairTint {
  std::array<float, 4> wash{};
  std::array<float, 4> frame{};
  friend bool operator==(const PairTint&, const PairTint&) = default;
};
PairTint pair_tint(PairVerdict verdict);

}  // namespace ajp::app::play
