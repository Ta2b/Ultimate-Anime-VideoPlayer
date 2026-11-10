// La lecture d'une édition ordonnée de Matroska dans son ordre (A14-03) : les parties sont des plages du
// fichier, jouées l'une après l'autre. À la fin d'une partie, un seek exact mène au début de la suivante,
// sauf si elle la prolonge dans le fichier ; après la dernière, la lecture finit. La barre de lecture reste
// celle du fichier. Les parties d'un autre fichier (générique lié) ne sont pas jouées.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace ajp::app::play {

struct PlayPart {
  static constexpr std::int64_t kToEnd = std::numeric_limits<std::int64_t>::max();
  std::int64_t start_us = 0;
  std::int64_t end_us = kToEnd;
};

class EditionPlan {
 public:
  struct Jump {
    std::int64_t to_us = 0;
  };
  struct End {};
  using Step = std::variant<std::monostate, Jump, End>;

  EditionPlan() = default;
  explicit EditionPlan(std::vector<PlayPart> parts) : parts_(std::move(parts)) {}

  bool active() const { return !parts_.empty(); }

  // Un seek demandé (barre, touches, reprise) vers target_us : la partie qui le contient devient la partie en
  // cours. Hors de toute partie : le début de la première partie qui suit dans le fichier. std::nullopt :
  // après la dernière, la lecture est finie.
  std::optional<std::int64_t> seek(std::int64_t target_us);

  // Une image de l'époque courante est montrée à pts_us ; period_us : la période de sortie (0 inconnue). Rend
  // le saut vers la partie suivante quand l'image suivante sortirait de la partie, ou la fin.
  Step shown(std::int64_t pts_us, std::int64_t period_us);

 private:
  bool inside(std::size_t index, std::int64_t pts_us, std::int64_t period_us) const;

  std::vector<PlayPart> parts_;
  std::size_t index_ = 0;  // la partie en cours, dans l'ordre de l'édition
  bool entered_ = false;   // une image de la partie en cours est montrée depuis le dernier saut
  bool jumped_ = false;    // un saut vers la partie en cours attend sa première image
};

}  // namespace ajp::app::play
