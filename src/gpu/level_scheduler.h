// Ordonnanceur d'un niveau d'interpolation : logique pure, sans GPU ni allocation.
//
// Un niveau du Plan reçoit une suite d'entrées (les sources décodées, ou les
// sorties du niveau précédent) et rend la suite de ses sorties, dans l'ordre :
//   - l'accumulateur de cadence de core donne les instants de sortie de chaque
//     paire (gauche, entrée) et leurs horodatages ;
//   - la timeline des dessins de core classe chaque entrée (nouveau dessin,
//     dessin tenu, coupe de scène) et décide à chaque instant Source,
//     Interpolate ou Repeated ;
//   - une paire attend que son anticipation (preload du rythme) soit classée ;
//     les sorties partent donc par paquets, toujours dans l'ordre des pts.
// Chaque sortie reçoit un SourceId dérivé, unique, et une mesure synthétisée
// contre la sortie précédente : le niveau suivant la classe sans repasser par
// le GPU (un doublon reste un doublon, une coupe reste une coupe).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>

#include "core/cadence.h"
#include "core/drawings.h"
#include "core/plan.h"
#include "core/types.h"

namespace ajp::gpu {

// Au plus quatre niveaux : chaque facteur dépasse 1 et leur produit reste <= 16.
inline constexpr std::size_t kMaxLevels = 4;

// Identité de la sortie numéro `serial` du niveau `level` : bit 31 posé, niveau sur
// 3 bits, numéro sur 28 bits. Les identités du décodage restent sous 2^31.
constexpr SourceId derived_id(std::uint32_t level, std::uint32_t serial) {
  return SourceId{0x8000'0000U | ((level & 0x7U) << 28U) | (serial & 0x0FFF'FFFFU)};
}

constexpr bool is_derived(SourceId id) {
  return (id.value & 0x8000'0000U) != 0;
}

// Une entrée du niveau, dans l'ordre de présentation.
struct LevelInput {
  SourceId id;
  std::optional<std::int64_t> pts_us;  // absent : horodatage invalide
  std::uint32_t epoch = 0;
  core::PairMeasure measure;  // contre l'entrée précédente ; ignorée en tête de séquence
};

// Une sortie : la décision de la timeline à un instant de la grille.
struct LevelOutput {
  SourceId id;  // dérivé, unique
  std::optional<std::int64_t> pts_us;
  std::uint32_t epoch = 0;
  core::InstantDecision decision;  // a et b : identités des entrées du niveau
  // L'instant de la grille : la paire (pair_left, pair_right) et t dans ]0, 1]. Une origine
  // montre pair_right seule, à t = 1.
  SourceId pair_left;
  SourceId pair_right;
  core::Ratio t{1, 1};
  bool scene_cut = false;  // instant dans une paire coupée : l'image de gauche reste
  // Contre la sortie précédente, pour le niveau suivant. identical : la même image,
  // rien à recalculer en aval.
  core::PairMeasure measure;
};

// Compteurs pour l'OSD, depuis la création du niveau.
struct LevelCounters {
  std::uint64_t outputs = 0;
  std::uint64_t interpolated = 0;
  std::uint64_t repeated = 0;    // Show::Repeated : REPEATED (Fast) ou doublon exact (Quality)
  std::uint64_t scene_cuts = 0;  // entrées classées coupe de scène
};

class LevelScheduler {
 public:
  // Paires en attente de leur anticipation : preload (4 au plus) et une marge.
  static constexpr std::size_t kMaxPending = 8;
  static constexpr std::size_t kMaxOutputs = kMaxPending * core::kMaxInstantsPerPair;
  // Entrées encore référencées : la paire la plus ancienne en attente jusqu'à la
  // dernière entrée, plus la première source du dessin en Smooth.
  static constexpr std::size_t kMaxLive = kMaxPending + 2;

  // Refuse un facteur hors de l'accumulateur, Quality hors du rythme Source, ou un
  // indice de niveau hors de [0, kMaxLevels[.
  static std::expected<LevelScheduler, Refusal> create(const core::InterpLevel& level, std::uint32_t index);

  // L'entrée suivante. Rend les sorties décidées par elle (souvent une paire,
  // parfois aucune, plusieurs quand une anticipation se complète), valables
  // jusqu'au prochain appel. Un refus de la timeline (identité reprise, mesure
  // hors plage) remet le niveau à zéro : l'entrée suivante devient l'origine.
  std::expected<std::span<const LevelOutput>, Refusal> push(const LevelInput& input);

  // Fin du flux : l'anticipation qui manque vaut « aucun dessin suivant ».
  std::span<const LevelOutput> finish();

  // Seek ou changement de Plan : les paires en attente sont jetées ; la prochaine
  // entrée de `epoch` devient l'origine. Refuse une époque plus ancienne.
  std::expected<void, Refusal> flush(std::uint32_t epoch);

  // Entrées que les décisions à venir peuvent encore citer : l'appelant garde leurs images.
  std::span<const SourceId> live() const { return {live_.data(), live_count_}; }

  const LevelCounters& counters() const { return counters_; }
  const core::InterpLevel& level() const { return level_; }

 private:
  struct Record {
    SourceId id;
    std::uint64_t position = 0;
    core::PairMeasure measure;  // contre l'entrée précédente
    SourceId first;             // première source du dessin
    std::uint64_t first_position = 0;
  };

  // Une paire de la grille, ou une origine qui émet sa source seule.
  struct Pending {
    bool origin = false;
    SourceId left;  // paire seulement
    SourceId right;
    std::uint32_t epoch = 0;
    std::array<core::OutputInstant, core::kMaxInstantsPerPair> instants{};
    std::size_t count = 0;
  };

  // Les entrées que montre une sortie, en positions d'entrée : [lo, hi].
  struct Content {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
    bool single = true;  // une seule image, telle quelle
  };

  LevelScheduler(const core::InterpLevel& level, std::uint32_t index, core::CadenceAccumulator cadence)
      : level_(level), index_(index), cadence_(cadence) {}

  std::expected<void, Refusal> classify(const LevelInput& input, bool starts_sequence);
  std::expected<void, Refusal> enqueue(const Pending& pending);
  std::expected<void, Refusal> drain();
  std::expected<void, Refusal> decide(const Pending& pending);
  void emit(LevelOutput output, const Record& left);
  void reset();
  void update_live();

  const Record* find(SourceId id) const;
  const Record* at(std::uint64_t position) const;
  Content content_of(const core::InstantDecision& decision, const Record& left) const;
  core::PairMeasure synthesize(const Content& previous, const Content& current) const;

  core::InterpLevel level_;
  std::uint32_t index_ = 0;
  core::CadenceAccumulator cadence_;
  core::DrawingsTimeline timeline_;
  std::uint32_t epoch_ = 0;  // la plus récente vue : une remise à zéro ne la fait pas reculer

  std::array<Record, core::drawings::kWindow> records_{};
  std::uint64_t next_position_ = 0;  // position de la prochaine entrée

  std::array<Pending, kMaxPending> pending_{};
  std::size_t pending_head_ = 0;
  std::size_t pending_count_ = 0;

  std::array<LevelOutput, kMaxOutputs> outputs_{};
  std::size_t output_count_ = 0;
  std::optional<Content> last_;  // contenu de la dernière sortie de la séquence
  std::uint32_t serial_ = 0;

  std::array<SourceId, kMaxLive> live_{};
  std::size_t live_count_ = 0;

  LevelCounters counters_;
};

}  // namespace ajp::gpu
