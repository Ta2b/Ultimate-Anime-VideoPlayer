// Accumulateur de cadence : pour un facteur rationnel num/den, les instants de
// sortie de chaque paire de sources (k, k+1) et leurs horodatages.
//
// La sortie vit sur une grille uniforme de num/den fois la cadence source :
// l'image j se place à la position j·den/num de la source d'origine. Une
// position entière est la source elle-même ; une position fractionnaire est
// interpolée à t = sa partie fractionnaire. Tout se compte en entiers, en
// 1/num de paire : la grille ne dérive pas, même sur des heures.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>

#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

// num <= 8·den : au plus 8 instants par paire.
inline constexpr std::size_t kMaxInstantsPerPair = 8;

// La fraction sous sa forme irréductible, dénominateur positif.
Ratio reduced(Ratio ratio);

// Une source vue par l'accumulateur.
struct SourceStamp {
  SourceId id;
  std::optional<std::int64_t> pts_us;  // absent : horodatage invalide
  std::uint32_t epoch = 0;
};

// Un instant de sortie, à la position t de la paire (gauche, droite).
struct OutputInstant {
  Ratio t;                             // (0, 1], en 1/num non réduit ; 1 : la source droite elle-même
  std::optional<std::int64_t> pts_us;  // absent seulement pour une origine sans horodatage
  bool is_source() const { return t.num == t.den; }
};

// Ce que la source reçue fait de la grille.
enum class GridEvent : std::uint8_t {
  Pair,              // la grille continue : instants entre gauche et droite
  Start,             // première source : origine de la grille
  NewEpoch,          // première source après Flush, ou d'une époque plus récente
  InvalidPts,        // horodatage absent, ici ou à gauche : origine
  PtsNotIncreasing,  // horodatage qui ne croît pas : origine
  Stale,             // source d'une époque passée : jetée, rien à émettre
};

// Les instants qu'émet une source reçue. Une origine émet la source seule (t = 1).
class CadenceStep {
 public:
  CadenceStep(GridEvent kind, SourceId from, SourceId to) : event(kind), left(from), right(to) {}

  GridEvent event;
  SourceId left;   // Pair seulement
  SourceId right;  // la source reçue

  std::span<const OutputInstant> instants() const { return {slots_.data(), count_}; }

 private:
  friend class CadenceAccumulator;
  // num <= 8·den borne la boucle de push() à kMaxInstantsPerPair appels.
  void add(const OutputInstant& instant) { slots_[count_++] = instant; }

  std::array<OutputInstant, kMaxInstantsPerPair> slots_{};
  std::size_t count_ = 0;
};

// Repris de vf_animejanai.c (rife_acc, rife_num, rife_den) : voir cadence.cpp.
class CadenceAccumulator {
 public:
  // Refuse un facteur hors de 1 < num/den <= 8 ou des bornes de ses termes.
  static std::expected<CadenceAccumulator, Refusal> create(Ratio factor);

  Ratio factor() const { return {num_, den_}; }

  // La source suivante, dans l'ordre de présentation.
  CadenceStep push(const SourceStamp& source);

  // L'interpolation de la dernière paire a échoué : gpu émet sa source droite
  // seule, et la grille repart d'elle.
  void resync();

  // Seek ou changement de Plan : la prochaine source de `epoch` redevient l'origine.
  // Refuse une époque plus ancienne que la courante.
  std::expected<void, Refusal> flush(std::uint32_t epoch);

 private:
  CadenceAccumulator(std::int64_t num, std::int64_t den) : num_(num), den_(den), acc_(den) {}
  CadenceStep origin(const SourceStamp& source, GridEvent event);

  std::int64_t num_;
  std::int64_t den_;
  std::int64_t acc_;  // prochain point de grille dans la paire courante, en 1/num : (0, den]
  std::optional<SourceStamp> left_;
  std::uint32_t epoch_ = 0;
  bool started_ = false;
};

}  // namespace ajp::core
