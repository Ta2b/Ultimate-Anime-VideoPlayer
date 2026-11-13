// Le look en une table (GradeParams::look_table) : la pile entière, pied adouci et domaines compris, évaluée
// sur le GPU aux 129³ nœuds d'une table 3D, puis lue d'une seule lecture trilinéaire par pixel au lieu de
// traverser les vingt emplacements. Interne à render : Grade le pose à la place du hook du look.
//
// La table est construite par une passe de calcul au premier rendu qui la demande, et refaite au rendu qui
// suit un changement d'intensité, de LUT ou de domaine, sans relire de hook. Le passage du look lui-même
// (tuiles de 16 × 16, voisinage de --clean-looks) est le texte de anime-hdr-look.hook, où seul ajl_look()
// lit la table : le chemin exact reste celui des tests et des calculs en double (output_tune, sample_chain).
//
// Une approximation : à 129 points, quelques codes PQ 10 bits au pire sur les pliures de la pile (le bleu
// saturé), une fraction de code en moyenne (docs/hdr-anime.md, « Le look en une table »).
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"
#include "render/cube_lut.h"

namespace ajp::render {

// Les nœuds par axe : 4 par maille des LUT de 33 points, dont la grille tombe ainsi sur des nœuds (la
// première LUT de la pile y est exacte). 129³ × 16 octets : 34 Mo de mémoire vidéo.
inline constexpr int kLookTableSize = 129;

class LookTable {
 public:
  // Refus si le GPU n'a pas de format rgba32f lisible en trilinéaire et inscriptible par une passe de calcul.
  static std::expected<std::unique_ptr<LookTable>, Refusal> create(pl_gpu gpu);
  ~LookTable();
  LookTable(const LookTable&) = delete;
  LookTable& operator=(const LookTable&) = delete;
  LookTable(LookTable&&) = delete;
  LookTable& operator=(LookTable&&) = delete;

  // Le texte du hook du look tel que Grade le donne à libplacebo (marques remplacées) et ses LUT distinctes,
  // dans l'ordre de ses textures AJG_LUT, AJG_LUT2… Refait les textures des LUT ; la table sera refaite.
  std::expected<void, Refusal> set_looks(std::string_view look_hook_text,
                                         std::span<const CubeLut* const> textures);

  // Les //!PARAM du look, tels que Grade les écrit dans le hook exact.
  struct Inputs {
    std::array<float, kLookSlots> mixes{};
    float white = 203.0F;
    float pq = 0.0F;
    float toe = 0.0F;
    float gamma = 2.4F;
    float clean = 0.0F;
    bool on = false;
    friend bool operator==(const Inputs&, const Inputs&) = default;
  };
  void set_inputs(const Inputs& inputs);

  const pl_hook* hook() const;
  // Les constructions de la table depuis la création (tests).
  std::uint64_t builds() const;

 private:
  struct Impl;
  explicit LookTable(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
