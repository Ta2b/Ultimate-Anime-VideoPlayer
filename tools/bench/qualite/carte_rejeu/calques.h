// Les calques et la plaque de décor rejoués sur processeur (docs/calques.md), à la suite de la carte de
// chaque paire : ce que le lecteur déciderait (app/play/layer_plan.h, compilé tel quel) et ce que la garde
// de la composition (kernels/spec/layers.h) en refuserait, mesuré sur les lumas du flux.
//
// Une ligne par paire sur la sortie standard :
//   calques ÉTAT PART GARDE ÉCART VITESSE PLAQUE PGARDE PÉCART RAISON
// ÉTAT 1 : la paire a des calques, 2 : avec la caméra lk d'une paire voisine (layer_plan.h, with_neighbour) ;
// PART : part des tuiles décor ; GARDE : part du décor (pondéré par le masque) que la garde rend à RIFE, à
// l'instant ½ ; ÉCART : médiane de |Y(a déplacée) − Y(b déplacée)| sur le cœur du décor, en codes ; VITESSE :
// déplacement du centre de l'image par la caméra, pixels de travail ; PLAQUE 0 rien, 1 nouvelle clé, 2 décor
// repris de la clé ; PGARDE, PÉCART : la même garde et le même écart pour la plaque (la luma de la clé
// déplacée contre celle de b ; −1 sans reprise) ; RAISON : pourquoi la paire n'a pas de calques
// (layer_plan.h), « - » si elle en a. −1 : pas mesuré.
//
// Sur les lumas du flux, sans SPAN ni RIFE : la garde du lecteur compare des images agrandies, en RVB, et la
// plaque y est un SPAN de la clé ; les écarts mesurés ici en sont une approximation, pas une copie.
//
// Vues (--vues) : une image PPM toutes les PAS paires à calques, à gauche l'image à ½ composée par
// ref::compose_layers sur le fondu de a et b (qui tient la place de RIFE : là où il reste, les dessins
// paraissent doublés), à droite le fondu teinté : bleu, décor repris de la caméra ; rouge, décor rendu à RIFE
// par la garde ; sans teinte, RIFE. De même pour la plaque : b avec son décor repris de la clé, et b teinté.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "app/play/layer_plan.h"
#include "kernels/image.h"
#include "kernels/spec/scene_flow.h"

namespace carte_rejeu {

class LayerReplay {
 public:
  // divisor : le diviseur du travail au flux ; views : dossier des vues (vide : aucune), une toutes les every
  // paires.
  LayerReplay(int divisor, std::string views, int every, bool dump_grid);

  // La paire suivante du plan : sa ligne, et ses vues s'il y a lieu. late : sa caméra lk relue après la
  // décision (--sans-affinage), pour les paires suivantes. Faux si une vue ne s'écrit pas.
  bool pair(const kirari::kernels::Plane<std::uint8_t>& a, const kirari::kernels::Plane<std::uint8_t>& b,
            const kirari::kernels::SceneMotion& motion,
            const std::optional<kirari::app::play::NeighbourCamera>& late = std::nullopt);

 private:
  int divisor_;
  std::string views_;
  int every_;
  bool dump_grid_;
  int index_ = 0;
  kirari::app::play::PlateTrack plate_;
  kirari::app::play::NeighbourCameras neighbours_;
  kirari::kernels::PlaneF key_;  // la luma de l'image clé
};

}  // namespace carte_rejeu
