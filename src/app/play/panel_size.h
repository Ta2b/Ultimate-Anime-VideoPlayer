#pragma once

// La place du menu des réglages, gardée en parts de la place que la fenêtre lui laisse plutôt qu'en pixels :
// le menu suit la fenêtre quand elle change de taille, et une taille tirée aux bords ou une position où on
// l'a déplacé devient une nouvelle part.

namespace ajp::app::play {

struct PanelArea {
  float width = 0.0F;
  float height = 0.0F;
  bool operator==(const PanelArea&) const = default;
};

// Un rectangle dans la place : x et y depuis son coin haut gauche.
struct PanelRect {
  float x = 0.0F;
  float y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
  bool operator==(const PanelRect&) const = default;
};

// Le menu sans vidéo derrière lui (accueil, construction d'un moteur) : collé à droite sur toute la hauteur,
// d'une largeur fixe qui laisse `keep_left` à gauche pour la carte de l'écran (bornée à `narrowest` et
// `widest`). Il ne se déplace ni ne se redimensionne : rien à dévoiler derrière, et la carte reste visible.
PanelRect fixed_panel(PanelArea area, float widest, float narrowest, float keep_left);

class PanelSize {
 public:
  // Le rectangle voulu pour cette place, gardé dedans ; la première fois, `first` (en pixels).
  PanelRect target(PanelArea area, PanelRect first);
  // Vrai quand la place a changé depuis l'image précédente : le rectangle voulu s'impose alors au panneau.
  bool area_changed() const { return area_changed_; }
  // Le rectangle du panneau après son tracé. Un écart avec l'image précédente, la place restant la même,
  // vient des bords tirés ou d'un déplacement : il devient la nouvelle part.
  void observe(PanelRect rect);

 private:
  bool placed_ = false;  // faux : pas encore de part, `first`
  PanelRect share_;      // le rectangle en parts de la place
  PanelArea area_;
  PanelRect last_;
  bool area_changed_ = true;
};

}  // namespace ajp::app::play
