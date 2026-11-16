#pragma once

// La taille du menu des réglages, gardée en parts de la place que la fenêtre lui laisse plutôt qu'en pixels :
// le menu suit la fenêtre quand elle change de taille, et une taille tirée aux bords devient une nouvelle
// part.

namespace ajp::app::play {

struct PanelArea {
  float width = 0.0F;
  float height = 0.0F;
  bool operator==(const PanelArea&) const = default;
};

class PanelSize {
 public:
  // La taille voulue pour cette place ; la première fois, default_width de large et toute la hauteur.
  PanelArea target(PanelArea area, float default_width);
  // Vrai quand la place a changé depuis l'image précédente : la taille voulue s'impose alors au panneau.
  bool area_changed() const { return area_changed_; }
  // La taille du panneau après son tracé. Un écart avec l'image précédente, la place restant la même, vient
  // des bords tirés : il devient la nouvelle part.
  void observe(PanelArea size);

 private:
  float width_share_ = 0.0F;  // 0 : pas encore de part, le défaut
  float height_share_ = 0.0F;
  PanelArea area_;
  PanelArea last_;
  bool area_changed_ = true;
};

}  // namespace ajp::app::play
