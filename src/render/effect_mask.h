// Éclats HDR des effets (piste 6 du plan du flux NVIDIA) : le masque des effets que la passe anime du grade
// lit (3-anime.hook, fx_on). Un effet (explosion, magie, éclair) est clair, change vite et est mal suivi par
// le flux. La carte de la scène (kernels/spec/scene_flow.h) le dit par tuile ; ce masque en tire, par tuile,
// la part de la marge des éclats que la lumière garde : 1 hors des dessins sûrs (le grade d'aujourd'hui ; une
// lumière immobile ne se distingue pas du décor, app/play/effect_book.h), 0 sur un dessin sûr qui bouge (la
// lumière n'y garde que fx_rest de sa montée), et une montée plus haute sur un effet sûr.
//
// Piège majeur : un masque qui bouge d'une image à l'autre fait pulser la luminosité. Le masque est donc :
//  - dilaté d'une tuile (le halo d'une explosion déborde de la tuile où le flux la voit) ;
//  - lissé dans le temps, vite vers 1 (l'éclat d'une explosion part tout de suite), lentement vers 0 (une
//    tuile qui hésite d'une paire à l'autre reste haute) ; une paire tenue ou un fondu n'apprennent rien
//    (un effet animé en deux ne retombe pas une paire sur deux) ; une coupe pose le masque d'un coup, ce
//    qu'un changement de plan cache ;
//  - adouci d'un flou 3 × 3, puis lu en bilinéaire par le GPU.
// Sans carte, il revient lentement vers 1 partout, puis s'éteint (neutral) : le grade redevient celui
// d'aujourd'hui, image pour image.
//
// Une seconde couche, lissée, dilatée et floutée de même, suit les seuls effets sûrs (la classe effet de la
// carte) : là, la passe anime ouvre plus large la porte des lumières et monte plus haut (fx_gain, essai du
// PC du 19 novembre : l'effet se voyait, mais sur 2 % des pixels). Elle ne touche rien hors des effets.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace kirari::render {

// Ce que le masque lit d'une paire, sans dépendre de la carte de la scène (l'adaptateur est dans app/play,
// effect_book.h).
struct EffectTiles {
  int cols = 0;
  int rows = 0;
  // La part de l'image que couvrent cols tuiles (cols × côté d'une tuile / largeur), 1 ou un peu plus : la
  // dernière colonne et la dernière ligne peuvent déborder de l'image.
  double cover_x = 1.0;
  double cover_y = 1.0;
  // Par tuile, ligne par ligne : kEffectUnsure si la carte ne tranche pas, sinon de 0 (décor ou dessin
  // sûr) à 1 (effet sûr).
  std::vector<float> effect;
  // Par tuile, 1 sur un effet sûr (la classe effet de la carte), 0 ailleurs ; vide : aucun.
  std::vector<float> sure;
  bool cut = false;          // coupe de scène : le masque est posé d'un coup
  bool hold = false;         // dessin tenu ou fondu : rien n'est appris de la paire
  std::int64_t span_us = 0;  // durée de la paire (pts(b) − pts(a)) ; ≤ 0 : 41 708 µs (23,976 i/s)
};

inline constexpr float kEffectUnsure = -1.0F;

// Constantes de temps du lissage (µs) : vers l'effet, puis vers le décor (et vers le neutre sans carte).
inline constexpr double kEffectRiseUs = 40'000.0;
inline constexpr double kEffectFallUs = 700'000.0;
// Sous ce seuil d'écart au neutre partout, le masque est neutre (fx_on à 0).
inline constexpr float kEffectNeutral = 1.0F / 512.0F;

class EffectMask {
 public:
  // La carte de la paire suivante. Une coupe sans tuiles (pas de carte du nouveau plan) remet le masque au
  // neutre.
  void update(const EffectTiles& tiles);
  // Une paire sans carte (absente, pas encore relue) : retour lent vers le neutre.
  void relax(std::int64_t span_us);
  void reset();

  // Vrai sans masque, ou quand toutes les tuiles sont revenues à 1 et qu'aucun effet sûr ne reste : la passe
  // ne s'ajoute pas.
  bool neutral() const;
  int cols() const { return cols_; }
  int rows() const { return rows_; }
  double cover_x() const { return cover_x_; }
  double cover_y() const { return cover_y_; }
  // La part de la marge gardée (lissée, dilatée, floutée), ligne par ligne, de 0 à 1.
  std::span<const float> values() const { return shown_; }
  // L'effet sûr de la seconde couche (lissé, dilaté, flouté), ligne par ligne, de 0 à 1 ; vide avec values.
  std::span<const float> sure() const { return sure_shown_; }
  // Augmenté à chaque changement des valeurs ou de leur grille (la composition gardée de l'écran en dépend).
  std::uint64_t serial() const { return serial_; }

 private:
  void publish();
  int cols_ = 0;
  int rows_ = 0;
  double cover_x_ = 1.0;
  double cover_y_ = 1.0;
  std::vector<float> level_;  // l'état lissé, avant le flou
  std::vector<float> shown_;
  std::vector<float> sure_level_;  // la seconde couche (effets sûrs), avant le flou
  std::vector<float> sure_shown_;
  std::uint64_t serial_ = 0;
};

// Le pas d'un lissage exponentiel de constante tau_us sur une paire de span_us.
float effect_step(std::int64_t span_us, double tau_us);

}  // namespace kirari::render
