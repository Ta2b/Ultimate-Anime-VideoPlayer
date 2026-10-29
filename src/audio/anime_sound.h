// « Son anime » (docs/audio-anime.md) : un traitement à part pour les pistes STÉRÉO jouées sur une sortie
// stéréo, la forme de presque tous les animes (AAC 2.0). Les profils de mixage (mixer.h) agissent surtout sur
// le 5.1 et le 7.1 ; ici tout part de la paire gauche/droite.
//
//   1. Spatialisation : milieu/côtés. Le milieu (M = (G+D)/2) porte les voix et le centre, les côtés
//      (S = (G-D)/2) l'ambiance. Enceintes : l'ambiance est élargie (annulation de diaphonie transaurale
//      réduite aux côtés) et décorrélée par des passe-tout ; le milieu n'est jamais retiré, donc la somme
//      G+D (l'écoute mono) ne perd rien. Casque : la paire devient six enceintes virtuelles (avant ±30°, voix
//      sur le centre, ambiance sur les arrières ±110°) rendues par le binaural du mixeur (ou les réponses
//      HeSuVi de l'utilisateur).
//   2. Voix : la bande de la voix (cloche large autour de 1 kHz) du milieu est portée de +1 à +3 dB, avec un
//      léger égaliseur de présence à 3 kHz ; « voix en avant » creuse en plus les côtés dans la bande vocale.
//   3. Basses : harmoniques douces (3f, 5f) des graves sous 110 Hz, pour les petits haut-parleurs et le
//      casque, sans toucher au fondamental.
//   4. Dynamique : normalisation de sonie (ITU-R BS.1770-4 / EBU R128 : filtre K, fenêtres de 400 ms et de
//      3 s, porte à -50 LUFS, gain lent borné), compresseur doux, limiteur vrai-crête (suréchantillonnage
//      ×4, anticipation de 1,3 ms).
//
// Niveau « Naturel » = module absent : rien n'est construit, le signal passe tel quel (bit à bit).
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "audio/pcm.h"

namespace ajp::audio {

struct AnimeTuning {
  bool enabled = false;        // faux : « Naturel », aucun traitement
  float width = 0.5F;          // spatialisation (largeur, voix devant) : 0 à 1
  float ambience = 0.5F;       // niveau de l'ambiance décorrélée : 0 à 1
  float voice_db = 2.0F;       // centre porté (la voix) : 0 à 4 dB
  float presence_db = 1.5F;    // cloche de présence à 3 kHz : 0 à 4 dB
  float voice_forward = 0.0F;  // « voix en avant » : 0 à 1
  float bass = 0.3F;           // renfort des graves par harmoniques : 0 à 1
  bool loudness = true;        // normalisation de sonie
  float target_lufs = -20.0F;  // sa cible : -23 à -16 LUFS
  float compress = 0.3F;       // compresseur doux : 0 à 1
  float ceiling_db = -1.0F;    // plafond du limiteur vrai-crête : -3 à -0,5 dBTP
  bool operator==(const AnimeTuning&) const = default;
};
inline constexpr std::size_t kAnimeFields = 11;
// Le message cmd::SetListening porte ces réglages dans un tableau de 16 cases (les non utilisées à 0).
std::array<float, 16> to_wire(const AnimeTuning& tuning);
AnimeTuning anime_from_wire(const std::array<float, 16>& wire);
AnimeTuning clamped(AnimeTuning tuning);

// Un niveau nommé : Naturel (rien), Anime, Anime fort.
struct AnimeLevel {
  std::string_view key;  // « natural », « anime », « strong »
  std::string_view label;
  std::string_view detail;
  AnimeTuning tuning;
};
std::span<const AnimeLevel> anime_levels();
const AnimeLevel* find_anime_level(std::string_view key);

// Mesure de sonie ITU-R BS.1770-4 sur une paire gauche/droite. Flux : add() reçoit des trames entrelacées de
// n'importe quelle taille.
class LoudnessMeter {
 public:
  explicit LoudnessMeter(int sample_rate);
  // Une trame ; renvoie vrai quand un sous-bloc de 100 ms vient de se fermer.
  bool add(float left, float right);
  void reset();
  // LUFS sur les 400 ms (momentané) ou les 3 s (court terme) derniers ; -infini si rien n'est mesuré.
  double momentary() const;
  double short_term() const;
  int blocks() const { return filled_; }

 private:
  struct Stage {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    std::array<double, 2> z1{}, z2{};
    double run(int ch, double x) {
      const double y = b0 * x + z1[static_cast<std::size_t>(ch)];
      z1[static_cast<std::size_t>(ch)] = b1 * x - a1 * y + z2[static_cast<std::size_t>(ch)];
      z2[static_cast<std::size_t>(ch)] = b2 * x - a2 * y;
      return y;
    }
  };
  double mean_power(int blocks) const;
  Stage shelf_, high_pass_;
  int hop_ = 4800;
  int count_ = 0;
  double sum_ = 0.0;
  std::array<double, 30> power_{};  // les 30 derniers sous-blocs de 100 ms (3 s)
  int head_ = 0;
  int filled_ = 0;
};
// Sonie intégrée d'un enregistrement stéréo entrelacé (BS.1770-4 : blocs de 400 ms au pas de 100 ms,
// porte absolue -70 LUFS, porte relative -10 LU) ; pour les mesures hors ligne et les tests.
double integrated_lufs(std::span<const float> stereo, int sample_rate);
// Crête vraie en dBTP d'un enregistrement stéréo entrelacé (suréchantillonnage ×8 par sinc fenêtré).
double true_peak_db(std::span<const float> stereo, int sample_rate);

class AnimeSound {
 public:
  // headphones : la paire devient six enceintes virtuelles (étape 1 rend 6 canaux) ; sinon elle reste en
  // stéréo.
  AnimeSound(const AnimeTuning& tuning, int sample_rate, bool headphones);
  ~AnimeSound();
  AnimeSound(const AnimeSound&) = delete;
  AnimeSound& operator=(const AnimeSound&) = delete;

  // Étape 1 : spatialisation, voix. Entrée stéréo ; sortie stéréo (enceintes) ou 6 canaux FL FR FC LFE SL SR
  // (casque, à rendre par le binaural du mixeur).
  PcmBlock stage(PcmBlock in);
  // Étape 2 : graves, sonie, compresseur, limiteur. Entrée et sortie stéréo, même nombre de trames.
  void finish(PcmBlock& stereo);
  void reset();

  // Le gain de sonie en cours, en dB (diagnostic) et la latence propre au module, en trames.
  float loudness_gain_db() const;
  int latency_frames() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace ajp::audio
