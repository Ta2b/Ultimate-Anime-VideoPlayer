// « Comment écoutez-vous ? » (docs/audio-spatialisation.md) : le mixage entre ce que décode le lecteur et ce
// que l'appareil de sortie reçoit.
//
//   Casque    : crossfeed pour une source stéréo (le canal opposé, adouci et retardé, comme à l'enceinte),
//   binaural
//               pour une source 5.1 ou 7.1 (chaque haut-parleur placé autour de la tête : retard et ombre de
//               la tête).
//   Enceintes : rendu fidèle ; une source 5.1 ou 7.1 est mixée en stéréo avec le centre rehaussé (les voix).
//   Home cinéma : le multicanal passe tel quel vers l'appareil (libswresample adapte si les canaux
//   diffèrent). Fidèle    : aucun traitement, quel que soit le profil (la bascule A/B).
//
// Aucune HRTF mesurée n'est livrée (licences, poids) : le binaural est un modèle paramétrique de la tête
// (retard interaural de Woodworth, ombre en passe-bas), un seul profil. Une source ou un appareil mono n'est
// jamais spatialisé.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "audio/pcm.h"

namespace ajp::audio {

// Longueur de la ligne à retard, en trames : assez pour 0,65 ms à 96 kHz.
inline constexpr int kMixDelayMax = 64;

enum class Listen : std::uint8_t { Headphones, Speakers, Cinema };

struct MixConfig {
  Listen listen = Listen::Speakers;
  bool faithful = false;  // aucun traitement
  int sink_channels = 2;  // canaux que reçoit l'appareil
  int sample_rate = 48'000;
};

struct MixEar {
  std::array<float, kMixDelayMax> line{};  // ligne à retard circulaire
  int head = 0;
  float low = 0.0F;  // état du passe-bas
};
struct MixPath {  // un haut-parleur vers une oreille
  int delay = 0;
  float gain = 1.0F;
  float alpha = 1.0F;  // coefficient du passe-bas à un pôle (1 : aucun)
  MixEar ear;
};

class Mixer {
 public:
  explicit Mixer(MixConfig config);

  // Les canaux que le décodeur doit rendre pour une source de source_channels canaux : ceux de la source
  // quand le mixeur les traite lui-même (5.1 ou 7.1 vers un appareil stéréo, hors fidèle et hors Home
  // cinéma), sinon ceux de l'appareil.
  int decoder_channels(int source_channels) const;

  // Un bloc aux canaux de decoder_channels, rendu aux canaux de l'appareil.
  PcmBlock process(PcmBlock block);

  // Oublie l'état des filtres (seek).
  void reset();
  void set_faithful(bool faithful);
  bool faithful() const { return config_.faithful; }
  const MixConfig& config() const { return config_; }

 private:
  void build_binaural(int channels);
  PcmBlock downmix(const PcmBlock& in) const;
  void binaural(const PcmBlock& in, PcmBlock& out);
  void crossfeed(PcmBlock& block);

  MixConfig config_;
  int binaural_channels_ = 0;
  std::vector<std::array<MixPath, 2>> paths_;  // par canal source, [oreille gauche, droite]
  std::array<MixPath, 2> feed_;                // crossfeed : L vers R, R vers L
};

}  // namespace ajp::audio
