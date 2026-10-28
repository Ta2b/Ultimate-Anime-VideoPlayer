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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "audio/pcm.h"

namespace ajp::audio {

// Longueur de la ligne à retard, en trames : assez pour 0,65 ms à 96 kHz.
inline constexpr int kMixDelayMax = 64;

enum class Listen : std::uint8_t { Headphones, Speakers, Cinema };

// Les réglages fins du mixage ; les défauts sont ceux du profil « Naturel » (le mixage d'avant les profils).
struct MixTuning {
  float crossfeed_gain = 0.32F;        // casque, source stéréo : part du canal opposé (0 à 0,7)
  float crossfeed_cutoff_hz = 700.0F;  // adoucissement du canal opposé
  float crossfeed_delay_ms = 0.25F;    // son retard (0 à 0,6)
  float head_radius_m = 0.0875F;       // casque, source 5.1 ou 7.1 : taille de la tête (retard interaural)
  float head_shadow = 0.35F;           // atténuation de l'oreille éloignée (0 à 0,7)
  float surround_gain = 1.0F;          // niveau des canaux arrière et latéraux (0,5 à 1,6)
  float center_lift = 1.26F;           // enceintes : niveau du centre, les voix (1 à 2)
  float lfe_gain = 0.3F;               // niveau des basses (0 à 0,6)
  float downmix_scale = 0.6F;          // enceintes : gain du mixage stéréo (0,4 à 0,9)
  float distance = 0.0F;  // enceintes virtuelles plus loin : niveau direct baissé (jusqu'à -2 dB), pièce plus
                          // présente (0 à 1)
  float room = 0.0F;          // réverbération légère d'une pièce (0 à 1 ; 0 : aucune, aucun coût)
  float eq_bass_db = 0.0F;    // égaliseur : graves (étagère à 120 Hz), -8 à +8 dB
  float eq_voice_db = 0.0F;   // médiums, les voix (cloche à 2,5 kHz), -8 à +8 dB
  float eq_treble_db = 0.0F;  // aigus (étagère à 8 kHz), -8 à +8 dB
  float night = 0.0F;  // compression de dynamique « nuit » : explosions réduites, murmures relevés (0 à 1)
  bool operator==(const MixTuning&) const = default;
};
inline constexpr std::size_t kMixTuningFields = 15;
// Le tuple des 9 réglages, dans l'ordre des champs ci-dessus (le message cmd::SetListening porte ce tableau).
std::array<float, kMixTuningFields> to_array(const MixTuning& tuning);
MixTuning tuning_from(const std::array<float, kMixTuningFields>& fields);
// Les mêmes réglages dans le tableau du message cmd::SetListening (16 cases, les non utilisées à 0), et
// retour.
std::array<float, 16> to_wire(const MixTuning& tuning);
MixTuning tuning_from_wire(const std::array<float, 16>& wire);
// Les valeurs ramenées dans leurs plages.
MixTuning clamped(MixTuning tuning);

// Un profil de virtualisation nommé : ce que l'on choisit au lieu de régler à la main.
struct MixPreset {
  std::string_view key;  // « natural », « soft », « strong », « wide », « voices », « night »
  std::string_view label;
  std::string_view detail;
  bool headphones;  // proposé au casque
  bool speakers;    // proposé aux enceintes
  MixTuning tuning;
};
std::span<const MixPreset> mix_presets();
const MixPreset* find_preset(std::string_view key);

struct MixConfig {
  Listen listen = Listen::Speakers;
  MixTuning tuning;
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
  PcmBlock process_core(PcmBlock block);

  MixConfig config_;
  int binaural_channels_ = 0;
  std::vector<std::array<MixPath, 2>> paths_;  // par canal source, [oreille gauche, droite]
  std::array<MixPath, 2> feed_;                // crossfeed : L vers R, R vers L
  std::shared_ptr<struct MixPost>
      post_;  // égaliseur, pièce et compression (stéréo) ; absent si rien n'est demandé
};

}  // namespace ajp::audio
