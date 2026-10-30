// Convolution par réponses impulsionnelles « à la HeSuVi » (docs/audio-spatialisation.md) : un fichier WAV
// de réponses (une par haut-parleur et par oreille) remplace le binaural paramétrique du mixeur pour l'écoute
// Casque. Le lecteur ne livre aucune réponse : l'utilisateur choisit son fichier (voir les licences).
//
// Formats lus (PCM 16 ou 24 bits, ou float 32) :
//   14 canaux  HeSuVi, dans l'ordre documenté : FL→G, FL→D, SL→G, SL→D, BL→G, BL→D, FC→G, FR→D, FR→G, SR→D,
//              SR→G, BR→D, BR→G, FC→D (G, D : oreille gauche, droite ; pas de LFE).
//   16 canaux  huit haut-parleurs dans l'ordre du mixeur (FL FR FC LFE BL BR SL SR), chacun oreille gauche
//              puis droite.
//   2 canaux   HRIR stéréo : la réponse des deux oreilles pour une seule source (l'enceinte gauche) ; la
//              droite en est le miroir, le centre la moyenne.
// Rien de plus long que 32 768 échantillons par réponse. Le fichier est lu une fois à son taux ; prepared()
// le ramène au taux de sortie et le normalise.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "core/types.h"

namespace ajp::audio {

inline constexpr int kHrirMaxTaps = 32'768;       // échantillons par réponse
inline constexpr int kHrirBlock = 512;            // trames par bloc de convolution (latence : un bloc)
inline constexpr std::size_t kHrirPositions = 8;  // FL FR FC LFE BL BR SL SR (l'ordre du mixeur à 7.1)
inline constexpr float kHrirPeak = 0.9F;          // plafond après convolution d'un signal plein-échelle
// Le niveau des réponses de l'utilisateur est respecté comme dans HeSuVi, PipeWire ou JamesDSP (aucun gain) ;
// seule une réponse démesurée est ramenée ici. Le mixeur écrête doucement la sortie.
inline constexpr float kHrirUserCeiling = 8.0F;

// Les réponses d'un haut-parleur vers chaque oreille ; vide : ce haut-parleur n'a pas de réponse.
struct Hrir {
  int sample_rate = 48'000;
  std::array<std::array<std::vector<float>, 2>, kHrirPositions> taps;  // [position][oreille 0 G, 1 D]
  // Le plus long des filtres.
  std::size_t length() const;
  bool empty() const { return length() == 0; }
};

// Lit un WAV déjà en mémoire. Refus : FormatUnsupported (pas du WAV, format de PCM ou nombre de canaux
// inconnus), FileUnreadable (fichier tronqué ou vide), FieldOutOfRange (plus de 32 768 échantillons).
std::expected<Hrir, Refusal> parse_hrir(std::span<const std::byte> wav);
// Lit le fichier (FileUnreadable s'il est absent ou illisible, ou trop gros pour être une réponse).
std::expected<std::shared_ptr<const Hrir>, Refusal> load_hrir(const std::filesystem::path& path);

// Rééchantillonnage d'une réponse (sinc fenêtré, coupure sous le plus bas des deux taux), tronquée à
// kHrirMaxTaps.
std::vector<float> resample(std::span<const float> in, int from_rate, int to_rate);
// Le gain de sécurité : le plus fort des filtres est borné à kHrirPeak par la somme de ses valeurs absolues
// (le pire signal plein-échelle sur un seul haut-parleur ne dépasse alors pas kHrirPeak) ; jamais de gain
// au-dessus de 1. Rend le gain appliqué.
float normalize(Hrir& hrir, float ceiling = kHrirPeak);
// La réponse au taux de sortie, puis normalisée.
Hrir prepared(const Hrir& hrir, int sample_rate);

// La FFT radix-2 en place (taille : puissance de deux), exposée pour les tests.
struct Cpx {
  float re = 0.0F;
  float im = 0.0F;
};
class Fft {
 public:
  explicit Fft(std::size_t size);
  std::size_t size() const { return size_; }
  void forward(std::span<Cpx> data) const;
  // Inverse, normalisée par 1/N.
  void inverse(std::span<Cpx> data) const;

 private:
  void run(std::span<Cpx> data, bool inverse) const;
  std::size_t size_;
  std::vector<std::size_t> reverse_;
  std::vector<Cpx> twiddle_;  // exp(-2πik/N), k < N/2
};

// La convolution partitionnée uniforme (overlap-save, blocs de kHrirBlock trames, FFT de 2 blocs), un état
// par canal source. Latence du flux : un bloc (10,7 ms à 48 kHz). Les canaux source sont 2 (FL FR), 6 (FL FR
// FC LFE SL SR) ou 8 (FL FR FC LFE BL BR SL SR) ; la sortie est stéréo.
class HrirConvolver {
 public:
  // hrir est déjà au taux de sortie (prepared). lfe_gain : le LFE sans réponse passe directement aux deux
  // oreilles (comme le binaural paramétrique).
  HrirConvolver(const Hrir& hrir, int source_channels, float lfe_gain);

  // Un flux de trames de n'importe quelle taille : rend autant de trames stéréo que reçues, retardées d'un
  // bloc (le début est du silence).
  void process(std::span<const float> in, std::vector<float>& out);
  // Un seul bloc entier : kHrirBlock trames entrelacées vers kHrirBlock trames stéréo, sans retard ajouté.
  void process_block(std::span<const float> in, std::span<float> out);
  void reset();
  int source_channels() const { return channels_; }

 private:
  using Spectrum = std::vector<Cpx>;
  struct Filter {
    std::vector<Spectrum> parts;  // vide : aucun filtre
  };
  int channels_;
  std::size_t partitions_ = 1;
  Fft fft_;
  std::vector<std::array<Filter, 2>> filters_;  // [canal][oreille]
  std::vector<std::vector<Spectrum>> history_;  // [canal][anneau de partitions_ spectres]
  std::vector<std::vector<float>> previous_;    // [canal] le bloc précédent
  std::size_t head_ = 0;
  std::vector<Cpx> work_;
  std::vector<Cpx> accumulator_;
  std::vector<float> pending_in_;  // trames reçues, pas encore un bloc
  std::vector<float> ready_out_;   // trames stéréo prêtes
};

}  // namespace ajp::audio
