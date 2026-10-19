// Pourcentage affiché pendant la construction d'un moteur TensorRT.
//
// trtexec ne donne que quelques repères (prepare/trtexec_progress.h), et le choix des tactiques,
// l'essentiel du temps, n'en a aucun : une seule couche y prend parfois 40 s, et le repère du plan
// mémoire revient une fois par sous-moteur Myelin. La barre avance donc au temps écoulé rapporté à
// la durée attendue (celle des constructions précédentes du même modèle, mise à l'échelle des
// pixels), ralentit au-delà, ne recule jamais et n'atteint 100 % qu'à la fin. Les repères de
// trtexec servent de plancher.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace ajp::app::play {

// Durée attendue, en secondes, sans construction précédente du modèle.
inline constexpr double kDefaultBuildSeconds = 60.0;

class BuildProgress {
 public:
  BuildProgress(double expected_seconds, std::int64_t started_us);

  // Repère de trtexec (fraction 0 à 1) : plancher de la barre.
  void marker(float fraction);
  void finish() { finished_ = true; }
  // La fraction à afficher à cet instant, 0 à 1 ; 1 seulement une fois finie.
  float fraction(std::int64_t now_us) const;
  double elapsed_seconds(std::int64_t now_us) const;
  double expected_seconds() const { return expected_s_; }

 private:
  double expected_s_;
  std::int64_t started_us_;
  float marker_ = 0.0F;
  bool finished_ = false;
};

// Une construction passée, lue dans un journal de construction du cache (*.build.log : la ligne
// de commande de trtexec, puis « Engine built in 60.2 sec »).
struct PastBuild {
  std::filesystem::path onnx;
  int width = 0;  // forme optimale de l'entrée
  int height = 0;
  std::int64_t pixels = 0;  // hauteur × largeur
  double seconds = 0.0;
};
std::optional<PastBuild> read_build_log(std::string_view text);
// Les constructions lisibles du cache, dans l'ordre du dossier (128 au plus).
std::vector<PastBuild> past_builds(const std::filesystem::path& cache_dir);

// Durée attendue pour construire onnx à pixels (hauteur × largeur de l'entrée) : la construction
// passée du même modèle la plus proche en pixels, allongée en pixels^0,8 pour une entrée plus grande
// (SPAN : 60 s en 720p, 116 s en 1080p sur la 4070 Ti), jamais raccourcie ; kDefaultBuildSeconds
// sans historique.
double expected_build_seconds(const std::filesystem::path& cache_dir, const std::filesystem::path& onnx,
                              std::int64_t pixels);

}  // namespace ajp::app::play
