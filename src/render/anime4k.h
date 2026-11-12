// Anime4K (third_party/anime4k, MIT) : un agrandissement ×2 en shaders de mpv, lus par libplacebo comme les
// hooks du grade, sans moteur de réseau. C'est le plancher des petites cartes (iGPU, cartes sans unités
// matricielles) : il ne demande ni TensorRT ni Vulkan de calcul, seulement le rendu de la présentation.
//
// Les shaders se posent sur MAIN, avant le filtre qui étire vers la fenêtre, dans l'ordre de la chaîne
// « Mode X » d'Anime4K pour mpv : le plafonnement des hautes lumières (sa mesure sur MAIN, son plafond sur
// PREKERNEL), la restauration (A : Restore, B : Restore Soft ; C : aucune), puis un seul ×2 (Upscale, ou
// Upscale Denoise en C). Le second ×2 de mpv (et ses AutoDownscalePre) n'est pas repris : comme SPAN, Anime4K
// double l'image de travail, et libplacebo l'ajuste à la fenêtre. Le ×2 ne tourne que si la fenêtre est au
// moins 1,2 fois plus grande que l'image (sa condition //!WHEN) : sur une fenêtre plus petite, il serait
// réduit aussitôt.
#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <libplacebo/shaders/custom.h>

#include "core/types.h"

namespace ajp::render {

// La taille des réseaux : M (Rapide, le jeu « lower-end GPU » d'Anime4K) ou VL (Qualité, le jeu « higher-end
// GPU »). Chaque cran de taille double à peu près le temps (S, M, L, VL : ×2 à chaque pas, selon Anime4K).
enum class Anime4kSize : std::uint8_t { Fast, Quality };
// Les modes d'Anime4K (md/GLSL_Instructions_Advanced.md) : A pour une image floue et rééchantillonnée (la
// plupart des anime 1080p, les vieux SD) ; B pour peu de flou et des défauts de réduction (la plupart des
// 720p, un 1080p réduit en 720p) ; C pour une image sans dégradation (un 1080p réduit en 480p).
enum class Anime4kMode : std::uint8_t { A, B, C };

struct Anime4kChoice {
  Anime4kSize size = Anime4kSize::Fast;
  Anime4kMode mode = Anime4kMode::A;
  friend bool operator==(const Anime4kChoice&, const Anime4kChoice&) = default;
};

std::string_view name(Anime4kMode mode);  // « A », « B », « C »
std::string_view name(Anime4kSize size);  // « M », « VL » : la taille des réseaux d'Anime4K

// Le mode qui va à l'image de travail : une source réduite d'au moins 1,75 fois (1080p vers 540p ou 480p)
// arrive nette et sans défaut, C ; réduite moins (1080p vers 720p), B. Une source gardée telle quelle : B
// pour une 720p (de 600 à 899 lignes), A pour une 1080p et plus comme pour les SD sous 600 lignes. Hauteurs
// inconnues (0) : A.
Anime4kMode anime4k_mode(int source_height, int work_height);

// Les textes des shaders d'un choix, dans leur ordre de passage (pour la construction et les tests).
std::vector<std::string_view> anime4k_texts(Anime4kChoice choice);

class Anime4k {
 public:
  // VendorCallFailed : libplacebo refuse un des shaders (le nom du shader dans le détail).
  static std::expected<Anime4k, Refusal> create(pl_gpu gpu, Anime4kChoice choice);

  Anime4k(Anime4k&& other) noexcept;
  Anime4k& operator=(Anime4k&& other) noexcept;
  Anime4k(const Anime4k&) = delete;
  Anime4k& operator=(const Anime4k&) = delete;
  ~Anime4k();

  Anime4kChoice choice() const { return choice_; }
  // Les hooks à poser devant les autres (netteté, grade), dans cet ordre.
  std::span<const pl_hook* const> hooks() const { return hooks_; }

 private:
  Anime4k(Anime4kChoice choice, std::vector<const pl_hook*> hooks)
      : choice_(choice), hooks_(std::move(hooks)) {}
  void destroy();
  Anime4kChoice choice_;
  std::vector<const pl_hook*> hooks_;
};

// La passe est-elle d'Anime4K ? Ses descriptions portent toutes « Anime4K- » : le temps GPU de chaque
// image se compte à part, comme celui du grade.
bool is_anime4k_pass(std::string_view description);

}  // namespace ajp::render
