// Une cascade de visages LBP au format d'OpenCV (« opencv_storage », stageType BOOST, featureType LBP),
// évaluée ici sans OpenCV : celle de lbpcascade_animeface (third_party/lbpcascade_animeface, MIT), entraînée
// sur des visages d'anime. Elle juge la forme (yeux, menton) en niveaux de gris, pas la couleur.
//
// La méthode est celle des cascades à motifs binaires locaux sur blocs (MB-LBP) : dans une fenêtre de 24 ×
// 24, chaque trait compare la somme d'un bloc central à ses huit voisins (un code de 8 bits), et chaque
// classifieur faible donne sa feuille selon que ce code est dans son ensemble de 256 bits. Un étage somme ses
// feuilles et rejette la fenêtre sous son seuil. La fenêtre parcourt l'image à toutes les échelles
// (facteur 1,1), puis les fenêtres retenues se regroupent : un visage en a plusieurs voisines (minNeighbors).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "media/faces.h"

namespace ajp::media {

// La cascade embarquée, octet pour octet (face_cascade_data.cpp, généré).
extern const std::string_view kAnimeFaceCascade;

// Une image en niveaux de gris, 8 bits, rangée ligne par ligne.
struct GrayImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
  std::uint8_t at(int x, int y) const {
    return pixels[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
                  static_cast<std::size_t>(x)];
  }
};

// L'égalisation d'histogramme : les niveaux répartis sur 0 à 255 selon leur rang (comme l'image d'entrée
// de l'entraînement de la cascade).
void equalize(GrayImage& image);
// Réduite à width × height par interpolation bilinéaire, centres des points alignés : la réduction de chaque
// échelle de la recherche, comme pendant l'entraînement de la cascade.
GrayImage resize_linear(const GrayImage& image, int width, int height);

struct CascadeOptions {
  float scale_step = 1.1F;  // d'une échelle à la suivante
  int min_neighbors = 5;    // un visage : plus de min_neighbors fenêtres voisines retenues
  int min_size = 24;        // le plus petit visage, en points de l'image
};

class FaceCascade {
 public:
  // Lue d'un fichier XML d'OpenCV ; rien si le fichier n'est pas une cascade LBP à souches (maxDepth 1)
  // complète.
  static std::optional<FaceCascade> parse(std::string_view xml);
  // La cascade embarquée, lue une fois ; nulle si elle ne se lit pas.
  static const FaceCascade* anime();

  int window_width() const { return width_; }
  int window_height() const { return height_; }
  std::size_t stages() const { return stages_.size(); }

  // Les visages d'une image en niveaux de gris (égalisée ici), en coordonnées normalisées ; le score est la
  // part des voisines (1 à partir de 4 × min_neighbors).
  std::vector<FaceBox> detect(const GrayImage& gray, const CascadeOptions& options = {}) const;

 private:
  struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
  };
  struct Weak {
    int feature = 0;
    std::array<std::int32_t, 8> subset{};
    float left = 0.0F;   // le code est dans l'ensemble
    float right = 0.0F;  // il n'y est pas
  };
  struct Stage {
    float threshold = 0.0F;
    std::size_t first = 0;
    std::size_t count = 0;
  };
  struct Integral;

  static std::optional<Weak> weak_of(std::string_view nodes_text,
                                     std::optional<std::string_view> leaves_text);
  bool read_stage(std::string_view xml, std::size_t& pos);
  bool read_features(std::string_view xml, std::size_t pos);
  bool accepts(const Integral& sum, int x, int y) const;

  int width_ = 0;
  int height_ = 0;
  std::vector<Rect> features_;
  std::vector<Weak> weak_;
  std::vector<Stage> stages_;
};

}  // namespace ajp::media
