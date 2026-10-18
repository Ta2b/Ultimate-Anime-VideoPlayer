// Outils des tests de subs : script ASS de référence, toile qui compose une
// SubtitleFrame comme render le fera, puits qui enregistre, fichier temporaire.
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "subs/subtitle_frame.h"

namespace ajp::subs::test {

// Script ASS sur 640×360, une ligne par seconde :
//   [0, 1) s : style Default, en bas au centre, blanc bordé de noir ;
//   [1, 2) s : style Top, en haut au centre, jaune, sans bordure ;
//   [2, 3) s : \an7\pos(40,60), rouge, sans bordure ;
//   [3, 4) s : même place, \alpha&H80& ;
//   [4, 5) s : \move(40,60,440,60), blanc, sans bordure.
std::string_view reference_script();

// Script de n lignes distinctes, la ligne i sur [i, i + 1) s.
std::string numbered_script(int lines);

inline constexpr int kScriptWidth = 640;
inline constexpr int kScriptHeight = 360;

struct Box {
  int x0 = 0;
  int y0 = 0;
  int x1 = 0;  // exclus
  int y1 = 0;
  bool empty() const { return x1 <= x0 || y1 <= y0; }
};

// L'image de sortie après dessin des sous-titres sur fond noir.
struct Canvas {
  int width = 0;
  int height = 0;
  std::vector<float> opacity;             // 1 − Π(1 − o)
  std::vector<std::array<float, 3>> rgb;  // 0 à 1

  float opacity_at(int x, int y) const;
  const std::array<float, 3>& rgb_at(int x, int y) const;
  // Plus petite boîte des pixels d'opacité au-dessus du seuil.
  Box covered(float threshold = 0.1F) const;
};

Canvas draw(const SubtitleFrame& frame);

// Vérifie l'atlas : rectangles dedans, couverture non nulle dans au moins un,
// et rien hors des rectangles.
void check_atlas(const SubtitleFrame& frame);

// Police TrueType du système (DejaVu Sans), lue en entier ; vide si absente.
std::vector<std::byte> system_font_bytes();

class RecordingSink final : public SubtitleSink {
 public:
  bool try_push(SubtitleFrame&& frame) override;
  void refused(const Refusal& refusal) override;

  // Attend au moins count réponses ; false après le délai.
  bool wait_for_frames(std::size_t count, std::chrono::milliseconds timeout);
  std::size_t frame_count() const;
  std::vector<SubtitleFrame> take_frames();
  std::vector<Refusal> refusals() const;

 private:
  mutable std::mutex mutex_;
  std::condition_variable arrived_;
  std::vector<SubtitleFrame> frames_;
  std::vector<Refusal> refusals_;
};

class TempFile {
 public:
  TempFile(std::string_view name, std::string_view contents);
  ~TempFile();
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace ajp::subs::test
