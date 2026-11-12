// Les messages de FFmpeg dans le journal (A39-05) : sans cela ils partent sur stderr, perdus sous Windows
// (exécutable fenêtré) et absents du rapport. Seules les erreurs passent (blocs verts, HEVC abîmé, refus de
// NVDEC) ; une même ligne répétée est comptée, pas recopiée.
#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace ajp::media {

// À appeler une fois, après l'installation des puits du journal.
void route_ffmpeg_log_to_journal();

// Ce qui s'écrit des lignes d'erreur de FFmpeg (B15-08). Deux lignes qui ne diffèrent que par leurs nombres
// (« Invalid NAL unit size (123 > 45) ») sont la même ; et au plus kMaxLinesPerSecond lignes neuves par
// seconde sortent, le reste est compté. Un flux abîmé à 24 i/s n'écrit plus 24 lignes par seconde sur le
// thread qui décode. Sans verrou : l'appelant garde le sien.
class ErrorFolder {
 public:
  static constexpr int kMaxLinesPerSecond = 5;

  // Les lignes du journal pour `line` reçue à l'instant `now` : rien, la ligne, ou le compte des répétitions
  // d'avant puis la ligne.
  std::vector<std::string> push(const std::string& line, std::chrono::steady_clock::time_point now);

 private:
  std::string last_;        // la ligne précédente, nombres remplacés
  int repeats_ = 0;         // fois qu'elle s'est répétée après la première
  bool last_shown_ = true;  // la précédente est sortie ; sinon ses répétitions sont des lignes omises
  int omitted_ = 0;         // lignes tues par le plafond
  std::chrono::steady_clock::time_point window_start_{};
  int window_lines_ = 0;
};

}  // namespace ajp::media
