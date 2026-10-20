// L'écriture d'un export : les images converties par le rendu (YUV 4:2:0 en mémoire hôte), encodées
// par l'encodeur matériel du GPU (encoder_choice.h) et écrites en MKV avec tout ce que la source
// porte d'autre, copié tel quel : pistes audio et de sous-titres, polices jointes, chapitres et
// métadonnées. FFmpeg reste hors de cet en-tête.
//
// Un thread « export » encode et écrit : push() lui confie une image sans la copier (l'encodeur lit
// la mémoire de l'appelant, puis appelle release), finish() vide l'encodeur et clôt le fichier.
// Détruit sans finish(), l'export est abandonné et le fichier partiel supprimé.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

#include "core/frame.h"
#include "core/types.h"
#include "export/encoder_choice.h"

namespace ajp::exporter {

struct WriterConfig {
  std::filesystem::path source;  // le fichier lu : ses autres pistes, chapitres et métadonnées
  std::filesystem::path output;  // .mkv, remplacé s'il existe
  int width = 0;                 // taille encodée, paire
  int height = 0;
  int fps_num = 0;  // fréquence de sortie (source × interpolation) : un repère du contrôle de débit
  int fps_den = 1;
  std::optional<ColorInfo> color;  // celle des images converties ; rien : celle de la source (Muxer)
  Codec codec = Codec::Hevc;
  Quality quality = Quality::VeryHigh;
  GpuVendor vendor = GpuVendor::Other;  // GPU du rendu : l'encodeur essayé d'abord
};

// Une image convertie, dans la disposition de l'encodeur ouvert (layout()). La mémoire reste à
// l'appelant jusqu'à release, appelé une fois quand l'encodeur ne la lit plus, depuis n'importe
// quel thread.
struct Picture {
  std::array<std::uint8_t*, 3> planes{};
  std::array<int, 3> pitches{};  // octets par ligne
  std::size_t bytes = 0;         // étendue de la mémoire, du premier plan à la fin du dernier
  std::int64_t pts_us = 0;       // instant de présentation, depuis le début du fichier (RingFrame)
  std::function<void()> release;
};

struct WriterProgress {
  std::int64_t frames = 0;  // images encodées
  std::int64_t bytes = 0;   // octets écrits
};

class ExportWriter {
 public:
  // Ouvre la source, l'encodeur (le premier candidat qui accepte) et le fichier, écrit l'en-tête et
  // lance le thread. Refus : MediaUnreadable (source), EncoderUnavailable (avec les encodeurs
  // essayés), FileUnwritable, EncodeFailed, ThreadUnavailable.
  static std::expected<std::unique_ptr<ExportWriter>, Refusal> open(const WriterConfig& config);
  ~ExportWriter();
  ExportWriter(const ExportWriter&) = delete;
  ExportWriter& operator=(const ExportWriter&) = delete;

  Layout layout() const { return layout_; }
  const ColorInfo& color() const { return color_; }        // l'étiquette du flux : le rendu code ainsi
  const std::string& encoder() const { return encoder_; }  // « HEVC 10 bits · NVENC »

  // Confie une image au thread, sans attendre. false : file pleine (l'image reste à l'appelant,
  // release n'est pas appelé) ou export arrêté par un refus (failure()).
  bool try_push(Picture&& picture);
  // Toutes les images confiées encodées, l'encodeur vidé, les pistes copiées jusqu'au bout et le
  // fichier clos. Bloque jusque-là. Refus : le premier du thread.
  std::expected<void, Refusal> finish();
  // Le refus qui a arrêté le thread, s'il y en a un.
  std::optional<Refusal> failure() const;
  WriterProgress progress() const;

  struct State;  // FFmpeg : contexte d'encodeur et mux, dans export_writer.cpp

 private:
  ExportWriter() = default;
  void run(const std::stop_token& stop);

  static constexpr std::size_t kQueueDepth = 4;
  std::unique_ptr<State> state_;
  Layout layout_ = Layout::P010;
  ColorInfo color_;
  std::string encoder_;
  std::filesystem::path output_;
  mutable std::mutex mutex_;
  std::condition_variable_any wake_;
  std::deque<Picture> queue_;       // sous mutex_
  bool finishing_ = false;          // sous mutex_ : plus d'image, vider et clore
  bool done_ = false;               // sous mutex_ : le thread a fini (clos, ou refusé)
  std::optional<Refusal> failure_;  // sous mutex_
  std::atomic<std::int64_t> frames_{0};
  std::atomic<std::int64_t> bytes_{0};
  std::jthread thread_;
};

}  // namespace ajp::exporter
