// L'écriture d'un export : les images converties par le rendu (YUV 4:2:0 en mémoire hôte, ou en
// mémoire CUDA pour NVENC), encodées par l'encodeur matériel du GPU (encoder_choice.h) et écrites
// en MKV avec tout ce que la source porte d'autre, copié tel quel : pistes audio et de sous-titres,
// polices jointes, chapitres et métadonnées. FFmpeg reste hors de cet en-tête.
//
// Un thread « export » encode et écrit : push() lui confie une image sans la copier (l'encodeur lit
// la mémoire de l'appelant, puis appelle release), finish() vide l'encodeur et clôt le fichier.
// Détruit sans finish(), l'export est abandonné et le fichier partiel supprimé. À cadence fixe, il
// garde une référence sur la dernière image encodée, qu'il répète aux instants manquants : une
// image de l'appelant de plus tenue à la fois.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
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

struct AVBufferRef;
struct AVFrame;

namespace ajp::exporter {

class FrameEncoder;
class Muxer;

// Profondeur de la file d'écriture par défaut (WriterConfig::queue_depth).
inline constexpr std::size_t kDefaultQueueDepth = 4;

struct WriterConfig {
  std::filesystem::path source;  // le fichier lu : ses autres pistes, chapitres et métadonnées
  std::filesystem::path output;  // .mkv, .mp4 ou .mov (export/container.h), remplacé s'il existe
  int width = 0;                 // taille encodée, paire
  int height = 0;
  int fps_num = 0;  // fréquence de sortie (source × interpolation) : un repère du contrôle de débit
  int fps_den = 1;
  std::optional<ColorInfo> color;  // celle des images converties ; rien : celle de la source (Muxer)
  Codec codec = Codec::Hevc;
  Quality quality = Quality::VeryHigh;
  GpuVendor vendor = GpuVendor::Other;  // GPU du rendu : l'encodeur essayé d'abord
  int cuda_ordinal = -1;                // son ordinal CUDA, pour NVENC (EncoderSetup) ; -1 : inconnu
  // L'appareil Vulkan de l'écran (AVHWDeviceContext, emprunté) : l'encodeur Vulkan Video y reçoit
  // des images rendues sur le GPU (vulkan_frames()) au lieu d'en créer un autre et de charger
  // chaque image depuis l'hôte. nullptr : son propre appareil.
  AVBufferRef* vulkan_device = nullptr;
  // Images confiées et pas encore encodées : try_push refuse au-delà. 8 au profil « débit » de
  // l'export (app/play/export_depth.h).
  std::size_t queue_depth = kDefaultQueueDepth;
  // Images de l'encodeur (cuda_frames()) que l'appelant tient à la fois, rendues et pas encore
  // confiées (FrameExport : ses tampons) : le pool CUDA les compte en plus de la file.
  std::size_t caller_frames = 6;
  // Cadence fixe : l'image précédente répétée aux instants de la grille que la chaîne ne produit
  // pas (coupe de scène, dessin tenu : la paire n'est pas interpolée), pour un fichier que les
  // logiciels de montage lisent à fps_num / fps_den. Faux : ces instants manquent, cadence variable.
  bool constant_rate = true;
};

// Plus longue suite de répétitions comblée (cinq minutes à 120 i/s) : un écart plus grand est un
// saut d'horodatage de la source, pas une image tenue, et reste tel quel.
inline constexpr std::int64_t kMaxRepeatRun = 36'000;

// Cadence fixe : combien de fois l'image de previous_us se répète avant celle de next_us, pour
// que l'écart, partagé en parts égales, approche la période 10⁶ × fps_den / fps_num µs. Arrondi au
// plus proche : un écart de moins d'une période et demie ne répète rien. La k-ième répétition
// tombe à previous_us + k × (next_us − previous_us) / (n + 1).
constexpr std::int64_t repeats_between(std::int64_t previous_us, std::int64_t next_us, int fps_num,
                                       int fps_den) {
  if (fps_num <= 0 || fps_den <= 0 || next_us <= previous_us) return 0;
  constexpr std::int64_t kUs = 1'000'000;
  const std::int64_t gap = next_us - previous_us;
  const std::int64_t periods = (2 * gap * fps_num + kUs * fps_den) / (2 * kUs * fps_den);
  const std::int64_t repeats = periods - 1;
  return repeats > 0 && repeats <= kMaxRepeatRun ? repeats : 0;
}

// Une image convertie, dans la disposition de l'encodeur ouvert (layout()). La mémoire reste à
// l'appelant jusqu'à release, appelé une fois quand l'encodeur ne la lit plus, depuis n'importe
// quel thread. En mémoire hôte ; ou, si l'encodeur lit la mémoire CUDA (upload() Cuda), des
// adresses CUDA de son contexte, les plans à la suite (chroma après la luma, même pas), que
// l'encodeur lit sans copie.
struct Picture {
  std::array<std::uint8_t*, 3> planes{};
  std::array<int, 3> pitches{};  // octets par ligne
  std::size_t bytes = 0;         // étendue de la mémoire, du premier plan à la fin du dernier
  std::int64_t pts_us = 0;       // instant de présentation, depuis le début du fichier (RingFrame)
  bool device = false;           // planes : adresses CUDA (upload() Cuda), pas de l'hôte
  // Ou une image de l'encodeur (vulkan_frames(), cuda_frames()), déjà rendue ou copiée sur le
  // GPU : l'écriture la prend et la libère ; planes, bytes et release ne comptent pas.
  AVFrame* frame = nullptr;
  std::function<void()> release;
};

struct WriterProgress {
  std::int64_t frames = 0;    // images encodées, répétitions comprises
  std::int64_t repeated = 0;  // dont répétitions de la cadence fixe
  std::int64_t bytes = 0;     // octets écrits
};

class ExportWriter {
 public:
  // Ouvre la source, l'encodeur (le premier candidat qui accepte) et le fichier, écrit l'en-tête et
  // lance le thread. Refus : MediaUnreadable (source), EncoderUnavailable (avec les encodeurs
  // essayés), FileUnwritable, EncodeFailed, ThreadUnavailable.
  static std::expected<std::unique_ptr<ExportWriter>, Refusal> open(const WriterConfig& config);
  // De même avec un encodeur déjà ouvert à la place de celui de FFmpeg (video_encoder.h) : les
  // tests de l'écriture lui donnent un encodeur factice qui garde N images (export_writer_test.cpp).
  static std::expected<std::unique_ptr<ExportWriter>, Refusal> open(const WriterConfig& config,
                                                                    std::unique_ptr<FrameEncoder> encoder);
  ~ExportWriter();
  ExportWriter(const ExportWriter&) = delete;
  ExportWriter& operator=(const ExportWriter&) = delete;

  Layout layout() const { return layout_; }
  Upload upload() const { return upload_; }  // Cuda : les Picture peuvent être en mémoire CUDA
  // Le contexte d'images de l'encodeur Vulkan Video sur l'appareil partagé (WriterConfig), emprunté :
  // les Picture peuvent être ses images (Picture::frame). nullptr : autre encodeur, ou son propre
  // appareil.
  AVBufferRef* vulkan_frames() const;
  // Le contexte d'images CUDA de l'encodeur (NVENC), emprunté : les Picture peuvent être ses images
  // (Picture::frame), copiées sur le GPU. nullptr : autre encodeur.
  AVBufferRef* cuda_frames() const;
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
  // Après un finish() réussi : non vide si la lecture des pistes copiées (audio, sous-titres) a été
  // interrompue par une erreur de la source. Le fichier est écrit mais ces pistes s'arrêtent avant la fin.
  std::string warning() const;

  struct State;  // FFmpeg : contexte d'encodeur et mux, dans export_writer.cpp

 private:
  ExportWriter() = default;
  // Le fichier commencé (en-tête écrit) et le thread lancé, sur une source ouverte et un encodeur.
  static std::expected<std::unique_ptr<ExportWriter>, Refusal> start(const WriterConfig& config,
                                                                     Muxer&& muxer, const ColorInfo& color,
                                                                     std::unique_ptr<FrameEncoder> encoder);
  void run(const std::stop_token& stop);

  std::size_t queue_depth_ = kDefaultQueueDepth;
  std::unique_ptr<State> state_;
  Layout layout_ = Layout::P010;
  Upload upload_ = Upload::Host;
  ColorInfo color_;
  std::string encoder_;
  std::filesystem::path output_;
  mutable std::mutex mutex_;
  std::condition_variable_any wake_;
  std::deque<Picture> queue_;       // sous mutex_
  bool finishing_ = false;          // sous mutex_ : plus d'image, vider et clore
  bool done_ = false;               // sous mutex_ : le thread a fini (clos, ou refusé)
  std::optional<Refusal> failure_;  // sous mutex_
  std::string warning_;             // sous mutex_
  std::atomic<std::int64_t> frames_{0};
  std::atomic<std::int64_t> repeated_{0};
  std::atomic<std::int64_t> bytes_{0};
  std::jthread thread_;
};

}  // namespace ajp::exporter
