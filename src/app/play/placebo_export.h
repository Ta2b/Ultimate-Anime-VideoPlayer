// L'export vu par l'écran : chaque image du ring rendue par libplacebo à la taille de sortie, dans la
// disposition de l'encodeur (YUV 4:2:0, 10 bits en P010 ou en plans séparés), avec ses meilleurs
// filtres de mise à l'échelle, puis relue sur l'hôte sans attendre, dans des tampons que l'encodeur
// lit tels quels. Ni grade, ni HDR, ni sous-titres, ni interface : l'image du ring telle que la chaîne
// l'a faite, au moins aussi fine qu'à l'écran.
//
// Le slot repart au thread gpu dès le rendu soumis : la relecture lit les textures de sortie, pas le
// ring. Les tampons tournent : rendu et relecture → prêt → lu par l'encodeur → libre.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <optional>

#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

#include "app/play/placebo_screen.h"
#include "core/frame.h"
#include "core/ring.h"
#include "core/types.h"
#include "export/export_writer.h"

namespace ajp::app::play {

struct FrameExportSetup {
  int width = 0;  // taille encodée, paire
  int height = 0;
  exporter::Layout layout = exporter::Layout::P010;
  ColorInfo color;  // celle des images du ring, gardée telle quelle
};

class FrameExport {
 public:
  static constexpr std::size_t kBuffers = 6;

  // Refus : format de texture absent ou sans rendu ni relecture (FeatureUnavailable), mémoire.
  static std::expected<std::unique_ptr<FrameExport>, Refusal> open(PlaceboScreen& screen,
                                                                   const FrameExportSetup& setup);
  // Attend les relectures en cours. Les tampons encore lus par l'encodeur doivent être rendus avant :
  // l'écriture (ExportWriter) part la première.
  ~FrameExport();
  FrameExport(const FrameExport&) = delete;
  FrameExport& operator=(const FrameExport&) = delete;

  bool has_room() const;  // un tampon libre : convert() peut suivre
  // Rend l'image du ring dans un tampon libre et lance sa relecture. Le slot est rendu dans tous les
  // cas : freed, à poster au thread gpu. failure : VendorCallFailed, ou SequenceOrder sans tampon libre.
  struct Converted {
    SlotFree freed;
    std::optional<Refusal> failure;
  };
  Converted convert(const RingFrame& frame);
  // La plus ancienne image relue, dans l'ordre des convert(), confiée à push ; si push la refuse,
  // elle reste prête. false : rien de prêt, ou refusée.
  bool deliver(const std::function<bool(exporter::Picture&&)>& push);
  bool idle() const { return order_.empty(); }  // aucune image en relecture ni prête

 private:
  struct Buffer {
    std::uint8_t* data = nullptr;  // aligné sur 4 Kio : la relecture peut l'importer sans copie
    int planes_done = 0;           // relectures de plans finies (rappels de libplacebo)
    bool in_flight = false;        // en relecture ou prêt
    std::int64_t pts_us = 0;
    std::atomic<bool> held{false};  // lu par l'encodeur, rendu par Picture::release
  };
  FrameExport(PlaceboScreen& screen, const FrameExportSetup& setup);
  bool ready(Buffer& buffer);

  PlaceboScreen& screen_;
  FrameExportSetup setup_;
  pl_renderer renderer_ = nullptr;
  std::array<pl_tex, 3> planes_{};
  int plane_count_ = 0;
  std::array<std::size_t, 3> offsets_{};
  std::array<std::size_t, 3> pitches_{};
  std::size_t bytes_ = 0;
  bool async_ = true;  // relecture par rappel ; sinon bloquante (GPU sans rappels)
  std::array<Buffer, kBuffers> buffers_;
  std::deque<std::size_t> order_;  // tampons en relecture ou prêts, dans l'ordre des convert()
};

}  // namespace ajp::app::play
