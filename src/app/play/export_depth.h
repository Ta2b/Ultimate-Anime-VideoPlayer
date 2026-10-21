// Profondeur de l'export : combien d'images le ring, la conversion et la file d'écriture tiennent à
// la fois. Sans affichage, la latence ne compte pas ; avec plusieurs voies de calcul (gpu/lanes.h,
// cartes d'au moins 7,5 Gio), le profil « débit » garde deux paires en vol pour que les voies ne
// se vident jamais : 8 slots du ring au lieu de 4, 10 tampons de conversion au lieu de 6, une file
// d'écriture de 8 au lieu de 4. Le pool CUDA de NVENC grandit d'autant (WriterConfig) : environ
// 230 Mo de plus en 4K, d'où la profondeur d'avant sous 7,5 Gio.
#pragma once

#include <cstddef>
#include <cstdint>

#include "app/play/links.h"
#include "app/play/placebo_export.h"
#include "export/export_writer.h"

namespace ajp::app::play {

struct ExportDepth {
  std::uint32_t ring_slots = 4;                      // slots du ring de l'étage gpu
  std::size_t buffers = 6;                           // FrameExportSetup::buffers
  std::size_t queue = exporter::kDefaultQueueDepth;  // WriterConfig::queue_depth
  bool throughput = false;
};

constexpr ExportDepth export_depth(int lanes) {
  if (lanes <= 1) return {};
  return ExportDepth{.ring_slots = 8, .buffers = 10, .queue = 8, .throughput = true};
}

// La file d'écriture se remplit seulement si la conversion a plus de tampons qu'elle (en mémoire
// hôte, une image confiée garde son tampon) ; le ring tient au moins deux paires ×2.
static_assert(export_depth(3).buffers > export_depth(3).queue);
static_assert(export_depth(1).buffers > export_depth(1).queue);
static_assert(export_depth(3).buffers <= FrameExport::kMaxBuffers);
static_assert(export_depth(3).ring_slots <= kMaxRingSlots);

}  // namespace ajp::app::play
