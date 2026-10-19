// Thread export du lecteur : run_export() nourrie par sa boîte Inbox.
//
// Le producteur (render, sur le thread main, qui relit son image hors écran) pousse au plus
// kFramesInFlight images ; l'export rend un Credit pour chaque image dont il n'a plus besoin, et
// le producteur attend ce crédit dans sa propre boîte avant de pousser la suivante. ExportEnd
// termine l'encodage et le mux ; ExportCancel ou l'arrêt du thread l'abandonnent sans sortie.
// Le thread démarre par start_thread() (core/thread.h) : un refus de run_export() part au
// rapporteur du thread, un succès au rappel done.
#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <thread>

#include "core/inbox.h"
#include "core/messages.h"
#include "core/thread.h"
#include "core/types.h"
#include "export/export.h"
#include "export/frame_source.h"

namespace ajp::exporter {

// Une image à encoder ; keep_alive possède ses pixels jusqu'au Credit qui la rend.
struct ExportFrameMsg {
  HostFrame frame;
  std::shared_ptr<const void> keep_alive;
};
struct ExportEnd {};
struct ExportCancel {};

using ExportInbox = Inbox<ExportFrameMsg, ExportEnd, ExportCancel>;

inline constexpr std::size_t kFramesInFlight = 3;

struct ExportThreadHooks {
  std::function<void(Credit)> credit;                   // obligatoire ; appelé sur le thread export
  std::function<void(const ExportProgress&)> progress;  // facultatif
  std::function<void(const ExportSummary&)> done;       // facultatif : l'export a abouti
};

class ExportThread {
 public:
  // job.source et job.stop sont ignorés : les images viennent de la boîte, l'arrêt du thread
  // annule. Refus : ceux de validate_job, FieldMissing (crédit ou rapporteur absents),
  // ThreadUnavailable.
  static std::expected<std::unique_ptr<ExportThread>, Refusal> start(ExportJob job, ExportThreadHooks hooks,
                                                                     ThreadReport report);

  ExportThread(const ExportThread&) = delete;
  ExportThread& operator=(const ExportThread&) = delete;
  ~ExportThread();  // demande l'arrêt et attend la fin du thread

  ExportInbox& inbox() { return *inbox_; }

  // Attend la fin du thread : après ExportEnd, ExportCancel ou un refus.
  void join();

 private:
  ExportThread() = default;

  std::shared_ptr<ExportInbox> inbox_;
  std::jthread thread_;
};

}  // namespace ajp::exporter
