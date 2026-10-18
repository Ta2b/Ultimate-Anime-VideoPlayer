// Thread subs : attend sa boîte, passe chaque message au worker, s'arrête sur
// cmd::Quit ou à la demande d'arrêt du std::jthread.
#pragma once

#include <expected>
#include <memory>
#include <thread>

#include "core/types.h"
#include "subs/subs_messages.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_renderer.h"

namespace ajp::subs {

class SubsThread {
 public:
  // Crée la bibliothèque et le moteur libass (fontconfig y lit ses caches),
  // puis lance le thread. Le puits doit survivre au thread. La boîte est
  // partagée : les atlas encore en vol y rentrent tant qu'elle existe, puis
  // sont simplement détruits.
  static std::expected<std::unique_ptr<SubsThread>, Refusal> start(const RendererOptions& options,
                                                                   SubtitleSink& sink,
                                                                   std::shared_ptr<SubsInbox> inbox);

  SubsThread(const SubsThread&) = delete;
  SubsThread& operator=(const SubsThread&) = delete;
  ~SubsThread() = default;  // le std::jthread demande l'arrêt et attend

 private:
  SubsThread() = default;

  std::jthread thread_;
};

}  // namespace ajp::subs
