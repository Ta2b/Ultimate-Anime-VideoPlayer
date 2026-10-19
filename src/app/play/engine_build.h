// Construction du moteur SPAN en arrière-plan (trtexec, une minute la première fois, puis lu
// dans le cache) sur une instance de backend à elle : main garde la fenêtre vivante et
// affiche la progression pendant ce temps.
//
// Provisoire : le thread preparer (prepare/preparer.h), branché par gpu, en prendra la place.
#pragma once

#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "app/play/backend_library.h"
#include "core/types.h"
#include "gpu/backend.h"

namespace ajp::app::play {

struct EngineRequest {
  std::shared_ptr<BackendLibrary> library;
  gpu::BackendInit init;  // render, cache_dir et log ; render vit plus longtemps que la construction
  gpu::ModelSpec model;
  gpu::ShapeSet shapes;
};

class EngineBuild {
 public:
  static std::expected<std::unique_ptr<EngineBuild>, Refusal> start(EngineRequest request);
  ~EngineBuild();  // demande l'arrêt : trtexec est interrompu, la construction refusée (Cancelled)
  EngineBuild(const EngineBuild&) = delete;
  EngineBuild& operator=(const EngineBuild&) = delete;

  // Le résultat, une fois la construction finie ; rien avant.
  std::optional<std::expected<gpu::Artifact, Refusal>> result();
  // « 37 % : Building engine », pour l'OSD.
  std::string progress() const;

 private:
  EngineBuild() = default;
  void run(const std::stop_token& stop, EngineRequest request);

  mutable std::mutex mutex_;
  std::optional<std::expected<gpu::Artifact, Refusal>> result_;
  float fraction_ = 0.0F;
  std::string stage_ = "démarrage";
  std::jthread thread_;  // dernier membre
};

}  // namespace ajp::app::play
