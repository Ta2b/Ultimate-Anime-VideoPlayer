// Construction du moteur SPAN en arrière-plan (trtexec, une minute la première fois, puis lu
// dans le cache) sur une instance de backend à elle : main garde la fenêtre vivante et
// affiche la progression pendant ce temps.
//
// Provisoire : le thread preparer (prepare/preparer.h), branché par gpu, en prendra la place.
#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "app/play/backend_library.h"
#include "app/play/build_backend.h"
#include "core/types.h"
#include "gpu/backend.h"

namespace ajp::app::play {

struct EngineRequest {
  std::shared_ptr<BackendLibrary> library;
  gpu::BackendInit init;  // render, cache_dir et log ; render vit plus longtemps que la construction
  gpu::ModelSpec model;
  gpu::ShapeSet shapes;
  // L'instance de la préparation (build_backend.h) ; nulle : une instance à cette construction.
  std::shared_ptr<BuildBackend> builder;
  // Détruite depuis le thread de la fenêtre : la fermeture pompe ses événements en attendant la fin (A05-10).
  bool pump_on_close = false;
};

// A31-11 : une construction refusée à l'écriture (BuilderFailed, CacheIo, FileUnwritable, IoFailed) alors que
// le dossier du cache a moins de kBuildSpaceBytes libres devient InsufficientSpace : « disque plein » à
// l'écran au lieu d'une erreur de fichier ou de construction.
inline constexpr std::uintmax_t kBuildSpaceBytes = 256ULL * 1024 * 1024;
std::expected<gpu::Artifact, Refusal> with_space_reason(std::expected<gpu::Artifact, Refusal> outcome,
                                                        const std::filesystem::path& cache_dir,
                                                        std::uintmax_t needed = kBuildSpaceBytes);

class EngineBuild {
 public:
  static std::expected<std::unique_ptr<EngineBuild>, Refusal> start(EngineRequest request);
  // Demande l'arrêt : trtexec est interrompu, la construction refusée (Cancelled). Avec pump_on_close, la
  // fenêtre est pompée jusqu'à la fin du thread (lecture de l'ONNX, désérialisation de contrôle).
  ~EngineBuild();
  EngineBuild(const EngineBuild&) = delete;
  EngineBuild& operator=(const EngineBuild&) = delete;

  // Le résultat, une fois la construction finie ; rien avant.
  std::optional<std::expected<gpu::Artifact, Refusal>> result();
  // « 37 % : Building engine », pour l'OSD.
  std::string progress() const;
  // Le dernier repère de trtexec : fraction et étape.
  struct Status {
    float fraction = 0.0F;
    std::string stage;
  };
  Status status() const;

 private:
  EngineBuild() = default;
  void run(const std::stop_token& stop, const EngineRequest& request);

  mutable std::mutex mutex_;
  std::atomic<bool> told_build_{false};  // l'événement « build » du banc, dit une fois
  std::atomic<bool> done_{false};        // run est fini, son instance détruite
  bool pump_on_close_ = false;           // EngineRequest::pump_on_close
  std::optional<std::expected<gpu::Artifact, Refusal>> result_;
  float fraction_ = 0.0F;
  std::string stage_ = "démarrage";
  std::jthread thread_;  // dernier membre
};

}  // namespace ajp::app::play
