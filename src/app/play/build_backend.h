// Une instance de backend pour toutes les constructions et lectures du cache d'une préparation (A08-10,
// A05-10) : chaque EngineBuild en créait une à lui (contexte CUDA, chargement de nvinfer), jusqu'à six de
// plus par ouverture, même tout en cache, et la détruisait sur son thread avant que la fenêtre puisse le
// joindre.
//
// L'instance naît à la première construction et vit jusqu'à la fin de la préparation (prepare_chain). Les
// constructions passent l'une après l'autre : build tient un verrou, une instance ne sert qu'à un thread à la
// fois (BackendLibrary::create).
#pragma once

#include <expected>
#include <functional>
#include <mutex>
#include <optional>

#include "app/play/backend_library.h"
#include "core/types.h"
#include "gpu/backend.h"

namespace ajp::app::play {

class BuildBackend {
 public:
  using Create = std::function<std::expected<BackendLibrary::Instance, Refusal>()>;
  explicit BuildBackend(Create create) : create_(std::move(create)) {}

  // La construction sur l'instance partagée, créée au premier appel. Un refus de création est retenu : les
  // constructions suivantes le rendent sans réessayer.
  std::expected<gpu::Artifact, Refusal> build(const gpu::ModelSpec& model, const gpu::ShapeSet& shapes,
                                              gpu::Progress& progress) {
    const std::scoped_lock lock(mutex_);
    if (!instance_ && !refused_) {
      auto made = create_();
      ++created_;
      if (made)
        instance_ = std::move(*made);
      else
        refused_ = std::move(made.error());
    }
    if (refused_) return std::unexpected(*refused_);
    return (*instance_)->build(model, shapes, progress);
  }

  int created() const {
    const std::scoped_lock lock(mutex_);
    return created_;
  }

 private:
  Create create_;
  mutable std::mutex mutex_;
  std::optional<BackendLibrary::Instance> instance_;
  std::optional<Refusal> refused_;
  int created_ = 0;
};

}  // namespace ajp::app::play
