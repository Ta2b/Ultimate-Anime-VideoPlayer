// La carte de la scène dans l'étage gpu (upscale_stage.h) : lance le flux de NVIDIA sur une paire d'images
// de travail (KernelOps::scene_flow_start), le relit sans jamais attendre (scene_flow_take) et en tire la
// carte (kernels/ref/scene_flow.h, scene_motion_of), posée dans PairFacts::scene pour la paire qu'elle
// décrit.
//
// Deux cadences. Avec l'image suivante ingérée en avance (interpolation par RIFE), la paire (N, N+1) est
// lancée pendant l'image N et sa carte est là quand l'étage décide de cette paire, avant RIFE et SPAN. Sans
// elle (sans interpolation, DistilDRBA), la paire (N−1, N) est lancée à l'image N et sa carte n'arrive qu'à
// l'image suivante : latest() la garde, avec sa paire, mais PairFacts ne la reçoit pas.
//
// Absente (--scene-map éteint, backend sans NVOF, Turing, refus) : rien n'est lancé, chaque paire garde
// scene vide, et la chaîne fait exactement ce qu'elle faisait. Un refus éteint la carte pour la lecture, avec
// une ligne au journal.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "kernels/ref/scene_flow.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::app::play {

class SceneMap {
 public:
  explicit SceneMap(bool wanted) : on_(wanted) {}
  bool on() const { return on_; }

  // Lance la paire (left, right) des images de travail a et b, d'époque epoch. Les indices temporels de
  // NVOF repartent si la paire ne suit pas la précédente (début, saut, nouvelle époque) ou après une coupe.
  void start(gpu::KernelOps& ops, const gpu::Tensor& a, const gpu::Tensor& b, SourceId left, SourceId right,
             std::uint32_t epoch);
  // La carte de la paire (left, right) si elle est relue ; sinon rien. Relit sans attente ce qui est fini.
  std::optional<kernels::SceneMotion> for_pair(gpu::KernelOps& ops, SourceId left, SourceId right);
  // La paire en cours a été jugée coupée : la suivante repart sans indices ni classes.
  void cut();

  // La dernière carte relue et sa paire, pour qui accepte une paire de retard.
  struct Latest {
    SourceId left;
    SourceId right;
    kernels::SceneMotion motion;
  };
  const std::optional<Latest>& latest() const { return latest_; }

  // La ligne du journal de fin : paires lancées, relues, attachées, en retard, coupes et fondus vus.
  void log_summary() const;

 private:
  struct Pending {
    std::uint64_t ticket = 0;
    SourceId left;
    SourceId right;
  };
  void poll(gpu::KernelOps& ops);
  void stop(const Refusal& refusal, const char* where);

  bool on_ = false;
  std::vector<Pending> pending_;
  std::optional<Latest> latest_;
  kernels::ref::SceneHistory history_;
  std::optional<SourceId> last_right_;
  std::uint32_t last_epoch_ = 0;
  bool restart_ = true;
  std::uint64_t started_ = 0;
  std::uint64_t read_ = 0;
  std::uint64_t attached_ = 0;
  std::uint64_t dropped_ = 0;  // relectures abandonnées : NVOF plus lent que la chaîne
  std::uint64_t cuts_ = 0;
  std::uint64_t fades_ = 0;
  std::uint64_t camera_fits_ = 0;
};

}  // namespace kirari::app::play
