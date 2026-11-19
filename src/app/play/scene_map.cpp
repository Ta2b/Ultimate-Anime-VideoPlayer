// La carte de la scène dans l'étage gpu (scene_map.h).
#include "app/play/scene_map.h"

#include <algorithm>
#include <utility>

#include "core/log.h"

namespace kirari::app::play {

void SceneMap::stop(const Refusal& refusal, const char* where) {
  on_ = false;
  pending_.clear();
  log_channel("gpu")->info("carte de la scène éteinte pour la lecture ({} : {})", where, refusal.detail);
}

void SceneMap::start(gpu::KernelOps& ops, const gpu::Tensor& a, const gpu::Tensor& b, SourceId left,
                     SourceId right, std::uint32_t epoch) {
  if (!on_) return;
  poll(ops);
  const bool follows = last_right_ && *last_right_ == left && last_epoch_ == epoch;
  const bool restart = restart_ || !follows;
  auto ticket = ops.scene_flow_start(a, b, restart);
  if (!ticket) {
    stop(ticket.error(), "scene_flow_start");
    return;
  }
  if (*ticket == 0) {  // NVOF occupé : paire sautée, la suivante ne la suit pas (last_right_ inchangé)
    ++busy_;
    return;
  }
  // Le backend garde kSceneFlowReadbacks paires : la nouvelle a pris la place de la plus ancienne.
  if (pending_.size() >= gpu::kSceneFlowReadbacks) {
    pending_.erase(pending_.begin());
    ++lost_;
  }
  pending_.push_back({.ticket = *ticket, .left = left, .right = right, .restart = restart});
  last_left_ = left;
  last_right_ = right;
  last_epoch_ = epoch;
  restart_ = false;
  ++started_;
}

void SceneMap::poll(gpu::KernelOps& ops) {
  while (on_ && !pending_.empty()) {
    auto taken = ops.scene_flow_take(pending_.front().ticket);
    if (!taken) {
      if (taken.error().code != RefusalCode::SequenceOrder) {
        stop(taken.error(), "scene_flow_take");
        return;
      }
      pending_.erase(pending_.begin());  // périmée : perdue
      ++lost_;
      continue;
    }
    if (!*taken) return;                             // pas encore finie : les suivantes non plus
    if (pending_.front().restart) history_.clear();  // les classes d'un autre plan ne disent rien ici
    kernels::SceneMotion motion = kernels::ref::scene_motion_of(**taken, history_);
    ++read_;
    if (motion.cut_likely) {
      ++cuts_;
      restart_ = true;  // NVOF ne trouve rien à suivre : la paire suivante repart sans indices
    }
    if (motion.fade) ++fades_;
    if (motion.camera_fit) ++camera_fits_;
    latest_ =
        Latest{.left = pending_.front().left, .right = pending_.front().right, .motion = std::move(motion)};
    pending_.erase(pending_.begin());
  }
}

std::optional<kernels::SceneMotion> SceneMap::for_pair(gpu::KernelOps& ops, SourceId left, SourceId right) {
  if (!on_) return std::nullopt;
  poll(ops);
  if (!latest_ || !(latest_->left == left) || !(latest_->right == right)) {
    if (std::ranges::any_of(pending_, [&](const Pending& p) { return p.left == left && p.right == right; }))
      ++missed_;
    return std::nullopt;
  }
  ++attached_;
  return latest_->motion;
}

void SceneMap::cut() {
  restart_ = true;
}

void SceneMap::log_summary() const {
  if (started_ == 0) return;
  log_channel("gpu")->info(
      "carte de la scène : {} paires lancées, {} relues, {} posées sur leur paire ; {} sautées (NVOF "
      "occupé), "
      "{} relues trop tard, {} perdues ; caméra ajustée sur {}, {} coupes probables, {} fondus",
      started_, read_, attached_, busy_, missed_, lost_, camera_fits_, cuts_, fades_);
}

}  // namespace kirari::app::play
