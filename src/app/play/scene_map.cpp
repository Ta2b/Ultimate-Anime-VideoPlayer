// La carte de la scène dans l'étage gpu (scene_map.h).
#include "app/play/scene_map.h"

#include <algorithm>
#include <chrono>
#include <utility>

#include "core/log.h"
#include "kernels/ref/scene_camera.h"

namespace kirari::app::play {
namespace {

double since_us(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - from).count();
}

}  // namespace

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
  const auto asked = std::chrono::steady_clock::now();
  auto ticket = ops.scene_flow_start(a, b, restart);
  start_us_ += since_us(asked);
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

void SceneMap::stop_refine(const Refusal& refusal, const char* where) {
  refine_ = false;
  log_channel("gpu")->info(
      "affinage de la caméra de la carte éteint pour la lecture ({} : {}) : la caméra de "
      "NVOF reste",
      where, refusal.detail);
}

bool SceneMap::take_tiles(gpu::KernelOps& ops) {
  Pending& pair = pending_.front();
  auto taken = ops.scene_flow_take(pair.ticket);
  if (!taken) {
    if (taken.error().code != RefusalCode::SequenceOrder) {
      stop(taken.error(), "scene_flow_take");
      return false;
    }
    pending_.erase(pending_.begin());  // périmée : perdue
    ++lost_;
    return false;
  }
  if (!*taken) return false;  // pas encore finie
  pair.tiles = std::move(**taken);
  ++read_;
  const auto asked = std::chrono::steady_clock::now();
  pair.camera = kernels::ref::scene_camera_of(*pair.tiles);
  map_us_ += since_us(asked);
  return true;
}

void SceneMap::request_refine(gpu::KernelOps& ops, Pending& pair) {
  if (!refine_) return;
  const std::optional<kernels::SceneCameraRequest> request = kernels::ref::scene_camera_request(pair.camera);
  if (!request) return;
  auto ticket = ops.scene_camera_start(pair.ticket, *request);
  if (!ticket) {
    stop_refine(ticket.error(), "scene_camera_start");
    return;
  }
  if (*ticket == 0) {
    ++refine_missed_;
    return;
  }
  pair.refine = *ticket;
}

bool SceneMap::collect_refine(gpu::KernelOps& ops, Pending& pair) {
  auto taken = ops.scene_camera_take(pair.refine);
  if (!taken) {
    if (taken.error().code == RefusalCode::SequenceOrder) {
      ++refine_missed_;
    } else {
      stop_refine(taken.error(), "scene_camera_take");
    }
    pair.refine = 0;
    return true;
  }
  if (!*taken) return false;
  kernels::ref::scene_camera_apply(pair.camera, **taken);
  if (pair.camera.refined) ++refined_;
  pair.refine = 0;
  return true;
}

void SceneMap::finish(Pending& pair) {
  if (!pair.tiles) {  // jamais : poll et settle ne finissent qu'une paire relue
    pending_.erase(pending_.begin());
    return;
  }
  if (pair.restart) history_.clear();  // les classes d'un autre plan ne disent rien ici
  const auto asked = std::chrono::steady_clock::now();
  kernels::SceneMotion motion = kernels::ref::scene_motion_of(*pair.tiles, history_, pair.camera);
  map_us_ += since_us(asked);
  if (motion.cut_likely) {
    ++cuts_;
    restart_ = true;  // NVOF ne trouve rien à suivre : la paire suivante repart sans indices
  }
  if (motion.fade) ++fades_;
  if (motion.camera_fit) ++camera_fits_;
  latest_ = Latest{.left = pair.left, .right = pair.right, .motion = std::move(motion)};
  pending_.erase(pending_.begin());
}

void SceneMap::poll(gpu::KernelOps& ops) {
  while (on_ && !pending_.empty()) {
    Pending& pair = pending_.front();
    if (!pair.tiles) {
      const std::size_t before = pending_.size();
      if (!take_tiles(ops)) {
        if (on_ && pending_.size() < before) continue;  // perdue : la suivante
        return;                                         // pas encore finie : les suivantes non plus
      }
      request_refine(ops, pair);
    }
    if (pair.refine != 0 && !collect_refine(ops, pair)) return;  // caméra encore en affinage
    finish(pair);
  }
}

void SceneMap::settle(gpu::KernelOps& ops, SourceId left, SourceId right) {
  const auto wanted = [&](const Pending& p) { return p.left == left && p.right == right; };
  if (!std::ranges::any_of(pending_, wanted)) return;
  while (on_ && !pending_.empty() && pending_.front().tiles) {
    const bool reached = wanted(pending_.front());
    if (pending_.front().refine != 0) ++refine_late_;
    finish(pending_.front());
    if (reached) return;
    poll(ops);  // les suivantes, peut-être déjà relues
  }
}

std::optional<kernels::SceneMotion> SceneMap::for_pair(gpu::KernelOps& ops, SourceId left, SourceId right) {
  if (!on_) return std::nullopt;
  poll(ops);
  settle(ops, left, right);
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
      "{} relues trop tard, {} perdues ; caméra ajustée sur {}, affinée sur {} ({} affinages trop tard, {} "
      "pas lancés), {} coupes probables, {} fondus ; hôte : {:.0f} µs pour lancer une paire, {:.0f} µs pour "
      "tirer une carte, en moyenne",
      started_, read_, attached_, busy_, missed_, lost_, camera_fits_, refined_, refine_late_, refine_missed_,
      cuts_, fades_, start_us_ / static_cast<double>(started_ + busy_),
      read_ > 0 ? map_us_ / static_cast<double>(read_) : 0.0);
}

}  // namespace kirari::app::play
