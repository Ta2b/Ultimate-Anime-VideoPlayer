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

// L'affinage parti de la caméra du plan est gardé si sa vérification sur l'image (SceneCamera::check, pixels
// du flux) reste sous kEarlyFloor, ou sous kEarlyFactor fois celle de la dernière caméra affinée du plan sans
// passer kEarlyCeiling, et s'il reste à kEarlyGap pixels du flux de la caméra de NVOF de sa paire, aux coins
// (garde contre une vérification faite sur trop peu de texture). Rejeu de dix extraits (carte_rejeu, 1 479
// départs) : 1 346 gardés, dont 3 vérifiés à plus de 0,05 pixel au-delà de l'affinage parti de NVOF. L'écart
// seul à la caméra de NVOF (2 pixels) en gardait 1 416, dont 46 ainsi faux (un départ qui glisse vers une
// autre solution, souvent une image sur trois sur un dessin animé en trois), et écartait des départs justes
// là où NVOF biaise la caméra de 2 à 9 pixels (panoramique3).
constexpr double kEarlyFloor = 0.05;
constexpr double kEarlyFactor = 2.0;
constexpr double kEarlyCeiling = 0.15;
constexpr double kEarlyGap = 16.0;

// La vérification que l'affinage parti de la caméra du plan ne doit pas dépasser ; check : celle de la
// dernière caméra affinée du plan (négative : aucune).
double early_limit(double check) {
  return check < 0.0 ? kEarlyFloor : std::clamp(kEarlyFactor * check, kEarlyFloor, kEarlyCeiling);
}

// L'affinage parti de la caméra du plan, posé sur tried, est-il gardé ? nvof : la caméra de NVOF de la paire.
bool early_kept(const kernels::ref::SceneCamera& tried, const kernels::ref::SceneCamera& nvof, double limit,
                int flow_width, int flow_height) {
  return tried.refined && tried.check >= 0.0 && tried.check <= limit &&
         kernels::ref::scene_camera_gap(tried.model, nvof.model, flow_width, flow_height) <= kEarlyGap;
}

}  // namespace

void SceneMap::stop(const Refusal& refusal, const char* where) {
  on_ = false;
  pending_.clear();
  late_.clear();
  guess_.reset();
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
  if (restart) ++shot_;
  pending_.push_back({.ticket = *ticket, .left = left, .right = right, .restart = restart, .shot = shot_});
  request_early(ops, pending_.back());
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
  // La caméra du plan pour la paire suivante, en attendant l'affinage de celle-ci ; la vérification de la
  // dernière caméra affinée du plan reste.
  if (pair.camera.fit) {
    const double check = guess_ && guess_->shot == pair.shot ? guess_->check : -1.0;
    guess_ = Guess{.shot = pair.shot, .camera = pair.camera, .check = check};
  }
  return true;
}

void SceneMap::request_early(gpu::KernelOps& ops, Pending& pair) {
  if (!refine_ || pair.restart || !guess_ || guess_->shot != pair.shot) return;
  const double limit = early_limit(guess_->check);
  const std::optional<kernels::SceneCameraRequest> request =
      kernels::ref::scene_camera_request(guess_->camera);
  if (!request) return;
  auto ticket = ops.scene_camera_start(pair.ticket, *request);
  if (!ticket) {
    stop_refine(ticket.error(), "scene_camera_start");
    return;
  }
  if (*ticket == 0) return;  // relectures toutes en vol : il partira des tuiles relues
  pair.refine = *ticket;
  pair.early_limit = limit;
  ++refine_early_;
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
  pair.refine = 0;
  const Late where = late_of(pair);
  if (!apply_refined(pair.camera, **taken, std::exchange(pair.early_limit, 0.0), where)) {
    request_refine(ops, pair);  // depuis la caméra de NVOF, comme sans caméra du plan
    return pair.refine == 0;
  }
  if (pair.camera.refined)
    guess_ = Guess{.shot = pair.shot, .camera = pair.camera, .check = pair.camera.check};
  return true;
}

bool SceneMap::apply_refined(kernels::ref::SceneCamera& camera, const kernels::SceneCameraRefined& refined,
                             double limit, const Late& where) {
  if (limit > 0.0) {
    kernels::ref::SceneCamera tried = camera;
    kernels::ref::scene_camera_apply(tried, refined);
    if (!early_kept(tried, camera, limit, where.flow_width, where.flow_height)) {
      ++refine_off_;
      return false;
    }
    camera = std::move(tried);
  } else {
    kernels::ref::scene_camera_apply(camera, refined);
  }
  if (!camera.refined) return true;
  // La vérification du plan suit, pour l'affinage de la paire suivante (guess_ porte déjà une caméra de NVOF
  // plus récente quand cet affinage arrive en retard).
  if (guess_ && guess_->shot == where.shot && camera.check >= 0.0) guess_->check = camera.check;
  ++refined_;
  const kernels::ref::SceneCameraWork work =
      kernels::ref::scene_camera_work(camera, where.flow_width, where.flow_height, where.divisor);
  last_refined_ = Refined{.left = where.left,
                          .right = where.right,
                          .camera = work.camera,
                          .residual = work.residual,
                          .residual_tiles = work.residual_tiles};
  return true;
}

void SceneMap::poll_late(gpu::KernelOps& ops) {
  while (on_ && refine_ && !late_.empty()) {
    Late& late = late_.front();
    auto taken = ops.scene_camera_take(late.refine);
    if (!taken) {
      if (taken.error().code != RefusalCode::SequenceOrder) stop_refine(taken.error(), "scene_camera_take");
      late_.erase(late_.begin());
      continue;
    }
    if (!*taken) return;  // les suivants sont derrière lui sur la file de NVOF
    if (apply_refined(late.camera, **taken, late.early_limit, late)) ++refine_after_;
    late_.erase(late_.begin());
  }
}

SceneMap::Late SceneMap::late_of(const Pending& pair) {
  return Late{.early_limit = pair.early_limit,
              .shot = pair.shot,
              .left = pair.left,
              .right = pair.right,
              .camera = pair.camera,
              .flow_width = pair.tiles ? pair.tiles->flow_width : 0,
              .flow_height = pair.tiles ? pair.tiles->flow_height : 0,
              .divisor = pair.tiles ? pair.tiles->divisor : 1};
}

void SceneMap::keep_late(Pending& pair) {
  ++refine_late_;
  // Le backend garde un affinage par paire en vol : au-delà, le plus ancien est périmé.
  if (late_.size() >= gpu::kSceneFlowReadbacks) late_.erase(late_.begin());
  late_.push_back(late_of(pair));
  late_.back().refine = std::exchange(pair.refine, 0);
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
  if (pair.across_cut && !motion.cut_likely) kernels::ref::mark_cut(motion);
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
  poll_late(ops);  // d'abord les plus anciens : last_refined() garde le plus récent
  while (on_ && !pending_.empty()) {
    Pending& pair = pending_.front();
    if (!pair.tiles) {
      const std::size_t before = pending_.size();
      if (!take_tiles(ops)) {
        if (on_ && pending_.size() < before) continue;  // perdue : la suivante
        return;                                         // pas encore finie : les suivantes non plus
      }
      if (pair.refine == 0) request_refine(ops, pair);
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
    if (pending_.front().refine != 0) keep_late(pending_.front());
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

void SceneMap::cut(SourceId left, SourceId right) {
  restart_ = true;
  if (latest_ && latest_->left == left && latest_->right == right && !latest_->motion.cut_likely) {
    kernels::ref::mark_cut(latest_->motion);
    ++cuts_;
  }
  for (Pending& pair : pending_) {
    if (pair.left == left && pair.right == right) pair.across_cut = true;
  }
}

void SceneMap::log_summary() const {
  if (started_ == 0) return;
  log_channel("gpu")->info(
      "carte de la scène : {} paires lancées, {} relues, {} posées sur leur paire ; {} sautées (NVOF "
      "occupé), "
      "{} relues trop tard, {} perdues ; caméra ajustée sur {}, affinée sur {} ({} affinages trop tard, dont "
      "{} relus ensuite, {} pas lancés ; {} lancés derrière le flux depuis la caméra du plan, {} écartés), "
      "{} "
      "coupes probables, {} fondus ; hôte : {:.0f} µs pour lancer une paire, {:.0f} µs pour tirer une carte, "
      "en moyenne",
      started_, read_, attached_, busy_, missed_, lost_, camera_fits_, refined_, refine_late_, refine_after_,
      refine_missed_, refine_early_, refine_off_, cuts_, fades_,
      start_us_ / static_cast<double>(started_ + busy_),
      read_ > 0 ? map_us_ / static_cast<double>(read_) : 0.0);
}

}  // namespace kirari::app::play
