// La vsync où paraît chaque image d'une swapchain FIFO, pour PresentScheduler::on_vsync.
//
// FIFO montre les images dans l'ordre, une vsync au moins chacune : une image paraît donc au
// plus tôt à la vsync qui suit la précédente, même quand pl_swapchain_start_frame rend la main
// sans attendre (swapchain neuve ou recréée, images libres). Sinon, elle paraît « latence »
// vsyncs après celle où elle a été obtenue. Les rangs ne reculent jamais : quand l'estimation
// recommence (changement de mode), la numérotation continue.
#pragma once

#include <cstdint>

#include "present/scheduler.h"
#include "present/vsync_estimator.h"

namespace ajp::present {

class FrameVsyncs {
 public:
  // acquired_us : retour de pl_swapchain_start_frame ; latency : images en vol avant elle.
  // Sans estimation, une vsync par image et l'instant d'acquisition.
  VsyncTick next(const VsyncEstimator& estimator, std::int64_t acquired_us, int latency);

 private:
  std::int64_t last_index_ = -1;  // rang donné à l'image précédente
  std::int64_t offset_ = 0;       // rang donné moins rang de l'estimation
  std::int64_t last_raw_ = 0;     // dernier rang de l'estimation, latence comprise
  bool have_raw_ = false;         // last_raw_ vient de l'estimation en cours
};

}  // namespace ajp::present
