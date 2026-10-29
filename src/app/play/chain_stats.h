// Ce que la chaîne ouverte montre d'elle-même : les moteurs du cache pour l'onglet Moteurs du
// menu, et les mesures de l'OSD (compteurs de l'interpolation, mémoire vidéo).
#pragma once

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "app/play/gpu_telemetry.h"
#include "app/play/live_settings.h"
#include "app/play/options.h"
#include "app/play/session.h"
#include "app/play/settings_panel.h"
#include "core/auto_rate.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {

// Les moteurs du cache, ceux de cette lecture en tête (SPAN à span_size, RIFE à rife_size).
std::vector<EngineBadge> engine_badges(const std::filesystem::path& cache_dir,
                                       const std::filesystem::path& span, WorkSize span_size,
                                       const std::optional<std::filesystem::path>& rife,
                                       const kernels::RifeGeometry& rife_size);

// Le multiplicateur en centièmes (250) écrit à la française : « 2,5 », « 3,75 », « 4 ».
std::string auto_factor_text(int centi);
// Le palier du mode Auto et sa raison : « Auto ×3,5 (84 i/s) · plafond écran ×6 · prédit ×4 · limite GPU 78 %
// · scène lourde · dernier changement : scène allégée ».
std::string auto_status_text(const core::AutoInfo& info, double source_fps);

// Mesures de la chaîne pour l'OSD. interpolation : rien sans interpolation ; sinon son moteur,
// « RIFE » ou « flux optique », en tête de la ligne des compteurs. source_fps : images par
// seconde de la source (0 : inconnues).
class ChainStats final : public StatsSource {
 public:
  ChainStats(const LiveSettings& live, gpu::KernelOps& kernels, bool interpolates, std::string engine,
             double source_fps, GpuTelemetry* telemetry = nullptr)
      : live_(live),
        kernels_(kernels),
        interpolates_(interpolates),
        engine_(std::move(engine)),
        source_fps_(source_fps),
        telemetry_(telemetry) {}
  std::optional<std::string> rife() const override;
  // La mémoire vidéo et la ligne GPU de l'OSD : relues à 2 Hz par un fil à part (cudaMemGetInfo
  // prend le verrou du contexte CUDA, NVML dix appels : le fil d'affichage n'attend ni l'un ni
  // l'autre). Le fil naît au premier appel, donc seulement quand l'OSD les demande.
  std::optional<std::string> memory() const override;
  std::optional<std::string> gpu() const override;
  std::int64_t output_period_us() const override;
  double output_fps() const override;
  std::int64_t source_period_us() const override;
  std::uint64_t skipped() const override { return live_.skipped(); }

 private:
  const LiveSettings& live_;
  gpu::KernelOps& kernels_;
  bool interpolates_;
  std::string engine_;
  double source_fps_;
  GpuTelemetry* telemetry_;

  std::optional<std::string> read_memory() const;
  std::optional<std::string> read_gpu() const;
  void start_sampler() const;

  mutable std::mutex sampled_mutex_;
  mutable std::optional<std::string> memory_text_;
  mutable std::optional<std::string> gpu_text_;
  mutable std::once_flag sampler_started_;
  // En dernier : arrêté et joint avant que les membres qu'il lit ne partent.
  mutable std::jthread sampler_;
};

}  // namespace ajp::app::play
