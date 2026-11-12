// Ce que la chaîne ouverte montre d'elle-même : les moteurs du cache pour l'onglet Moteurs du
// menu, et les mesures de l'OSD (compteurs de l'interpolation, mémoire vidéo).
#pragma once

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "app/play/build_progress.h"
#include "app/play/gpu_telemetry.h"
#include "app/play/live_settings.h"
#include "app/play/options.h"
#include "app/play/sample_demand.h"
#include "app/play/session.h"
#include "app/play/settings_panel.h"
#include "core/auto_rate.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {

// Les moteurs du cache, ceux de cette lecture en tête : loaded, les moteurs que la chaîne a construits
// (B08-12 : la variante mouvement ou le fusionné de RIFE, pas seulement le modèle tel que livré).
std::vector<EngineBadge> engine_badges(const std::filesystem::path& cache_dir,
                                       std::span<const LoadedEngine> loaded);

// Le multiplicateur en centièmes (250) écrit à la française : « 2,5 », « 3,75 », « 4 ».
std::string auto_factor_text(int centi);
// Le palier du mode Auto et sa raison : « Auto ×3,5 (84 i/s) · plafond écran ×6 · prédit ×4 · limite GPU 78 %
// · scène lourde · dernier changement : scène allégée ».
std::string auto_status_text(const core::AutoInfo& info, double source_fps);
// Le mode Rapide : « Rapide · 87 i/s · limite GPU 78 % · écart service 11,4 ms ».
std::string free_status_text(const core::FreeInfo& info);
// Rapide à pas constant : N, l'échelle mesurée, le budget, les pertes.
std::string steady_status_text(const core::SteadyInfo& info);

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
        telemetry_(telemetry) {
    check_memory();
    publish_memory();
  }
  ~ChainStats() override { stop(); }
  std::optional<std::string> rife() const override;
  // La mémoire vidéo et la ligne GPU de l'OSD : relues à 2 Hz par un fil à part (cudaMemGetInfo
  // prend le verrou du contexte CUDA, NVML dix appels : le fil d'affichage n'attend ni l'un ni
  // l'autre). Le fil naît au premier appel, donc seulement quand l'OSD les demande.
  std::optional<std::string> memory() override;
  std::optional<std::string> gpu() override;
  std::int64_t output_period_us() const override;
  double output_fps() const override;
  bool uncapped() const override { return live_.free_active(); }
  // Arrête et joint le fil de relevé, qui ne renaît plus : à appeler avant de détruire le backend dont il
  // lit les noyaux (A08-02). Sans fil lancé, rien à attendre.
  void stop();
  std::int64_t source_period_us() const override;
  std::uint64_t skipped() const override { return live_.skipped(); }
  std::optional<PairVerdict> pair_verdict(std::uint32_t epoch, std::int64_t pts_us) const override {
    return live_.pair_marks().at(epoch, pts_us);
  }

 private:
  const LiveSettings& live_;
  gpu::KernelOps& kernels_;
  bool interpolates_;
  std::string engine_;
  double source_fps_;
  GpuTelemetry* telemetry_;

  std::optional<std::string> read_memory() const;
  // A32-10 : la mémoire vidéo pleine (moins de 256 Mio libres) se dit une fois au journal ; sous Windows, le
  // pilote déborde alors en mémoire système sans refuser, et la chaîne ralentit sans autre signe. Vrai si
  // pleine.
  bool note_memory(const gpu::KernelOps::DeviceMemory& memory) const;
  void check_memory() const;
  void publish_memory();  // B33-05 : la mémoire vidéo pour la ligne des 5 minutes (resource_usage.h)
  bool stopped_ = false;
  mutable std::atomic<bool> memory_full_told_{false};
  std::optional<std::string> read_gpu() const;
  void start_sampler();

  std::mutex sampled_mutex_;
  std::optional<std::string> memory_text_;
  std::optional<std::string> gpu_text_;
  std::once_flag sampler_started_;
  SampleDemand demand_;  // le relevé dort quand l'OSD ne le lit plus
  // En dernier : arrêté et joint avant que les membres qu'il lit ne partent.
  std::jthread sampler_;
};

}  // namespace ajp::app::play
