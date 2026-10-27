// Télémétrie de la carte NVIDIA pendant la lecture : horloges, état de performance, puissance,
// température, charge, lien PCIe et limites actives, lus par NVML (libnvidia-ml.so.1, fournie par
// le pilote : rien à redistribuer). Absente sans NVML ou hors Linux : l'OSD s'en passe.
//
// Sert à voir ce que les rapports relevaient à la main avec nvidia-smi : la carte descend-elle
// en P-state, ou plafonne-t-elle en puissance, pendant une lecture chargée ?
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "core/gpu_device.h"

namespace ajp::app::play {

struct GpuSample {
  unsigned sm_mhz = 0;
  unsigned sm_max_mhz = 0;
  unsigned mem_mhz = 0;
  int pstate = -1;  // 0 à 15 ; -1 : inconnu
  unsigned power_mw = 0;
  unsigned power_limit_mw = 0;
  unsigned temperature_c = 0;
  unsigned gpu_percent = 0;
  unsigned pcie_generation = 0;
  unsigned pcie_width = 0;
  std::uint64_t throttle = 0;  // masque des raisons d'écrêtage d'NVML
};

// L'UUID d'un appareil Vulkan tel que NVML le nomme : « GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx ».
std::string nvml_uuid_text(const DeviceUuid& uuid);

// Les raisons d'écrêtage qui comptent pour une lecture (puissance, chaleur, freinage matériel),
// en clair, séparées par « + » ; vide : aucune. Le ralenti et les horloges d'application sont
// normaux et ne comptent pas.
std::string throttle_text(std::uint64_t mask);

// La ligne de l'OSD : « GPU 2 520 / 3 105 MHz · P2 · 188 / 285 W · 61 °C · 93 % · PCIe 4 x16 ».
std::string telemetry_line(const GpuSample& sample);

// Un avertissement quand la carte est sous-cadencée alors qu'elle est chargée (horloge SM sous
// 70 % du maximum à plus de 80 % de charge) ; vide sinon. L'écrêtage se lit déjà dans la ligne.
std::string telemetry_warning(const GpuSample& sample);

// Le pilote NVIDIA « 580.82.07 » est-il plus ancien que major.minor.patch ? Faux si la version ne se
// lit pas (pilote Mesa, texte libre) : on ne s'alarme pas sans la connaître.
bool nvidia_driver_older_than(std::string_view version, int major, int minor = 0, int patch = 0);

// Les versions minimales annoncées par la release : RTX HDR (Maxine) 550, Smooth Motion 580.82.07.
inline constexpr int kRtxHdrDriverMajor = 550;

class GpuTelemetry {
 public:
  // Rien si la carte n'est pas NVIDIA, si NVML manque, ou si l'UUID n'y est pas connu.
  static std::unique_ptr<GpuTelemetry> open(const GpuDevice& device);
  ~GpuTelemetry();
  GpuTelemetry(const GpuTelemetry&) = delete;
  GpuTelemetry& operator=(const GpuTelemetry&) = delete;

  // Relue au plus toutes les 500 ms (host_us : l'horloge de l'hôte), sinon la précédente.
  std::optional<GpuSample> sample(std::int64_t host_us);

 private:
  struct Impl;
  explicit GpuTelemetry(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::app::play
