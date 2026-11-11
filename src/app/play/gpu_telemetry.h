// Télémétrie de la carte pendant la lecture : horloges, état de performance, puissance,
// température, charge, lien PCIe et limites actives, lus par NVML (libnvidia-ml.so.1, fournie par
// le pilote : rien à redistribuer). Sur une carte AMD sous Linux, les mêmes mesures viennent de
// sysfs (amdgpu : gpu_busy_percent, pp_dpm_sclk, hwmon). Absente sinon (Windows : ADLX n'est pas
// lu) : l'OSD s'en passe.
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
  unsigned vram_used_mib = 0;
  unsigned vram_total_mib = 0;  // 0 : inconnue
  std::uint64_t throttle = 0;   // masque des raisons d'écrêtage d'NVML
  std::uint64_t energy_mj =
      0;  // énergie totale depuis le chargement du pilote (NVML, Volta et plus) ; 0 : inconnue
};

// Millijoules par image neuve entre deux relevés du compteur d'énergie. Rien si un relevé manque (0), si le
// compteur recule ou si aucune image n'est neuve.
std::optional<double> millijoules_per_frame(std::uint64_t start_mj, std::uint64_t end_mj,
                                            std::int64_t frames);

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

// Les extrêmes des relevés d'une lecture (fréquence, température), pour la ligne de fin du journal (A39-06).
// note_gpu_range retient un relevé ; take_gpu_range_text rend « GPU … » et repart de zéro ; vide : aucun
// relevé.
void note_gpu_range(const GpuSample& sample);
std::string take_gpu_range_text();

// Une ligne de pp_dpm_sclk ou pp_dpm_mclk du pilote amdgpu (« 0: 500Mhz \n1: 1900Mhz \n2: 2900Mhz * ») :
// l'horloge courante (la ligne étoilée) et la plus haute, en MHz ; 0 et 0 si le texte ne se lit pas.
struct DpmClocks {
  unsigned current_mhz = 0;
  unsigned max_mhz = 0;
};
DpmClocks parse_amdgpu_dpm(std::string_view text);

// La génération PCIe d'un current_link_speed de sysfs (« 16.0 GT/s PCIe » : 4) ; 0 si le texte ne se lit pas.
unsigned parse_pcie_generation(std::string_view text);

// Le pilote NVIDIA « 580.82.07 » est-il plus ancien que major.minor.patch ? Faux si la version ne se
// lit pas (pilote Mesa, texte libre) : on ne s'alarme pas sans la connaître.
bool nvidia_driver_older_than(std::string_view version, int major, int minor = 0, int patch = 0);

class GpuTelemetry {
 public:
  // Rien si la carte n'est ni NVIDIA (NVML présent, UUID connu) ni AMD sous Linux (sysfs amdgpu).
  static std::unique_ptr<GpuTelemetry> open(const GpuDevice& device);
  ~GpuTelemetry();
  GpuTelemetry(const GpuTelemetry&) = delete;
  GpuTelemetry& operator=(const GpuTelemetry&) = delete;

  // Relue au plus toutes les 500 ms (host_us : l'horloge de l'hôte), sinon la précédente. fresh : relue
  // maintenant, pour les compteurs d'énergie du début et de la fin d'une lecture (B11-14).
  std::optional<GpuSample> sample(std::int64_t host_us, bool fresh = false);
  // Le compteur d'énergie relu maintenant (mJ), 0 s'il ne se lit pas.
  std::uint64_t energy_mj(std::int64_t host_us) {
    return sample(host_us, true).value_or(GpuSample{}).energy_mj;
  }

 private:
  struct Impl;
  explicit GpuTelemetry(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::app::play
