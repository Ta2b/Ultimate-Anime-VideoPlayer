#include "gpu/vulkan/backend/family.h"

#include "core/gpu_device.h"

namespace ajp::gpu::vk {
namespace {

struct Range {
  std::uint32_t first;
  std::uint32_t last;
  GpuFamily family;
};

// Identifiants d'appareils PCI par puce (bases de pci.ids), du plus ancien au plus récent.
constexpr Range kNvidia[] = {
    {0x1E00, 0x1FFF, GpuFamily::NvidiaTuring},     // TU102, TU104, TU106, TU117
    {0x2080, 0x20FF, GpuFamily::NvidiaAmpere},     // GA100
    {0x2180, 0x21FF, GpuFamily::NvidiaTuring},     // TU116
    {0x2200, 0x25FF, GpuFamily::NvidiaAmpere},     // GA102 à GA107
    {0x2680, 0x28FF, GpuFamily::NvidiaAda},        // AD102 à AD107
    {0x2B00, 0x2FFF, GpuFamily::NvidiaBlackwell},  // GB202 à GB207
};
constexpr Range kAmd[] = {
    {0x7440, 0x745F, GpuFamily::AmdRdna3},  // Navi 31
    {0x7460, 0x747F, GpuFamily::AmdRdna3},  // Navi 32
    {0x7480, 0x749F, GpuFamily::AmdRdna3},  // Navi 33
    {0x15BF, 0x15BF, GpuFamily::AmdRdna3},  // Phoenix
    {0x15C8, 0x15C8, GpuFamily::AmdRdna3},  // Phoenix 2
    {0x150E, 0x150E, GpuFamily::AmdRdna3},  // Strix
    {0x1114, 0x1114, GpuFamily::AmdRdna3},  // Krackan
    {0x1586, 0x1586, GpuFamily::AmdRdna3},  // Strix Halo
    {0x7550, 0x755F, GpuFamily::AmdRdna4},  // Navi 48
    {0x7590, 0x759F, GpuFamily::AmdRdna4},  // Navi 44
};
constexpr Range kIntel[] = {
    {0x4680, 0x46DF, GpuFamily::IntelXeLp},       // Alder Lake S, P et N
    {0x4905, 0x4909, GpuFamily::IntelXeLp},       // DG1
    {0x4C80, 0x4C9F, GpuFamily::IntelXeLp},       // Rocket Lake
    {0x5690, 0x56CF, GpuFamily::IntelAlchemist},  // DG2, Arc A et Flex
    {0x7D40, 0x7D7F, GpuFamily::IntelXeLpg},      // Meteor Lake, Arrow Lake H et U
    {0x7DC0, 0x7DDF, GpuFamily::IntelXeLpg},      // Meteor Lake, Arrow Lake H
    {0x9A40, 0x9AFF, GpuFamily::IntelXeLp},       // Tiger Lake
    {0xA720, 0xA7AF, GpuFamily::IntelXeLp},       // Raptor Lake S et P (UHD 770)
    {0x6400, 0x64FF, GpuFamily::IntelXe2},        // Lunar Lake
    {0xE200, 0xE2FF, GpuFamily::IntelXe2},        // Battlemage
};

template <std::size_t N>
GpuFamily find(const Range (&ranges)[N], std::uint32_t device, GpuFamily otherwise) {
  for (const Range& r : ranges) {
    if (device >= r.first && device <= r.last) return r.family;
  }
  return otherwise;
}

// La puce que RADV met entre parenthèses : NAVI3x et GFX11xx sont des RDNA3 (3.5 compris),
// NAVI4x et GFX12xx des RDNA4 ; le reste, ou un autre pilote, reste avant RDNA3.
GpuFamily radv_family(std::string_view device_name) {
  const std::size_t at = device_name.find("(RADV ");
  if (at == std::string_view::npos) return GpuFamily::AmdBeforeRdna3;
  const std::string_view chip = device_name.substr(at + 6);
  if (chip.starts_with("NAVI3") || chip.starts_with("GFX11")) return GpuFamily::AmdRdna3;
  if (chip.starts_with("NAVI4") || chip.starts_with("GFX12")) return GpuFamily::AmdRdna4;
  return GpuFamily::AmdBeforeRdna3;
}

}  // namespace

std::string_view name(GpuFamily family) {
  switch (family) {
    case GpuFamily::Unknown:
      return "inconnue";
    case GpuFamily::Software:
      return "logicielle";
    case GpuFamily::NvidiaTuring:
      return "NVIDIA Turing";
    case GpuFamily::NvidiaAmpere:
      return "NVIDIA Ampere";
    case GpuFamily::NvidiaAda:
      return "NVIDIA Ada";
    case GpuFamily::NvidiaBlackwell:
      return "NVIDIA Blackwell";
    case GpuFamily::AmdBeforeRdna3:
      return "AMD avant RDNA3";
    case GpuFamily::AmdRdna3:
      return "AMD RDNA3";
    case GpuFamily::AmdRdna4:
      return "AMD RDNA4";
    case GpuFamily::IntelXeLp:
      return "Intel Xe-LP";
    case GpuFamily::IntelXeLpg:
      return "Intel Xe-LPG";
    case GpuFamily::IntelAlchemist:
      return "Intel Alchemist";
    case GpuFamily::IntelXe2:
      return "Intel Xe2";
    case GpuFamily::Apple:
      return "Apple";
  }
  return "inconnue";
}

GpuFamily detect_family(std::uint32_t vendor_id, std::uint32_t device_id, bool cpu,
                        std::string_view device_name) {
  if (cpu) return GpuFamily::Software;
  switch (vendor_id) {
    case pci::kNvidia:
      return find(kNvidia, device_id, GpuFamily::Unknown);
    case pci::kAmd:
      return find(kAmd, device_id, radv_family(device_name));
    case pci::kIntel:
      return find(kIntel, device_id, GpuFamily::Unknown);
    case pci::kApple:
      return GpuFamily::Apple;
    default:
      return GpuFamily::Unknown;
  }
}

DriverKind driver_kind(std::string_view driver_name) {
  if (driver_name == "NVIDIA") return DriverKind::NvidiaProprietary;
  if (driver_name == "radv") return DriverKind::Radv;
  if (driver_name == "AMD proprietary driver" || driver_name == "AMD open-source driver") {
    return DriverKind::AmdProprietary;
  }
  if (driver_name.starts_with("Intel open-source Mesa")) return DriverKind::IntelMesa;
  if (driver_name.starts_with("Intel Corporation")) return DriverKind::IntelWindows;
  return DriverKind::Other;
}

bool matrix_units_allowed(GpuFamily family, DriverKind driver) {
  switch (family) {
    case GpuFamily::NvidiaTuring:
    case GpuFamily::NvidiaAmpere:
    case GpuFamily::NvidiaAda:
    case GpuFamily::NvidiaBlackwell:
      return driver == DriverKind::NvidiaProprietary;
    // WMMA des RDNA3 et RDNA4, sous RADV comme sous le pilote AMD (llama.cpp fait de même ; le
    // pilote AMD annonce aussi l'extension sur RDNA2, sans unité matricielle). L'autotuner
    // compare chaque sortie à celle du noyau direct avant de la garder.
    case GpuFamily::AmdRdna3:
    case GpuFamily::AmdRdna4:
      return driver == DriverKind::Radv || driver == DriverKind::AmdProprietary;
    default:
      return false;
  }
}

bool matrix_units_unlisted(GpuFamily family, DriverKind driver) {
  return family == GpuFamily::Unknown && driver == DriverKind::NvidiaProprietary;
}

}  // namespace ajp::gpu::vk
