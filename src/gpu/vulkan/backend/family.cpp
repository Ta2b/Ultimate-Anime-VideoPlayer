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
    {0x7550, 0x755F, GpuFamily::AmdRdna4},  // Navi 48
    {0x7590, 0x759F, GpuFamily::AmdRdna4},  // Navi 44
};
constexpr Range kIntel[] = {
    {0x5690, 0x56CF, GpuFamily::IntelAlchemist},  // DG2, Arc A et Flex
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
    case GpuFamily::IntelAlchemist:
      return "Intel Alchemist";
    case GpuFamily::IntelXe2:
      return "Intel Xe2";
  }
  return "inconnue";
}

GpuFamily detect_family(std::uint32_t vendor_id, std::uint32_t device_id, bool cpu) {
  if (cpu) return GpuFamily::Software;
  switch (vendor_id) {
    case pci::kNvidia:
      return find(kNvidia, device_id, GpuFamily::Unknown);
    case pci::kAmd:
      return find(kAmd, device_id, GpuFamily::AmdBeforeRdna3);
    case pci::kIntel:
      return find(kIntel, device_id, GpuFamily::Unknown);
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
    default:
      return false;
  }
}

}  // namespace ajp::gpu::vk
