#include "app/play/gpu_telemetry.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <utility>

#include "core/charconv.h"
#include "vendor_hdr/dynamic_library.h"

namespace ajp::app::play {
namespace {

// Raisons d'écrêtage d'NVML (nvmlClocksThrottleReason*).
constexpr std::uint64_t kSoftwarePowerCap = 0x4;
constexpr std::uint64_t kHardwareSlowdown = 0x8;
constexpr std::uint64_t kSoftwareThermal = 0x20;
constexpr std::uint64_t kHardwareThermal = 0x40;
constexpr std::uint64_t kPowerBrake = 0x80;

}  // namespace

std::string nvml_uuid_text(const DeviceUuid& uuid) {
  std::string text = "GPU-";
  for (std::size_t i = 0; i < uuid.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) text += '-';
    text += std::format("{:02x}", uuid[i]);
  }
  return text;
}

std::string throttle_text(std::uint64_t mask) {
  std::string text;
  const auto add = [&](std::uint64_t bit, const char* what) {
    if ((mask & bit) == 0) return;
    if (!text.empty()) text += " + ";
    text += what;
  };
  add(kSoftwarePowerCap, "limite de puissance");
  add(kHardwareSlowdown, "ralentissement matériel");
  add(kSoftwareThermal | kHardwareThermal, "chaleur");
  add(kPowerBrake, "frein de puissance");
  return text;
}

std::string telemetry_line(const GpuSample& s) {
  std::string line = std::format("GPU {} / {} MHz", s.sm_mhz, s.sm_max_mhz);
  if (s.pstate >= 0) line += std::format(" · P{}", s.pstate);
  if (s.power_limit_mw > 0)
    line += std::format(" · {} / {} W", s.power_mw / 1000, s.power_limit_mw / 1000);
  else if (s.power_mw > 0)
    line += std::format(" · {} W", s.power_mw / 1000);
  line += std::format(" · {} °C · {} %", s.temperature_c, s.gpu_percent);
  if (s.pcie_generation > 0) line += std::format(" · PCIe {} x{}", s.pcie_generation, s.pcie_width);
  if (s.vram_total_mib > 0) {
    std::string vram = std::format("{:.1f} / {:.1f} Go", static_cast<double>(s.vram_used_mib) / 1024.0,
                                   static_cast<double>(s.vram_total_mib) / 1024.0);
    std::ranges::replace(vram, '.', ',');  // la virgule décimale, comme le reste de l'OSD
    line += " · VRAM " + vram;
  }
  if (const std::string why = throttle_text(s.throttle); !why.empty()) line += " · " + why;
  return line;
}

DpmClocks parse_amdgpu_dpm(std::string_view text) {
  DpmClocks clocks;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t end = text.find('\n', at);
    if (end == std::string_view::npos) end = text.size();
    const std::string_view line = text.substr(at, end - at);
    at = end + 1;
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos) continue;
    unsigned mhz = 0;
    bool any = false;
    for (std::size_t i = colon + 1; i < line.size() && (line[i] == ' ' || (line[i] >= '0' && line[i] <= '9'));
         ++i) {
      if (line[i] == ' ') {
        if (any) break;
        continue;
      }
      mhz = (mhz * 10) + static_cast<unsigned>(line[i] - '0');
      any = true;
    }
    if (!any) continue;
    clocks.max_mhz = std::max(clocks.max_mhz, mhz);
    if (line.ends_with('*')) clocks.current_mhz = mhz;
  }
  return clocks;
}

unsigned parse_pcie_generation(std::string_view text) {
  double gts = 0.0;
  const auto [end, error] = core::from_chars(text.data(), text.data() + text.size(), gts);
  if (error != std::errc{} || end == text.data()) return 0;
  // Débits par voie : 2,5 / 5 / 8 / 16 / 32 / 64 GT/s pour les générations 1 à 6.
  constexpr std::array<double, 6> kRates{2.5, 5.0, 8.0, 16.0, 32.0, 64.0};
  for (std::size_t i = 0; i < kRates.size(); ++i) {
    if (gts >= kRates[i] - 0.01 && gts <= kRates[i] + 0.01) return static_cast<unsigned>(i + 1);
  }
  return 0;
}

std::string telemetry_warning(const GpuSample& s) {
  if (s.sm_max_mhz > 0 && s.gpu_percent > 80 && s.sm_mhz * 10 < s.sm_max_mhz * 7)
    return std::format("horloges basses sous charge ({} sur {} MHz) : mode « performances maximales » ?",
                       s.sm_mhz, s.sm_max_mhz);
  return {};
}

bool nvidia_driver_older_than(std::string_view version, int major, int minor, int patch) {
  std::array<int, 3> parts{};
  std::size_t index = 0;
  bool any = false;
  for (const char c : version) {
    if (c >= '0' && c <= '9') {
      parts[index] = parts[index] * 10 + (c - '0');
      any = true;
    } else if (c == '.' && index + 1 < parts.size() && any) {
      ++index;
    } else {
      break;
    }
  }
  if (!any) return false;
  return parts < std::array<int, 3>{major, minor, patch};
}

#if defined(__linux__)

namespace {

using Handle = struct NvmlDevice*;
struct Utilization {
  unsigned gpu;
  unsigned memory;
};

struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;
constexpr int kClockSm = 1;
constexpr int kClockMem = 2;
constexpr std::int64_t kRefreshUs = 500'000;

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

// Un entier de sysfs, 0 s'il manque ou ne se lit pas.
std::uint64_t read_number(const std::filesystem::path& path, int base = 10) {
  const std::string text = read_text(path);
  try {
    return text.empty() ? 0 : std::stoull(text, nullptr, base);
  } catch (const std::exception&) {
    return 0;
  }
}

// Le dossier sysfs de la carte AMD (vendeur 0x1002, ce deviceID) ; vide si introuvable. Plusieurs
// cartes identiques : la première (l'UUID Vulkan ne se retrouve pas dans sysfs).
std::filesystem::path find_amd_card(std::uint32_t pci_device) {
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", error)) {
    const std::string name = entry.path().filename().string();
    if (!name.starts_with("card") || name.find('-') != std::string::npos) continue;
    const std::filesystem::path device = entry.path() / "device";
    if (read_number(device / "vendor", 16) == pci::kAmd && read_number(device / "device", 16) == pci_device)
      return device;
  }
  return {};
}

}  // namespace

struct GpuTelemetry::Impl {
  std::optional<vendor_hdr::DynamicLibrary> library;
  std::filesystem::path amd_card;  // sysfs amdgpu, à la place de NVML
  std::filesystem::path amd_hwmon;
  int (*shutdown)() = nullptr;
  Handle device = nullptr;
  int (*clock)(Handle, int, unsigned*) = nullptr;
  int (*max_clock)(Handle, int, unsigned*) = nullptr;
  int (*pstate)(Handle, int*) = nullptr;
  int (*power)(Handle, unsigned*) = nullptr;
  int (*power_limit)(Handle, unsigned*) = nullptr;
  int (*temperature)(Handle, int, unsigned*) = nullptr;
  int (*utilization)(Handle, Utilization*) = nullptr;
  int (*pcie_generation)(Handle, unsigned*) = nullptr;
  int (*pcie_width)(Handle, unsigned*) = nullptr;
  int (*throttle)(Handle, unsigned long long*) = nullptr;
  int (*memory)(Handle, NvmlMemory*) = nullptr;
  std::optional<GpuSample> last;
  std::int64_t last_us = 0;

  Impl() = default;
  explicit Impl(vendor_hdr::DynamicLibrary lib) : library(std::move(lib)) {}
  ~Impl() {
    if (shutdown != nullptr) shutdown();
  }
};

GpuTelemetry::GpuTelemetry(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GpuTelemetry::~GpuTelemetry() = default;

std::unique_ptr<GpuTelemetry> GpuTelemetry::open(const GpuDevice& device) {
  if (device.vendor() == GpuVendor::Amd) {
    const std::filesystem::path card = find_amd_card(device.pci_device);
    if (card.empty()) return nullptr;
    auto impl = std::make_unique<Impl>();
    impl->amd_card = card;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(card / "hwmon", error)) {
      impl->amd_hwmon = entry.path();
      break;
    }
    return std::unique_ptr<GpuTelemetry>(new GpuTelemetry(std::move(impl)));
  }
  if (device.vendor() != GpuVendor::Nvidia) return nullptr;
  // Une bibliothèque du pilote s'ouvre par son chemin absolu (dynamic_library.h) : les dossiers
  // où les distributions la rangent.
  static constexpr std::array<const char*, 6> kFolders{
      "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/lib/nvidia", "/usr/lib/wsl/lib", "/lib64"};
  std::optional<vendor_hdr::DynamicLibrary> library;
  for (const char* folder : kFolders) {
    const std::filesystem::path path = std::filesystem::path(folder) / "libnvidia-ml.so.1";
    std::error_code error;
    if (!std::filesystem::exists(path, error)) continue;
    if (auto opened = vendor_hdr::DynamicLibrary::open(path)) {
      library.emplace(std::move(*opened));
      break;
    }
  }
  if (!library) return nullptr;
  auto impl = std::make_unique<Impl>(std::move(*library));
  const vendor_hdr::DynamicLibrary& lib = *impl->library;
  const auto init = lib.symbol<int()>("nvmlInit_v2");
  const auto shutdown = lib.symbol<int()>("nvmlShutdown");
  const auto by_uuid = lib.symbol<int(const char*, Handle*)>("nvmlDeviceGetHandleByUUID");
  if (!init || !shutdown || !by_uuid || (*init)() != 0) return nullptr;
  impl->shutdown = *shutdown;
  const std::string uuid = nvml_uuid_text(device.uuid);
  if ((*by_uuid)(uuid.c_str(), &impl->device) != 0) return nullptr;
  const auto bind = [&](auto& slot, const char* name) {
    using Fn = std::remove_pointer_t<std::remove_reference_t<decltype(slot)>>;
    if (auto found = lib.symbol<Fn>(name)) slot = *found;
  };
  bind(impl->clock, "nvmlDeviceGetClockInfo");
  bind(impl->max_clock, "nvmlDeviceGetMaxClockInfo");
  bind(impl->pstate, "nvmlDeviceGetPerformanceState");
  bind(impl->power, "nvmlDeviceGetPowerUsage");
  bind(impl->power_limit, "nvmlDeviceGetEnforcedPowerLimit");
  bind(impl->temperature, "nvmlDeviceGetTemperature");
  bind(impl->utilization, "nvmlDeviceGetUtilizationRates");
  bind(impl->pcie_generation, "nvmlDeviceGetCurrPcieLinkGeneration");
  bind(impl->pcie_width, "nvmlDeviceGetCurrPcieLinkWidth");
  bind(impl->throttle, "nvmlDeviceGetCurrentClocksThrottleReasons");
  bind(impl->memory, "nvmlDeviceGetMemoryInfo");
  if (impl->clock == nullptr) return nullptr;
  return std::unique_ptr<GpuTelemetry>(new GpuTelemetry(std::move(impl)));
}

std::optional<GpuSample> GpuTelemetry::sample(std::int64_t host_us) {
  Impl& i = *impl_;
  if (i.last && host_us - i.last_us < kRefreshUs) return i.last;
  GpuSample s;
  if (!i.amd_card.empty()) {
    const DpmClocks core = parse_amdgpu_dpm(read_text(i.amd_card / "pp_dpm_sclk"));
    s.sm_mhz = core.current_mhz;
    s.sm_max_mhz = core.max_mhz;
    s.mem_mhz = parse_amdgpu_dpm(read_text(i.amd_card / "pp_dpm_mclk")).current_mhz;
    s.gpu_percent = static_cast<unsigned>(read_number(i.amd_card / "gpu_busy_percent"));
    if (!i.amd_hwmon.empty()) {
      // En microwatts et en millidegrés.
      const std::uint64_t watts = read_number(i.amd_hwmon / "power1_average") != 0
                                      ? read_number(i.amd_hwmon / "power1_average")
                                      : read_number(i.amd_hwmon / "power1_input");
      s.power_mw = static_cast<unsigned>(watts / 1000);
      s.power_limit_mw = static_cast<unsigned>(read_number(i.amd_hwmon / "power1_cap") / 1000);
      s.temperature_c = static_cast<unsigned>(read_number(i.amd_hwmon / "temp1_input") / 1000);
    }
    s.vram_total_mib = static_cast<unsigned>(read_number(i.amd_card / "mem_info_vram_total") / kMiB);
    s.vram_used_mib = static_cast<unsigned>(read_number(i.amd_card / "mem_info_vram_used") / kMiB);
    s.pcie_generation = parse_pcie_generation(read_text(i.amd_card / "current_link_speed"));
    if (s.pcie_generation > 0)
      s.pcie_width = static_cast<unsigned>(read_number(i.amd_card / "current_link_width"));
    i.last = s;
    i.last_us = host_us;
    return s;
  }
  i.clock(i.device, kClockSm, &s.sm_mhz);
  if (i.clock != nullptr) i.clock(i.device, kClockMem, &s.mem_mhz);
  if (i.max_clock != nullptr) i.max_clock(i.device, kClockSm, &s.sm_max_mhz);
  if (i.pstate != nullptr) i.pstate(i.device, &s.pstate);
  if (i.power != nullptr) i.power(i.device, &s.power_mw);
  if (i.power_limit != nullptr) i.power_limit(i.device, &s.power_limit_mw);
  if (i.temperature != nullptr) i.temperature(i.device, 0, &s.temperature_c);
  if (i.utilization != nullptr) {
    Utilization usage{};
    if (i.utilization(i.device, &usage) == 0) s.gpu_percent = usage.gpu;
  }
  if (i.pcie_generation != nullptr) i.pcie_generation(i.device, &s.pcie_generation);
  if (i.pcie_width != nullptr) i.pcie_width(i.device, &s.pcie_width);
  if (i.memory != nullptr) {
    NvmlMemory memory{};
    if (i.memory(i.device, &memory) == 0) {
      s.vram_total_mib = static_cast<unsigned>(memory.total / kMiB);
      s.vram_used_mib = static_cast<unsigned>(memory.used / kMiB);
    }
  }
  if (i.throttle != nullptr) {
    unsigned long long mask = 0;
    if (i.throttle(i.device, &mask) == 0) s.throttle = mask;
  }
  i.last = s;
  i.last_us = host_us;
  return s;
}

#else

struct GpuTelemetry::Impl {};
GpuTelemetry::GpuTelemetry(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GpuTelemetry::~GpuTelemetry() = default;
std::unique_ptr<GpuTelemetry> GpuTelemetry::open(const GpuDevice&) {
  return nullptr;
}
std::optional<GpuSample> GpuTelemetry::sample(std::int64_t) {
  return std::nullopt;
}

#endif

}  // namespace ajp::app::play
