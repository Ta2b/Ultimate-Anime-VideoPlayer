#include "app/play/gpu_telemetry.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <format>
#include <utility>

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
  if (const std::string why = throttle_text(s.throttle); !why.empty()) line += " · " + why;
  return line;
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

constexpr int kClockSm = 1;
constexpr int kClockMem = 2;
constexpr std::int64_t kRefreshUs = 500'000;

}  // namespace

struct GpuTelemetry::Impl {
  vendor_hdr::DynamicLibrary library;
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
  std::optional<GpuSample> last;
  std::int64_t last_us = 0;

  explicit Impl(vendor_hdr::DynamicLibrary lib) : library(std::move(lib)) {}
  ~Impl() {
    if (shutdown != nullptr) shutdown();
  }
};

GpuTelemetry::GpuTelemetry(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GpuTelemetry::~GpuTelemetry() = default;

std::unique_ptr<GpuTelemetry> GpuTelemetry::open(const GpuDevice& device) {
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
  const vendor_hdr::DynamicLibrary& lib = impl->library;
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
  if (impl->clock == nullptr) return nullptr;
  return std::unique_ptr<GpuTelemetry>(new GpuTelemetry(std::move(impl)));
}

std::optional<GpuSample> GpuTelemetry::sample(std::int64_t host_us) {
  Impl& i = *impl_;
  if (i.last && host_us - i.last_us < kRefreshUs) return i.last;
  GpuSample s;
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
