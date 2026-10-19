#include "app/system_report.h"

#include <algorithm>
#include <format>
#include <numeric>

#include "app/engine_text.h"
#include "core/model_index.h"

namespace ajp::app {
namespace {

// Rang d'un type d'appareil pour le rendu : plus petit, meilleur.
int type_rank(DeviceType type) {
  switch (type) {
    case DeviceType::Discrete:
      return 0;
    case DeviceType::Integrated:
      return 1;
    case DeviceType::Virtual:
      return 2;
    case DeviceType::Other:
      return 3;
    case DeviceType::Cpu:
      return 4;
  }
  return 4;
}

std::string_view type_text(DeviceType type) {
  switch (type) {
    case DeviceType::Discrete:
      return "dédié";
    case DeviceType::Integrated:
      return "intégré";
    case DeviceType::Virtual:
      return "virtuel";
    case DeviceType::Cpu:
      return "logiciel";
    case DeviceType::Other:
      return "autre";
  }
  return "autre";
}

std::string presence(std::string_view what, const std::optional<std::string>& library) {
  return library ? std::format("{} : présent ({})", what, *library) : std::format("{} : absent", what);
}

}  // namespace

SystemReport detect_system(const LibraryLoader& loader, core::Os os) {
  SystemReport report{.os = os};
  if (auto devices = probe_vulkan(loader, os)) {
    report.devices = std::move(*devices);
  } else {
    report.vulkan_refusal = devices.error();
  }
  report.runtimes = probe_runtimes(loader, os);
  return report;
}

std::optional<std::size_t> render_device(const std::vector<VulkanDevice>& devices) {
  if (devices.empty()) return std::nullopt;
  std::vector<std::size_t> order(devices.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
    const int rank_a = type_rank(devices[a].type);
    const int rank_b = type_rank(devices[b].type);
    if (rank_a != rank_b) return rank_a < rank_b;
    return devices[a].vram_bytes > devices[b].vram_bytes;
  });
  return order.front();
}

core::GpuInfo gpu_info(const VulkanDevice& device) {
  const core::GpuVendor vendor = vendor_of(device.vendor_id);
  const bool matrix = device.cooperative_matrix;
  return core::GpuInfo{
      .vendor = vendor,
      .name = device.name,
      .vram_bytes = device.vram_bytes,
      .tensor_cores = vendor == core::GpuVendor::Nvidia && matrix,
      .rocm_supported = vendor == core::GpuVendor::Amd && matrix,
      .openvino_gpu = vendor == core::GpuVendor::Intel && (matrix || device.type == DeviceType::Discrete)};
}

core::SystemInfo to_system_info(const SystemReport& report) {
  core::SystemInfo info{.os = report.os};
  const std::optional<std::size_t> render = render_device(report.devices);
  if (!render) return info;
  info.gpus.push_back(gpu_info(report.devices[*render]));
  for (std::size_t i = 0; i < report.devices.size(); ++i) {
    if (i != *render) info.gpus.push_back(gpu_info(report.devices[i]));
  }
  return info;
}

std::optional<bool> cuda_fits_tensorrt(const RuntimeReport& runtimes) {
  if (!runtimes.cuda) return std::nullopt;
  return runtimes.cuda->version >= kTensorRtMinCuda;
}

std::vector<EngineBlocker> engine_blockers(const SystemReport& report) {
  if (report.os == core::Os::MacOs) return {};  // ni CUDA ni TensorRT
  const RuntimeReport& r = report.runtimes;
  const std::optional<bool> fits = cuda_fits_tensorrt(r);
  if (fits == true) return {};
  const std::string driver =
      fits ? std::format("pilote CUDA {}.{} trop ancien", r.cuda->major(), r.cuda->minor())
           : std::string("pilote CUDA introuvable");
  return {
      EngineBlocker{.engine = core::EngineId::TensorRt,
                    .reason = driver + " : TensorRT 11 exige CUDA 13 (pilote NVIDIA R580 ou plus récent)"}};
}

std::vector<std::string> describe(const SystemReport& report) {
  std::vector<std::string> lines;
  lines.push_back(std::format("Système : {}", core::os_name(report.os)));
  const std::optional<std::size_t> render = render_device(report.devices);
  for (std::size_t i = 0; i < report.devices.size(); ++i) {
    const VulkanDevice& d = report.devices[i];
    lines.push_back(std::format(
        "GPU {}{} : {} ({}, {}), PCI {:04x}:{:04x}, pilote {} {}, Vulkan {}.{}, UUID {}{}", i,
        render == i ? " (rendu)" : "", d.name, type_text(d.type), size_text(d.vram_bytes), d.vendor_id,
        d.device_id, d.driver_name.empty() ? "?" : d.driver_name, driver_version_text(d, report.os),
        d.api_version >> 22, (d.api_version >> 12) & 0x3FF, d.uuid_known ? uuid_text(d.uuid) : "inconnu",
        d.cooperative_matrix ? ", unités matricielles" : ""));
  }
  if (report.vulkan_refusal) {
    lines.push_back(std::format("Vulkan : aucun GPU lu ({} : {})", name(report.vulkan_refusal->code),
                                report.vulkan_refusal->detail));
  } else if (report.devices.empty()) {
    lines.emplace_back("Vulkan : aucun GPU");
  }
  const RuntimeReport& r = report.runtimes;
  if (r.cuda) {
    lines.push_back(std::format("Pilote CUDA : {}.{} ({}){}", r.cuda->major(), r.cuda->minor(),
                                r.cuda->library,
                                r.cuda->version >= kTensorRtMinCuda ? "" : ", trop ancien pour TensorRT 11"));
  } else {
    lines.emplace_back("Pilote CUDA : absent");
  }
  lines.push_back(presence("ROCm (HIP)", r.hip));
  lines.push_back(presence("MIGraphX", r.migraphx));
  lines.push_back(presence("OpenVINO", r.openvino));
  for (const std::string& note : r.notes) lines.push_back("Note : " + note);
  return lines;
}

}  // namespace ajp::app
