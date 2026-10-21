// MIGraphX sur la mémoire de l'appareil : les tenseurs de la chaîne Vulkan, exportés en
// descripteurs opaques et importés par HIP, servent directement d'entrées et de sorties au
// programme (compilé sans offload_copy). Plus de copies par le bus PCIe à chaque image : sur
// une RX 9070 XT en 1440p, deux aller-retours de plusieurs dizaines de Mo évités par réseau. Avec
// la timeline Vulkan importée, le réseau s'enchaîne aussi côté GPU : plus d'attente de l'hôte.
#include <cstring>
#include <exception>
#include <format>
#include <hip/hip_runtime_api.h>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/migraphx/migraphx_detail.h"

namespace ajp::gpu::migraphx_rt::detail {
namespace {

using vk::net::HostBinding;

class ImportedHip final : public vk::net::ImportedMemory {
 public:
  ImportedHip(hipExternalMemory_t memory, void* pointer) : memory_(memory), pointer_(pointer) {}
  ImportedHip(const ImportedHip&) = delete;
  ImportedHip& operator=(const ImportedHip&) = delete;
  ImportedHip(ImportedHip&&) = delete;
  ImportedHip& operator=(ImportedHip&&) = delete;
  ~ImportedHip() override {
    (void)hipFree(pointer_);
    (void)hipDestroyExternalMemory(memory_);
  }
  void* device() const override { return pointer_; }

 private:
  hipExternalMemory_t memory_;
  void* pointer_;
};

// Tampon HIP à soi : paramètres internes du programme (scratch), conversions de type.
class HipBuffer {
 public:
  HipBuffer() = default;
  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
  HipBuffer& operator=(HipBuffer&&) = delete;
  ~HipBuffer() {
    if (pointer_ != nullptr) (void)hipFree(pointer_);
  }
  // Au moins bytes octets ; nullptr si HIP refuse.
  void* get(std::size_t bytes) {
    if (bytes_ >= bytes && pointer_ != nullptr) return pointer_;
    if (pointer_ != nullptr) (void)hipFree(pointer_);
    pointer_ = nullptr;
    bytes_ = 0;
    if (hipMalloc(&pointer_, bytes) != hipSuccess) {
      pointer_ = nullptr;
      return nullptr;
    }
    bytes_ = bytes;
    return pointer_;
  }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

std::size_t element_bytes(TensorType type) {
  return type == TensorType::F32 ? 4U : 2U;
}

// La timeline de la file Vulkan, importée par HIP.
class ImportedTimeline final : public vk::net::ImportedSemaphore {
 public:
  explicit ImportedTimeline(hipExternalSemaphore_t semaphore) : semaphore_(semaphore) {}
  ImportedTimeline(const ImportedTimeline&) = delete;
  ImportedTimeline& operator=(const ImportedTimeline&) = delete;
  ImportedTimeline(ImportedTimeline&&) = delete;
  ImportedTimeline& operator=(ImportedTimeline&&) = delete;
  ~ImportedTimeline() override { (void)hipDestroyExternalSemaphore(semaphore_); }
  hipExternalSemaphore_t handle() const { return semaphore_; }

 private:
  hipExternalSemaphore_t semaphore_;
};

class DeviceNet final : public vk::net::RuntimeNet {
 public:
  DeviceNet(int device, migraphx::program program, vk::net::IoNames names)
      : device_(device), program_(std::move(program)), names_(std::move(names)) {}
  DeviceNet(const DeviceNet&) = delete;
  DeviceNet& operator=(const DeviceNet&) = delete;
  DeviceNet(DeviceNet&&) = delete;
  DeviceNet& operator=(DeviceNet&&) = delete;
  ~DeviceNet() override {
    if (stream_ == nullptr) return;
    (void)hipStreamSynchronize(stream_);  // scratch et sorties encore en usage sur le flux
    (void)hipStreamDestroy(stream_);
  }
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "MIGraphX : nombre d'entrées ou de sorties");
    try {
      if (hipSetDevice(device_) != hipSuccess) return refuse(RefusalCode::DeviceUnavailable, "hipSetDevice");
      // Les tampons à soi servent aussi au flux de run_ordered.
      if (stream_ != nullptr && hipStreamSynchronize(stream_) != hipSuccess)
        return refuse(RefusalCode::DeviceUnavailable, "hipStreamSynchronize");
      auto params = bind(inputs, outputs);
      if (!params) return std::unexpected(params.error());
      const migraphx::arguments results = program_.eval(*params);
      if (hipDeviceSynchronize() != hipSuccess)
        return refuse(RefusalCode::DeviceUnavailable, "hipDeviceSynchronize");
      if (results.size() != outputs.size()) {
        return refuse(RefusalCode::InputMismatch,
                      std::format("MIGraphX : {} sorties, {} attendues", results.size(), outputs.size()));
      }
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const migraphx::argument result = results[i];
        if (result.data() == outputs[i].device) continue;  // écrite en place
        if (auto ok = deliver(result, outputs[i], names_.outputs[i]); !ok) return ok;
      }
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable, std::format("MIGraphX, exécution : {}", e.what()));
    }
  }

  // Comme run, mis en file sur le flux HIP du réseau entre l'attente de wait et le signal de
  // signal sur la timeline Vulkan : ni attente ni copie par l'hôte. Refus FeatureUnavailable avant
  // toute mise en file si une entrée ou une sortie passerait par l'hôte (type à convertir).
  std::expected<void, Refusal> run_ordered(std::span<const HostBinding> inputs,
                                           std::span<const HostBinding> outputs,
                                           vk::net::ImportedSemaphore& semaphore, std::uint64_t wait,
                                           std::uint64_t signal) override {
    if (inputs.size() != names_.inputs.size() || outputs.size() != names_.outputs.size())
      return refuse(RefusalCode::InputMismatch, "MIGraphX : nombre d'entrées ou de sorties");
    auto* timeline = dynamic_cast<ImportedTimeline*>(&semaphore);
    if (timeline == nullptr) return refuse(RefusalCode::FeatureUnavailable, "MIGraphX : sémaphore étranger");
    try {
      if (hipSetDevice(device_) != hipSuccess) return refuse(RefusalCode::DeviceUnavailable, "hipSetDevice");
      const migraphx::program_parameter_shapes shapes = program_.get_parameter_shapes();
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!matches(shapes[names_.inputs[i].c_str()], inputs[i]))
          return refuse(RefusalCode::FeatureUnavailable,
                        std::format("{} : conversion par l'hôte", names_.inputs[i]));
      }
      const migraphx::shapes produced = program_.get_output_shapes();
      if (produced.size() != outputs.size())
        return refuse(RefusalCode::FeatureUnavailable, "MIGraphX : sorties du programme");
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (!matches(produced[i], outputs[i]))
          return refuse(RefusalCode::FeatureUnavailable,
                        std::format("{} : conversion par l'hôte", names_.outputs[i]));
      }
      auto params = bind(inputs, outputs);
      if (!params) return std::unexpected(params.error());
      if (stream_ == nullptr && hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess) {
        stream_ = nullptr;
        return refuse(RefusalCode::DeviceUnavailable, "hipStreamCreateWithFlags");
      }
      // Après un refus d'ici, rien ne signale : l'appelant le fait par l'hôte.
      const hipExternalSemaphore_t handle = timeline->handle();
      hipExternalSemaphoreWaitParams waits{};
      waits.params.fence.value = wait;
      if (const hipError_t e = hipWaitExternalSemaphoresAsync(&handle, &waits, 1, stream_); e != hipSuccess)
        return refuse(RefusalCode::DeviceUnavailable,
                      std::format("hipWaitExternalSemaphoresAsync : {}", hipGetErrorString(e)));
      const migraphx::arguments results = program_.run_async(*params, stream_);
      if (results.size() != outputs.size()) {
        return refuse(RefusalCode::InputMismatch,
                      std::format("MIGraphX : {} sorties, {} attendues", results.size(), outputs.size()));
      }
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const migraphx::argument result = results[i];
        if (result.data() == outputs[i].device) continue;  // écrite en place
        if (hipMemcpyAsync(outputs[i].device, result.data(), outputs[i].bytes, hipMemcpyDeviceToDevice,
                           stream_) != hipSuccess)
          return refuse(RefusalCode::DeviceUnavailable,
                        std::format("{} : hipMemcpyAsync", names_.outputs[i]));
      }
      hipExternalSemaphoreSignalParams signals{};
      signals.params.fence.value = signal;
      if (const hipError_t e = hipSignalExternalSemaphoresAsync(&handle, &signals, 1, stream_);
          e != hipSuccess)
        return refuse(RefusalCode::DeviceUnavailable,
                      std::format("hipSignalExternalSemaphoresAsync : {}", hipGetErrorString(e)));
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable, std::format("MIGraphX, exécution : {}", e.what()));
    }
  }

 private:
  static bool matches(const migraphx::shape& shape, const HostBinding& b) {
    const auto type = type_of(shape.type());
    return type && *type == b.type && shape.bytes() == b.bytes && b.device != nullptr;
  }

  // Les paramètres du programme : entrées (converties si besoin), sorties en place quand le type
  // concorde, le reste dans des tampons à soi.
  std::expected<migraphx::program_parameters, Refusal> bind(std::span<const HostBinding> inputs,
                                                            std::span<const HostBinding> outputs) {
    const migraphx::program_parameter_shapes shapes = program_.get_parameter_shapes();
    migraphx::program_parameters params;
    std::vector<std::string_view> bound;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const std::string& name = names_.inputs[i];
      const migraphx::shape shape = shapes[name.c_str()];
      auto data = input(name, shape, inputs[i]);
      if (!data) return std::unexpected(data.error());
      params.add(name.c_str(), migraphx::argument(shape, *data));
      bound.emplace_back(name);
    }
    // Sorties en paramètres (« main:#output_0 »…, dans l'ordre de l'ONNX) : écrites en place
    // quand le type concorde.
    std::vector<const char*> output_params(outputs.size(), nullptr);
    for (const char* pname : shapes.names()) {
      const std::string_view p(pname);
      const std::size_t at = p.rfind("#output_");
      if (at == std::string_view::npos) continue;
      const std::size_t index = std::stoul(std::string(p.substr(at + 8)));
      if (index < outputs.size()) output_params[index] = pname;
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (output_params[i] == nullptr) continue;
      const migraphx::shape shape = shapes[output_params[i]];
      void* data =
          matches(shape, outputs[i]) ? outputs[i].device : scratch_[output_params[i]].get(shape.bytes());
      if (data == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "MIGraphX : tampon de sortie");
      params.add(output_params[i], migraphx::argument(shape, data));
      bound.emplace_back(output_params[i]);
    }
    // Le reste (scratch selon la version de MIGraphX) : des tampons à soi, gardés.
    for (const char* pname : shapes.names()) {
      if (std::ranges::contains(bound, std::string_view(pname))) continue;
      void* data = scratch_[pname].get(shapes[pname].bytes());
      if (data == nullptr)
        return refuse(RefusalCode::GpuMemoryExhausted, std::format("MIGraphX : {}", pname));
      params.add(pname, migraphx::argument(shapes[pname], data));
    }
    return params;
  }

  // L'adresse d'une entrée : le tenseur importé, ou une copie convertie par l'hôte (programme en
  // float pour des tenseurs F16 : interface non réécrite à la construction).
  std::expected<void*, Refusal> input(const std::string& name, const migraphx::shape& shape,
                                      const HostBinding& b) {
    if (b.device == nullptr)
      return refuse(RefusalCode::InputMismatch, std::format("{} : tenseur non partagé", name));
    if (matches(shape, b)) return b.device;
    const auto want = type_of(shape.type());
    if (!want) return refuse(RefusalCode::FormatUnsupported, std::format("{} : type MIGraphX", name));
    std::vector<std::byte>& host = host_[name];
    host.resize(b.bytes);
    std::vector<std::byte>& converted = converted_[name];
    converted.resize(shape.bytes());
    if (hipMemcpy(host.data(), b.device, b.bytes, hipMemcpyDeviceToHost) != hipSuccess)
      return refuse(RefusalCode::DeviceUnavailable, std::format("{} : hipMemcpy", name));
    if (auto ok = vk::net::copy_converted(host.data(), b.type, converted.data(), *want, shape.elements());
        !ok)
      return std::unexpected(ok.error());
    void* staged = scratch_[name].get(converted.size());
    if (staged == nullptr ||
        hipMemcpy(staged, converted.data(), converted.size(), hipMemcpyHostToDevice) != hipSuccess)
      return refuse(RefusalCode::DeviceUnavailable, std::format("{} : copie convertie", name));
    return staged;
  }

  // Une sortie hors de son tenseur : copiée sur l'appareil si le type concorde, sinon convertie
  // par l'hôte.
  std::expected<void, Refusal> deliver(const migraphx::argument& result, const HostBinding& out,
                                       const std::string& name) {
    const migraphx::shape shape = result.get_shape();
    const auto type = type_of(shape.type());
    if (!type) return refuse(RefusalCode::FormatUnsupported, "sortie MIGraphX d'un type inattendu");
    const std::size_t count = shape.elements();
    if (count * element_bytes(out.type) != out.bytes || out.device == nullptr)
      return refuse(RefusalCode::InputMismatch, std::format("{} : {} éléments", name, count));
    if (*type == out.type) {
      if (hipMemcpy(out.device, result.data(), out.bytes, hipMemcpyDeviceToDevice) != hipSuccess)
        return refuse(RefusalCode::DeviceUnavailable, std::format("{} : hipMemcpy", name));
      return {};
    }
    std::vector<std::byte>& host = host_[name];
    host.resize(shape.bytes());
    std::vector<std::byte>& converted = converted_[name];
    converted.resize(out.bytes);
    if (hipMemcpy(host.data(), result.data(), host.size(), hipMemcpyDeviceToHost) != hipSuccess)
      return refuse(RefusalCode::DeviceUnavailable, std::format("{} : hipMemcpy", name));
    if (auto ok = vk::net::copy_converted(host.data(), *type, converted.data(), out.type, count); !ok)
      return ok;
    if (hipMemcpy(out.device, converted.data(), converted.size(), hipMemcpyHostToDevice) != hipSuccess)
      return refuse(RefusalCode::DeviceUnavailable, std::format("{} : hipMemcpy", name));
    return {};
  }

  int device_;
  migraphx::program program_;
  vk::net::IoNames names_;
  hipStream_t stream_ = nullptr;  // celui de run_ordered, créé au premier appel
  std::map<std::string, HipBuffer, std::less<>> scratch_;
  std::map<std::string, std::vector<std::byte>, std::less<>> host_;
  std::map<std::string, std::vector<std::byte>, std::less<>> converted_;
};

}  // namespace

std::unique_ptr<vk::net::RuntimeNet> device_net(int device, migraphx::program program,
                                                vk::net::IoNames names) {
  return std::make_unique<DeviceNet>(device, std::move(program), std::move(names));
}

std::expected<std::unique_ptr<vk::net::ImportedMemory>, Refusal> import_memory(int device, int fd,
                                                                               std::uint64_t bytes) {
  if (hipSetDevice(device) != hipSuccess) return refuse(RefusalCode::DeviceUnavailable, "hipSetDevice");
  hipExternalMemoryHandleDesc desc{};
  desc.type = hipExternalMemoryHandleTypeOpaqueFd;
  desc.handle.fd = fd;
  desc.size = bytes;
  desc.flags = hipExternalMemoryDedicated;  // allocation Vulkan dédiée (shared_tensor.cpp)
  hipExternalMemory_t memory = nullptr;
  if (const hipError_t e = hipImportExternalMemory(&memory, &desc); e != hipSuccess)
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("hipImportExternalMemory : {}", hipGetErrorString(e)));
  hipExternalMemoryBufferDesc buffer{};
  buffer.offset = 0;
  buffer.size = bytes;
  void* pointer = nullptr;
  if (const hipError_t e = hipExternalMemoryGetMappedBuffer(&pointer, memory, &buffer); e != hipSuccess) {
    (void)hipDestroyExternalMemory(memory);
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("hipExternalMemoryGetMappedBuffer : {}", hipGetErrorString(e)));
  }
  return std::make_unique<ImportedHip>(memory, pointer);
}

std::expected<std::unique_ptr<vk::net::ImportedSemaphore>, Refusal> import_semaphore(int device, int fd) {
  if (hipSetDevice(device) != hipSuccess) return refuse(RefusalCode::DeviceUnavailable, "hipSetDevice");
  hipExternalSemaphoreHandleDesc desc{};
  desc.type = hipExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  desc.handle.fd = fd;
  hipExternalSemaphore_t semaphore = nullptr;
  if (const hipError_t e = hipImportExternalSemaphore(&semaphore, &desc); e != hipSuccess)
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("hipImportExternalSemaphore : {}", hipGetErrorString(e)));
  return std::make_unique<ImportedTimeline>(semaphore);
}

}  // namespace ajp::gpu::migraphx_rt::detail
