// Les réseaux du runtime OpenVINO (openvino_runtime.cpp), interne au module : un refus nommé, le
// type d'élément d'un tenseur, les mémoires USM hôte du GPU que Vulkan importe, et le réseau
// compilé qui les reconnaît, comme les mémoires d'appareil de la chaîne qu'OpenVINO importe
// (openvino_import.h) : openvino_net.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <string>
#include <utility>

#include "gpu/openvino/openvino_import.h"
#include "gpu/openvino/openvino_runtime.h"

namespace ajp::gpu::openvino_rt {

inline std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

inline ov::element::Type element_of(TensorType type) {
  return type == TensorType::F32 ? ov::element::f32 : ov::element::f16;
}

// Les mémoires USM hôte du contexte GPU (Runtime::allocate_host), que Vulkan importe sur un GPU
// intégré : le greffon GPU les lit et les écrit sans copie. Les réseaux les reconnaissent à leur
// adresse et les lui donnent pour ce qu'elles sont (USM_USER_BUFFER), pas comme mémoire hôte
// ordinaire, qu'il recopierait.
class UsmPool {
 public:
  explicit UsmPool(ov::RemoteContext context) : context_(std::move(context)) {}

  // Lève les exceptions d'OpenVINO (pilote sans USM).
  std::pair<ov::RemoteTensor, std::byte*> allocate(std::uint64_t bytes) {
    ov::RemoteTensor tensor = context_.create_tensor(
        ov::element::u8, {static_cast<std::size_t>(bytes)},
        {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}});
    auto* data = static_cast<std::byte*>(
        tensor.get_params().at(ov::intel_gpu::mem_handle.name()).as<ov::intel_gpu::gpu_handle_param>());
    const std::scoped_lock lock(mutex_);
    blocks_[data] = bytes;
    return {std::move(tensor), data};
  }
  void forget(const std::byte* data) {
    const std::scoped_lock lock(mutex_);
    blocks_.erase(data);
  }
  // L'adresse est-elle dans un bloc USM vivant (les tenseurs de la chaîne y ont un décalage) ?
  bool owns(const std::byte* data, std::uint64_t bytes) const {
    const std::scoped_lock lock(mutex_);
    auto it = blocks_.upper_bound(data);
    if (it == blocks_.begin()) return false;
    --it;
    return data + bytes <= it->first + it->second;
  }
  ov::RemoteTensor wrap(ov::element::Type type, const ov::Shape& shape, std::byte* data) {
    return context_.create_tensor(
        type, shape,
        {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
         {ov::intel_gpu::mem_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(data)}});
  }

 private:
  ov::RemoteContext context_;
  mutable std::mutex mutex_;
  std::map<const std::byte*, std::uint64_t> blocks_;  // début, taille
};

class UsmMemory final : public vk::net::HostMemory {
 public:
  UsmMemory(std::shared_ptr<UsmPool> pool, ov::RemoteTensor tensor, std::byte* data)
      : pool_(std::move(pool)), tensor_(std::move(tensor)), data_(data) {}
  UsmMemory(const UsmMemory&) = delete;
  UsmMemory& operator=(const UsmMemory&) = delete;
  UsmMemory(UsmMemory&&) = delete;
  UsmMemory& operator=(UsmMemory&&) = delete;
  ~UsmMemory() override { pool_->forget(data_); }
  std::byte* data() const override { return data_; }

 private:
  std::shared_ptr<UsmPool> pool_;
  ov::RemoteTensor tensor_;  // possède la mémoire
  std::byte* data_;
};

// Un réseau compilé prêt à servir la chaîne : ses tenseurs enveloppent la mémoire du lecteur sans
// copie côté hôte. usm et imports : les mémoires USM du GPU et celles de la chaîne importées sur
// une carte, pour un réseau compilé sur ce GPU ; nuls sur le NPU, qui lit toute mémoire hôte de la
// même façon.
std::unique_ptr<vk::net::RuntimeNet> make_net(ov::CompiledModel compiled, vk::net::IoNames names,
                                              std::shared_ptr<UsmPool> usm,
                                              std::shared_ptr<DeviceImports> imports);

}  // namespace ajp::gpu::openvino_rt
