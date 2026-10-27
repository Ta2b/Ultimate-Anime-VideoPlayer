// La mémoire d'appareil de la chaîne Vulkan importée par OpenVINO sur une carte Arc (interop,
// net_runtime.h) : le dma-buf (Linux) ou le HANDLE NT (Windows) qu'elle exporte devient un tenseur
// distant du GPU (BUFFER_FROM_HANDLE, cl_khr_external_memory), que les réseaux lient tel quel :
// plus aucune copie par l'hôte ni par le bus PCIe. Interne au module.
#pragma once

#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <optional>
#include <utility>

#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::openvino_rt {

// Les imports vivants du contexte GPU, par leur cl_mem (ImportedMemory::device, que la chaîne
// passe en HostBinding::device) : un réseau (openvino_net.cpp) y retrouve le tenseur distant
// d'une liaison.
class DeviceImports : public std::enable_shared_from_this<DeviceImports> {
 public:
  // clSetMemObjectDestructorCallback (OpenCL 1.1), sous Windows : chaque HANDLE est rendu à la
  // destruction de sa mémoire. Conventions d'appel de x64, où CL_API_CALL et CL_CALLBACK ne
  // changent rien.
  using SetDestructor = std::int32_t (*)(void* mem, void (*notify)(void* mem, void* user), void* user);

  // Nul si l'import est impossible ici : sous Windows, sans le chargeur OpenCL du greffon GPU.
  static std::shared_ptr<DeviceImports> open(ov::RemoteContext context);
  DeviceImports(ov::RemoteContext context, SetDestructor set_destructor)
      : context_(std::move(context)), set_destructor_(set_destructor) {}

  // Importe la mémoire, au type et à la forme du tenseur qu'elle porte ; prend le descripteur en
  // cas de succès, le laisse à l'appelant sinon. Refus FeatureUnavailable : pilote sans
  // cl_khr_external_memory, descripteur refusé.
  std::expected<std::unique_ptr<vk::net::ImportedMemory>, Refusal> import(vk::net::HandleValue handle,
                                                                          TensorType type,
                                                                          const Shape& shape);
  // Le tenseur d'une liaison sur un import : l'import lui-même au même type et à la même forme,
  // sinon une vue du même cl_mem (OCL_BUFFER) qui y tient. nullopt : device n'est le début d'aucun
  // import vivant (décalage dans un tenseur, que le cl_mem ne dit pas), ou la liaison déborde.
  // Lève les exceptions d'OpenVINO.
  std::optional<ov::RemoteTensor> tensor_for(void* device, ov::element::Type type, const ov::Shape& shape,
                                             std::uint64_t bytes);
  void forget(void* device);

 private:
  ov::RemoteContext context_;
  [[maybe_unused]] SetDestructor set_destructor_;  // nul sous Linux
  std::mutex mutex_;
  std::map<void*, ov::RemoteTensor> live_;  // cl_mem → tenseur de l'import
};

}  // namespace ajp::gpu::openvino_rt
