#include "gpu/openvino/openvino_import.h"

#include <cstring>
#include <exception>
#include <format>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <string>

#include "gpu/openvino/openvino_net.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ajp::gpu::openvino_rt {
namespace {

using vk::net::HandleValue;

void* mem_of(const ov::RemoteTensor& tensor) {
  return tensor.get_params().at(ov::intel_gpu::mem_handle.name()).as<ov::intel_gpu::gpu_handle_param>();
}

// Une mémoire importée, inscrite dans DeviceImports tant qu'elle vit ; à sa destruction, les
// réseaux qui la lient encore gardent leur tenseur, et le cl_mem avec, jusqu'à la liaison suivante.
class DeviceImport final : public vk::net::ImportedMemory {
 public:
  DeviceImport(std::shared_ptr<DeviceImports> owner, void* mem, ov::RemoteTensor tensor)
      : owner_(std::move(owner)), mem_(mem), tensor_(std::move(tensor)) {}
  DeviceImport(const DeviceImport&) = delete;
  DeviceImport& operator=(const DeviceImport&) = delete;
  DeviceImport(DeviceImport&&) = delete;
  DeviceImport& operator=(DeviceImport&&) = delete;
  ~DeviceImport() override { owner_->forget(mem_); }
  void* device() const override { return mem_; }

  // Par une copie bloquante de tout le tenseur vers l'hôte et retour (remote_tensor.cpp :
  // is_blocking), une seule fois, au premier import.
  std::expected<std::uint32_t, Refusal> exchange(std::uint32_t value) override {
    try {
      if (tensor_.get_byte_size() < 2 * sizeof(std::uint32_t))
        return refuse(RefusalCode::FormatUnsupported, "OpenVINO : tenseur trop petit pour l'échange");
      ov::Tensor host(tensor_.get_element_type(), tensor_.get_shape());
      tensor_.copy_to(host);
      auto* bytes = static_cast<std::byte*>(host.data());
      std::uint32_t seen = 0;
      std::memcpy(&seen, bytes, sizeof(seen));
      std::memcpy(bytes + sizeof(seen), &value, sizeof(value));
      tensor_.copy_from(host);
      return seen;
    } catch (const std::exception& e) {
      return refuse(RefusalCode::DeviceUnavailable, std::format("OpenVINO, échange : {}", e.what()));
    }
  }

 private:
  std::shared_ptr<DeviceImports> owner_;
  void* mem_;
  ov::RemoteTensor tensor_;
};

ov::RemoteTensor import_tensor(ov::RemoteContext& context, ov::element::Type type, const ov::Shape& shape,
                               ov::intel_gpu::os_handle_param handle) {
  return context.create_tensor(
      type, shape,
      {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::BUFFER_FROM_HANDLE},
       {ov::intel_gpu::os_handle.name(), handle}});
}

#if defined(_WIN32)

// Le HANDLE rendu une fois le cl_mem détruit (mem_obj.cpp de NEO : après sa mémoire), pas
// avant : OpenVINO garde les mémoires importées en cache par la valeur du HANDLE, et une valeur
// rendue trop tôt pourrait revenir, pour une autre mémoire, tant qu'un réseau tient encore celle-ci.
void close_on_destroy(void* /*mem*/, void* user) {
  CloseHandle(static_cast<HANDLE>(user));
}

#else

// Le même fichier ouvert (dma-buf : un inode par mémoire), que descriptor désigne encore.
bool same_file(int descriptor, const struct stat& file) {
  struct stat now{};
  return fstat(descriptor, &now) == 0 && now.st_dev == file.st_dev && now.st_ino == file.st_ino;
}

#endif

}  // namespace

std::shared_ptr<DeviceImports> DeviceImports::open(ov::RemoteContext context) {
#if defined(_WIN32)
  // Le chargeur que le greffon GPU a déjà chargé, celui de ses cl_mem.
  HMODULE opencl = GetModuleHandleW(L"OpenCL.dll");
  if (opencl == nullptr) return nullptr;
  auto set_destructor =
      reinterpret_cast<SetDestructor>(GetProcAddress(opencl, "clSetMemObjectDestructorCallback"));
  if (set_destructor == nullptr) return nullptr;
  return std::make_shared<DeviceImports>(std::move(context), set_destructor);
#else
  return std::make_shared<DeviceImports>(std::move(context), nullptr);
#endif
}

std::expected<std::unique_ptr<vk::net::ImportedMemory>, Refusal> DeviceImports::import(HandleValue handle,
                                                                                       TensorType type,
                                                                                       const Shape& shape) {
  ov::Shape dims;
  for (const std::int64_t d : shape.dims) dims.push_back(static_cast<std::size_t>(d));
  const ov::element::Type element = element_of(type);
  ov::RemoteTensor tensor;
#if defined(_WIN32)
  // NEO ouvre la ressource par le HANDLE sans le garder : il reste à l'appelant jusqu'au rappel.
  auto* const nt = reinterpret_cast<HANDLE>(handle);
  try {
    tensor = import_tensor(context_, element, dims, nt);
  } catch (const std::exception& e) {
    return refuse(RefusalCode::FeatureUnavailable, std::format("OpenVINO, import : {}", e.what()));
  }
#else
  // NEO garde le descripteur qu'il importe et le ferme avec le cl_mem (drm_memory_manager.cpp :
  // closeSharedHandle), même quand OpenVINO lève après l'import : une copie lui est donnée, et
  // l'original, fermé ici au succès, reste à l'appelant sinon.
  const int original = static_cast<int>(handle);
  struct stat file{};
  if (fstat(original, &file) != 0)
    return refuse(RefusalCode::FeatureUnavailable, "OpenVINO, import : descripteur invalide");
  const int copy = fcntl(original, F_DUPFD_CLOEXEC, 0);
  if (copy < 0) return refuse(RefusalCode::FeatureUnavailable, "OpenVINO, import : dup du descripteur");
  try {
    tensor = import_tensor(context_, element, dims, copy);
  } catch (const std::exception& e) {
    if (same_file(copy, file)) close(copy);  // pas encore fermé par le pilote, ni repris
    return refuse(RefusalCode::FeatureUnavailable, std::format("OpenVINO, import : {}", e.what()));
  }
  // OpenVINO garde les mémoires importées en cache par la valeur du descripteur : celle-ci ne doit
  // pas revenir à un autre import tant que la mémoire vit, donc rester ouverte au pilote.
  if (!same_file(copy, file))
    return refuse(RefusalCode::FeatureUnavailable, "OpenVINO, import : descripteur fermé par le pilote");
#endif
  void* const mem = mem_of(tensor);
  {
    const std::scoped_lock lock(mutex_);
    live_.insert_or_assign(mem, tensor);
  }
  auto imported = std::make_unique<DeviceImport>(shared_from_this(), mem, std::move(tensor));
#if defined(_WIN32)
  // Dernière étape faillible : une fois le rappel inscrit, le HANDLE est au cl_mem ; refusé, la
  // mémoire part avec imported, et le HANDLE reste à l'appelant.
  if (set_destructor_(mem, &close_on_destroy, nt) != 0)
    return refuse(RefusalCode::FeatureUnavailable, "OpenVINO, import : rappel de destruction refusé");
#else
  close(original);
#endif
  return imported;
}

std::optional<ov::RemoteTensor> DeviceImports::tensor_for(void* device, ov::element::Type type,
                                                          const ov::Shape& shape, std::uint64_t bytes) {
  const std::scoped_lock lock(mutex_);
  const auto found = live_.find(device);
  if (found == live_.end()) return std::nullopt;
  const ov::RemoteTensor& tensor = found->second;
  if (tensor.get_element_type() == type && tensor.get_shape() == shape) return tensor;
  if (bytes > tensor.get_byte_size()) return std::nullopt;
  return context_.create_tensor(
      type, shape,
      {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::OCL_BUFFER},
       {ov::intel_gpu::mem_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(device)}});
}

void DeviceImports::forget(void* device) {
  const std::scoped_lock lock(mutex_);
  live_.erase(device);
}

}  // namespace ajp::gpu::openvino_rt
