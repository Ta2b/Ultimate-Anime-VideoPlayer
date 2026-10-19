// Backend CPU de référence (ajp_backend_cpu) : Backend et KernelOps sur l'hôte, par kernels/ref
// et l'interpréteur FP32 de gpu_ir. Jamais proposé à l'utilisateur ni livré avec le lecteur :
// il sert aux tests de bout en bout du module gpu et au débogage d'un port, image par image.
//
// Tout s'exécute sur le thread appelant, au moment de l'appel : enqueue() rend la main quand
// le moteur a calculé, signal() et wait() n'attendent rien. Le journal ne sert pas : chaque
// erreur revient en refus. Les tenseurs vivent en mémoire
// hôte (host_tensor.h) ; les images décodées doivent l'être aussi (FrameMemory::Host).
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/cpu/cpu_kernels.h"
#include "gpu/cpu/host_tensor.h"
#include "gpu/rife_input.h"

namespace ajp::gpu::cpu {

class CpuBackend final : public Backend {
 public:
  // Rien à lier : ni appareil, ni cache, ni journal.
  explicit CpuBackend(const BackendInit&) {}
  CpuBackend(const CpuBackend&) = delete;
  CpuBackend& operator=(const CpuBackend&) = delete;
  ~CpuBackend() override = default;

  BackendCaps capabilities() const override;
  // Le CPU n'est lié à aucun appareil : jamais Same ni Different.
  DeviceMatch match(const DeviceUuid&) const override { return DeviceMatch::Unknown; }
  // Rien à construire : l'artefact est l'ONNX lui-même, lu une fois pour le valider.
  std::expected<Artifact, Refusal> build(const ModelSpec& model, const ShapeSet& shapes,
                                         Progress& progress) override;
  std::expected<std::unique_ptr<Engine>, Refusal> load(const Artifact& artifact) override;
  std::expected<Tensor, Refusal> import(const ExternalImage& image) override;
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_tensor(TensorType type,
                                                                     const Shape& shape) override;
  KernelOps* kernels() override { return &kernels_; }
  // Exécute le moteur tout de suite. Refus : moteur d'un autre backend, nombre ou forme des
  // tenseurs, refus de l'interpréteur ; les sorties restent alors intactes.
  std::expected<void, Refusal> enqueue(Engine& engine, std::span<const Tensor> inputs,
                                       std::span<Tensor> outputs) override;
  std::expected<Fence, Refusal> signal() override { return Fence{.handle = 0, .value = ++fence_}; }
  std::expected<void, Refusal> wait(const Fence&) override { return {}; }

  RifeInputOps* rife_input() { return &kernels_; }

 private:
  HostMemory memory_;  // avant les noyaux qui le lisent
  CpuKernels kernels_{memory_};
  std::uint64_t fence_ = 0;
};

}  // namespace ajp::gpu::cpu
