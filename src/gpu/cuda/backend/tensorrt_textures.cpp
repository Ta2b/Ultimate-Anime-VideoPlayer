// Objets texture des tenseurs FP16 lus par le filtre bilinéaire matériel (bilinear.cuh) : le
// tenseur 1 × C × H × W vu en W × (C · H) texels, pitch dense de 2W octets. Un objet par adresse et
// par forme, créé à la première lecture et gardé : les tenseurs de la chaîne (sorties des moteurs,
// tampons des étages) gardent leur adresse d'une image à l'autre. Une adresse libérée puis rendue
// à un autre tenseur de même forme décrit la même mémoire : l'objet reste juste.
#include <cstdint>
#include <format>

#include "gpu/cuda/backend/tensorrt_backend.h"

namespace ajp::gpu::cuda {

cudaTextureObject_t TensorRtBackend::texture(const Tensor& tensor) {
  if (tensor.type != TensorType::F16 || tensor.handle == 0) return 0;
  const std::uintptr_t address = tensor.handle + tensor.offset;
  const std::int64_t width = tensor.shape.dims[3];
  const std::int64_t rows = tensor.shape.dims[0] * tensor.shape.dims[1] * tensor.shape.dims[2];
  const TextureKey key{.address = address, .width = width, .rows = rows};
  if (const auto found = textures_.find(key); found != textures_.end()) return found->second;
  // Pitch multiple de texturePitchAlignment (32 octets : 16 pixels FP16), adresse alignée sur
  // textureAlignment, tailles dans maxTexture2DLinear ; sinon 0, gardé aussi : le noyau lit exact.
  const auto pitch = static_cast<std::size_t>(width) * 2;
  cudaTextureObject_t object = 0;
  if (width > 0 && rows > 0 && pitch % properties_.texturePitchAlignment == 0 &&
      address % properties_.textureAlignment == 0 && width <= properties_.maxTexture2DLinear[0] &&
      rows <= properties_.maxTexture2DLinear[1] &&
      pitch <= static_cast<std::size_t>(properties_.maxTexture2DLinear[2])) {
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
    resource.res.pitch2D.desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);  // un __half
    resource.res.pitch2D.width = static_cast<std::size_t>(width);
    resource.res.pitch2D.height = static_cast<std::size_t>(rows);
    resource.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 0;
    if (cudaCreateTextureObject(&object, &resource, &description, nullptr) != cudaSuccess) {
      cudaGetLastError();  // l'erreur ne doit pas remonter au lancement suivant
      object = 0;
      log(BackendLogLevel::Warn,
          std::format("objet texture refusé pour un tenseur {}x{} : lecture exacte", width, rows));
    }
  }
  textures_.emplace(key, object);
  return object;
}

void TensorRtBackend::trim_textures() {
  // Avant les objets d'un lancement, jamais pendant : ceux déjà pris pour ses paramètres restent.
  if (textures_.size() >= kTextures) destroy_textures();
}

void TensorRtBackend::destroy_textures() {
  // Un noyau en vol peut encore lire un objet : les files se vident avant.
  for (cudaStream_t lane : streams_) {
    if (lane != nullptr) cudaStreamSynchronize(lane);
  }
  for (const auto& [key, object] : textures_) {
    if (object != 0) cudaDestroyTextureObject(object);
  }
  textures_.clear();
}

}  // namespace ajp::gpu::cuda
