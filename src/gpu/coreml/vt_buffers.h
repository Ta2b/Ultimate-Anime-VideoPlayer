// Les tampons de Core Video de l'interpolation d'Apple (coreml_vt.h) : leurs réserves, leurs plans
// pour le CPU, et les conversions du GPU (vt_gpu.h), vérifiées contre celles du CPU (vt_pixels.h)
// avant de servir. Objective-C++, sur un Mac seulement.
#pragma once

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "core/types.h"
#include "gpu/coreml/vt_gpu.h"
#include "gpu/coreml/vt_pixels.h"

namespace ajp::gpu::coreml_rt {

// Un refus de l'interpolation d'Apple, nommée dans le détail.
std::unexpected<Refusal> refuse_vt(RefusalCode code, std::string text);

// Un tampon de pixels retenu, rendu à la destruction.
class PixelBuffer {
 public:
  PixelBuffer() = default;
  explicit PixelBuffer(CVPixelBufferRef ref) : ref_(ref) {}
  PixelBuffer(const PixelBuffer&) = delete;
  PixelBuffer& operator=(const PixelBuffer&) = delete;
  PixelBuffer(PixelBuffer&& other) noexcept : ref_(std::exchange(other.ref_, nullptr)) {}
  PixelBuffer& operator=(PixelBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ref_ = std::exchange(other.ref_, nullptr);
    }
    return *this;
  }
  ~PixelBuffer() { reset(); }
  CVPixelBufferRef get() const { return ref_; }

 private:
  void reset() {
    if (ref_ != nullptr) CVPixelBufferRelease(ref_);
    ref_ = nullptr;
  }
  CVPixelBufferRef ref_ = nullptr;
};

// Une réserve de tampons retenue ; take en donne un neuf, ou un rendu par VideoToolbox.
class BufferPool {
 public:
  BufferPool() = default;
  explicit BufferPool(CVPixelBufferPoolRef ref) : ref_(ref) {}
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;
  BufferPool(BufferPool&& other) noexcept : ref_(std::exchange(other.ref_, nullptr)) {}
  BufferPool& operator=(BufferPool&& other) noexcept {
    if (this != &other) {
      if (ref_ != nullptr) CVPixelBufferPoolRelease(ref_);
      ref_ = std::exchange(other.ref_, nullptr);
    }
    return *this;
  }
  ~BufferPool() {
    if (ref_ != nullptr) CVPixelBufferPoolRelease(ref_);
  }

  std::expected<PixelBuffer, Refusal> take() const {
    CVPixelBufferRef raw = nullptr;
    if (const CVReturn status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, ref_, &raw);
        status != kCVReturnSuccess || raw == nullptr) {
      return refuse_vt(RefusalCode::GpuMemoryExhausted, std::format("tampon de pixels refusé ({})", status));
    }
    return PixelBuffer(raw);
  }

 private:
  CVPixelBufferPoolRef ref_ = nullptr;
};

// Les tampons d'une sorte (source ou destination) : les exigences de l'interpolateur (nil : aucune),
// au format et à la taille choisis, en IOSurface que Metal prend pour textures (vt_gpu.h).
std::expected<BufferPool, Refusal> make_pool(NSDictionary<NSString*, id>* wanted, vt::Pixels format,
                                             int width, int height);

// Les couleurs de la chaîne pour RIFE : BT.709 (kernels/spec/rife.h), la chroma au centre de chaque
// bloc 2 × 2 (vt_pixels.h).
void tag_colors(CVPixelBufferRef buffer, bool yuv);

// Les plans du tampon, verrouillés le temps de use.
std::expected<void, Refusal> with_planes(CVPixelBufferRef buffer, bool read_only,
                                         const std::function<void(const vt::Planes&)>& use);

// Les rangées en bandes de hauteur paire, sur les cœurs de la puce.
void by_bands(int height, const std::function<void(vt::Rows)>& work);

// Les conversions sur le GPU pour ce format et cette taille, vérifiées contre celles du CPU sur une
// paire de test ; sinon nul, et la raison dans why.
std::unique_ptr<VtGpu> open_gpu(vt::Pixels format, int width, int height, const BufferPool& sources,
                                const BufferPool& destinations, std::string& why);

}  // namespace ajp::gpu::coreml_rt
