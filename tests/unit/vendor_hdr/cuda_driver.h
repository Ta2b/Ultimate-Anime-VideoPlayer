// API pilote CUDA pour les tests [nvidia] : libcuda.so.1 ouverte par dlopen, sans lien à la
// construction ni en-tête du toolkit, pour que la CI (sans pilote NVIDIA) construise et SKIP.
// Prototypes repris de cuda.h (CUDA 13.4, /opt/cuda/include) ; les symboles _v2 sont ceux que
// cuda.h désigne par #define.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vendor_hdr/true_hdr.h"

namespace ajp::vendor_hdr::test {

struct CudaApi;

// Contexte primaire de l'appareil 0, courant sur le thread qui l'ouvre.
class Cuda {
 public:
  // nullptr et why rempli quand le pilote, un appareil ou le contexte manque.
  static std::unique_ptr<Cuda> open(std::string& why);
  Cuda(const Cuda&) = delete;
  Cuda& operator=(const Cuda&) = delete;
  ~Cuda();

  std::string device_name() const;
  std::uintptr_t context() const { return context_; }
  const CudaApi& api() const { return *api_; }

  // Échoue le test (REQUIRE) sur toute erreur du pilote.
  void check(int result, const char* call) const;
  void synchronize() const;

 private:
  Cuda(std::unique_ptr<CudaApi> api, int device, std::uintptr_t context);
  std::unique_ptr<CudaApi> api_;
  int device_ = 0;
  std::uintptr_t context_ = 0;
};

// Tampon linéaire en mémoire CUDA, de pas width × 4 (ou pitch s'il est donné).
class DeviceImage {
 public:
  DeviceImage(const Cuda& cuda, int width, int height, TrueHdrPixel pixel, std::size_t pitch = 0);
  DeviceImage(const DeviceImage&) = delete;
  DeviceImage& operator=(const DeviceImage&) = delete;
  ~DeviceImage();

  TrueHdrImage view() const;
  void upload(const std::vector<std::uint32_t>& pixels) const;  // un mot par pixel, lignes serrées
  std::vector<std::uint32_t> download() const;

 private:
  const Cuda& cuda_;
  std::uintptr_t device_ = 0;
  int width_;
  int height_;
  std::size_t pitch_;
  TrueHdrPixel pixel_;
};

// Le flux de gpu tel que true_hdr.h le décrit : flux non bloquant et deux événements sans
// chronométrage, le flux 0 attendant l'entrée puis le flux attendant la sortie.
class CudaGpuStream final : public GpuStream {
 public:
  explicit CudaGpuStream(const Cuda& cuda);
  CudaGpuStream(const CudaGpuStream&) = delete;
  CudaGpuStream& operator=(const CudaGpuStream&) = delete;
  ~CudaGpuStream() override;

  Done hand_to_legacy() override;
  Done take_from_legacy() override;

  // Copies ordonnées sur ce flux seulement : le test n'a ainsi aucune autre synchronisation.
  void upload(const DeviceImage& image, const std::vector<std::uint32_t>& pixels) const;
  std::vector<std::uint32_t> download(const DeviceImage& image) const;

 private:
  Done call(int result, const char* what) const;
  const Cuda& cuda_;
  std::uintptr_t stream_ = 0;
  std::uintptr_t input_ready_ = 0;
  std::uintptr_t output_ready_ = 0;
};

}  // namespace ajp::vendor_hdr::test
