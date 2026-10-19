// Runtime TensorRT : désérialise les moteurs que prepare a construits avec
// trtexec et leur crée un contexte d'exécution.
//
// Interface interne du backend TensorRT : l'adaptation au contrat gpu::Backend
// (ABI 2, w2/fondations) se fera au-dessus, sans changer ce runtime.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "core/types.h"
#include "gpu/cuda/trt/engine.h"

namespace spdlog {
class logger;
}

namespace ajp::gpu::trt {

struct RuntimeState;
class TrtLog;

// Plus gros moteur accepté : RIFE 4.26 en 1088x1920 pèse 113 Mo, et rien de
// livré n'approche cette borne.
inline constexpr std::uintmax_t kMaxEngineBytes = std::uintmax_t{4} << 30U;

// Version de la libnvinfer chargée (getInferLibVersion, getInferLibBuildVersion),
// et numéro de build des en-têtes compilés (NV_TENSORRT_BUILD), que le jeton
// de cache des moteurs reprend du fork.
struct LibraryVersion {
  int major = 0;
  int minor = 0;
  int patch = 0;
  int build = 0;
  int headers_build = 0;
  std::string text() const;  // « 11.3.0.99 »
};

// TensorRT n'a qu'un journal par processus : le premier IRuntime, IBuilder ou
// IRefitter créé fixe celui de tous les autres tant que l'un d'eux vit, et
// ignore le journal des suivants. Un seul TrtRuntime vit donc à la fois, et
// tout IBuilder créé pendant sa vie (tests) reçoit son journal, log().
class TrtRuntime {
 public:
  // Crée l'IRuntime sur l'appareil CUDA courant du thread. Refuse une
  // libnvinfer d'une autre version majeure que nos en-têtes (TensorRT 11), et
  // un second runtime tant qu'un objet TensorRT d'un autre journal vit encore.
  static std::expected<TrtRuntime, Refusal> create(std::shared_ptr<spdlog::logger> log);

  LibraryVersion version() const;
  TrtLog& log() const;

  // Lit et désérialise un moteur. CacheIo si le fichier est illisible,
  // EngineInvalid si TensorRT le refuse (octets abîmés, autre version de
  // TensorRT, autre GPU), avec l'erreur de TensorRT dans le détail.
  std::expected<TrtEngine, Refusal> load_file(const std::filesystem::path& engine) const;
  std::expected<TrtEngine, Refusal> load(std::span<const std::byte> plan) const;

 private:
  explicit TrtRuntime(std::shared_ptr<RuntimeState> state);
  std::shared_ptr<RuntimeState> state_;
};

}  // namespace ajp::gpu::trt
