// Journal de TensorRT relié à spdlog, interne au runtime et aux tests.
//
// TensorRT appelle log() depuis n'importe lequel de ses threads : la dernière
// erreur est gardée sous verrou pour compléter le refus de l'appel qui a échoué
// (un appel TensorRT ne rend qu'un booléen ou un pointeur nul).
#pragma once

#include <NvInferRuntime.h>
#include <memory>
#include <mutex>
#include <string>

namespace spdlog {
class logger;
}

namespace ajp::gpu::trt {

class TrtLog final : public nvinfer1::ILogger {
 public:
  explicit TrtLog(std::shared_ptr<spdlog::logger> log);

  // Erreurs en error, avertissements en warn, informations en debug (TensorRT
  // en écrit des dizaines par moteur chargé), détails en trace.
  void log(Severity severity, nvinfer1::AsciiChar const* message) noexcept override;

  // Rend la dernière erreur reçue et l'oublie ; vide s'il n'y en a pas eu.
  std::string take_last_error();

 private:
  std::shared_ptr<spdlog::logger> log_;
  std::mutex mutex_;
  std::string last_error_;
};

// « ; TensorRT : <erreur> » ou rien, pour finir le détail d'un refus.
std::string error_suffix(TrtLog& log);

}  // namespace ajp::gpu::trt
