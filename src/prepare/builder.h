// Construction d'un moteur TensorRT par trtexec en sous-processus.
//
// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:542-582
// (engine_path_for), 691-760 (run_build_process), 762-806 (run_build_spec) et
// 862-890 (make_build_spec). Un moteur à la fois : l'appelant est le thread
// preparer. Un plantage de trtexec devient un refus et épargne le lecteur ;
// un moteur n'est publié dans le cache que complet et vérifié, et son timing
// cache seulement après une construction réussie.
#pragma once

#include <chrono>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "core/types.h"
#include "gpu/backend.h"
#include "prepare/identity.h"
#include "prepare/model_source.h"
#include "prepare/trtexec_command.h"

namespace spdlog {
class logger;
}

namespace ajp::prepare {

struct TrtexecConfig {
  std::filesystem::path trtexec;    // exécutable livré avec le backend TensorRT
  std::filesystem::path cache_dir;  // dossier des moteurs
  TrtRuntime runtime;               // 11.3 au moins
  CudaDevice device;
  std::chrono::milliseconds timeout = std::chrono::minutes(20);
  std::string input_name = "input";  // nom de l'entrée des modèles livrés
};

// Vérifie un moteur avant sa publication, par exemple en le désérialisant.
using EngineCheck = std::function<std::expected<void, Refusal>(const std::filesystem::path& engine)>;

// Ce que la construction fera, calculé sans rien lancer.
struct BuildPlan {
  std::string settings;              // hachés dans la clé, ceux du fork
  SanitizedSettings command;         // réglages de la ligne de commande, TensorRT 11
  std::string engine_file;           // « aji-v3-<clé>.engine »
  std::filesystem::path engine;      // cache_dir / engine_file
  std::filesystem::path log_file;    // « <moteur>.build.log », gardé pour le diagnostic
  std::filesystem::path layer_info;  // « <moteur>.layers.json », rapport des couches (trtexec_command.h)
  std::string label;                 // « 1920x1080 » pour l'OSD
};

std::expected<BuildPlan, Refusal> plan_build(const TrtexecConfig& config, const gpu::ModelSpec& model,
                                             const ModelSource& source, const gpu::ShapeSet& shapes);

class TrtexecBuilder {
 public:
  explicit TrtexecBuilder(TrtexecConfig config, EngineCheck check = {});

  // Contrat de Backend::build : un backend TensorRT y délègue sa construction.
  std::expected<gpu::Artifact, Refusal> build(const gpu::ModelSpec& model, const gpu::ShapeSet& shapes,
                                              gpu::Progress& progress) const;

  // Reconstruit un moteur publié que load() n'a pas pu désérialiser, dont le
  // fichier a ce SHA-256 (rejected_digest du fork). Il reste en place jusqu'à
  // son remplacement réussi ; s'il a déjà été remplacé, le remplaçant se reprend.
  std::expected<gpu::Artifact, Refusal> rebuild(const gpu::ModelSpec& model, const gpu::ShapeSet& shapes,
                                                gpu::Progress& progress,
                                                const std::string& rejected_sha256) const;

 private:
  std::expected<void, Refusal> run(const BuildPlan& plan, const gpu::ModelSpec& model,
                                   const ModelSource& source, const std::filesystem::path& output,
                                   gpu::Progress& progress) const;

  TrtexecConfig config_;
  EngineCheck check_;
  std::shared_ptr<spdlog::logger> log_;
};

}  // namespace ajp::prepare
