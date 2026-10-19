// Program : un Plan prêt à exécuter, avec ses moteurs chargés dans le backend du thread gpu.
// Immuable. app envoie par Install une ProgramRequest (le Plan et les artefacts que prepare a
// construits) ; gpu possède son Backend et y charge les moteurs, sur son thread. Le ring
// arrive à part, à chaque génération (RingGeneration, output_ring.h).
//
// Les étapes suivent les ordonnancements de core/plan.h, dans l'ordre :
//   ingestion (YUV → RGB de la chaîne) et préparation de RIFE s'il y a des niveaux ;
//   niveaux S1 : RIFE prédit chaque instant à la résolution de la source ;
//   réduction éventuelle, puis passes d'agrandissement : une fois par image ;
//   niveau S3 : mouvement à la résolution de la source, warp et mélange à la sortie ;
//   écriture dans le ring.
// La sortie d'un niveau devient la source du suivant sous des SourceId dérivés.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/messages.h"
#include "core/plan.h"
#include "core/ring.h"
#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/rife_input.h"

namespace ajp::gpu {

// Slots d'un ring : au plus 16.
inline constexpr std::size_t kMaxRingSlots = 16;

// Un modèle chargé par Backend::load, sous l'identifiant du Plan.
struct LoadedModel {
  core::ModelId id;
  std::shared_ptr<Engine> engine;
};

// Un moteur que prepare a construit (Backend::build) pour un modèle du Plan.
struct ModelArtifact {
  core::ModelId id;
  Artifact artifact;
};

// Ce que app envoie par Install : le Plan et les artefacts de ses modèles.
class ProgramRequest final : public ProgramBase {
 public:
  ProgramRequest(core::Plan plan, std::vector<ModelArtifact> models, TensorType precision = TensorType::F16)
      : plan_(std::move(plan)), models_(std::move(models)), precision_(precision) {}

  std::uint64_t plan_hash() const override { return plan_.hash; }
  const core::Plan& plan() const { return plan_; }
  std::span<const ModelArtifact> models() const { return models_; }
  TensorType precision() const { return precision_; }

 private:
  core::Plan plan_;
  std::vector<ModelArtifact> models_;
  TensorType precision_;
};

struct ProgramParts {
  core::Plan plan;
  Backend* backend = nullptr;              // vit plus longtemps que le Program
  RifeInputOps* rife = nullptr;            // LoadedBackend::rife_input() ; exigé dès qu'il y a un niveau
  std::vector<LoadedModel> models;         // chargés par ce backend : ils meurent avant lui
  TensorType precision = TensorType::F16;  // tenseurs d'image entre les étapes : F16 ou F32
};

enum class StageKind : std::uint8_t {
  Ingest,       // YUV décodé → RGB de la chaîne (KernelOps::ingest)
  RifePad,      // YUV décodé → RGB rempli de RIFE (KernelOps::rife_pad)
  Interpolate,  // S1 : RIFE prédit l'instant, puis rife_crop
  Downscale,    // KernelOps::resample
  Upscale,      // une passe de modèle
  Motion,       // S3 : RIFE estime le mouvement au milieu de la paire
  WarpBlend,    // S3 : warp et mélange à la résolution de sortie
  WriteRing,    // KernelOps::write_ring
};

struct Stage {
  StageKind kind = StageKind::Ingest;
  std::uint32_t level = 0;  // Interpolate, Motion, WarpBlend : indice dans plan.levels
  core::ModelId model;      // Interpolate, Upscale, Motion
  int factor = 1;           // Upscale
};

// Une passe d'agrandissement : 1 × 3 × h × w → 1 × 3 × fh × fw.
struct UpscalePass {
  Engine* engine = nullptr;
  int factor = 1;
};

class Program final : public ProgramBase {
 public:
  // Refus : backend ou noyaux absents (FieldMissing), sortie vide ou impaire
  // (FieldOutOfRange), ordonnancement non exécutable (StepOrder : S2, S3 suivi d'un autre
  // niveau), modèle absent (ModelUnavailable) ou d'une autre forme d'entrées et sorties
  // (ModelMismatch), noyau d'entrée de RIFE absent ou format de ring, précision ou route HDR
  // non exécutés ici (FormatUnsupported, HdrUnavailable).
  static std::expected<std::shared_ptr<const Program>, Refusal> create(ProgramParts parts);

  std::uint64_t plan_hash() const override { return plan_.hash; }
  const core::Plan& plan() const { return plan_; }
  std::span<const Stage> stages() const { return stages_; }

  Backend& backend() const { return *backend_; }
  KernelOps& kernels() const { return *kernels_; }
  RifeInputOps* rife() const { return rife_; }
  TensorType precision() const { return precision_; }

  // Moteur RIFE du niveau : image prédite (S1) ou mouvement et masque (S3).
  Engine& level_engine(std::size_t level) const { return *level_engines_[level]; }
  std::span<const UpscalePass> upscale() const { return upscale_; }
  int upscale_factor() const { return upscale_factor_; }

 private:
  Program() = default;

  core::Plan plan_;
  Backend* backend_ = nullptr;
  KernelOps* kernels_ = nullptr;
  RifeInputOps* rife_ = nullptr;
  TensorType precision_ = TensorType::F16;
  std::vector<std::shared_ptr<Engine>> engines_;  // possédés : vivants autant que le Program
  std::vector<Engine*> level_engines_;
  std::vector<UpscalePass> upscale_;
  int upscale_factor_ = 1;
  std::vector<Stage> stages_;
};

// Charge chaque moteur de la demande dans backend (Backend::load), puis crée le Program.
// Refus : celui de Backend::load, qui cite le modèle, ou celui de Program::create.
std::expected<std::shared_ptr<const Program>, Refusal> load_program(const ProgramRequest& request,
                                                                    Backend& backend, RifeInputOps* rife);

// Une ligne lisible des étapes, pour le journal et l'OSD :
// « ingestion · RIFE · S1 x2 rife · agrandissement x2 span · ring ».
std::string describe(const Program& program);

}  // namespace ajp::gpu
