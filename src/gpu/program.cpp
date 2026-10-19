#include "gpu/program.h"

#include <algorithm>
#include <format>
#include <utility>

#include "gpu/level_scheduler.h"

namespace ajp::gpu {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Le moteur d'un modèle du Plan, avec le nombre d'entrées et de sorties que son étape attend.
std::expected<std::shared_ptr<Engine>, Refusal> find_engine(const std::vector<LoadedModel>& models,
                                                            const core::ModelId& id, std::size_t inputs,
                                                            std::size_t outputs, std::string_view role) {
  const auto it = std::ranges::find(models, id, &LoadedModel::id);
  if (it == models.end() || it->engine == nullptr) {
    return refuse(RefusalCode::ModelUnavailable, std::format("{} : moteur {} non chargé", role, id));
  }
  const Engine& engine = *it->engine;
  if (engine.input_names().size() != inputs || engine.output_names().size() != outputs) {
    return refuse(RefusalCode::ModelMismatch,
                  std::format("{} : {} a {} entrée(s) et {} sortie(s), l'étape en attend {} et {}", role, id,
                              engine.input_names().size(), engine.output_names().size(), inputs, outputs));
  }
  return it->engine;
}

// S1 en tête, puis au plus un S3 en dernier : la sortie d'un S3 n'existe qu'à la résolution de
// sortie, sans l'image préparée que RIFE lirait au niveau suivant.
std::expected<void, Refusal> check_schedules(const std::vector<core::InterpLevel>& levels) {
  if (levels.size() > kMaxLevels) {
    return refuse(RefusalCode::StepOrder, std::format("{} niveaux, au plus {}", levels.size(), kMaxLevels));
  }
  for (std::size_t i = 0; i < levels.size(); ++i) {
    switch (levels[i].schedule) {
      case core::Schedule::BeforeUpscale:
        if (i > 0 && levels[i - 1].schedule != core::Schedule::BeforeUpscale) {
          return refuse(RefusalCode::StepOrder, std::format("niveau {} : S1 après un S3", i));
        }
        break;
      case core::Schedule::AfterUpscale:
        return refuse(
            RefusalCode::StepOrder,
            std::format("niveau {} : S2 attend un remplissage RGB pour RIFE, absent de KernelOps (ABI 3)",
                        i));
      case core::Schedule::OutputRes:
        if (i + 1 != levels.size()) {
          return refuse(RefusalCode::StepOrder,
                        std::format("niveau {} : un S3 ne peut être suivi d'un autre niveau", i));
        }
        break;
    }
    // Facteur, politique et rythme : ce que l'ordonnanceur du niveau acceptera.
    if (auto scheduler = LevelScheduler::create(levels[i], static_cast<std::uint32_t>(i)); !scheduler) {
      return std::unexpected(std::move(scheduler.error()));
    }
  }
  return {};
}

std::expected<void, Refusal> check_parts(const ProgramParts& parts) {
  const core::Plan& plan = parts.plan;
  if (parts.backend == nullptr || parts.backend->kernels() == nullptr) {
    return refuse(RefusalCode::FieldMissing, "backend ou noyaux de la chaîne absents");
  }
  if (plan.output_width <= 0 || plan.output_height <= 0 || plan.output_width % 2 != 0 ||
      plan.output_height % 2 != 0) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("sortie {}×{} : dimensions positives et paires",
                                                            plan.output_width, plan.output_height));
  }
  if (parts.precision != TensorType::F16 && parts.precision != TensorType::F32) {
    return refuse(RefusalCode::FormatUnsupported, "précision des images : F16 ou F32");
  }
  if (plan.hdr.kind == core::HdrMode::TrueHdr) {
    return refuse(RefusalCode::HdrUnavailable, "TrueHDR n'est pas encore exécuté par gpu");
  }
  if (!plan.levels.empty() && parts.rife == nullptr) {
    return refuse(RefusalCode::FormatUnsupported,
                  "le backend n'exporte pas ajp_backend_rife_input : pas d'interpolation");
  }
  if (plan.ring != RingFormat::Rgba16Unorm) {
    return refuse(RefusalCode::FormatUnsupported, "ring Yuv444p16 : write_ring n'écrit que RGBA16 UNORM");
  }
  return check_schedules(plan.levels);
}

}  // namespace

std::expected<std::shared_ptr<const Program>, Refusal> Program::create(ProgramParts parts) {
  if (auto ok = check_parts(parts); !ok) return std::unexpected(std::move(ok.error()));
  std::shared_ptr<Program> program(new Program());
  program->backend_ = parts.backend;
  program->kernels_ = parts.backend->kernels();
  program->rife_ = parts.rife;
  program->precision_ = parts.precision;
  std::vector<Stage>& stages = program->stages_;
  stages.push_back({.kind = StageKind::Ingest});
  if (!parts.plan.levels.empty()) stages.push_back({.kind = StageKind::RifePad});

  const std::vector<core::InterpLevel>& levels = parts.plan.levels;
  const auto add_level = [&](std::size_t i) -> std::expected<void, Refusal> {
    const bool motion = levels[i].schedule == core::Schedule::OutputRes;
    auto engine = find_engine(parts.models, levels[i].model, 1, motion ? 2 : 1, std::format("niveau {}", i));
    if (!engine) return std::unexpected(std::move(engine.error()));
    program->engines_.push_back(*engine);
    program->level_engines_.push_back(engine->get());
    const auto level = static_cast<std::uint32_t>(i);
    if (motion) {
      stages.push_back({.kind = StageKind::Motion, .level = level, .model = levels[i].model});
      stages.push_back({.kind = StageKind::WarpBlend, .level = level});
    } else {
      stages.push_back({.kind = StageKind::Interpolate, .level = level, .model = levels[i].model});
    }
    return {};
  };
  std::size_t next = 0;
  for (; next < levels.size() && levels[next].schedule == core::Schedule::BeforeUpscale; ++next) {
    if (auto ok = add_level(next); !ok) return std::unexpected(std::move(ok.error()));
  }
  if (parts.plan.downscale) stages.push_back({.kind = StageKind::Downscale});
  if (parts.plan.upscale) {
    for (const core::ModelPass& pass : parts.plan.upscale->passes) {
      if (pass.factor < 1 || pass.factor > 8) {
        return refuse(RefusalCode::FieldOutOfRange,
                      std::format("passe {} : facteur {} hors de [1, 8]", pass.model, pass.factor));
      }
      auto engine = find_engine(parts.models, pass.model, 1, 1, "agrandissement");
      if (!engine) return std::unexpected(std::move(engine.error()));
      program->engines_.push_back(*engine);
      program->upscale_.push_back({.engine = engine->get(), .factor = pass.factor});
      program->upscale_factor_ *= pass.factor;
      stages.push_back({.kind = StageKind::Upscale, .model = pass.model, .factor = pass.factor});
    }
  }
  for (; next < levels.size(); ++next) {
    if (auto ok = add_level(next); !ok) return std::unexpected(std::move(ok.error()));
  }
  if (parts.plan.output_width % (2 * program->upscale_factor_) != 0 ||
      parts.plan.output_height % (2 * program->upscale_factor_) != 0) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("sortie {}×{} : pas une image paire agrandie x{}", parts.plan.output_width,
                              parts.plan.output_height, program->upscale_factor_));
  }
  stages.push_back({.kind = StageKind::WriteRing});
  program->plan_ = std::move(parts.plan);
  return program;
}

std::expected<std::shared_ptr<const Program>, Refusal> load_program(const ProgramRequest& request,
                                                                    Backend& backend, RifeInputOps* rife) {
  ProgramParts parts{
      .plan = request.plan(), .backend = &backend, .rife = rife, .precision = request.precision()};
  for (const ModelArtifact& model : request.models()) {
    auto engine = backend.load(model.artifact);
    if (!engine) {
      return refuse(engine.error().code, std::format("moteur {} : {}", model.id, engine.error().detail));
    }
    parts.models.push_back({.id = model.id, .engine = std::shared_ptr<Engine>(std::move(*engine))});
  }
  return Program::create(std::move(parts));
}

std::string describe(const Program& program) {
  std::string text;
  const auto add = [&text](std::string_view part) {
    if (!text.empty()) text += " · ";
    text += part;
  };
  for (const Stage& stage : program.stages()) {
    switch (stage.kind) {
      case StageKind::Ingest:
        add("ingestion");
        break;
      case StageKind::RifePad:
        add("préparation RIFE");
        break;
      case StageKind::Interpolate:
      case StageKind::Motion: {
        const core::InterpLevel& level = program.plan().levels[stage.level];
        add(std::format("{} x{}/{} {}", stage.kind == StageKind::Motion ? "S3" : "S1", level.factor.num,
                        level.factor.den, stage.model));
        break;
      }
      case StageKind::Downscale:
        add(std::format("réduction {}p", program.plan().downscale.value_or(core::DownscaleStep{}).height));
        break;
      case StageKind::Upscale:
        add(std::format("agrandissement x{} {}", stage.factor, stage.model));
        break;
      case StageKind::WarpBlend:
        add("warp et mélange");
        break;
      case StageKind::WriteRing:
        add("ring");
        break;
    }
  }
  return text;
}

}  // namespace ajp::gpu
