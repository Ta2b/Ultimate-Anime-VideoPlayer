// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:542-582, 691-806
// et 862-890 : clé du moteur, journal de construction à côté du moteur, code
// de sortie et place libre dans le refus, aucun moteur incomplet publié.
#include "prepare/builder.h"

#include <format>
#include <fstream>
#include <system_error>
#include <utility>

#include <spdlog/logger.h>

#include "prepare/cache.h"
#include "prepare/log.h"
#include "prepare/process.h"
#include "prepare/trtexec_progress.h"

namespace ajp::prepare {
namespace {

namespace fs = std::filesystem;

void report(gpu::Progress& progress, float fraction, const std::string& stage) {
  if (progress.report) progress.report(fraction, stage);
}

std::string free_space(const fs::path& folder) {
  std::error_code error;
  const auto space = fs::space(folder, error);
  return error ? "inconnue" : std::format("{} octets", space.available);
}

// Traduit la fin de trtexec en refus ; rien quand il a réussi.
std::expected<void, Refusal> judge(const ProcessExit& exit, const TrtexecProgress& parsed,
                                   const BuildPlan& plan, std::chrono::milliseconds timeout) {
  const std::string error = parsed.last_error().empty() ? "" : " ; dernière erreur : " + parsed.last_error();
  const std::string where = " ; journal : " + plan.log_file.string();
  switch (exit.kind) {
    case ExitKind::Cancelled:
      return std::unexpected(Refusal{RefusalCode::Cancelled, "construction annulée de " + plan.engine_file});
    case ExitKind::TimedOut:
      return std::unexpected(
          Refusal{RefusalCode::BuilderTimeout,
                  std::format("trtexec a dépassé son délai de {:.1f} s{}{}",
                              std::chrono::duration<double>(timeout).count(), error, where)});
    case ExitKind::Crashed:
      return std::unexpected(
          Refusal{RefusalCode::BuilderCrashed, "trtexec : " + describe(exit) + error + where});
    case ExitKind::Exited:
      break;
  }
  if (exit.code != 0 || parsed.failed()) {
    return std::unexpected(Refusal{
        RefusalCode::BuilderFailed,
        std::format(
            "trtexec : {}{}{} ; place libre du cache : {}. Vérifiez la place, les droits du cache et le "
            "journal, puis rouvrez la vidéo.",
            describe(exit), error, where, free_space(plan.engine.parent_path()))});
  }
  return {};
}

}  // namespace

std::expected<BuildPlan, Refusal> plan_build(const TrtexecConfig& config, const gpu::ModelSpec& model,
                                             const ModelSource& source, const gpu::ShapeSet& shapes) {
  auto recipe = recipe_for(model, source.summary);
  if (!recipe) return std::unexpected(recipe.error());
  auto profile = profile_shapes(source.input, shapes);
  if (!profile) return std::unexpected(profile.error());
  BuildPlan plan{.settings = engine_settings(*recipe, config.input_name, *profile),
                 .label = shapes_label(shapes)};
  plan.command = sanitize_trt11(plan.settings);
  auto name = engine_name({.content = source.content_key,
                           .settings = plan.settings,
                           .runtime = config.runtime.token(),
                           .device = config.device.token()});
  if (!name) return std::unexpected(name.error());
  plan.engine_file = std::move(*name);
  plan.engine = config.cache_dir / plan.engine_file;
  plan.log_file = plan.engine;
  plan.log_file += ".build.log";
  return plan;
}

TrtexecBuilder::TrtexecBuilder(TrtexecConfig config, EngineCheck check)
    : config_(std::move(config)), check_(std::move(check)), log_(make_logger()) {}

std::expected<gpu::Artifact, Refusal> TrtexecBuilder::build(const gpu::ModelSpec& model,
                                                            const gpu::ShapeSet& shapes,
                                                            gpu::Progress& progress) const {
  if (config_.runtime.major < 11) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   "TensorRT " + config_.runtime.token() + " : trtexec 11 ou plus attendu"});
  }
  auto source = read_model(model, config_.input_name);
  if (!source) return std::unexpected(source.error());
  auto plan = plan_build(config_, model, *source, shapes);
  if (!plan) return std::unexpected(plan.error());
  for (const std::string& flag : plan->command.removed) {
    log_->info("{} : « {} » retiré de la ligne de commande (TensorRT 11)", model.id, flag);
  }

  report(progress, 0.0F, "Préparation");
  const CacheEntry entry{.destination = plan->engine};
  auto produced = produce_once(
      entry, [&](const fs::path& output) { return run(*plan, model, *source, output, progress); },
      progress.stop);
  if (!produced) {
    log_->warn("{} {} : {} ({})", model.id, plan->label, produced.error().detail,
               name(produced.error().code));
    return std::unexpected(produced.error());
  }
  if (*produced == Production::Reused) {
    log_->info("{} {} : moteur en cache {}", model.id, plan->label, plan->engine_file);
    report(progress, 1.0F, "Moteur en cache");
  }
  return gpu::Artifact{.cache_key = plan->engine_file, .path = plan->engine};
}

std::expected<void, Refusal> TrtexecBuilder::run(const BuildPlan& plan, const gpu::ModelSpec& model,
                                                 const ModelSource& source, const fs::path& output,
                                                 gpu::Progress& progress) const {
  const ProcessSpec spec{
      .program = config_.trtexec,
      .args = trtexec_arguments(
          {.onnx = model.onnx_path, .engine = output, .timing_cache = timing_cache_for(plan.engine)},
          plan.command.kept),
      .timeout = config_.timeout,
  };
  log_->info("{} {} : construction de {}", model.id, plan.label, plan.engine_file);
  std::ofstream log_file(plan.log_file, std::ios::trunc);
  TrtexecProgress parsed;
  auto exit = run_process(
      spec,
      [&](std::string_view line) {
        log_file << line << '\n';
        if (auto step = parsed.feed(line)) report(progress, step->fraction, step->stage);
      },
      progress.stop);
  if (!exit) {
    log_file << "[ajp] " << exit.error().detail << '\n';
    return std::unexpected(exit.error());
  }
  log_file << "[ajp] trtexec : " << describe(*exit) << '\n';
  if (auto verdict = judge(*exit, parsed, plan, config_.timeout); !verdict) {
    log_file << "[ajp] " << verdict.error().detail << '\n';
    return verdict;
  }
  // Le moteur doit correspondre au modèle haché avant la construction.
  auto after = read_model(model, config_.input_name);
  if (!after || after->content_key != source.content_key) {
    return std::unexpected(
        Refusal{RefusalCode::ModelMismatch, model.onnx_path.string() + " a changé pendant la construction"});
  }
  if (check_) return check_(output);
  return {};
}

}  // namespace ajp::prepare
