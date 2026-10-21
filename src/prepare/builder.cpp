// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:542-582, 691-806
// et 862-890 : clé du moteur, journal de construction à côté du moteur, code
// de sortie et place libre dans le refus, aucun moteur incomplet publié.
#include "prepare/builder.h"

#include <format>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>
#include <utility>

#include <spdlog/logger.h>

#include "prepare/cache.h"
#include "prepare/cache_io.h"
#include "prepare/key_lock.h"
#include "prepare/log.h"
#include "prepare/process.h"
#include "prepare/sha256.h"
#include "prepare/trtexec_progress.h"
#include "prepare/utf8.h"

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
  const std::string where = " ; journal : " + utf8_path(plan.log_file);
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

// Le verrou du GPU (device_timing_cache), sans attendre s'il est libre ; sinon l'OSD dit
// pourquoi la construction ne démarre pas, et l'attente reste annulable.
std::expected<KeyLock, Refusal> lock_device(const fs::path& timing_cache, gpu::Progress& progress) {
  fs::path lock_file = timing_cache;
  lock_file += ".build-lock";
  auto held = KeyLock::acquire(lock_file, progress.stop, LockWait::TryOnly);
  if (held || held.error().code != RefusalCode::CacheBusy) return held;
  report(progress, 0.0F, "En attente d'une autre construction sur ce GPU");
  return KeyLock::acquire(lock_file, progress.stop);
}

// trtexec réécrit son timing cache sur place, sans temporaire : tué pendant
// cette écriture, il laisserait un cache corrompu que chaque construction
// suivante refuserait. Il travaille donc sur une copie du dossier de travail,
// publiée par renommage après une construction réussie, sous le verrou du GPU.
// Le fork lui passait le cache persistant (aji_trt.cpp:876-889) ; la clé du
// moteur n'en dépend pas.
void stage_timing_cache(const fs::path& persistent, const fs::path& staged, spdlog::logger& log) {
  std::error_code error;
  if (!fs::exists(persistent, error)) return;
  fs::copy_file(persistent, staged, error);
  if (!error) return;
  log.warn("timing cache {} illisible ({}), construction sans lui", utf8_path(persistent), error.message());
  fs::remove(staged, error);
}

void keep_timing_cache(const fs::path& staged, const fs::path& persistent, spdlog::logger& log) {
  std::error_code error;
  if (!fs::exists(staged, error)) return;
  if (auto kept = publish_file(staged, persistent); !kept) {
    log.warn("timing cache non gardé : {}", kept.error().detail);  // la construction suivante sera plus lente
  }
}

// Le rapport des couches publié à côté du moteur, et son résumé au journal : un reformatage hors
// des E/S est une fusion manquée, à corriger dans l'export ONNX (tools/trt/layer_report.py).
void keep_layer_info(const fs::path& staged, const BuildPlan& plan, std::string_view model,
                     spdlog::logger& log) {
  std::ifstream in(staged, std::ios::binary);
  if (!in) {
    log.warn("{} {} : trtexec n'a pas écrit de rapport des couches", model, plan.label);
    return;
  }
  const std::string json{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  in.close();
  const LayerCounts counts = layer_counts(json);
  if (auto kept = publish_file(staged, plan.layer_info); !kept) {
    log.warn("{} {} : rapport des couches non gardé : {}", model, plan.label, kept.error().detail);
    return;
  }
  log.info("{} {} : {} couches dont {} reformatages ; rapport {}", model, plan.label, counts.layers,
           counts.reformats, utf8_path(plan.layer_info));
}

}  // namespace

std::filesystem::path device_timing_cache(const TrtexecConfig& config) {
  const std::string id = sha256_hex(config.runtime.token() + "\n" + config.device.token()).substr(0, 16);
  return config.cache_dir / std::format("aji-timing-{}.cache", id);
}

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
  plan.layer_info = layer_info_for(plan.engine);
  return plan;
}

TrtexecBuilder::TrtexecBuilder(TrtexecConfig config, EngineCheck check)
    : config_(std::move(config)), check_(std::move(check)), log_(make_logger()) {}

std::expected<gpu::Artifact, Refusal> TrtexecBuilder::build(const gpu::ModelSpec& model,
                                                            const gpu::ShapeSet& shapes,
                                                            gpu::Progress& progress) const {
  return rebuild(model, shapes, progress, {});
}

std::expected<gpu::Artifact, Refusal> TrtexecBuilder::rebuild(const gpu::ModelSpec& model,
                                                              const gpu::ShapeSet& shapes,
                                                              gpu::Progress& progress,
                                                              const std::string& rejected_sha256) const {
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
  const CacheEntry entry{.destination = plan->engine, .rejected_sha256 = rejected_sha256};
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
  const fs::path timing_cache = device_timing_cache(config_);
  auto device = lock_device(timing_cache, progress);
  if (!device) return std::unexpected(std::move(device.error()));
  const fs::path staged_cache = output.parent_path() / "timing.cache";
  const fs::path staged_layers = output.parent_path() / "layers.json";
  stage_timing_cache(timing_cache, staged_cache, *log_);
  const ProcessSpec spec{
      .program = config_.trtexec,
      .args = trtexec_arguments({.onnx = model.onnx_path,
                                 .engine = output,
                                 .timing_cache = staged_cache,
                                 .layer_info = staged_layers},
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
  if (!is_published(output)) {
    return std::unexpected(
        Refusal{RefusalCode::BuilderFailed,
                "trtexec a réussi sans écrire de moteur ; journal : " + utf8_path(plan.log_file)});
  }
  // Le moteur doit correspondre au modèle haché avant la construction.
  auto after = read_model(model, config_.input_name);
  if (!after) return std::unexpected(after.error());
  if (after->content_key != source.content_key) {
    return std::unexpected(Refusal{RefusalCode::ModelMismatch,
                                   utf8_path(model.onnx_path) + " a changé pendant la construction"});
  }
  if (check_) {
    if (auto usable = check_(output); !usable) return usable;
  }
  keep_timing_cache(staged_cache, timing_cache, *log_);
  keep_layer_info(staged_layers, plan, model.id, *log_);
  return {};
}

}  // namespace ajp::prepare
