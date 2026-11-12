// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:542-582, 691-806
// et 862-890 : clé du moteur, journal de construction à côté du moteur, code
// de sortie et place libre dans le refus, aucun moteur incomplet publié.
#include "prepare/builder.h"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <spdlog/logger.h>

#include "core/file_sync.h"
#include "core/text.h"
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

// Sous ce seuil de place libre, un moteur ne s'écrit pas : l'échec de trtexec est un disque plein (A39-10).
constexpr std::uintmax_t kMinFreeBytes = std::uintmax_t{512} * 1024 * 1024;

bool disk_nearly_full(const fs::path& folder) {
  std::error_code error;
  const auto space = fs::space(folder, error);
  return !error && space.available < kMinFreeBytes;
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
  if ((exit.code != 0 || parsed.failed()) && disk_nearly_full(plan.engine.parent_path())) {
    return std::unexpected(Refusal{
        RefusalCode::InsufficientSpace,
        std::format(
            "trtexec a échoué faute de place : {} libres dans le dossier du moteur, il en faut au moins "
            "512 Mo{}{}",
            free_space(plan.engine.parent_path()), error, where)});
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

// Sous Windows, trtexec relit sa ligne de commande dans la page ANSI du système : un chemin qu'elle ne sait
// pas écrire et qui n'a pas de nom court 8.3 (volume sans 8.3) y deviendrait « ? ». On le sait avant de
// lancer : le refus dit quoi déplacer au lieu d'envoyer vers la place libre (B05-06). Toujours vide hors de
// Windows.
std::string unreachable_by_trtexec(const std::vector<fs::path>& paths) {
  for (const fs::path& path : paths) {
    if (!ansi_command_path(path)) return utf8_path(path);
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
  report(progress, 0.0F, "Waiting for another build on this GPU");
  return KeyLock::acquire(lock_file, progress.stop);
}

// Cette ligne d'erreur de trtexec parle-t-elle du timing cache ?
bool mentions_cache(std::string_view line) {
  const std::string lower = core::ascii_lower(line);
  return lower.find("timing cache") != std::string::npos || lower.find("timingcache") != std::string::npos ||
         lower.find("timing-cache") != std::string::npos;
}

// Un échec ou un plantage avec le cache se refait sans lui ; pas un délai (le refaire doublerait l'attente),
// pas un disque plein, pas une annulation.
bool retries_without_cache(RefusalCode code) {
  return code == RefusalCode::BuilderFailed || code == RefusalCode::BuilderCrashed;
}

// Le cache accusé est mis de côté (« .bad ») : il ne gêne plus, et reste pour qui voudra le regarder.
void set_aside_bad_cache(const fs::path& persistent, spdlog::logger& log) {
  fs::path bad = persistent;
  bad += ".bad";
  if (const std::error_code error = core::rename_retrying(persistent, bad)) {
    log.warn("timing cache fautif {} non mis de côté : {}", utf8_path(persistent.filename()),
             error.message());
  }
}

// trtexec réécrit son timing cache sur place, sans temporaire : tué pendant
// cette écriture, il laisserait un cache corrompu que chaque construction
// suivante refuserait. Il travaille donc sur une copie du dossier de travail,
// publiée par renommage après une construction réussie, sous le verrou du GPU.
// Le fork lui passait le cache persistant (aji_trt.cpp:876-889) ; la clé du
// moteur n'en dépend pas.
// Rend vrai quand la copie est faite.
bool stage_timing_cache(const fs::path& persistent, const fs::path& staged, spdlog::logger& log) {
  std::error_code error;
  if (!fs::exists(persistent, error)) return false;
  fs::copy_file(persistent, staged, error);
  if (!error) return true;
  log.warn("timing cache {} illisible ({}), construction sans lui", utf8_path(persistent), error.message());
  fs::remove(staged, error);
  return false;
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
  if (config_.runtime.major < 11) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   "TensorRT " + config_.runtime.token() + " : trtexec 11 ou plus attendu"});
  }
  if (config_.trtexec.empty()) {
    return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                   "trtexec absent du dossier du moteur, ou d'empreinte inattendue : "
                                   "réinstallez le moteur TensorRT"});
  }
  auto source = read_model(model, config_.input_name);
  if (!source) return std::unexpected(source.error());
  auto plan = plan_build(config_, model, *source, shapes);
  if (!plan) return std::unexpected(plan.error());
  for (const std::string& flag : plan->command.removed) {
    log_->info("{} : « {} » retiré de la ligne de commande (TensorRT 11)", model.id, flag);
  }

  report(progress, 0.0F, "Preparing");
  // Le détenteur de la clé attend le GPU (kMaxLockWait au plus) puis lance trtexec, deux fois au pire (le
  // nouvel essai sans timing cache) ; une minute de plus pour la vérification et la publication (A05-02,
  // B05-07).
  const CacheEntry entry{.destination = plan->engine,
                         .max_wait = std::chrono::duration_cast<std::chrono::milliseconds>(
                             kMaxLockWait + 2 * config_.timeout + std::chrono::minutes(1))};
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
    report(progress, 1.0F, "Engine in cache");
  }
  return gpu::Artifact{.cache_key = plan->engine_file, .path = plan->engine};
}

std::expected<void, Refusal> TrtexecBuilder::run(const BuildPlan& plan, const gpu::ModelSpec& model,
                                                 const ModelSource& source, const fs::path& output,
                                                 gpu::Progress& progress) const {
  const fs::path timing_cache = device_timing_cache(config_);
  auto device = lock_device(timing_cache, progress);
  if (!device) return std::unexpected(std::move(device.error()));
  if (!config_.trtexec_sha256.empty()) {
    const auto now = sha256_file_known(config_.trtexec);
    if (!now || *now != config_.trtexec_sha256) {
      return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                     "trtexec remplacé depuis l'ouverture du moteur : réinstallez le moteur "
                                     "TensorRT, puis rouvrez la vidéo"});
    }
  }
  const fs::path staged_cache = output.parent_path() / "timing.cache";
  const fs::path staged_layers = output.parent_path() / "layers.json";
  if (const std::string bad = unreachable_by_trtexec({model.onnx_path, output, staged_layers, staged_cache});
      !bad.empty()) {
    return std::unexpected(
        Refusal{RefusalCode::CacheIo,
                "trtexec ne sait pas lire ce chemin (nom hors de la page ANSI de Windows, pas de nom court "
                "8.3 sur ce "
                "volume) : " +
                    bad +
                    ". Placez les modèles et le dossier des moteurs dans un chemin en lettres latines, "
                    "ou activez « Utiliser Unicode UTF-8 » dans les paramètres régionaux de Windows."});
  }
  const bool cached = stage_timing_cache(timing_cache, staged_cache, *log_);
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
  // Une ligne d'erreur de trtexec qui parle du cache : le cache est alors le coupable probable (B05-03).
  bool cache_complaint = false;
  // Un essai de trtexec, son journal ajouté à celui du moteur (vidé au premier).
  const auto attempt = [&](std::ios::openmode mode) -> std::expected<void, Refusal> {
    std::ofstream log_file(plan.log_file, mode);
    TrtexecProgress parsed;
    auto exit = run_process(
        spec,
        [&](std::string_view line) {
          log_file << line << '\n';
          if (line.find("[E]") != std::string_view::npos && mentions_cache(line)) cache_complaint = true;
          if (auto step = parsed.feed(line)) report(progress, step->fraction, step->stage);
        },
        progress.stop);
    if (!exit) {
      log_file << "[ajp] " << exit.error().detail << '\n';
      return std::unexpected(exit.error());
    }
    log_file << "[ajp] trtexec : " << describe(*exit) << '\n';
    log_file.flush();
    if (!log_file)
      log_->warn("journal {} illisible ou incomplet (droits ou disque plein, voir la place libre)",
                 utf8_path(plan.log_file));
    auto verdict = judge(*exit, parsed, plan, config_.timeout);
    if (!verdict) log_file << "[ajp] " << verdict.error().detail << '\n';
    return verdict;
  };
  auto built = attempt(std::ios::trunc);
  // Le timing cache persistant n'est remplacé qu'après une réussite : s'il fait échouer ou planter trtexec
  // (abîmé, écrit par un autre build), chaque construction sur ce GPU échouerait. Un échec ou un plantage
  // avec lui se refait une fois sans lui, sur les deux systèmes (sous Windows un abort() sort en code 3, sous
  // Linux c'est un signal : B05-02). Le cache neuf ne remplace l'ancien que si l'on a de quoi croire l'ancien
  // fautif (B05-03).
  bool replace_cache = true;
  bool cache_suspect = false;
  if (!built && cached && retries_without_cache(built.error().code)) {
    const RefusalCode first = built.error().code;
    cache_suspect = first == RefusalCode::BuilderCrashed || cache_complaint;
    log_->warn("{} {} : trtexec a échoué avec le timing cache {} ({}), nouvel essai sans lui", model.id,
               plan.label, utf8_path(timing_cache.filename()), name(first));
    std::error_code removed;
    fs::remove(staged_cache, removed);
    report(progress, 0.0F, "Preparing");
    built = attempt(std::ios::app);
    // Réussite sans le cache : un plantage ou une plainte sur le cache l'accuse ; un échec d'une autre sorte
    // (mémoire du GPU, pilote) est passager, et l'ancien cache, riche de toutes les mesures, reste.
    replace_cache = cache_suspect;
  }
  if (!built) return built;
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
  if (replace_cache) {
    if (cache_suspect) set_aside_bad_cache(timing_cache, *log_);
    keep_timing_cache(staged_cache, timing_cache, *log_);
  } else {
    log_->info(
        "{} {} : l'échec avec le timing cache venait d'ailleurs (aucune mention du cache) : cache gardé",
        model.id, plan.label);
  }
  keep_layer_info(staged_layers, plan, model.id, *log_);
  return {};
}

}  // namespace ajp::prepare
