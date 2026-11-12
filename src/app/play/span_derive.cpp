#include "app/play/span_derive.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "app/play/i18n.h"
#include "app/play/span_int8.h"
#include "core/file_sync.h"
#include "gpu/span_fp8/hard_cases.h"
#include "gpu/span_fp8/onnx_weights.h"
#include "gpu/span_fp8/package.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

// C02-11 : la découpe des images de la dérivation n'a qu'une valeur, celle de gpu/span_fp8/derive.h ;
// make_span_prepare la passe à media::StillsSpec, dont les valeurs par défaut doivent rester les mêmes.
static_assert(media::StillsSpec{}.max_width == gpu::span_fp8::kCropMaxWidth &&
              media::StillsSpec{}.max_height == gpu::span_fp8::kCropMaxHeight);

namespace {

// « 52,0 dB » : une décimale, virgule française.
std::string db(double value) {
  std::string text = std::format("{:.1f}", value);
  std::ranges::replace(text, '.', ',');
  return text;
}

std::string ratio(double value) {
  std::string text = std::format("{:.4f}", value);
  std::ranges::replace(text, '.', ',');
  return text;
}

// « 1,18 » : un rapport de temps, deux décimales.
std::string times(double value) {
  std::string text = std::format("{:.2f}", value);
  std::ranges::replace(text, '.', ',');
  return text;
}

}  // namespace

StillsSplit split_stills(std::size_t count) {
  StillsSplit split;
  for (std::size_t i = 0; i < count; ++i) (i % 3 == 2 ? split.validation : split.calibration).push_back(i);
  return split;
}

namespace {

// Les étapes que le backend nomme (en français, il ne connaît pas la langue de l'interface) : dites dans
// celle-ci.
std::string gpu_stage_text(const std::string& stage) {
  if (stage == "référence FP16") return i18n::tr("FP16 reference");
  if (stage == "mesure du temps") return i18n::tr("timing");
  if (stage == "terminé") return i18n::tr("done");
  return stage;  // « calibration » et « validation » sont les mêmes dans les deux langues
}

}  // namespace

namespace {

std::string verdict_text(const gpu::span_fp8::Verdict& verdict) {
  std::string text = i18n::tr_format("mean PSNR {} dB, worst {} dB, worst SSIM {}", db(verdict.psnr_mean_db),
                                     db(verdict.psnr_worst_db), ratio(verdict.ssim_worst));
  text += i18n::tr_format(", 99.9th percentile error {}/255 (dark {}/255), hard frames worst {} dB",
                          db(255.0 * verdict.error_p999_worst), db(255.0 * verdict.dark_error_p999_worst),
                          db(verdict.hard_psnr_worst_db));
  return text;
}

std::vector<gpu::span_fp8::ImageScore> scores_of(const std::vector<gpu::SpanFp8ImageScore>& from) {
  std::vector<gpu::span_fp8::ImageScore> out;
  out.reserve(from.size());
  for (const gpu::SpanFp8ImageScore& score : from)
    out.push_back({.psnr_db = score.psnr_db,
                   .ssim = score.ssim,
                   .error_p999 = score.error_p999,
                   .dark_error_p999 = score.dark_error_p999});
  return out;
}

// B02-01 : le gain des noyaux FP8 sur le SPAN FP16 de TensorRT, que le backend a chronométré avec eux
// (C02-01). Rien quand la qualité refuse déjà, ou sans mesure (pas de moteur FP16 dans la requête). Un gain
// sous kMinSpeedup refuse le paquet : sa perte de qualité ne s'achèterait par rien.
void judge_speed(const gpu::SpanFp8Derivation& derived, double kernels_ms, SpanDeriveReport& report) {
  if (!report.verdict.passed || derived.trt_fp16_ms <= 0.0) return;
  report.trt_fp16_ms = derived.trt_fp16_ms;
  report.speedup = kernels_ms > 0.0 ? derived.trt_fp16_ms / kernels_ms : 0.0;
  report.too_slow = report.speedup < gpu::span_fp8::kMinSpeedup;
  if (report.too_slow) report.verdict.passed = false;
}

// La requête : le moteur FP16 à chronométrer et la taille où le faire (C02-01, C02-05).
std::expected<void, Refusal> ask_timing(const SpanDerivePlan& plan, SpanDeriveProgress& progress,
                                        const std::stop_token& stop, gpu::SpanFp8Request& request) {
  const gpu::SpanStill& still = request.validation.front();
  request.timing_width = (plan.work_width > 0 ? plan.work_width : still.width) / 2 * 2;
  request.timing_height = (plan.work_height > 0 ? plan.work_height : still.height) / 2 * 2;
  if (!plan.fp16_engine) return {};
  progress.set(0.3F, i18n::tr("TensorRT FP16 engine"));
  gpu::Progress building{.report = {}, .stop = stop};
  auto engine = plan.fp16_engine(request.timing_width, request.timing_height, building);
  if (!engine) return std::unexpected(std::move(engine.error()));
  request.fp16 = std::move(*engine);
  return {};
}

// Ce que la variante à accumulateur f16 a coûté et rapporté, quand elle est gardée (C02-03).
std::string acc16_text(const SpanDeriveReport& report) {
  if (!report.acc16) return {};
  const gpu::SpanFp8Derivation& d = report.derivation;
  const double gain = d.kernels_ms > 0.0 ? 100.0 * (1.0 - d.kernels_ms_acc16 / d.kernels_ms) : 0.0;
  return " " +
         i18n::tr_format(
             "The f16 accumulator variant is kept: it costs {} dB of mean PSNR and saves {} % of the time.",
             db(report.acc16_loss_db), db(gain));
}

// Ce que le gain mesuré dit à l'utilisateur : rien sans mesure ; le refus quand il manque.
std::string speed_text(const SpanDeriveReport& report, double kernels_ms) {
  if (report.trt_fp16_ms <= 0.0) return {};
  if (report.too_slow)
    return i18n::tr_format(
        "Performance mode rejected: FP8 kernels {} ms against {} ms for TensorRT FP16 ({}× as fast, at least "
        "{}× is required). Playback stays on TensorRT FP16.",
        db(kernels_ms), db(report.trt_fp16_ms), times(report.speedup), times(gpu::span_fp8::kMinSpeedup));
  return " " + i18n::tr_format(
                   "TensorRT FP16 takes {} ms here: the FP8 kernels are {}× as fast, for the "
                   "quality given above.",
                   db(report.trt_fp16_ms), times(report.speedup));
}

// La porte d'une variante : images de validation, cas durs, puis l'écrêtage relevé sur la validation
// (C02-04 ; les deux variantes ont les mêmes échelles).
gpu::span_fp8::Verdict judge_variant(const std::vector<gpu::SpanFp8ImageScore>& scores,
                                     const std::vector<gpu::SpanFp8ImageScore>& hard,
                                     const gpu::SpanFp8Derivation& derived, std::size_t blocks) {
  gpu::span_fp8::Verdict verdict = gpu::span_fp8::judge_hardened(scores_of(scores), scores_of(hard));
  gpu::span_fp8::judge_peaks(verdict, derived.validation_peaks, derived.scales, blocks);
  return verdict;
}

// Étape 3.5a : la variante à accumulateur f16 hors blocs, si elle passe la porte, perd au plus 0,3 dB de PSNR
// moyen et va plus vite (C02-03) ; c'est alors elle que le paquet, le reçu et le temps décrivent. Rend le
// temps de la variante gardée.
double keep_variant(const gpu::SpanFp8Derivation& derived, std::size_t blocks, SpanDeriveReport& report) {
  report.verdict = judge_variant(derived.scores, derived.hard_scores, derived, blocks);
  report.passed = report.verdict.passed;
  report.acc16 = false;
  if (derived.scores_acc16.empty()) return derived.kernels_ms;
  const gpu::span_fp8::Verdict verdict16 =
      judge_variant(derived.scores_acc16, derived.hard_scores_acc16, derived, blocks);
  if (!gpu::span_fp8::keep_acc16(report.verdict, verdict16, derived.kernels_ms, derived.kernels_ms_acc16))
    return derived.kernels_ms;
  report.acc16 = true;
  report.acc16_loss_db = report.verdict.psnr_mean_db - verdict16.psnr_mean_db;
  report.verdict = verdict16;
  return derived.kernels_ms_acc16;
}

// Le refus pour écrêtage (C02-04) : les images de la vidéo sortent de ce que la calibration a vu.
std::string clipped_text(const gpu::span_fp8::Verdict& verdict) {
  return i18n::tr_format(
      "Performance mode rejected: {} FP8 tensors would clip on the validation frames (peak at {}× their "
      "range): the calibration frames do not cover this video. Playback stays on TensorRT FP16.",
      verdict.clipped_tensors, times(verdict.peak_ratio));
}

// Ce que la dérivation dit à l'utilisateur : prêt avec ses mesures et son gain, ou pourquoi elle refuse.
std::string derived_text(const SpanDeriveReport& report, double kernels_ms) {
  if (report.too_slow) return speed_text(report, kernels_ms);
  const std::string scores_text = verdict_text(report.verdict);
  if (report.passed)
    return i18n::tr_format(
               "Performance mode ready: {} ({} calibration frames, {} validation frames at {}, kernels {} "
               "ms). It is used the next time a video opens.",
               scores_text, report.calibration_images, report.validation_images, report.size,
               db(kernels_ms)) +
           acc16_text(report) + speed_text(report, kernels_ms);
  if (report.verdict.clipped_tensors > 0) return clipped_text(report.verdict);
  return i18n::tr_format(
      "Performance mode rejected: {}; {:.0f} dB, {:.0f} dB and {} are required. Playback stays on TensorRT "
      "FP16.",
      scores_text, gpu::span_fp8::kMinPsnrMeanDb, gpu::span_fp8::kMinPsnrWorstDb,
      ratio(gpu::span_fp8::kMinSsimWorst));
}

void add_hard_cases(gpu::SpanFp8Request& request) {
  for (gpu::span_fp8::HardCase& hard :
       gpu::span_fp8::hard_cases(request.validation.front().width, request.validation.front().height))
    request.hard.push_back(
        gpu::SpanStill{.width = hard.width, .height = hard.height, .rgb = std::move(hard.rgb)});
}

}  // namespace

std::expected<SpanDeriveReport, Refusal> derive_span_fp8(const SpanDerivePlan& plan,
                                                         const SpanStillsSource& stills,
                                                         const SpanDeriver& deriver,
                                                         SpanDeriveProgress& progress,
                                                         const std::stop_token& stop) {
  progress.set(0.0F, i18n::tr("reading the model"));
  auto weights = plan.load_weights ? plan.load_weights() : gpu::span_fp8::package_from_onnx_file(plan.model);
  if (!weights) return std::unexpected(std::move(weights.error()));
  auto source_sha = prepare::sha256_file(plan.model);
  if (!source_sha) return std::unexpected(std::move(source_sha.error()));

  progress.set(0.02F, i18n::tr("video frames"));
  auto taken =
      stills(stop, [&](float fraction) { progress.set(0.02F + 0.28F * fraction, i18n::tr("video frames")); });
  if (!taken) return std::unexpected(std::move(taken.error()));
  const StillsSplit split = split_stills(taken->size());
  if (split.calibration.empty() || split.validation.empty())
    return refuse(RefusalCode::DecodeFailed, i18n::tr("too few usable frames in the video"));

  gpu::SpanFp8Request request;
  request.package = gpu::span_fp8::serialize(*weights);
  request.min_free_ratio = plan.min_free_ratio;
  prepare::Sha256 corpus;
  const auto still_of = [](media::Still& still) {
    return gpu::SpanStill{.width = still.width, .height = still.height, .rgb = std::move(still.rgb)};
  };
  for (const std::size_t i : split.calibration) {
    corpus.update(
        std::span(reinterpret_cast<const std::byte*>((*taken)[i].rgb.data()), (*taken)[i].rgb.size()));
    request.calibration.push_back(still_of((*taken)[i]));
  }
  for (const std::size_t i : split.validation) request.validation.push_back(still_of((*taken)[i]));
  taken->clear();
  add_hard_cases(request);  // étape 3.7a : les cas durs, à la taille des images de validation
  if (auto ok = ask_timing(plan, progress, stop, request); !ok) return std::unexpected(std::move(ok.error()));

  progress.set(0.3F, i18n::tr("calibration"));
  gpu::Progress backend_progress{.report =
                                     [&](float fraction, const std::string& stage) {
                                       progress.set(0.3F + 0.65F * fraction, gpu_stage_text(stage));
                                     },
                                 .stop = stop};
  auto derived = deriver(request, backend_progress);
  if (!derived) return std::unexpected(std::move(derived.error()));

  SpanDeriveReport report;
  report.derivation = *derived;
  report.calibration_images = static_cast<int>(request.calibration.size());
  report.validation_images = static_cast<int>(request.validation.size());
  const int width = request.validation.front().width;
  report.size = gpu::span_fp8::size_key(request.timing_width, request.timing_height);
  const double kernels_ms = keep_variant(*derived, weights->blocks.size(), report);
  const bool acc16 = report.acc16;
  judge_speed(*derived, kernels_ms, report);
  report.passed = report.verdict.passed;

  // Le paquet jugé est celui dont les échelles viennent de cette calibration, et pas un autre.
  gpu::span_fp8::Package package = std::move(*weights);
  package.scales = derived->scales;
  package.acc16 = acc16;
  const std::vector<std::byte> bytes = gpu::span_fp8::serialize(package);
  const std::string package_sha =
      prepare::sha256_hex(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));

  report.package = span_variant_path(SpanVariantKind::Fp8Cutlass, plan.model, plan.models_dir, *source_sha);
  report.receipt = report.package;
  report.receipt.replace_extension(".toml");
  gpu::span_fp8::Receipt receipt;
  receipt.source = prepare::utf8_path(plan.model.filename());
  receipt.source_sha256 = *source_sha;
  receipt.model_sha256 = report.passed ? package_sha : std::string();
  receipt.quantizer = plan.quantizer;
  receipt.kernels_ms = kernels_ms;
  receipt.trt_fp16_ms = report.trt_fp16_ms;
  receipt.speedup = report.speedup;
  receipt.timed_during_playback = plan.during_playback;
  receipt.calibration_images = report.calibration_images;
  receipt.calibration_crops = report.calibration_images;  // des images entières, pas des découpes de 576²
  receipt.calibration_crop = width;
  receipt.corpus_sha256 = corpus.hex();
  receipt.validation_images = report.validation_images;
  receipt.validation_size = std::format("{}x{}", width, request.validation.front().height);
  receipt.timed_size = derived->timed_size;
  if (acc16) {
    receipt.acc16_psnr_loss_db = report.acc16_loss_db;
    receipt.kernels_ms_f32 = derived->kernels_ms;
  }
  receipt.verdict = report.verdict;
  receipt.reference = derived->reference;
  receipt.derived_by = "ajp (natif)";
  receipt.gpu_arch = derived->gpu_arch;
  receipt.kernels_version = derived->kernels_version;
  receipt.blocks = static_cast<int>(package.blocks.size());
  receipt.channels = package.real_channels;
  receipt.acc16 = acc16;
  (report.passed ? receipt.validated_sizes : receipt.refused_sizes).push_back(report.size);
  progress.set(0.97F, i18n::tr("writing"));
  if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, i18n::tr("preparation interrupted"));
  const std::string text =
      gpu::span_fp8::receipt_text(receipt,
                                  "dérivation native du lecteur (app/play/span_derive.h) : SPAN dérivé des "
                                  "poids d'AnimeJaNai, à ne pas publier.");
  if (auto ok = commit_derivation(report.package, report.receipt, bytes, text, report.passed); !ok)
    return std::unexpected(std::move(ok.error()));
  if (!report.passed) report.package.clear();

  report.text = derived_text(report, kernels_ms);
  progress.set(1.0F, i18n::tr("done"));
  return report;
}

std::expected<SpanDeriveReport, Refusal> validate_span_size(const SpanDerivePlan& plan,
                                                            const SpanStillsSource& stills,
                                                            const SpanDeriver& deriver,
                                                            SpanDeriveProgress& progress,
                                                            const std::stop_token& stop) {
  progress.set(0.0F, i18n::tr("reading the model"));
  auto source_sha = prepare::sha256_file(plan.model);
  if (!source_sha) return std::unexpected(std::move(source_sha.error()));
  SpanDeriveReport report;
  report.package = span_variant_path(SpanVariantKind::Fp8Cutlass, plan.model, plan.models_dir, *source_sha);
  report.receipt = report.package;
  report.receipt.replace_extension(".toml");
  auto sizes = read_receipt_sizes(report.receipt);
  if (!sizes) return std::unexpected(std::move(sizes.error()));
  std::vector<std::byte> bytes;
  {
    std::ifstream in(report.package, std::ios::binary);
    if (!in) return refuse(RefusalCode::FileUnreadable, prepare::utf8_path(report.package.filename()));
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto* first = reinterpret_cast<const std::byte*>(content.data());
    bytes.assign(first, first + content.size());
  }

  progress.set(0.02F, i18n::tr("video frames"));
  auto taken =
      stills(stop, [&](float fraction) { progress.set(0.02F + 0.28F * fraction, i18n::tr("video frames")); });
  if (!taken) return std::unexpected(std::move(taken.error()));
  const StillsSplit split = split_stills(taken->size());
  if (split.validation.empty())
    return refuse(RefusalCode::DecodeFailed, i18n::tr("too few usable frames in the video"));
  gpu::SpanFp8Request request;
  request.package = bytes;
  request.validate_only = true;
  request.min_free_ratio = plan.min_free_ratio;
  for (const std::size_t i : split.validation) {
    media::Still& still = (*taken)[i];
    request.validation.push_back(
        gpu::SpanStill{.width = still.width, .height = still.height, .rgb = std::move(still.rgb)});
  }
  taken->clear();
  add_hard_cases(request);
  if (auto ok = ask_timing(plan, progress, stop, request); !ok) return std::unexpected(std::move(ok.error()));

  progress.set(0.3F, i18n::tr("validation"));
  gpu::Progress backend_progress{.report =
                                     [&](float fraction, const std::string& stage) {
                                       progress.set(0.3F + 0.65F * fraction, gpu_stage_text(stage));
                                     },
                                 .stop = stop};
  auto derived = deriver(request, backend_progress);
  if (!derived) return std::unexpected(std::move(derived.error()));
  report.derivation = *derived;
  report.validation_images = static_cast<int>(request.validation.size());
  report.size = gpu::span_fp8::size_key(request.timing_width, request.timing_height);
  auto parsed = gpu::span_fp8::parse(bytes);
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  report.verdict = judge_variant(derived->scores, derived->hard_scores, *derived, parsed->blocks.size());
  judge_speed(*derived, derived->kernels_ms, report);
  report.passed = report.verdict.passed;
  const auto drop = [&](std::vector<std::string>& from) { std::erase(from, report.size); };
  drop(sizes->validated);
  drop(sizes->refused);
  (report.passed ? sizes->validated : sizes->refused).push_back(report.size);
  const std::string text = with_gate_refusals(
      gpu::span_fp8::with_sizes(sizes->text, sizes->validated, sizes->refused), sizes->refused);
  progress.set(0.97F, i18n::tr("writing"));
  if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, i18n::tr("preparation interrupted"));
  // Le paquet garde ses octets : il vaut toujours aux autres tailles de son reçu.
  if (auto ok = commit_derivation(report.package, report.receipt, bytes, text, true); !ok)
    return std::unexpected(std::move(ok.error()));
  const std::string scores_text = verdict_text(report.verdict);
  if (report.too_slow)
    report.text = speed_text(report, derived->kernels_ms);
  else if (report.verdict.clipped_tensors > 0)
    report.text = clipped_text(report.verdict);
  else
    report.text = report.passed
                      ? i18n::tr_format("Performance mode validated at {}: {}.", report.size, scores_text) +
                            speed_text(report, derived->kernels_ms)
                      : i18n::tr_format(
                            "Performance mode rejected at {}: {}. Playback stays on TensorRT FP16 at this "
                            "size.",
                            report.size, scores_text);
  if (!report.passed) report.package.clear();
  progress.set(1.0F, i18n::tr("done"));
  return report;
}

void pause_for_playback(std::string_view stage, const std::stop_token& stop) {
  // Le backend rend compte après chaque image de calibration ou de validation, et après chaque lot de la
  // mesure du temps (C02-02) : la lecture reprend la carte le temps d'une pause, interrompue au plus tard 10
  // ms après une demande d'arrêt.
  if (stage != "calibration" && stage != "validation" && stage != "mesure du temps") return;
  const auto until = std::chrono::steady_clock::now() + kBackgroundPause;
  while (!stop.stop_requested() && std::chrono::steady_clock::now() < until)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

SpanPrepare make_span_prepare(const std::shared_ptr<BackendLibrary>& library, const gpu::BackendInit& init,
                              bool supported, const std::filesystem::path& model,
                              const std::filesystem::path& models_dir, const std::filesystem::path& video,
                              const std::string& unavailable, const media::StillReduce& reduce,
                              const SpanFp8Context& context, bool automatic, bool during_playback) {
  SpanPrepare prepare;
  if (!supported) {
    prepare.note =
        i18n::tr("Performance mode (FP8) is not available: ") +
        (unavailable.empty() ? std::string(i18n::tr("it needs an RTX 40 (Ada) with the TensorRT engine"))
                             : unavailable) +
        ".";
    return prepare;
  }
  if (model.empty()) {
    prepare.note = i18n::tr("No SPAN model in use: nothing to prepare.");
    return prepare;
  }
  if (!span_fp8_cutlass_model(model)) {
    prepare.note =
        i18n::tr("Performance mode (FP8) is for the AnimeJaNai SPAN models: ArtCNN stays in FP16.");
    return prepare;
  }
  const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, model, models_dir,
                                                 context.width > 0 ? &context : nullptr);
  const bool validate = !found.unvalidated_size.empty();
  if (found.variant) {
    prepare.ready = true;
    prepare.note = i18n::tr("Ready: ") + found.note;
  } else if (validate) {
    prepare.note = i18n::tr_format(
        "Not validated at {} yet: a validation pass (about ten seconds) is offered.", found.unvalidated_size);
  } else if (!found.note.empty() && !found.stale) {
    prepare.refused = true;
    prepare.note = i18n::tr("Rejected: ") + found.note;
  } else {
    prepare.note = i18n::tr("Not prepared yet: playback stays on TensorRT FP16 at the Performance level.");
  }
  if (video.empty()) {
    prepare.note += i18n::tr(" Open a video to prepare: the player calibrates on it.");
    return prepare;
  }
  prepare.offered = true;
  prepare.run = [library, init, model, models_dir, video, reduce, validate, automatic, during_playback,
                 work = context](SpanDeriveProgress& progress,
                                 const std::stop_token& stop) -> std::expected<std::string, Refusal> {
    SpanDerivePlan plan{.model = model, .models_dir = models_dir};
    if (automatic) plan.min_free_ratio = kAutoMinFreeRatio;
    plan.during_playback = during_playback;
    plan.work_width = work.width;
    plan.work_height = work.height;
    plan.fp16_engine = [&](int width, int height,
                           gpu::Progress& building) -> std::expected<gpu::Artifact, Refusal> {
      auto backend = BackendLibrary::create(library, init);
      if (!backend) return std::unexpected(std::move(backend.error()));
      return build_span_fp16(**backend, model, width, height, building);
    };
    const SpanStillsSource stills = [&](const std::stop_token& token,
                                        const std::function<void(float)>& report) {
      const media::StillsSpec spec{.max_width = gpu::span_fp8::kCropMaxWidth,
                                   .max_height = gpu::span_fp8::kCropMaxHeight,
                                   .reduce = reduce};
      return media::extract_stills(video, spec, token, report);
    };
    const SpanDeriver deriver = [&](const gpu::SpanFp8Request& request, gpu::Progress& backend_progress) {
      auto backend = BackendLibrary::create(library, init);
      if (!backend)
        return std::expected<gpu::SpanFp8Derivation, Refusal>(std::unexpected(std::move(backend.error())));
      if (!during_playback) return (*backend)->derive_span_fp8(request, backend_progress);
      gpu::Progress paced{.report =
                              [&](float fraction, const std::string& stage) {
                                if (backend_progress.report) backend_progress.report(fraction, stage);
                                pause_for_playback(stage, backend_progress.stop);
                              },
                          .stop = backend_progress.stop};
      return (*backend)->derive_span_fp8(request, paced);
    };
    auto report = validate ? validate_span_size(plan, stills, deriver, progress, stop)
                           : derive_span_fp8(plan, stills, deriver, progress, stop);
    if (!report) return std::unexpected(std::move(report.error()));
    if (!report->passed) return refuse(RefusalCode::ReceiptInvalid, report->text);
    return report->text;
  };
  return prepare;
}

media::StillReduce span_still_reduce(int work_width, int work_height, core::DownscaleFilter filter) {
  using core::DownscaleFilter;
  media::StillReduce out{.width = work_width, .height = work_height};
  switch (filter) {
    case DownscaleFilter::Area:
      out.filter = media::StillFilter::Area;
      break;
    case DownscaleFilter::Spline36:
      out.filter = media::StillFilter::Spline36;
      break;
    case DownscaleFilter::Bilinear:
      out.filter = media::StillFilter::Triangle;
      break;
    case DownscaleFilter::Bicubic:
      out = {.width = work_width,
             .height = work_height,
             .filter = media::StillFilter::Cubic,
             .b = 1.0F,
             .c = 0.0F};
      break;
    case DownscaleFilter::Mitchell:
      out = {.width = work_width,
             .height = work_height,
             .filter = media::StillFilter::Cubic,
             .b = 1.0F / 3.0F,
             .c = 1.0F / 3.0F};
      break;
    case DownscaleFilter::CatmullRom:
      out = {.width = work_width,
             .height = work_height,
             .filter = media::StillFilter::Cubic,
             .b = 0.0F,
             .c = 0.5F};
      break;
    case DownscaleFilter::Hermite:
      out = {.width = work_width,
             .height = work_height,
             .filter = media::StillFilter::Cubic,
             .b = 0.0F,
             .c = 0.0F};
      break;
  }
  return out;
}

}  // namespace ajp::app::play
