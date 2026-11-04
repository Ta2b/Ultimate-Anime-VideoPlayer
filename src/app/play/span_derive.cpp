#include "app/play/span_derive.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <vector>

#include "app/play/i18n.h"
#include "app/play/span_int8.h"
#include "gpu/span_fp8/hard_cases.h"
#include "gpu/span_fp8/onnx_weights.h"
#include "gpu/span_fp8/package.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::filesystem::path partial_of(const std::filesystem::path& path) {
  std::filesystem::path out = path;
  out += ".part";
  return out;
}

std::expected<void, Refusal> write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (file)
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  file.close();
  if (!file)
    return refuse(RefusalCode::IoFailed, std::format("{} : écriture impossible", prepare::utf8_path(path)));
  return {};
}

// Un fichier remplacé par un autre déjà écrit sous un nom provisoire : l'ancien passe à « .old » le temps de
// l'opération, et revient si elle échoue.
class Swap {
 public:
  explicit Swap(std::filesystem::path target)
      : target_(std::move(target)), part_(partial_of(target_)), old_(target_) {
    old_ += ".old";
  }
  Swap(const Swap&) = delete;
  Swap& operator=(const Swap&) = delete;
  const std::filesystem::path& part() const { return part_; }

  std::expected<void, Refusal> install() {
    std::error_code error;
    if (std::filesystem::exists(target_, error)) {
      std::filesystem::rename(target_, old_, error);
      if (error) return failure(error);
      had_old_ = true;
    }
    std::filesystem::rename(part_, target_, error);
    if (error) return failure(error);
    installed_ = true;
    return {};
  }
  // Remet l'état d'avant : le nouveau fichier part, l'ancien revient, le provisoire disparaît.
  void rollback() {
    std::error_code error;
    if (installed_) std::filesystem::remove(target_, error);
    if (had_old_) std::filesystem::rename(old_, target_, error);
    std::filesystem::remove(part_, error);
    installed_ = false;
    had_old_ = false;
  }
  // L'opération a réussi : l'ancien fichier n'a plus de raison d'être.
  void finish() {
    std::error_code error;
    if (had_old_) std::filesystem::remove(old_, error);
    had_old_ = false;
  }

 private:
  std::unexpected<Refusal> failure(const std::error_code& error) const {
    return refuse(RefusalCode::IoFailed,
                  std::format("{} : {}", prepare::utf8_path(target_), error.message()));
  }

  std::filesystem::path target_;
  std::filesystem::path part_;
  std::filesystem::path old_;
  bool had_old_ = false;
  bool installed_ = false;
};

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

}  // namespace

StillsSplit split_stills(std::size_t count) {
  StillsSplit split;
  for (std::size_t i = 0; i < count; ++i) (i % 3 == 2 ? split.validation : split.calibration).push_back(i);
  return split;
}

std::expected<void, Refusal> commit_derivation(const std::filesystem::path& package,
                                               const std::filesystem::path& receipt,
                                               std::span<const std::byte> package_bytes,
                                               std::string_view receipt_text, bool passed) {
  std::error_code error;
  std::filesystem::create_directories(receipt.parent_path(), error);
  Swap package_swap(package);
  Swap receipt_swap(receipt);
  const auto fail = [&](std::expected<void, Refusal> why) {
    package_swap.rollback();
    receipt_swap.rollback();
    return why;
  };
  // Tout s'écrit d'abord sous un nom provisoire : un disque plein ou une coupure laisse l'ancien état entier.
  if (passed) {
    if (auto ok = write_file(package_swap.part(), package_bytes); !ok) return fail(ok);
  }
  const auto* text = reinterpret_cast<const std::byte*>(receipt_text.data());
  if (auto ok = write_file(receipt_swap.part(), std::span(text, receipt_text.size())); !ok) return fail(ok);
  if (passed) {
    if (auto ok = package_swap.install(); !ok) return fail(ok);
  }
  if (auto ok = receipt_swap.install(); !ok) return fail(ok);
  package_swap.finish();
  receipt_swap.finish();
  // Refusé : le reçu dit « passed = false », et le paquet d'une dérivation d'avant n'y correspond plus.
  if (!passed) std::filesystem::remove(package, error);
  return {};
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
  for (const gpu::SpanFp8ImageScore& score : from)
    out.push_back({.psnr_db = score.psnr_db,
                   .ssim = score.ssim,
                   .error_p999 = score.error_p999,
                   .dark_error_p999 = score.dark_error_p999});
  return out;
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
  const int height = request.validation.front().height;
  report.size = std::format("{}x{}", width, height);
  report.verdict = gpu::span_fp8::judge_hardened(scores_of(derived->scores), scores_of(derived->hard_scores));
  report.passed = report.verdict.passed;
  // Étape 3.5a : la variante à accumulateur f16 hors blocs, si elle passe la porte et perd au plus 0,3 dB de
  // PSNR moyen ; c'est alors elle que le paquet, le reçu et le temps décrivent.
  bool acc16 = false;
  double kernels_ms = derived->kernels_ms;
  if (!derived->scores_acc16.empty()) {
    const gpu::span_fp8::Verdict verdict16 = gpu::span_fp8::judge_hardened(
        scores_of(derived->scores_acc16), scores_of(derived->hard_scores_acc16));
    if (gpu::span_fp8::keep_acc16(report.verdict, verdict16)) {
      acc16 = true;
      report.verdict = verdict16;
      kernels_ms = derived->kernels_ms_acc16;
    }
  }
  report.acc16 = acc16;

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
  receipt.calibration_images = report.calibration_images;
  receipt.calibration_crops = report.calibration_images;  // des images entières, pas des découpes de 576²
  receipt.calibration_crop = width;
  receipt.corpus_sha256 = corpus.hex();
  receipt.validation_images = report.validation_images;
  receipt.validation_size = report.size;
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

  const std::string scores_text = verdict_text(report.verdict);
  if (report.passed) {
    report.text = i18n::tr_format(
        "Performance mode ready: {} ({} calibration frames, {} validation frames at {}, kernels {} ms). It "
        "is used the next time a video opens.",
        scores_text, report.calibration_images, report.validation_images, report.size, db(kernels_ms));
  } else {
    report.text = i18n::tr_format(
        "Performance mode rejected: {}; {:.0f} dB, {:.0f} dB and {} are required. Playback stays on TensorRT "
        "FP16.",
        scores_text, gpu::span_fp8::kMinPsnrMeanDb, gpu::span_fp8::kMinPsnrWorstDb,
        ratio(gpu::span_fp8::kMinSsimWorst));
  }
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
  report.size = std::format("{}x{}", request.validation.front().width, request.validation.front().height);
  report.verdict = gpu::span_fp8::judge_hardened(scores_of(derived->scores), scores_of(derived->hard_scores));
  report.passed = report.verdict.passed;
  const auto drop = [&](std::vector<std::string>& from) { std::erase(from, report.size); };
  drop(sizes->validated);
  drop(sizes->refused);
  (report.passed ? sizes->validated : sizes->refused).push_back(report.size);
  const std::string text =
      with_gate_refusals(gpu::span_fp8::with_sizes(sizes->text, sizes->validated, sizes->refused), sizes->refused);
  progress.set(0.97F, i18n::tr("writing"));
  if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, i18n::tr("preparation interrupted"));
  // Le paquet garde ses octets : il vaut toujours aux autres tailles de son reçu.
  if (auto ok = commit_derivation(report.package, report.receipt, bytes, text, true); !ok)
    return std::unexpected(std::move(ok.error()));
  const std::string scores_text = verdict_text(report.verdict);
  report.text = report.passed
                    ? i18n::tr_format("Performance mode validated at {}: {}.", report.size, scores_text)
                    : i18n::tr_format(
                          "Performance mode rejected at {}: {}. Playback stays on TensorRT FP16 at this "
                          "size.",
                          report.size, scores_text);
  if (!report.passed) report.package.clear();
  progress.set(1.0F, i18n::tr("done"));
  return report;
}

SpanPrepare make_span_prepare(const std::shared_ptr<BackendLibrary>& library, const gpu::BackendInit& init,
                              bool supported, const std::filesystem::path& model,
                              const std::filesystem::path& models_dir, const std::filesystem::path& video,
                              const std::string& unavailable, const media::StillReduce& reduce,
                              const SpanFp8Context& context, bool automatic) {
  SpanPrepare prepare;
  if (!supported) {
    prepare.note =
        i18n::tr("Performance mode (FP8) is not available: ") +
        (unavailable.empty() ? std::string(i18n::tr(
                                   "it needs an RTX 40 (Ada) or RTX 50 (Blackwell) with the TensorRT engine"))
                             : unavailable) +
        ".";
    return prepare;
  }
  if (model.empty()) {
    prepare.note = i18n::tr("No SPAN model in use: nothing to prepare.");
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
  } else if (found.stale) {
    prepare.note = i18n::tr("Not prepared yet: playback stays on TensorRT FP16 at the Performance level.");
  } else if (!found.note.empty()) {
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
  prepare.run = [library, init, model, models_dir, video, reduce, validate, automatic](
                    SpanDeriveProgress& progress,
                    const std::stop_token& stop) -> std::expected<std::string, Refusal> {
    SpanDerivePlan plan{.model = model, .models_dir = models_dir};
    if (automatic) plan.min_free_ratio = kAutoMinFreeRatio;
    const SpanStillsSource stills = [&](const std::stop_token& token,
                                        const std::function<void(float)>& report) {
      return media::extract_stills(video, media::StillsSpec{.reduce = reduce}, token, report);
    };
    const SpanDeriver deriver = [&](const gpu::SpanFp8Request& request, gpu::Progress& backend_progress) {
      auto backend = BackendLibrary::create(library, init);
      if (!backend)
        return std::expected<gpu::SpanFp8Derivation, Refusal>(std::unexpected(std::move(backend.error())));
      return (*backend)->derive_span_fp8(request, backend_progress);
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
