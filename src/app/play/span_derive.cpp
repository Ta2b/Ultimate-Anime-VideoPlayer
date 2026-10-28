#include "app/play/span_derive.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <system_error>

#include "app/play/span_int8.h"
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

std::expected<SpanDeriveReport, Refusal> derive_span_fp8(const SpanDerivePlan& plan,
                                                         const SpanStillsSource& stills,
                                                         const SpanDeriver& deriver,
                                                         SpanDeriveProgress& progress,
                                                         const std::stop_token& stop) {
  progress.set(0.0F, "lecture du modèle");
  auto weights = plan.load_weights ? plan.load_weights() : gpu::span_fp8::package_from_onnx_file(plan.model);
  if (!weights) return std::unexpected(std::move(weights.error()));
  auto source_sha = prepare::sha256_file(plan.model);
  if (!source_sha) return std::unexpected(std::move(source_sha.error()));

  progress.set(0.02F, "images de la vidéo");
  auto taken =
      stills(stop, [&](float fraction) { progress.set(0.02F + 0.28F * fraction, "images de la vidéo"); });
  if (!taken) return std::unexpected(std::move(taken.error()));
  const StillsSplit split = split_stills(taken->size());
  if (split.calibration.empty() || split.validation.empty())
    return refuse(RefusalCode::DecodeFailed, "trop peu d'images exploitables dans la vidéo");

  gpu::SpanFp8Request request;
  request.package = gpu::span_fp8::serialize(*weights);
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

  progress.set(0.3F, "calibration");
  gpu::Progress backend_progress{
      .report = [&](float fraction,
                    const std::string& stage) { progress.set(0.3F + 0.65F * fraction, stage); },
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
  std::vector<gpu::span_fp8::ImageScore> scores;
  for (const gpu::SpanFp8ImageScore& score : derived->scores)
    scores.push_back({.psnr_db = score.psnr_db, .ssim = score.ssim});
  report.verdict = gpu::span_fp8::judge(scores);
  report.passed = report.verdict.passed;

  // Le paquet jugé est celui dont les échelles viennent de cette calibration, et pas un autre.
  gpu::span_fp8::Package package = std::move(*weights);
  package.scales = derived->scales;
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
  receipt.kernels_ms = derived->kernels_ms;
  receipt.calibration_images = report.calibration_images;
  receipt.calibration_crops = report.calibration_images;  // des images entières, pas des découpes de 576²
  receipt.calibration_crop = width;
  receipt.corpus_sha256 = corpus.hex();
  receipt.validation_images = report.validation_images;
  receipt.validation_size = report.size;
  receipt.verdict = report.verdict;
  receipt.reference = derived->reference;
  receipt.derived_by = "ajp (natif)";
  progress.set(0.97F, "écriture");
  if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, "préparation interrompue");
  const std::string text =
      gpu::span_fp8::receipt_text(receipt,
                                  "dérivation native du lecteur (app/play/span_derive.h) : SPAN dérivé des "
                                  "poids d'AnimeJaNai, à ne pas publier.");
  if (auto ok = commit_derivation(report.package, report.receipt, bytes, text, report.passed); !ok)
    return std::unexpected(std::move(ok.error()));
  if (!report.passed) report.package.clear();

  const std::string scores_text =
      std::format("PSNR moyen {} dB, pire {} dB, pire SSIM {}", db(report.verdict.psnr_mean_db),
                  db(report.verdict.psnr_worst_db), ratio(report.verdict.ssim_worst));
  if (report.passed) {
    report.text = std::format(
        "Mode Performance prêt : {} ({} images de calibration, {} de validation en {}, noyaux {} ms). Il "
        "sert à "
        "la prochaine ouverture d'une vidéo.",
        scores_text, report.calibration_images, report.validation_images, report.size,
        db(derived->kernels_ms));
  } else {
    report.text = std::format(
        "Mode Performance refusé : {} ; il faut {:.0f} dB, {:.0f} dB et {}. La lecture reste sur TensorRT "
        "FP16.",
        scores_text, gpu::span_fp8::kMinPsnrMeanDb, gpu::span_fp8::kMinPsnrWorstDb,
        ratio(gpu::span_fp8::kMinSsimWorst));
  }
  progress.set(1.0F, "terminé");
  return report;
}

SpanPrepare make_span_prepare(const std::shared_ptr<BackendLibrary>& library, const gpu::BackendInit& init,
                              bool supported, const std::filesystem::path& model,
                              const std::filesystem::path& models_dir, const std::filesystem::path& video) {
  SpanPrepare prepare;
  if (!supported) {
    prepare.note =
        "Le mode Performance par noyaux FP8 demande une RTX 40 (Ada) avec le moteur TensorRT : ce moteur ou "
        "cette "
        "carte ne le calcule pas.";
    return prepare;
  }
  if (model.empty()) {
    prepare.note = "Aucun modèle SPAN en service : rien à préparer.";
    return prepare;
  }
  // Le modèle « Performance » b5f48 a 48 canaux et 5 blocs : les noyaux prennent le b8f64 seulement.
  if (model.filename().string().contains("b5f48")) {
    prepare.note =
        "Ce modèle SPAN (b5f48, 48 canaux) n'est pas pris par les noyaux FP8 : choisissez un modèle "
        "Balanced.";
    return prepare;
  }
  if (const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, model, models_dir);
      found.variant) {
    prepare.ready = true;
    prepare.note = "Prêt : " + found.note;
  } else if (!found.note.empty()) {
    prepare.note = "Refusé : " + found.note;
  } else {
    prepare.note = "Pas encore préparé : la lecture reste sur TensorRT FP16 au cran Performance.";
  }
  if (video.empty()) {
    prepare.note += " Ouvrez une vidéo pour préparer : le lecteur s'y calibre.";
    return prepare;
  }
  prepare.offered = true;
  prepare.run = [library, init, model, models_dir, video](
                    SpanDeriveProgress& progress,
                    const std::stop_token& stop) -> std::expected<std::string, Refusal> {
    const SpanDerivePlan plan{.model = model, .models_dir = models_dir};
    const SpanStillsSource stills = [&](const std::stop_token& token,
                                        const std::function<void(float)>& report) {
      return media::extract_stills(video, media::StillsSpec{}, token, report);
    };
    const SpanDeriver deriver = [&](const gpu::SpanFp8Request& request, gpu::Progress& backend_progress) {
      auto backend = BackendLibrary::create(library, init);
      if (!backend)
        return std::expected<gpu::SpanFp8Derivation, Refusal>(std::unexpected(std::move(backend.error())));
      return (*backend)->derive_span_fp8(request, backend_progress);
    };
    auto report = derive_span_fp8(plan, stills, deriver, progress, stop);
    if (!report) return std::unexpected(std::move(report.error()));
    if (!report->passed) return refuse(RefusalCode::ReceiptInvalid, report->text);
    return report->text;
  };
  return prepare;
}

}  // namespace ajp::app::play
