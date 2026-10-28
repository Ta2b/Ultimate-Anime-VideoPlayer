// « Préparer le mode Performance (SPAN FP8) » : la dérivation, chez l'utilisateur et sans Python, du paquet
// « .span8 » que le cran Performance de SPAN sert par les noyaux CUTLASS (docs/span-fp8-cutlass.md). C'est ce
// que tools/span/derive_span_fp8.py fait, mais sur la vidéo ouverte et par les noyaux du backend :
//
//   1. les poids et le graphe du SPAN choisi sont lus dans son ONNX (le b8f64 de 64 canaux seulement) ;
//   2. douze images SDR sont tirées de la vidéo ouverte par le décodeur du lecteur (media/stills.h), quatre
//   mises
//      de côté pour la validation, aucune gardée ensuite ;
//   3. le backend (Backend::derive_span_fp8) relève les maxima des 25 tenseurs sur les huit autres par une
//      référence FP16, en tire les échelles, puis juge le réseau FP8 contre la même référence sur les quatre
//      mises de côté ;
//   4. le paquet « .span8 » et son reçu « .toml » (mêmes formats que l'outil Python, aux mêmes garde-fous)
//   sont
//      écrits dans le dossier des modèles, par renommage : un fichier n'existe jamais à moitié écrit.
//
// Le chemin FP16 par défaut ne change pas tant que le reçu n'existe pas ou est refusé (span_int8.h) ; en cas
// d'échec technique, rien n'est écrit. Un échec des garde-fous écrit le reçu « passed = false » (le paquet
// d'avant ne correspond plus à lui) et laisse la lecture sur TensorRT FP16.
#pragma once

#include <atomic>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include "app/play/backend_library.h"
#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/span_fp8/derive.h"
#include "gpu/span_fp8/package.h"
#include "media/stills.h"

namespace ajp::app::play {

// L'avancement d'une préparation, lu par l'interface pendant que la tâche l'écrit.
class SpanDeriveProgress {
 public:
  void set(float fraction, std::string stage) {
    const std::scoped_lock lock(mutex_);
    fraction_ = fraction;
    stage_ = std::move(stage);
  }
  std::pair<float, std::string> get() const {
    const std::scoped_lock lock(mutex_);
    return {fraction_, stage_};
  }

 private:
  mutable std::mutex mutex_;
  float fraction_ = 0.0F;
  std::string stage_ = "démarrage";
};

// Les images de la vidéo (media::extract_stills), ou ce qui en tient lieu dans les tests.
using SpanStillsSource = std::function<std::expected<std::vector<media::Still>, Refusal>(
    const std::stop_token&, const std::function<void(float)>&)>;
// Le backend qui calcule (Backend::derive_span_fp8), ou ce qui en tient lieu dans les tests.
using SpanDeriver =
    std::function<std::expected<gpu::SpanFp8Derivation, Refusal>(const gpu::SpanFp8Request&, gpu::Progress&)>;

struct SpanDerivePlan {
  std::filesystem::path model;       // le SPAN ONNX choisi
  std::filesystem::path models_dir;  // « modeles » : où le paquet et son reçu s'écrivent
  std::string quantizer = "ajp (dérivation native, CUTLASS E4M3, maximum)";
  // Les poids du modèle rangés pour le paquet ; vide : lus dans l'ONNX (gpu/span_fp8/onnx_weights.h). Les
  // essais y mettent des poids de leur façon.
  std::function<std::expected<gpu::span_fp8::Package, Refusal>()> load_weights;
};

struct SpanDeriveReport {
  bool passed = false;
  gpu::SpanFp8Derivation derivation;
  gpu::span_fp8::Verdict verdict;
  int calibration_images = 0;
  int validation_images = 0;
  std::string size;               // « 1920x1080 »
  std::filesystem::path package;  // le paquet écrit ; vide si les garde-fous l'ont refusé
  std::filesystem::path receipt;
  std::string text;  // pour l'utilisateur : ce qui a été fait ou pourquoi pas
};

// Les images 0.. n-1 dans l'ordre du temps : une sur trois (la troisième, la sixième…) est mise de côté pour
// la validation, le reste calibre. Douze images donnent huit et quatre.
struct StillsSplit {
  std::vector<std::size_t> calibration;
  std::vector<std::size_t> validation;
};
StillsSplit split_stills(std::size_t count);

// La dérivation de bout en bout (étapes 1 à 4 ci-dessus). Refus : le modèle n'est pas le b8f64
// (FormatUnsupported), vidéo HDR ou sans corpus, backend qui refuse, Cancelled, écriture impossible
// (IoFailed) ; dans ces cas rien n'est écrit. Les garde-fous qui échouent ne sont pas un refus :
// report.passed est faux.
std::expected<SpanDeriveReport, Refusal> derive_span_fp8(const SpanDerivePlan& plan,
                                                         const SpanStillsSource& stills,
                                                         const SpanDeriver& deriver,
                                                         SpanDeriveProgress& progress,
                                                         const std::stop_token& stop);

// Écrit le paquet et le reçu à côté l'un de l'autre, par fichiers temporaires renommés ensuite. passed : les
// deux ; sinon le reçu seul, et un paquet d'avant est retiré. Refus IoFailed, sans fichier laissé à moitié
// écrit.
std::expected<void, Refusal> commit_derivation(const std::filesystem::path& package,
                                               const std::filesystem::path& receipt,
                                               std::span<const std::byte> package_bytes,
                                               std::string_view receipt_text, bool passed);

// Ce que l'interface sait d'une préparation possible (SettingsChoices::span_prepare).
struct SpanPrepare {
  bool offered = false;  // le moteur calcule le SPAN FP8 CUTLASS (BackendCaps::span_fp8_cutlass) et une vidéo
                         // est ouverte
  bool ready = false;    // un paquet dérivé passe déjà ses garde-fous : rien à refaire
  std::string note;      // sa situation : prêt (avec ses mesures), refusé (pourquoi), pas encore préparé, ou
                         // pourquoi la préparation n'est pas offerte
  // La tâche : à lancer sur un autre fil que celui de la présentation. Rend le message de réussite, ou un
  // refus dont le détail dit pourquoi (les garde-fous échoués en font partie).
  std::function<std::expected<std::string, Refusal>(SpanDeriveProgress&, const std::stop_token&)> run;
};

// La préparation de ce SPAN sur cette vidéo. library et init : le backend en service (la tâche s'en ouvre une
// instance à elle, comme la construction d'un moteur) ; supported : BackendCaps::span_fp8_cutlass ; video :
// le fichier ouvert (vide : l'écran d'accueil, rien à préparer).
SpanPrepare make_span_prepare(const std::shared_ptr<BackendLibrary>& library, const gpu::BackendInit& init,
                              bool supported, const std::filesystem::path& model,
                              const std::filesystem::path& models_dir, const std::filesystem::path& video);

}  // namespace ajp::app::play
