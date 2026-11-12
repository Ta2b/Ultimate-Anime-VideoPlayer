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
#include <chrono>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "app/play/backend_library.h"
#include "app/play/i18n.h"
#include "app/play/span_int8.h"
#include "core/profile.h"
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
  std::string stage_ = std::string(i18n::tr("starting"));  // affiché tel quel par la barre (C07-09)
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
  // Étape 3.6 : la VRAM libre doit valoir ce multiple du besoin (SpanFp8Request::min_free_ratio) ; 0 : pas
  // de contrôle. La préparation lancée par un préréglage met kAutoMinFreeRatio.
  double min_free_ratio = 0.0;
  // Les poids du modèle rangés pour le paquet ; vide : lus dans l'ONNX (gpu/span_fp8/onnx_weights.h). Les
  // essais y mettent des poids de leur façon.
  std::function<std::expected<gpu::span_fp8::Package, Refusal>()> load_weights;
  // B02-01 : le temps d'une passe du SPAN FP16 de TensorRT à cette taille (time_span_fp16), le moteur que la
  // lecture garde sans le paquet. Un paquet qui passe la porte de qualité n'est gardé que s'il va au moins
  // gpu::span_fp8::kMinSpeedup fois plus vite. Vide : rien n'est mesuré ni exigé (essais, outils) ; le
  // lecteur le donne toujours.
  std::function<std::expected<double, Refusal>(int width, int height, gpu::Progress&)> time_fp16;
  // B02-08 : la préparation tourne pendant la lecture (préréglage) : le reçu le dit, ses temps étant pris
  // sous la charge de la lecture.
  bool during_playback = false;
};

struct SpanDeriveReport {
  bool passed = false;
  gpu::SpanFp8Derivation derivation;
  gpu::span_fp8::Verdict verdict;  // celui de la variante gardée
  bool acc16 = false;              // la variante à accumulateur f16 hors blocs est gardée (étape 3.5a)
  int calibration_images = 0;
  int validation_images = 0;
  std::string size;               // « 1920x1080 »
  double trt_fp16_ms = 0.0;       // B02-01 : une passe du SPAN FP16 de TensorRT ; 0 : non mesuré
  double speedup = 0.0;           // trt_fp16_ms sur le temps des noyaux FP8 ; 0 : non mesuré
  bool too_slow = false;          // la qualité passait, le gain non : le paquet est refusé
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

// Validation seule à une taille nouvelle (étape 3.7b) : le paquet déjà dérivé, ses échelles telles quelles,
// jugé par la porte durcie sur les images de validation de cette vidéo (à la taille de travail) et les cas
// durs ; la taille rejoint validated_sizes ou refused_sizes du reçu, le paquet ne change pas. Refus :
// paquet ou reçu absents ou illisibles, vidéo sans corpus, backend qui refuse, Cancelled, écriture.
std::expected<SpanDeriveReport, Refusal> validate_span_size(const SpanDerivePlan& plan,
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
  bool refused =
      false;         // un reçu de cette vidéo l'a déjà refusé (garde-fous) : l'automatique ne le refait pas
  std::string note;  // sa situation : prêt (avec ses mesures), refusé (pourquoi), pas encore préparé, ou
                     // pourquoi la préparation n'est pas offerte
  // La tâche : à lancer sur un autre fil que celui de la présentation. Rend le message de réussite, ou un
  // refus dont le détail dit pourquoi (les garde-fous échoués en font partie).
  std::function<std::expected<std::string, Refusal>(SpanDeriveProgress&, const std::stop_token&)> run;
};

// La préparation de ce SPAN sur cette vidéo. library et init : le backend en service (la tâche s'en ouvre une
// instance à elle, comme la construction d'un moteur) ; supported : BackendCaps::span_fp8_cutlass ; video :
// le fichier ouvert (vide : l'écran d'accueil, rien à préparer). unavailable : pourquoi le moteur ne le
// calcule pas (span_auto.h : span_fp8_unavailable_reason), pour la phrase qui l'explique quand supported est
// faux.
// reduce : la taille de travail de la chaîne et son filtre de réduction (span_still_reduce) ; les images de
// calibration et de validation y sont ramenées (étape 3.7a). Vide : la découpe à la taille de la source.
// during_playback (B02-08) : la tâche tourne pendant la lecture (panneau, préparation automatique) ; elle
// laisse la carte à la lecture entre deux images (kBackgroundPause) et le reçu dit ses temps pris sous
// charge.
SpanPrepare make_span_prepare(const std::shared_ptr<BackendLibrary>& library, const gpu::BackendInit& init,
                              bool supported, const std::filesystem::path& model,
                              const std::filesystem::path& models_dir, const std::filesystem::path& video,
                              const std::string& unavailable = {}, const media::StillReduce& reduce = {},
                              const SpanFp8Context& context = {}, bool automatic = false,
                              bool during_playback = false);

// La marge de VRAM d'une préparation lancée par un préréglage (étape 3.6).
inline constexpr double kAutoMinFreeRatio = 1.5;

// B02-08 : la préparation lancée par un préréglage tourne pendant la lecture, sur le même GPU. Après chaque
// image qu'elle passe (trois passes du SPAN), elle rend la carte à la lecture ce temps-là.
inline constexpr std::chrono::milliseconds kBackgroundPause{120};

// Le temps d'une passe du SPAN FP16 de TensorRT à width × height sur backend (B02-01) : le moteur que la
// chaîne construit pour cette taille (lu dans le cache, construit s'il manque), kSpanTimingWarmup passes de
// chauffe, puis la médiane de kSpanTimingBatches lots de kSpanTimingBatchRuns passes enchaînées. Refus : ceux
// de la construction, du chargement ou d'une passe ; Cancelled si progress.stop le demande.
inline constexpr int kSpanTimingWarmup = 5;
inline constexpr int kSpanTimingBatches = 5;
inline constexpr int kSpanTimingBatchRuns = 5;
// B02-08 : la pause de kBackgroundPause après une image de calibration ou de validation (stage, l'étape que
// le backend nomme), pour une préparation qui tourne pendant la lecture.
void pause_for_playback(std::string_view stage, const std::stop_token& stop);
std::expected<double, Refusal> time_span_fp16(gpu::Backend& backend, const std::filesystem::path& model,
                                              int width, int height, gpu::Progress& progress);

// La réduction des images de la préparation : la taille de travail et le filtre du menu « Réduction », comme
// le noyau d'ingestion les applique (live_settings.h, reduction_of).
media::StillReduce span_still_reduce(int work_width, int work_height, core::DownscaleFilter filter);

}  // namespace ajp::app::play
