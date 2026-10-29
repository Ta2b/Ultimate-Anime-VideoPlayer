#include "app/play/span_auto.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "app/play/engine_install.h"
#include "app/play/span_int8.h"
#include "french_ui.h"

using namespace ajp;
using namespace ajp::app::play;
namespace fs = std::filesystem;

TEST_CASE("le mode Performance dit pourquoi il manque, et ce qu'il faut faire", "[span_auto]") {
  ajp::app::play::FrenchUi french;
  // Possible : rien à dire.
  CHECK(span_fp8_unavailable_reason({.backend_name = "tensorrt", .gpu_name = "RTX 4050", .capability = true})
            .empty());
  // Moteur TensorRT absent.
  const std::string absent = span_fp8_unavailable_reason({.backend_name = "vulkan", .gpu_name = "RTX 4050"});
  CHECK(absent.contains("moteur TensorRT non installé"));
  CHECK(absent.contains("Paramètres > Export et moteurs > Moteurs"));
  // Installé mais pas celui en service.
  const std::string other = span_fp8_unavailable_reason(
      {.backend_name = "vulkan", .gpu_name = "RTX 4050", .tensorrt_present = true});
  CHECK(other.contains("installé"));
  CHECK(other.contains("choisissez TensorRT"));
  // TensorRT en service, mais la carte n'est pas une Ada.
  const std::string old = span_fp8_unavailable_reason(
      {.backend_name = "tensorrt", .gpu_name = "NVIDIA GeForce RTX 3060", .tensorrt_present = true});
  CHECK(old.contains("RTX 3060"));
  CHECK(old.contains("RTX 40"));
}

TEST_CASE("rien n'exclut à tort une RTX 4050 de portable : Ada, calcul 8.9", "[span_auto]") {
  ajp::app::play::FrenchUi french;
  // Le nom de la carte suffit à la reconnaître quand nvidia-smi manque (graphique hybride, Optimus).
  CHECK(cuda_arch_from_name("NVIDIA GeForce RTX 4050 Laptop GPU") == "sm89");
  CHECK(cuda_arch_from_name("NVIDIA GeForce RTX 4060 Laptop GPU") == "sm89");
  CHECK(cuda_arch_from_name("NVIDIA GeForce RTX 4070 Ti") == "sm89");
  CHECK(cuda_arch_from_name("NVIDIA RTX 2000 Ada Generation Laptop GPU") != "sm120");
  CHECK(effective_cuda_arch("", "NVIDIA GeForce RTX 4050 Laptop GPU") == "sm89");
  CHECK(effective_cuda_arch("sm89", "NVIDIA GeForce RTX 4050 Laptop GPU") == "sm89");
}

TEST_CASE("le cran Performance de SPAN est offert dès que le moteur sait les noyaux, sans paquet",
          "[span_auto]") {
  ajp::app::play::FrenchUi french;
  const fs::path dir = fs::temp_directory_path() / "ajp-span-auto-tiers";
  fs::create_directories(dir);
  const fs::path model = dir / "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx";
  std::ofstream(model) << "modele";
  // Pas de paquet dans le dossier : l'offre ne dépend plus de lui.
  CHECK(span_tiers(model, dir / "modeles", false, true).performance);
  CHECK_FALSE(span_tiers(model, dir / "modeles", false, false).performance);
  // Le modèle de 48 canaux n'est pas pris par les noyaux.
  const fs::path small = dir / "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx";
  std::ofstream(small) << "modele";
  CHECK_FALSE(span_tiers(small, dir / "modeles", false, true).performance);
  fs::remove_all(dir);
}

namespace {

SpanPrepare offered_with(int& runs, bool succeed = true) {
  SpanPrepare prepare;
  prepare.offered = true;
  prepare.run = [&runs, succeed](SpanDeriveProgress& progress,
                                 const std::stop_token&) -> std::expected<std::string, Refusal> {
    ++runs;
    progress.set(0.5F, "calibration");
    if (!succeed)
      return std::unexpected(Refusal{RefusalCode::ReceiptInvalid, "Mode Performance refusé : PSNR 20 dB"});
    return std::string("Mode Performance prêt");
  };
  return prepare;
}

}  // namespace

TEST_CASE("la préparation automatique part au cran Performance, une fois par modèle et vidéo",
          "[span_auto]") {
  ajp::app::play::FrenchUi french;
  int runs = 0;
  SpanPrepare prepare = offered_with(runs);
  CHECK(should_start_span_auto(PrecisionTier::Performance, prepare, false, "m|v", ""));
  CHECK_FALSE(should_start_span_auto(PrecisionTier::Balanced, prepare, false, "m|v", ""));  // autre cran
  CHECK_FALSE(should_start_span_auto(PrecisionTier::Quality, prepare, false, "m|v", ""));
  CHECK_FALSE(should_start_span_auto(PrecisionTier::Performance, prepare, true, "m|v", ""));  // déjà en cours
  CHECK_FALSE(
      should_start_span_auto(PrecisionTier::Performance, prepare, false, "m|v", "m|v"));  // déjà tentée
  CHECK(should_start_span_auto(PrecisionTier::Performance, prepare, false, "m|autre", "m|v"));
  SpanPrepare ready = prepare;
  ready.ready = true;
  CHECK_FALSE(should_start_span_auto(PrecisionTier::Performance, ready, false, "m|v", ""));
  SpanPrepare refused = prepare;
  refused.refused =
      true;  // un reçu de cette vidéo l'a déjà refusée : pas de boucle de 40 s à chaque ouverture
  CHECK_FALSE(should_start_span_auto(PrecisionTier::Performance, refused, false, "m|v", ""));
  SpanPrepare not_offered = prepare;
  not_offered.offered = false;
  CHECK_FALSE(should_start_span_auto(PrecisionTier::Performance, not_offered, false, "m|v", ""));
}

TEST_CASE("la boucle du lecteur annonce la progression puis rouvre la chaîne à la fin", "[span_auto]") {
  ajp::app::play::FrenchUi french;
  int runs = 0;
  const SpanPrepare prepare = offered_with(runs);
  SpanJob job;
  const SpanAutoTick first = tick_span_auto(PrecisionTier::Performance, prepare, job, "m|v");
  REQUIRE(first.notify);
  CHECK(first.notify->starts_with("Préparation du mode Performance…"));
  CHECK(first.notify->ends_with(" %"));
  CHECK_FALSE(first.reopen);
  job.note->pending->wait();
  const SpanAutoTick done = tick_span_auto(PrecisionTier::Performance, prepare, job, "m|v");
  CHECK(done.reopen);
  REQUIRE(done.notify);
  CHECK(done.notify->contains("prêt"));
  // Une fois lu, rien de plus : pas de nouvelle tâche pour la même clé.
  const SpanAutoTick after = tick_span_auto(PrecisionTier::Performance, prepare, job, "m|v");
  CHECK_FALSE(after.reopen);
  CHECK_FALSE(after.notify.has_value());
  CHECK(runs == 1);
}

TEST_CASE("un refus des garde-fous est dit avec ses dB, la lecture reste en FP16, sans réessai",
          "[span_auto]") {
  ajp::app::play::FrenchUi french;
  int runs = 0;
  const SpanPrepare prepare = offered_with(runs, false);
  SpanJob job;
  tick_span_auto(PrecisionTier::Performance, prepare, job, "m|v");
  job.note->pending->wait();
  const SpanAutoTick done = tick_span_auto(PrecisionTier::Performance, prepare, job, "m|v");
  CHECK_FALSE(done.reopen);
  REQUIRE(done.notify);
  CHECK(done.notify->contains("lecture en FP16"));
  CHECK(done.notify->contains("PSNR 20 dB"));
  tick_span_auto(PrecisionTier::Performance, prepare, job, "m|v");
  CHECK(runs == 1);
}

TEST_CASE("fermer le lecteur arrête et attend la tâche", "[span_auto]") {
  ajp::app::play::FrenchUi french;
  PrepareNote note;
  SpanPrepare prepare;
  prepare.offered = true;
  prepare.run = [](SpanDeriveProgress&, const std::stop_token& stop) -> std::expected<std::string, Refusal> {
    while (!stop.stop_requested()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation interrompue"});
  };
  REQUIRE(start_span_prepare(prepare, note));
  stop_and_wait(note);
  CHECK(note.pending == nullptr);
}
