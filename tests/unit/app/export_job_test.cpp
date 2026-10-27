// Les calculs purs de l'export : chaîne, taille encodée, fréquence, nom du fichier, profondeur.
#include "app/play/export_job.h"

#include <filesystem>
#include <format>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "app/play/export_depth.h"
#include "app/play/scripted_input.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

// Un dossier vide, effacé à la fin du test.
struct TempDir {
  explicit TempDir(const char* test)
      : path(std::filesystem::temp_directory_path() / std::format("ajp-export-job-{}", test)) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::filesystem::path path;
};

}  // namespace

TEST_CASE("Export : la chaîne sans RTX HDR, HDR adaptatif, grade ni Smooth Motion", "[app][export]") {
  PlaySettings s;
  s.true_hdr = true;
  s.hdr_native = true;
  s.grade = true;
  s.smooth_motion = true;
  s.rife_factor = 4;
  const PlaySettings out = export_chain(s);
  CHECK_FALSE(out.true_hdr);
  CHECK_FALSE(out.hdr_native);
  CHECK_FALSE(out.grade);
  CHECK_FALSE(out.smooth_motion);
  CHECK(out.upscale == s.upscale);
  CHECK(out.rife_factor == 4);
  CHECK(out.work_height == s.work_height);
}

TEST_CASE("Export : la sortie de la chaîne est la taille de travail, doublée par SPAN", "[app][export]") {
  PlaySettings s;
  s.work_height = 720;
  CHECK(chain_output(s, 1920, 1080) == ExportSize{.width = 2560, .height = 1440});
  CHECK(chain_output(s, 1280, 720) == ExportSize{.width = 2560, .height = 1440});
  CHECK(chain_output(s, 720, 480) == ExportSize{.width = 1440, .height = 960});  // sous 720 : telle quelle
  s.upscale = false;
  CHECK(chain_output(s, 1920, 1080) == ExportSize{.width = 1280, .height = 720});
}

TEST_CASE("Export : taille encodée de la chaîne, de la source ou d'une hauteur fixe, paire",
          "[app][export]") {
  const ExportSize chain{.width = 2560, .height = 1440};
  const ExportSize source{.width = 1920, .height = 1080};
  CHECK(export_size(chain, source, kExportChainHeight) == chain);
  CHECK(export_size(chain, source, kExportSourceHeight) == source);
  CHECK(export_size(chain, source, 2160) == ExportSize{.width = 3840, .height = 2160});
  CHECK(export_size(chain, source, 720) == ExportSize{.width = 1280, .height = 720});
  // 853x480 (16:9 arrondi) : chaque côté pair pour le 4:2:0.
  CHECK(export_size({.width = 853, .height = 480}, source, kExportChainHeight) ==
        ExportSize{.width = 854, .height = 480});
  CHECK(export_size({.width = 1706, .height = 960}, source, 1080) ==
        ExportSize{.width = 1920, .height = 1080});
  // Jamais sous 2 × 2.
  CHECK(export_size({.width = 1, .height = 1}, source, kExportChainHeight) ==
        ExportSize{.width = 2, .height = 2});
}

TEST_CASE("Export : fréquence de la source multipliée par l'interpolation, réduite", "[app][export]") {
  const media::Rational ntsc{.num = 24'000, .den = 1'001};
  CHECK(export_rate(ntsc, Factor{.num = 3, .den = 1}) == media::Rational{.num = 72'000, .den = 1'001});
  CHECK(export_rate(ntsc, Factor{.num = 2, .den = 1}) == media::Rational{.num = 48'000, .den = 1'001});
  CHECK(export_rate(ntsc, Factor{.num = 5, .den = 2}) == media::Rational{.num = 60'000, .den = 1'001});
  CHECK(export_rate({.num = 25, .den = 1}, Factor{.num = 3, .den = 2}) ==
        media::Rational{.num = 75, .den = 2});
  // Sans interpolation (0 ou ×1) : la fréquence de la source, réduite.
  CHECK(export_rate(ntsc, Factor{}) == ntsc);
  CHECK(export_rate(ntsc, Factor{.num = 1, .den = 1}) == ntsc);
  CHECK(export_rate({.num = 48, .den = 2}, Factor{}) == media::Rational{.num = 24, .den = 1});
  // Inconnue : {0, 1}.
  CHECK(export_rate({.num = 0, .den = 1}, Factor{.num = 2, .den = 1}).num == 0);
}

TEST_CASE("Export : nom proposé à côté de la source, numéroté s'il est pris", "[app][export]") {
  const TempDir dir("name");
  const std::filesystem::path source = dir.path / prepare::path_from_utf8("Épisode 01.mkv");
  const ExportSize size{.width = 3840, .height = 2160};
  const std::filesystem::path first = default_export_path(source, size, {.num = 48'000, .den = 1'001});
  CHECK(first.parent_path() == dir.path);
  CHECK(prepare::utf8_path(first.filename()) == "Épisode 01 [ajp 3840x2160 47.952fps].mkv");
  CHECK(prepare::utf8_path(default_export_path(source, size, {.num = 48, .den = 1}).filename()) ==
        "Épisode 01 [ajp 3840x2160 48fps].mkv");
  CHECK(prepare::utf8_path(default_export_path(source, size, {}).filename()) ==
        "Épisode 01 [ajp 3840x2160].mkv");

  std::ofstream(first).put('x');
  CHECK(prepare::utf8_path(default_export_path(source, size, {.num = 48'000, .den = 1'001}).filename()) ==
        "Épisode 01 [ajp 3840x2160 47.952fps] (2).mkv");
}

TEST_CASE("Export : « export » de --keys lance l'export de la chaîne en cours sans clic", "[app][export]") {
  const TempDir dir("keys");
  const std::filesystem::path source = dir.path / "clip.mkv";
  PlaySettings running;
  running.rife_factor = 2;
  running.rife_factor_den = 1;
  running.true_hdr = true;
  media::TrackInfo video;
  video.width = 1920;
  video.height = 1080;
  video.frame_rate = {.num = 24'000, .den = 1'001};

  CHECK_FALSE(scripted_export("Space", running, video, source));
  CHECK_FALSE(scripted_export("exporter", running, video, source));
  const auto job = scripted_export("export", running, video, source);
  REQUIRE(job);
  CHECK(job->chain == running);  // export_run.h en retire RTX HDR lui-même (export_chain)
  CHECK(job->height == kExportChainHeight);
  CHECK(job->codec == exporter::Codec::Hevc);
  CHECK(job->quality == exporter::Quality::VeryHigh);
  // Le nom proposé : la sortie de la chaîne (720p ×2 par défaut) à la fréquence interpolée.
  CHECK(job->output == dir.path / "clip [ajp 2560x1440 47.952fps].mkv");

  const auto named = scripted_export("export=/tmp/sortie.mkv", running, video, source);
  REQUIRE(named);
  CHECK(named->output == std::filesystem::path("/tmp/sortie.mkv"));
}

TEST_CASE("Export : libellé du codec et de la qualité", "[app][export]") {
  CHECK(export_label(exporter::Codec::Hevc, exporter::Quality::VeryHigh) ==
        "HEVC 10 bits · qualité très haute");
  CHECK(export_label(exporter::Codec::Av1, exporter::Quality::Standard) == "AV1 10 bits · qualité standard");
  CHECK(export_label(exporter::Codec::Lossless, exporter::Quality::High) == "FFV1 4:2:0 10 bits, sans perte du codec");
}

TEST_CASE("Export : profil débit avec plusieurs voies de calcul, profondeur d'avant sinon", "[app][export]") {
  const ExportDepth one = export_depth(1);
  CHECK_FALSE(one.throughput);
  CHECK(one.ring_slots == 4);
  CHECK(one.buffers == 7);
  CHECK(one.queue == 4);
  const ExportDepth three = export_depth(3);
  CHECK(three.throughput);
  CHECK(three.ring_slots == 8);  // deux paires ×3 et de quoi convertir
  CHECK(three.queue == 8);
  // La file se remplit : une image confiée garde son tampon, comme celle en cours d'encodage et la
  // précédente que la cadence fixe répète.
  CHECK(three.buffers > three.queue + kWriterHeldBeyondQueue);
}

}  // namespace ajp::app::play
