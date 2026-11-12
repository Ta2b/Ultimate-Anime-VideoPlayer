// B40-04 : la chaîne livrée de bout en bout, sous Vulkan logiciel : un SPAN ×2 et un RIFE synthétiques
// (écrits ici, sans fichier binaire), l'agencement des étages et l'image juste ; et la matrice de couleur que
// le flux annonce, BT.601 contre BT.709. Caché ([.e2e]) comme les autres cas du lecteur.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "clip_writer.h"
#include "e2e_util.h"
#include "onnx_builder.h"
#include "prepare/utf8.h"

namespace {

using namespace ajp::e2e;  // NOLINT(google-build-using-namespace)
using ajp::gpu::ir::test::attr_int;
using ajp::gpu::ir::test::attr_string;
using ajp::gpu::ir::test::ModelBuilder;
using ajp::gpu::ir::test::tensor_f16;
namespace onnx = ajp::gpu::ir::onnx;

void write_model(const fs::path& path, const ModelBuilder& model) {
  const std::string bytes = model.model().str();
  std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Les graphes en FP16, comme les modèles livrés : la chaîne leur donne des tenseurs F16.
// Un SPAN ×2 d'identité : chaque pixel recopié sur ses quatre sous-pixels (Conv 1×1 vers 12 plans, puis
// DepthToSpace en CRD, comme les vrais SPAN).
void write_span(const fs::path& path) {
  ModelBuilder m;
  m.input("input", {1, 3, -1, -1}, onnx::kFloat16);
  std::vector<float> w(12 * 3, 0.0F);
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < 4; ++k) w[static_cast<std::size_t>((c * 4 + k) * 3 + c)] = 1.0F;
  }
  m.initializer(tensor_f16("w", {12, 3, 1, 1}, w))
      .initializer(tensor_f16("b", {12}, std::vector<float>(12, 0.0F)));
  m.node("Conv", {"input", "w", "b"}, {"c"});
  m.node("DepthToSpace", {"c"}, {"output"}, {attr_int("blocksize", 2), attr_string("mode", "CRD")});
  m.output("output", onnx::kFloat16);
  write_model(path, m);
}

// Un RIFE v2 (7 plans : deux images et l'instant) qui rend la moyenne des deux images : sur un plan fixe,
// l'image interpolée est la source.
void write_rife(const fs::path& path) {
  ModelBuilder m;
  m.input("input", {1, 7, -1, -1}, onnx::kFloat16);
  std::vector<float> w(3 * 7, 0.0F);
  for (int c = 0; c < 3; ++c) {
    w[static_cast<std::size_t>(c * 7 + c)] = 0.5F;
    w[static_cast<std::size_t>(c * 7 + 3 + c)] = 0.5F;
  }
  m.initializer(tensor_f16("w", {3, 7, 1, 1}, w))
      .initializer(tensor_f16("b", {3}, std::vector<float>(3, 0.0F)));
  m.node("Conv", {"input", "w", "b"}, {"output"});
  m.output("output", onnx::kFloat16);
  write_model(path, m);
}

// La moyenne absolue de l'écart entre deux captures, sur la bande que la barre de lecture ne couvre pas.
double band_difference(const Picture& a, const Picture& b) {
  if (a.width != b.width || a.height != b.height) return 255.0;
  double sum = 0;
  int count = 0;
  for (int y = a.height * 30 / 100; y < a.height * 45 / 100; ++y) {
    for (int x = a.width * 3 / 8; x < a.width * 5 / 8; ++x) {
      for (int c = 0; c < 3; ++c) {
        const std::size_t at =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(a.width) + static_cast<std::size_t>(x)) *
                3 +
            static_cast<std::size_t>(c);
        sum += std::abs(static_cast<int>(a.rgb[at]) - static_cast<int>(b.rgb[at]));
        ++count;
      }
    }
  }
  return count > 0 ? sum / count : 255.0;
}

// Le plus grand nombre d'images interpolées des lignes « RIFE ×2 : N paires, M interpolées » du journal.
int most_interpolated(const std::string& output) {
  int most = 0;
  for (std::size_t at = output.find(" interpolées"); at != std::string::npos;
       at = output.find(" interpolées", at + 1)) {
    std::size_t from = at;
    while (from > 0 && output[from - 1] >= '0' && output[from - 1] <= '9') --from;
    if (from < at) most = std::max(most, std::stoi(output.substr(from, at - from)));
  }
  return most;
}

ajp::media::test::ClipSpec plain_clip(std::string name) {
  return {.name = std::move(name),
          .width = 320,
          .height = 180,
          .frames = 50,
          .shot_frames = 50,
          .shot_lumas = {200}};
}

std::vector<std::string> base_args(const fs::path& clip, const fs::path& dir, const fs::path& snapshot) {
  return {"play",
          ajp::prepare::utf8_path(clip),
          "--software-gpu",
          "--mute",
          "--no-resume",
          "--window=640x360",
          "--seconds=2",
          "--data-dir=" + ajp::prepare::utf8_path(dir / "reglages"),
          "--snapshot=" + ajp::prepare::utf8_path(snapshot),
          "--snapshot-at=1.5"};
}

}  // namespace

TEST_CASE("Bout en bout : SPAN ×2 et RIFE ×2 synthétiques sous Vulkan, l'image de la chaîne nue",
          "[.e2e][app]") {
  const auto clip = ajp::media::test::write_clip(plain_clip("e2e-chaine.mkv"));
  REQUIRE(clip);
  const fs::path dir = fresh_dir("ajp-e2e-chaine");
  const fs::path source = alone(*clip, dir);
  const fs::path span = dir / "span_identite.onnx";
  const fs::path rife = dir / "rife_moyenne.onnx";
  write_span(span);
  write_rife(rife);
  // La chaîne nue (ni SPAN ni RIFE) : la référence.
  std::vector<std::string> bare = base_args(source, dir, dir / "nue.ppm");
  bare.insert(bare.end(), {"--no-upscale", "--rife=0"});
  const Run reference = run_player(bare);
  INFO(reference.output);
  REQUIRE(reference.code == 0);
  // La chaîne livrée : SPAN puis RIFE ×2, au facteur fixe (pas l'Auto, qui suit l'écran de Xvfb). Le premier
  // lancement construit les moteurs (la capture peut tomber sur l'écran de préparation), le second les lit
  // du cache et capture la lecture.
  const std::vector<std::string> engines = {"--model=" + ajp::prepare::utf8_path(span),
                                            "--rife-model=" + ajp::prepare::utf8_path(rife), "--rife=2"};
  for (const char* name : {"construction.ppm", "chaine.ppm"}) {
    std::vector<std::string> full = base_args(source, dir, dir / name);
    full.insert(full.end(), engines.begin(), engines.end());
    const Run chain = run_player(full);
    INFO(chain.output);
    REQUIRE(chain.code == 0);
    // Les deux étages ont tourné, sans repli : SPAN chargé, RIFE compté (un plan fixe : dessins tenus).
    CHECK(chain.output.find("moteur SPAN") != std::string::npos);
    CHECK(chain.output.find("RIFE ×2 :") != std::string::npos);
    CHECK(chain.output.find("refusé après la préparation") == std::string::npos);
  }
  // Un SPAN d'identité et un RIFE qui moyenne deux images égales : l'image est celle de la chaîne nue, à
  // l'arrondi du rendu 8 bits près (et au filtre de la mise à l'échelle finale, qui part d'une image deux
  // fois plus grande).
  const Picture expected = read_ppm(dir / "nue.ppm");
  const Picture got = read_ppm(dir / "chaine.ppm");
  REQUIRE(expected.width > 0);
  REQUIRE(got.width == expected.width);
  CHECK(band_difference(expected, got) < 3.0);
  // Une image qui change à chaque image (MPEG-4, luma propre à chacune) : RIFE interpole des paires.
  ajp::media::test::ClipSpec moving{.name = "e2e-chaine-mouvement.mkv",
                                    .codec = ajp::media::test::VideoCodec::Mpeg4,
                                    .width = 320,
                                    .height = 180,
                                    .frames = 50};
  const auto moving_clip = ajp::media::test::write_clip(moving);
  REQUIRE(moving_clip);
  std::vector<std::string> args = base_args(alone(*moving_clip, dir), dir, dir / "mouvement.ppm");
  args.insert(args.end(), engines.begin(), engines.end());
  const Run run = run_player(args);
  INFO(run.output);
  REQUIRE(run.code == 0);
  CHECK(most_interpolated(run.output) > 0);
}

TEST_CASE("Bout en bout : un flux BT.601 se rend en BT.601, pas en BT.709", "[.e2e][app]") {
  const fs::path dir = fresh_dir("ajp-e2e-matrice");
  ajp::media::test::ClipSpec bt709 = plain_clip("e2e-bt709.mkv");
  bt709.color = {.primaries = AVCOL_PRI_BT709,
                 .transfer = AVCOL_TRC_BT709,
                 .matrix = AVCOL_SPC_BT709,
                 .range = AVCOL_RANGE_MPEG};
  ajp::media::test::ClipSpec bt601 = plain_clip("e2e-bt601.mkv");
  bt601.color = {.primaries = AVCOL_PRI_SMPTE170M,
                 .transfer = AVCOL_TRC_SMPTE170M,
                 .matrix = AVCOL_SPC_SMPTE170M,
                 .range = AVCOL_RANGE_MPEG};
  const auto clip709 = ajp::media::test::write_clip(bt709);
  const auto clip601 = ajp::media::test::write_clip(bt601);
  REQUIRE(clip709);
  REQUIRE(clip601);
  std::vector<std::string> args709 = base_args(alone(*clip709, dir), dir, dir / "709.ppm");
  args709.insert(args709.end(), {"--no-upscale", "--rife=0"});
  const Run run709 = run_player(args709);
  INFO(run709.output);
  REQUIRE(run709.code == 0);
  std::vector<std::string> args601 = base_args(alone(*clip601, dir), dir, dir / "601.ppm");
  args601.insert(args601.end(), {"--no-upscale", "--rife=0"});
  const Run run601 = run_player(args601);
  INFO(run601.output);
  REQUIRE(run601.code == 0);
  const Picture a = read_ppm(dir / "709.ppm");
  const Picture b = read_ppm(dir / "601.ppm");
  REQUIRE(a.width > 0);
  REQUIRE(b.width == a.width);
  // Les mêmes YUV (le vert pâle du plan 0) : en BT.709, RVB 155, 238, 151 environ ; en BT.601, 161, 253,
  // 154. Le vert s'écarte d'une quinzaine : une matrice ignorée (les deux rendus en 709) le laisse à 0.
  const std::size_t probe = (static_cast<std::size_t>(a.height * 3 / 8) * static_cast<std::size_t>(a.width) +
                             static_cast<std::size_t>(a.width / 2)) *
                            3;
  const int green = static_cast<int>(b.rgb[probe + 1]) - static_cast<int>(a.rgb[probe + 1]);
  const int red = static_cast<int>(b.rgb[probe]) - static_cast<int>(a.rgb[probe]);
  CHECK(green >= 8);
  CHECK(green <= 22);
  CHECK(red >= 2);
  CHECK(red <= 12);
}
