// Le SPAN entier en FP8 (noyaux CUTLASS, docs/span-fp8-cutlass.md) dans le backend TensorRT : refus nommés
// d'un paquet inutilisable (sans passer par les noyaux) et, en test caché [.span_fp8], l'image de sortie
// contre celle du SPAN d'origine en FP16 par TensorRT.
//
// Le test caché demande un paquet dérivé par tools/span/derive_span_fp8.py et son modèle d'origine, passés à
// CMake (AJP_SPAN_FP8_PACKAGE, AJP_SPAN_FP8_MODEL, jamais copiés dans le dépôt) ; sans eux, il se SKIP. Une
// image d'anime en PPM (AJP_SPAN_FP8_IMAGE, hauteur et largeur que les noyaux prennent) remplace le dessin
// synthétique et donne le PSNR et le SSIM de la sortie contre TensorRT FP16 :
//   cmake -DAJP_SPAN_FP8_PACKAGE=.../xxx-fp8c-....span8 -DAJP_SPAN_FP8_MODEL=.../xxx.onnx ...
//   ajp-gpu ajp_backend_tensorrt_tests "[.span_fp8]"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "gpu/span_fp8/package.h"
#include "kernels/ref/half.h"
#include "prepare/sha256.h"

namespace ajp::test {
namespace {

namespace fs = std::filesystem;

struct Scratch {
  Scratch() : root(fs::temp_directory_path() / "ajp-span-fp8-backend-test") {
    fs::remove_all(root);
    fs::create_directories(root);
  }
  ~Scratch() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  fs::path root;
};

void write_bytes(const fs::path& path, const std::vector<std::byte>& bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

gpu::ModelSpec spec_of(const fs::path& path) {
  return gpu::ModelSpec{.id = path.stem().string(), .onnx_path = path, .sha256 = "0"};
}

gpu::ShapeSet fixed(int h, int w) {
  const gpu::Shape shape{.dims = {1, 3, h, w}};
  return {.min = shape, .opt = shape, .max = shape};
}

// Une image qui ressemble à du dessin : aplats, aplat dégradé, traits nets et un peu de grain.
std::vector<float> drawing(int h, int w) {
  std::vector<float> plane(static_cast<std::size_t>(3 * h * w));
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float v = 0.25F + 0.1F * static_cast<float>(c);
        if (x > w / 3 && x < 2 * w / 3 && y > h / 4) v = 0.7F - 0.1F * static_cast<float>(c);
        if (y > h / 2) v += 0.25F * static_cast<float>(x) / static_cast<float>(w);  // dégradé
        if ((x + 2 * y) % 97 < 2) v = 0.05F;                                        // traits
        v += 0.004F * static_cast<float>((x * 7 + y * 13 + c * 5) % 5);             // grain
        plane[static_cast<std::size_t>((c * h + y) * w + x)] = std::clamp(v, 0.F, 1.F);
      }
    }
  }
  return plane;
}

// Image PPM binaire (P6, 255) en plans RGB de 0 à 1 ; vide si illisible.
std::vector<float> read_ppm(const fs::path& path, int& h, int& w) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  file >> magic >> w >> h >> maxval;
  if (!file || magic != "P6" || maxval != 255 || w <= 0 || h <= 0) return {};
  file.get();
  std::vector<unsigned char> raw(static_cast<std::size_t>(3 * h * w));
  file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (!file) return {};
  std::vector<float> planes(raw.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c)
        planes[static_cast<std::size_t>((c * h + y) * w + x)] =
            static_cast<float>(raw[static_cast<std::size_t>((y * w + x) * 3 + c)]) / 255.F;
  return planes;
}

// SSIM de la luminance BT.709 par blocs de 8x8 sans recouvrement, moyenné (assez pour un garde-fou de test ;
// l'outil Python mesure le SSIM à fenêtre gaussienne du reçu).
double block_ssim(const std::vector<float>& a, const std::vector<float>& b, int h, int w) {
  const auto luma = [&](const std::vector<float>& v, int y, int x) {
    const auto at = [&](int c) {
      return static_cast<double>(std::clamp(v[static_cast<std::size_t>((c * h + y) * w + x)], 0.F, 1.F));
    };
    return 0.2126 * at(0) + 0.7152 * at(1) + 0.0722 * at(2);
  };
  constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
  double total = 0.0;
  int blocks = 0;
  for (int by = 0; by + 8 <= h; by += 8) {
    for (int bx = 0; bx + 8 <= w; bx += 8) {
      double ma = 0, mb = 0, saa = 0, sbb = 0, sab = 0;
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
          const double va = luma(a, by + y, bx + x), vb = luma(b, by + y, bx + x);
          ma += va, mb += vb, saa += va * va, sbb += vb * vb, sab += va * vb;
        }
      ma /= 64, mb /= 64;
      const double vara = saa / 64 - ma * ma, varb = sbb / 64 - mb * mb, cov = sab / 64 - ma * mb;
      total += ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (vara + varb + c2));
      ++blocks;
    }
  }
  return blocks == 0 ? 1.0 : total / blocks;
}

double psnr(const std::vector<float>& a, const std::vector<float>& b) {
  double mse = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d =
        static_cast<double>(std::clamp(a[i], 0.F, 1.F)) - static_cast<double>(std::clamp(b[i], 0.F, 1.F));
    mse += d * d;
  }
  mse /= static_cast<double>(a.size());
  return mse == 0.0 ? 99.0 : 10.0 * std::log10(1.0 / mse);
}

}  // namespace

TEST_CASE("Backend TensorRT : un paquet SPAN FP8 inutilisable est refusé par son nom, sans noyau lancé",
          "[backend_tensorrt][cuda][span_fp8]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  const Scratch scratch;
  gpu::Progress progress;

  const fs::path good = scratch.root / "modele-fp8c-0123456789abcdef.span8";
  write_bytes(good, gpu::span_fp8::serialize(gpu::span_fp8::synthetic()));
  const bool supported = loaded.backend->capabilities().span_fp8_cutlass;

  // Un fichier absent, tronqué ou abîmé : jamais une construction réussie.
  CHECK_FALSE(loaded.backend->build(spec_of(scratch.root / "absent-fp8c-0123456789abcdef.span8"),
                                    fixed(1088, 1920), progress));
  std::vector<std::byte> bytes = gpu::span_fp8::serialize(gpu::span_fp8::synthetic());
  bytes.resize(bytes.size() / 2);
  const fs::path truncated = scratch.root / "tronque-fp8c-0123456789abcdef.span8";
  write_bytes(truncated, bytes);
  CHECK_FALSE(loaded.backend->build(spec_of(truncated), fixed(1088, 1920), progress));

  // Une taille que les noyaux ne prennent pas (540x960 : 129 600 pixels de caractéristiques) ; une forme
  // libre.
  const auto odd = loaded.backend->build(spec_of(good), fixed(540, 960), progress);
  REQUIRE_FALSE(odd);
  CHECK(odd.error().code == RefusalCode::FormatUnsupported);
  const gpu::Shape low{.dims = {1, 3, 544, 960}}, high{.dims = {1, 3, 1088, 1920}};
  CHECK_FALSE(loaded.backend->build(spec_of(good), {.min = low, .opt = low, .max = high}, progress));

  // Une taille prise : construit si le GPU est un sm 8.9 avec les noyaux, refus nommé sinon.
  const auto built = loaded.backend->build(spec_of(good), fixed(1088, 1920), progress);
  CHECK(built.has_value() == supported);
  if (!built) {
    CHECK(built.error().code == RefusalCode::FormatUnsupported);
    CHECK_FALSE(built.error().detail.empty());
  }
}

TEST_CASE("Backend TensorRT : le SPAN FP8 CUTLASS contre le SPAN FP16 de TensorRT",
          "[.span_fp8][backend_tensorrt][cuda][span]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const fs::path package(AJP_SPAN_FP8_PACKAGE), model(AJP_SPAN_FP8_MODEL);
  if (!fs::is_regular_file(package) || !fs::is_regular_file(model))
    SKIP("AJP_SPAN_FP8_PACKAGE et AJP_SPAN_FP8_MODEL (CMake) : paquet dérivé et modèle d'origine");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  if (!loaded.backend->capabilities().span_fp8_cutlass) SKIP("appareil sans les noyaux FP8 CUTLASS (sm 8.9)");
  int h = 384, w = 640;
  std::vector<float> picture;
  if (const fs::path image(AJP_SPAN_FP8_IMAGE); !image.empty() && fs::is_regular_file(image)) {
    picture = read_ppm(image, h, w);
    if (picture.empty()) SKIP("AJP_SPAN_FP8_IMAGE : PPM P6 de 255 illisible");
  } else {
    picture = drawing(h, w);
  }
  if (!gpu::span_fp8::size_supported(h, w)) SKIP("taille de l'image que les noyaux ne prennent pas");
  gpu::Progress progress;
  const gpu::Shape in_shape{.dims = {1, 3, h, w}}, out_shape{.dims = {1, 3, 2 * h, 2 * w}};

  const auto sha = prepare::sha256_file(model);
  REQUIRE(sha);
  auto reference = loaded.backend->build(
      gpu::ModelSpec{.id = model.stem().string(), .onnx_path = model, .sha256 = *sha}, fixed(h, w), progress);
  REQUIRE(reference);
  auto fp16 = loaded.backend->load(*reference);
  REQUIRE(fp16);
  auto artifact = loaded.backend->build(spec_of(package), fixed(h, w), progress);
  REQUIRE(artifact);
  auto fp8 = loaded.backend->load(*artifact);
  REQUIRE(fp8);

  auto input = loaded.backend->alloc_tensor(gpu::TensorType::F16, in_shape);
  auto out16 = loaded.backend->alloc_tensor(gpu::TensorType::F16, out_shape);
  auto out8 = loaded.backend->alloc_tensor(gpu::TensorType::F16, out_shape);
  REQUIRE(input);
  REQUIRE(out16);
  REQUIRE(out8);
  upload((*input)->tensor(), picture);
  const std::array inputs{(*input)->tensor()};
  std::array reference_out{(*out16)->tensor()};
  std::array candidate_out{(*out8)->tensor()};
  REQUIRE(loaded.backend->enqueue(**fp16, inputs, reference_out));
  REQUIRE(loaded.backend->enqueue(**fp8, inputs, candidate_out));
  // Deux lancements de suite : la zone de travail, partagée, ne doit rien garder d'un lancement à l'autre.
  REQUIRE(loaded.backend->enqueue(**fp8, inputs, candidate_out));
  const std::vector<float> expected = download((*out16)->tensor());
  const std::vector<float> got = download((*out8)->tensor());
  REQUIRE(got.size() == expected.size());
  const double score = psnr(expected, got);
  const double similarity = block_ssim(expected, got, 2 * h, 2 * w);
  UNSCOPED_INFO("sortie " << 2 * w << "x" << 2 * h << " contre TensorRT FP16 : PSNR " << score
                          << " dB, SSIM par blocs " << similarity);
  WARN("PSNR " << score << " dB, SSIM par blocs " << similarity << " contre TensorRT FP16");
  CHECK(score >= 38.0);        // le pire cas toléré par le reçu (span_int8.h)
  CHECK(similarity >= 0.985);  // idem
}

}  // namespace ajp::test
