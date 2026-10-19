// TrueHDR réel sur la RTX : SDK Maxine VFX installé et GPU NVIDIA, SKIP sinon (la CI n'a ni
// l'un ni l'autre). Les noms commencent par « Maxine sur la RTX » pour que ctest -R les isole sous le
// verrou GPU (ajp-gpu). Fige ce qui a été mesuré le 19 octobre 2026 (rapport
// ~/ajp-reports/truehdr.md) : contrat de flux réel, déterminisme, absence d'état d'une image à
// l'autre, recréation, plusieurs instances, rampe de 32 marches, entrée tramée, pic toujours
// passé et débandage coupé avec rechargement.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "real_sdk.h"
#include "support.h"
#include "vendor_hdr/maxine_vfx.h"
#include "vendor_hdr/open_true_hdr.h"
#include "vendor_hdr/sdk_store.h"

using ajp::ColorMatrix;
using ajp::ColorRange;
using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;

namespace {

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr std::uint32_t kRampSteps = 32;

// Le SDK d'abord : sans lui, le test SKIP sans même toucher au GPU.
struct Rig {
  std::filesystem::path sdk = require_maxine_sdk();
  std::unique_ptr<Cuda> cuda = require_cuda();
  DeviceImage in{*cuda, kWidth, kHeight, TrueHdrPixel::Rgba8};
  DeviceImage out{*cuda, kWidth, kHeight, TrueHdrPixel::Rgb10A2};
  CudaGpuStream stream{*cuda};

  std::unique_ptr<TrueHdr> open(const TrueHdrConfig& config) const {
    auto opened = detail::open_maxine_at(sdk, config);
    REQUIRE(opened);
    return std::move(*opened);
  }

  // L'entrée est posée et la sortie relue sur le flux de gpu seulement : seul l'encadrement
  // d'événements de process() ordonne ces copies avec le travail du SDK sur le flux 0.
  std::vector<std::uint32_t> run(TrueHdr& hdr, const std::vector<std::uint32_t>& source,
                                 const InputPreparation& prepared = {.dithered = true}) {
    stream.upload(in, source);
    const Done done = hdr.process({.image = in.view(), .prepared = prepared}, out.view(), stream);
    REQUIRE(done);
    return stream.download(out);
  }
};

// Moyenne du rouge 10 bits par colonne sur les lignes [first, last).
std::vector<double> column_red(const std::vector<std::uint32_t>& image, int first, int last) {
  constexpr auto kColumns = static_cast<std::size_t>(kWidth);
  std::vector<double> means(kColumns, 0.0);
  for (auto y = static_cast<std::size_t>(first); y < static_cast<std::size_t>(last); ++y) {
    for (std::size_t x = 0; x < kColumns; ++x) means[x] += red10(image[(y * kColumns) + x]);
  }
  for (double& mean : means) mean /= last - first;
  return means;
}

// Plus grand saut entre deux colonnes voisines : un cran de quantification se voit ici.
double largest_step(const std::vector<double>& means) {
  double largest = 0.0;
  for (std::size_t x = 1; x < means.size(); ++x) largest = std::max(largest, means[x] - means[x - 1]);
  return largest;
}

std::uint32_t center_red(const std::vector<std::uint32_t>& image) {
  return red10(image[(static_cast<std::size_t>(kHeight / 2) * kWidth) + (kWidth / 2)]);
}

// Rouge au centre d'une image unie, pour chaque marche quasi noire 0 à 31 (8 bits).
std::vector<std::uint32_t> ramp(Rig& rig, TrueHdr& hdr, bool descending) {
  std::vector<std::uint32_t> response(kRampSteps);
  for (std::uint32_t i = 0; i < kRampSteps; ++i) {
    const std::uint32_t step = descending ? kRampSteps - 1 - i : i;
    response[step] = center_red(rig.run(hdr, flat_rgba8(kWidth, kHeight, step)));
  }
  return response;
}

}  // namespace

TEST_CASE("Maxine sur la RTX : du magasin au GPU, TrueHDR Maxine réel", "[nvidia][cuda]") {
  Rig rig;
  TempDir data;
  SdkStore store(data.path(), SdkKind::MaxineVfx);
  REQUIRE(store.import(rig.sdk));
  const SdkStatus status = store.status();
  REQUIRE(status.installed);
  // Ouvert depuis la copie du magasin : le runtime NGX y trouve son modèle sans LD_LIBRARY_PATH.
  auto opened = open_true_hdr(*status.installed, sdr_config(kWidth, kHeight, rig.cuda->context()));
  REQUIRE(opened);
  TrueHdr& hdr = **opened;
  CHECK(hdr.input_pixel() == TrueHdrPixel::Rgba8);

  const std::vector<std::uint32_t> source = source_rgba8(kWidth, kHeight, true);
  const std::vector<std::uint32_t> first = rig.run(hdr, source);
  CHECK(std::ranges::all_of(first, [](std::uint32_t pixel) { return alpha2(pixel) == 3; }));
  // PQ BT.2020 : la rampe de gris monte de bout en bout, du noir vers le pic.
  const std::vector<double> gray = column_red(first, 0, kHeight / 4);
  CHECK(gray.front() < 64.0);
  CHECK(gray.back() > 600.0);
  CHECK(gray.back() > gray[kWidth / 2]);
  CHECK(gray[kWidth / 2] > gray.front());

  // Déterministe : mêmes octets au second passage, et le premier passage est déjà juste.
  CHECK(rig.run(hdr, source) == first);
  CHECK(hdr.vendor_time().calls == 2);
  CHECK(hdr.vendor_time().host.count() > 0);

  // Un réglage s'applique sans recharger le modèle, et revient aux mêmes octets.
  REQUIRE(hdr.set_params({.contrast = 200, .saturation = 100, .middle_gray = 50, .peak_nits = 1000}));
  CHECK(rig.run(hdr, source) != first);
  REQUIRE(hdr.set_params(TrueHdrParams{}));
  CHECK(rig.run(hdr, source) == first);
}

TEST_CASE("Maxine sur la RTX : pas d'état d'une image à l'autre, ni d'une instance à l'autre",
          "[nvidia][cuda]") {
  Rig rig;
  const TrueHdrConfig config = sdr_config(kWidth, kHeight, rig.cuda->context());
  const std::vector<std::uint32_t> a = source_rgba8(kWidth, kHeight, true);
  const std::vector<std::uint32_t> b = flat_rgba8(kWidth, kHeight, 128);

  std::vector<std::uint32_t> reference;
  {
    std::unique_ptr<TrueHdr> hdr = rig.open(config);
    reference = rig.run(*hdr, a);
    // A, B, A : la seconde sortie de A ne dépend pas de B.
    CHECK(rig.run(*hdr, b) != reference);
    CHECK(rig.run(*hdr, a) == reference);
  }
  // Instance détruite puis recréée (bibliothèques refermées puis rouvertes) : mêmes octets.
  std::unique_ptr<TrueHdr> second = rig.open(config);
  CHECK(rig.run(*second, a) == reference);
  // Maxine admet plusieurs instances vivantes ensemble (la règle d'instance unique est celle
  // du RTX Video SDK) ; chacune donne les mêmes octets.
  std::unique_ptr<TrueHdr> third = rig.open(config);
  CHECK(rig.run(*third, a) == reference);
  CHECK(rig.run(*second, a) == reference);
}

TEST_CASE("Maxine sur la RTX : rampe de 32 marches quasi noires sur une seule instance", "[nvidia][cuda]") {
  Rig rig;
  std::unique_ptr<TrueHdr> hdr = rig.open(sdr_config(kWidth, kHeight, rig.cuda->context()));
  // Plan, grade : la rampe traverse TrueHDR à chaque réglage pour fixer les seuils en nits réels.
  const std::vector<std::uint32_t> up = ramp(rig, *hdr, false);
  // Même réponse dans les deux sens : aucune image n'influence la suivante.
  CHECK(ramp(rig, *hdr, true) == up);
  CHECK(up.front() == 0);
  CHECK(std::ranges::is_sorted(up));
  CHECK(std::ranges::adjacent_find(up) == up.end());  // 32 marches distinctes

  // Après un changement de réglage, la même instance rend une autre rampe, aussi croissante.
  REQUIRE(hdr->set_params({.contrast = 200, .saturation = 200, .middle_gray = 50, .peak_nits = 1000}));
  const std::vector<std::uint32_t> graded = ramp(rig, *hdr, false);
  CHECK(graded != up);
  CHECK(std::ranges::is_sorted(graded));
}

TEST_CASE("Maxine sur la RTX : entrée 8 bits tramée avec la vraie matrice", "[nvidia][cuda]") {
  Rig rig;
  std::unique_ptr<TrueHdr> hdr = rig.open(sdr_config(kWidth, kHeight, rig.cuda->context()));
  const int first = near_black_row(kHeight);
  // Rampe quasi noire en 10 bits : 4 codes par niveau 8 bits. Sans tramage, la sortie garde
  // des crans de 4 colonnes ; tramée, sa moyenne par colonne suit la rampe 10 bits. L'entrée
  // crantée se déclare tramée pour passer le contrat : elle montre ce que celui-ci évite.
  const std::vector<double> banded =
      column_red(rig.run(*hdr, source_rgba8(kWidth, kHeight, false), {.dithered = true}), first, kHeight);
  const std::vector<double> dithered =
      column_red(rig.run(*hdr, source_rgba8(kWidth, kHeight, true), {.dithered = true}), first, kHeight);
  INFO("plus grand cran sans tramage " << largest_step(banded) << ", tramé " << largest_step(dithered));
  CHECK(largest_step(dithered) * 2.0 < largest_step(banded));
  CHECK(dithered.back() > dithered.front());

  // Le contrat d'entrée tient sur le vrai SDK : rien n'est lancé sans tramage ni avec une
  // autre matrice ou une autre plage que la source.
  rig.stream.upload(rig.in, source_rgba8(kWidth, kHeight, false));
  const auto calls = hdr->vendor_time().calls;
  for (const InputPreparation& wrong :
       {InputPreparation{.dithered = false}, InputPreparation{.matrix = ColorMatrix::Bt601, .dithered = true},
        InputPreparation{.range = ColorRange::Full, .dithered = true}}) {
    const Done refused =
        hdr->process({.image = rig.in.view(), .prepared = wrong}, rig.out.view(), rig.stream);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::InputMismatch);
  }
  CHECK(hdr->vendor_time().calls == calls);
}

TEST_CASE("Maxine sur la RTX : lignes à marge, le pas est transmis au SDK", "[nvidia][cuda]") {
  Rig rig;
  std::unique_ptr<TrueHdr> hdr = rig.open(sdr_config(kWidth, kHeight, rig.cuda->context()));
  const std::vector<std::uint32_t> source = source_rgba8(kWidth, kHeight, true);
  const std::vector<std::uint32_t> tight = rig.run(*hdr, source);

  const std::size_t pitch = (static_cast<std::size_t>(kWidth) * 4) + 256;
  const DeviceImage in(*rig.cuda, kWidth, kHeight, TrueHdrPixel::Rgba8, pitch);
  const DeviceImage out(*rig.cuda, kWidth, kHeight, TrueHdrPixel::Rgb10A2, pitch);
  in.upload(source);
  rig.cuda->synchronize();
  REQUIRE(hdr->process({.image = in.view(), .prepared = {.dithered = true}}, out.view(), rig.stream));
  rig.cuda->synchronize();
  CHECK(out.download() == tight);
}

TEST_CASE("Maxine sur la RTX : le pic passé fait foi, jamais les 650 nits du SDK", "[nvidia][cuda]") {
  Rig rig;
  const TrueHdrConfig config = sdr_config(kWidth, kHeight, rig.cuda->context());
  REQUIRE(config.params.peak_nits == 1000);  // plan
  const std::vector<std::uint32_t> source = source_rgba8(kWidth, kHeight, true);
  std::unique_ptr<TrueHdr> hdr = rig.open(config);
  const std::vector<std::uint32_t> plan = rig.run(*hdr, source);
  // 650 nits, le défaut du SDK, donne une autre image : 1000 est donc bien celui appliqué.
  REQUIRE(hdr->set_params({.peak_nits = 650}));
  const std::vector<std::uint32_t> sdk_default = rig.run(*hdr, source);
  CHECK(sdk_default != plan);
  CHECK(column_red(sdk_default, 0, kHeight / 4).back() < column_red(plan, 0, kHeight / 4).back());
  REQUIRE(hdr->set_params(TrueHdrParams{}));
  CHECK(rig.run(*hdr, source) == plan);
}

TEST_CASE("Maxine sur la RTX : débandage coupé, modèle rechargé, octets stables", "[nvidia][cuda]") {
  Rig rig;
  const TrueHdrConfig config = sdr_config(kWidth, kHeight, rig.cuda->context());
  const std::vector<std::uint32_t> source = source_rgba8(kWidth, kHeight, true);
  std::unique_ptr<TrueHdr> hdr = rig.open(config);
  const std::vector<std::uint32_t> debanded = rig.run(*hdr, source);

  // Coupé : rechargé à l'image suivante, dont la sortie est déjà celle d'une instance ouverte
  // d'emblée sans débandage.
  REQUIRE(hdr->set_params({.debanding = false}));
  const std::vector<std::uint32_t> raw = rig.run(*hdr, source);
  CHECK(raw != debanded);
  CHECK(rig.run(*hdr, source) == raw);
  TrueHdrConfig off = config;
  off.params.debanding = false;
  std::unique_ptr<TrueHdr> fresh = rig.open(off);
  CHECK(rig.run(*fresh, source) == raw);

  // Un réglage changé avec le débandage tient après le rechargement : même image qu'une
  // instance neuve ouverte avec lui.
  TrueHdrConfig graded = config;
  graded.params = {.contrast = 150, .saturation = 120};
  REQUIRE(hdr->set_params(graded.params));
  std::unique_ptr<TrueHdr> reference = rig.open(graded);
  CHECK(rig.run(*hdr, source) == rig.run(*reference, source));

  // Rétabli : les octets du début reviennent.
  REQUIRE(hdr->set_params(TrueHdrParams{}));
  CHECK(rig.run(*hdr, source) == debanded);
}
