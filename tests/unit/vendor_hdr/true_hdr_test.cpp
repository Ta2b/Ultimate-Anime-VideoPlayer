// Contrats de flux et d'entrée de TrueHdr, avec un effet factice qui note ce qu'il reçoit.
#include "vendor_hdr/true_hdr.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "support.h"

using ajp::ColorMatrix;
using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;

namespace {

struct EffectLog {
  std::vector<TrueHdrParams> applied;
  int runs = 0;
  bool fail_run = false;
  bool fail_apply = false;
  std::chrono::milliseconds delay{0};  // temps hôte bloqué par chaque exécution
};

class ScriptedEffect final : public VendorEffect {
 public:
  ScriptedEffect(EffectLog& log, TrueHdrPixel input) : log_(log), input_(input) {}
  TrueHdrPixel input_pixel() const override { return input_; }
  Done apply(const TrueHdrParams& params) override {
    if (log_.fail_apply) return refuse(RefusalCode::VendorCallFailed, "réglage refusé");
    log_.applied.push_back(params);
    return {};
  }
  Done run_on_legacy(const TrueHdrImage&, const TrueHdrImage&) override {
    ++log_.runs;
    std::this_thread::sleep_for(log_.delay);
    if (log_.fail_run) return refuse(RefusalCode::VendorCallFailed, "exécution refusée");
    return {};
  }

 private:
  EffectLog& log_;
  TrueHdrPixel input_;
};

constexpr int kWidth = 8;
constexpr int kHeight = 4;

TrueHdrConfig config(ColorMatrix matrix = ColorMatrix::Bt709) {
  TrueHdrConfig made{.width = kWidth, .height = kHeight};
  made.source.matrix = matrix;
  return made;
}

std::unique_ptr<TrueHdr> open(EffectLog& log, TrueHdrPixel input = TrueHdrPixel::Rgba8,
                              const TrueHdrConfig& made = config()) {
  auto opened = make_true_hdr(made, std::make_unique<ScriptedEffect>(log, input));
  REQUIRE(opened);
  return std::move(*opened);
}

TrueHdrInput dithered(HostImage& image, ColorMatrix matrix = ColorMatrix::Bt709) {
  return TrueHdrInput{.image = image.view(), .prepared = {.matrix = matrix, .dithered = true}};
}

}  // namespace

TEST_CASE("TrueHdr : la création applique les réglages de la configuration", "[true_hdr]") {
  EffectLog log;
  TrueHdrConfig made = config();
  made.params.contrast = 120;
  auto hdr = open(log, TrueHdrPixel::Rgba8, made);
  REQUIRE(log.applied.size() == 1);
  CHECK(log.applied[0].contrast == 120);
  CHECK(hdr->input_pixel() == TrueHdrPixel::Rgba8);
  CHECK(hdr->config().width == kWidth);
}

TEST_CASE("TrueHdr : création refusée", "[true_hdr]") {
  EffectLog log;
  CHECK(make_true_hdr(config(), nullptr).error().code == RefusalCode::LibraryUnavailable);

  TrueHdrConfig hdr_source = config();
  hdr_source.source.transfer = ajp::Transfer::Pq;
  CHECK(make_true_hdr(hdr_source, std::make_unique<ScriptedEffect>(log, TrueHdrPixel::Rgba8)).error().code ==
        RefusalCode::HdrUnavailable);

  log.fail_apply = true;
  CHECK(make_true_hdr(config(), std::make_unique<ScriptedEffect>(log, TrueHdrPixel::Rgba8)).error().code ==
        RefusalCode::VendorCallFailed);
}

TEST_CASE("TrueHdr : le SDK tourne entre les deux événements du flux 0", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  REQUIRE(hdr->process(dithered(in), out.view(), stream));
  CHECK(stream.events == std::vector<std::string>{"hand", "take"});
  CHECK(log.runs == 1);
}

TEST_CASE("TrueHdr : le flux de gpu reprend la main même si le SDK échoue", "[true_hdr]") {
  EffectLog log;
  log.fail_run = true;
  auto hdr = open(log);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  auto processed = hdr->process(dithered(in), out.view(), stream);
  REQUIRE_FALSE(processed);
  CHECK(processed.error().detail == "exécution refusée");
  CHECK(stream.events == std::vector<std::string>{"hand", "take"});
}

TEST_CASE("TrueHdr : sans passage au flux 0, le SDK ne tourne pas", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  stream.fail_hand = true;
  CHECK_FALSE(hdr->process(dithered(in), out.view(), stream));
  CHECK(log.runs == 0);
  CHECK(stream.events == std::vector<std::string>{"hand"});
}

TEST_CASE("TrueHdr : l'entrée suit la vraie matrice de la source (AH-371)", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log, TrueHdrPixel::Rgba8, config(ColorMatrix::Bt601));
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  auto wrong = hdr->process(dithered(in, ColorMatrix::Bt709), out.view(), stream);
  REQUIRE_FALSE(wrong);
  CHECK(wrong.error().code == RefusalCode::InputMismatch);
  CHECK(wrong.error().detail == "entrée convertie avec la matrice BT.709, la source est en BT.601 (AH-371)");
  CHECK(stream.events.empty());
  CHECK(hdr->process(dithered(in, ColorMatrix::Bt601), out.view(), stream));
}

TEST_CASE("TrueHdr : une entrée 8 bits doit être tramée (AH-392)", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  TrueHdrInput plain = dithered(in);
  plain.prepared.dithered = false;
  auto refused = hdr->process(plain, out.view(), stream);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().detail == "entrée 8 bits quantifiée sans tramage (AH-392)");
  CHECK(log.runs == 0);
}

TEST_CASE("TrueHdr : une entrée 10 bits n'a pas besoin de tramage", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log, TrueHdrPixel::Rgb10A2);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  TrueHdrInput input = dithered(in);
  input.prepared.dithered = false;
  CHECK(hdr->process(input, out.view(), stream));
}

TEST_CASE("TrueHdr : tampons hors contrat", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;

  SECTION("entrée d'un autre format") {
    HostImage packed(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
    CHECK(hdr->process(dithered(packed), out.view(), stream).error().detail ==
          "entrée : RGB10A2 au lieu de RGBA 8 bits");
  }
  SECTION("sortie d'un autre format") {
    HostImage rgba(kWidth, kHeight, TrueHdrPixel::Rgba8);
    CHECK(hdr->process(dithered(in), rgba.view(), stream).error().detail ==
          "sortie : RGBA 8 bits au lieu de RGB10A2");
  }
  SECTION("autre taille") {
    HostImage small(kWidth / 2, kHeight, TrueHdrPixel::Rgba8);
    CHECK(hdr->process(dithered(small), out.view(), stream).error().detail == "entrée : 4x4 au lieu de 8x4");
  }
  SECTION("tampon nul") {
    TrueHdrImage empty = out.view();
    empty.device = 0;
    CHECK(hdr->process(dithered(in), empty, stream).error().detail == "sortie : tampon nul");
  }
  SECTION("pas de ligne trop court") {
    TrueHdrInput input = dithered(in);
    input.image.pitch = kWidth * 4 - 4;
    CHECK(hdr->process(input, out.view(), stream).error().code == RefusalCode::InputMismatch);
  }
  SECTION("entrée et sortie confondues") {
    TrueHdrImage same = in.view();
    same.pixel = TrueHdrPixel::Rgb10A2;
    CHECK(hdr->process(dithered(in), same, stream).error().detail ==
          "entrée et sortie se recouvrent dans le même tampon");
  }
  CHECK(stream.events.empty());
  CHECK(log.runs == 0);
}

TEST_CASE("TrueHdr : set_params refuse hors plage sans rien changer", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log);
  REQUIRE(log.applied.size() == 1);

  auto refused = hdr->set_params({.saturation = 250});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(log.applied.size() == 1);
  CHECK(hdr->config().params == TrueHdrParams{});

  REQUIRE(hdr->set_params({.contrast = 80, .peak_nits = 600}));
  CHECK(log.applied.back().contrast == 80);
  CHECK(hdr->config().params.peak_nits == 600);

  log.fail_apply = true;
  CHECK(hdr->set_params({.contrast = 90}).error().code == RefusalCode::VendorCallFailed);
  CHECK(hdr->config().params.contrast == 80);
}

TEST_CASE("TrueHdr : entrée et sortie qui se recouvrent en partie", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log);
  RecordingStream stream;
  // L'effet scripté ne lit pas les pixels : des adresses suffisent. 4 lignes de 32 octets.
  constexpr std::uintptr_t kBase = 0x100000;
  const TrueHdrImage in{.device = kBase, .pitch = kWidth * 4, .width = kWidth, .height = kHeight};
  const TrueHdrInput input{.image = in, .prepared = {.dithered = true}};
  TrueHdrImage out = in;
  out.pixel = TrueHdrPixel::Rgb10A2;

  out.device = kBase + (2 * kWidth * 4);  // commence sur la troisième ligne de l'entrée
  CHECK(hdr->process(input, out, stream).error().detail ==
        "entrée et sortie se recouvrent dans le même tampon");
  out.device = kBase - (2 * kWidth * 4);  // finit dans la première ligne de l'entrée
  CHECK_FALSE(hdr->process(input, out, stream));
  CHECK(log.runs == 0);

  out.device = kBase + (kHeight * kWidth * 4);  // juste après la dernière ligne : accepté
  CHECK(hdr->process(input, out, stream));
  CHECK(log.runs == 1);
}

TEST_CASE("TrueHdr : pas de ligne hors contrat", "[true_hdr]") {
  EffectLog log;
  auto hdr = open(log);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  TrueHdrInput input = dithered(in);

  input.image.pitch = (kWidth * 4) + 2;  // pixels de 4 octets désalignés
  CHECK(hdr->process(input, out.view(), stream).error().detail ==
        "entrée : pas de 34 octets, attendu un multiple de 4 entre 32 et 2147483647");
  input.image.pitch = limits::kMaxPitch + 1;  // ne tient plus dans le int de NvCVImage
  CHECK(hdr->process(input, out.view(), stream).error().code == RefusalCode::InputMismatch);

  input = dithered(in);
  input.image.device = UINTPTR_MAX - 64;  // la dernière ligne déborderait l'espace d'adresses
  CHECK(hdr->process(input, out.view(), stream).error().detail ==
        "entrée : tampon hors de l'espace d'adresses");
  CHECK(stream.events.empty());
  CHECK(log.runs == 0);
}

TEST_CASE("TrueHdr : l'entrée suit la vraie plage de la source (AH-371)", "[true_hdr]") {
  EffectLog log;
  TrueHdrConfig full = config();
  full.source.range = ajp::ColorRange::Full;
  auto hdr = open(log, TrueHdrPixel::Rgba8, full);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  TrueHdrInput input = dithered(in);  // convertie en plage limitée
  CHECK(hdr->process(input, out.view(), stream).error().detail ==
        "entrée convertie en plage limitée, la source est en plage pleine (AH-371)");
  input.prepared.range = ajp::ColorRange::Full;
  CHECK(hdr->process(input, out.view(), stream));
}

TEST_CASE("TrueHdr : le temps hôte du SDK est mesuré à part", "[true_hdr]") {
  using namespace std::chrono_literals;
  EffectLog log;
  log.delay = 2ms;
  auto hdr = open(log);
  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  RecordingStream stream;
  CHECK(hdr->vendor_time().calls == 0);

  REQUIRE(hdr->process(dithered(in), out.view(), stream));
  REQUIRE(hdr->process(dithered(in), out.view(), stream));
  CHECK(hdr->vendor_time().calls == 2);
  CHECK(hdr->vendor_time().host >= 4ms);

  // Une entrée refusée n'appelle pas le SDK ; un appel qui échoue compte quand même.
  CHECK_FALSE(hdr->process(dithered(in, ColorMatrix::Bt601), out.view(), stream));
  CHECK(hdr->vendor_time().calls == 2);
  log.fail_run = true;
  CHECK_FALSE(hdr->process(dithered(in), out.view(), stream));
  CHECK(hdr->vendor_time().calls == 3);
  CHECK(hdr->vendor_time().host >= 6ms);
}
