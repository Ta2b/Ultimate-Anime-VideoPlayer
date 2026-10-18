// Chargeurs des SDK : refus propres quand une bibliothèque, un symbole ou un appel manque, et
// chemin complet sur les faux SDK (fakes/). Fige l'ordre d'appels repris de
// animejanai-inference-rtx@118c272 src/pipeline_hdr.c:57-86 : réglages, liaison des images,
// modèle chargé une fois, exécution synchrone sur le flux 0.
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "support.h"
#include "vendor_hdr/dynamic_library.h"
#include "vendor_hdr/maxine_vfx.h"
#include "vendor_hdr/open_true_hdr.h"
#include "vendor_hdr/rtx_video.h"
#include "vendor_hdr/rtx_video_abi.h"
#include "vendor_hdr/sdk_store.h"

using ajp::RefusalCode;
using namespace ajp::vendor_hdr;
using namespace ajp::vendor_hdr::test;
namespace fs = std::filesystem;

namespace {

constexpr int kWidth = 4;
constexpr int kHeight = 2;

TrueHdrConfig small_config() {
  return TrueHdrConfig{.width = kWidth, .height = kHeight};
}

std::uint32_t rgba(std::uint32_t r, std::uint32_t g, std::uint32_t b) {
  return r | (g << 8U) | (b << 16U) | (255U << 24U);
}

std::uint32_t packed(std::uint32_t r, std::uint32_t g, std::uint32_t b) {
  return r | (g << 10U) | (b << 20U) | (3U << 30U);
}

std::string refusal_of(const std::expected<std::unique_ptr<TrueHdr>, ajp::Refusal>& opened,
                       RefusalCode code) {
  REQUIRE_FALSE(opened);
  CHECK(opened.error().code == code);
  return opened.error().detail;
}

// Le faux RTX Video garde son état dans sa bibliothèque : les tests le lisent par ses fonctions.
struct RtxProbe {
  DynamicLibrary library = DynamicLibrary::open(AJP_FAKE_RTX_PATH).value();
  bool created() const { return (*library.symbol<bool()>("fake_rtx_video_is_created"))(); }
  rtx_video::CUcontext context() const {
    return (*library.symbol<rtx_video::CUcontext()>("fake_rtx_video_context"))();
  }
  int evaluations() const { return (*library.symbol<int()>("fake_rtx_video_evaluations"))(); }
  void refuse_create(bool refuse) const {
    (*library.symbol<void(bool)>("fake_rtx_video_refuse_create"))(refuse);
  }
};

}  // namespace

TEST_CASE("chargeur : bibliothèque absente", "[loader]") {
  TempDir dir;
  auto missing = DynamicLibrary::open(dir.path() / "absent.so");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::LibraryUnavailable);
  CHECK(missing.error().detail.starts_with("absent.so ne se charge pas : "));

  write_file(dir.path() / "stub.so", elf_header());
  CHECK(DynamicLibrary::open(dir.path() / "stub.so").error().code == RefusalCode::LibraryUnavailable);
}

TEST_CASE("chargeur : symbole absent", "[loader]") {
  auto library = DynamicLibrary::open(AJP_FAKE_DEP_PATH);
  REQUIRE(library);
  CHECK((*library->symbol<int()>("fake_sdk_dep_marker"))() == 1);
  auto missing = library->symbol<int()>("NvVFX_Run");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().detail.starts_with("symbole NvVFX_Run absent de "));
}

TEST_CASE("Maxine : SDK absent ou inutilisable, refus propre", "[loader]") {
  TempDir dir;
  SECTION("dossier vide") {
    CHECK(refusal_of(detail::open_maxine_at(dir.path(), small_config()), RefusalCode::LibraryUnavailable) ==
          "lib/libNVCVImage.so absent du SDK");
  }
  SECTION("bibliothèques qui ne se chargent pas") {
    make_maxine_sdk(dir.path());
    CHECK(refusal_of(detail::open_maxine_at(dir.path(), small_config()), RefusalCode::LibraryUnavailable)
              .contains("ne se charge pas"));
  }
  SECTION("libNVCVImage sans NvCVImage_Init") {
    make_maxine_sdk(dir.path(), {.libraries = LibraryContent::Fakes});
    fs::copy_file(AJP_FAKE_DEP_PATH, dir.path() / "lib/libNVCVImage.so",
                  fs::copy_options::overwrite_existing);
    CHECK(refusal_of(detail::open_maxine_at(dir.path(), small_config()), RefusalCode::LibraryUnavailable) ==
          "symbole NvCVImage_Init absent de libNVCVImage.so");
  }
  SECTION("configuration refusée avant tout chargement") {
    TrueHdrConfig config = small_config();
    config.params.contrast = 500;
    CHECK(refusal_of(detail::open_maxine_at(dir.path(), config), RefusalCode::FieldOutOfRange) ==
          "truehdr.contrast = 500, hors de 0-200");
  }
}

TEST_CASE("Maxine : importé par le magasin, ouvert et exécuté (faux SDK)", "[loader]") {
  TempDir dir;
  make_maxine_sdk(dir.path() / "VideoFX", {.libraries = LibraryContent::Fakes});
  SdkStore store(dir.path() / "data", SdkKind::MaxineVfx);
  auto imported = store.import(dir.path() / "VideoFX");
  REQUIRE(imported);
  const SdkStatus status = store.status();
  REQUIRE(status.installed);

  auto opened = open_true_hdr(*status.installed, small_config());
  REQUIRE(opened);
  TrueHdr& hdr = **opened;
  CHECK(hdr.input_pixel() == TrueHdrPixel::Rgba8);

  HostImage in(kWidth, kHeight, TrueHdrPixel::Rgba8);
  HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
  in.pixels[0] = rgba(100, 50, 25);
  in.pixels[7] = rgba(255, 255, 255);
  const TrueHdrInput input{.image = in.view(), .prepared = {.dithered = true}};
  RecordingStream stream;

  REQUIRE(hdr.process(input, out.view(), stream));
  CHECK(stream.events == std::vector<std::string>{"hand", "take"});
  CHECK(out.pixels[0] == packed(400, 200, 100));
  CHECK(out.pixels[7] == packed(1020, 1020, 1020));

  // Deuxième image : le faux refuse un second NvVFX_Load, le modèle n'est donc chargé qu'une fois.
  REQUIRE(hdr.set_params({.contrast = 200, .saturation = 50, .middle_gray = 100}));
  REQUIRE(hdr.process(input, out.view(), stream));
  CHECK(out.pixels[0] == packed(800, 100, 200));
  CHECK(out.pixels[7] == packed(1023, 510, 1023));
  CHECK(stream.events.size() == 4);
}

TEST_CASE("Maxine : un autre SDK est refusé", "[loader]") {
  TempDir dir;
  make_rtx_sdk(dir.path() / "rtx", SdkVersion{.parts = {1, 1, 0, 0}});
  SdkStore store(dir.path() / "data", SdkKind::RtxVideo);
  auto imported = store.import(dir.path() / "rtx");
  REQUIRE(imported);
  CHECK(refusal_of(open_maxine(*imported->installed, small_config()), RefusalCode::LibraryUnavailable) ==
        "le SDK fourni n'est pas Maxine VFX");
}

TEST_CASE("RTX Video : la DLL ne se charge pas hors Windows, refus propre", "[loader]") {
  TempDir dir;
  make_rtx_sdk(dir.path() / "rtx", SdkVersion{.parts = {1, 1, 0, 0}});
  SdkStore store(dir.path() / "data", SdkKind::RtxVideo);
  auto imported = store.import(dir.path() / "rtx");
  REQUIRE(imported);
  CHECK(refusal_of(open_true_hdr(*imported->installed, small_config()), RefusalCode::LibraryUnavailable)
            .starts_with("rtx_video_api.dll ne se charge pas"));
  CHECK(refusal_of(detail::open_rtx_video_at(dir.path() / "absent.dll", small_config()),
                   RefusalCode::LibraryUnavailable)
            .starts_with("absent.dll ne se charge pas"));
  CHECK(refusal_of(detail::open_rtx_video_at(AJP_FAKE_DEP_PATH, small_config()),
                   RefusalCode::LibraryUnavailable)
            .starts_with("symbole rtx_video_api_cuda_create absent de "));
}

TEST_CASE("RTX Video : créé sur le flux 0, exécuté, arrêté (faux SDK)", "[loader]") {
  const RtxProbe probe;
  REQUIRE_FALSE(probe.created());
  TrueHdrConfig config = small_config();
  config.cuda_context = 0x1234;
  {
    auto opened = detail::open_rtx_video_at(AJP_FAKE_RTX_PATH, config);
    REQUIRE(opened);
    CHECK(probe.created());
    CHECK(reinterpret_cast<std::uintptr_t>(probe.context()) == 0x1234);
    TrueHdr& hdr = **opened;
    CHECK(hdr.input_pixel() == TrueHdrPixel::Rgb10A2);

    // Une seule instance à la fois : l'API du SDK n'a pas de poignée.
    CHECK(refusal_of(detail::open_rtx_video_at(AJP_FAKE_RTX_PATH, config), RefusalCode::VendorCallFailed)
              .starts_with("TrueHDR RTX Video : création refusée"));
    CHECK(probe.created());

    HostImage in(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
    HostImage out(kWidth, kHeight, TrueHdrPixel::Rgb10A2);
    in.pixels[1] = packed(400, 200, 100);
    RecordingStream stream;
    const int before = probe.evaluations();
    REQUIRE(hdr.process(TrueHdrInput{.image = in.view()}, out.view(), stream));
    CHECK(out.pixels[1] == packed(400, 200, 100));
    REQUIRE(hdr.set_params({.contrast = 50, .saturation = 200, .middle_gray = 25}));
    REQUIRE(hdr.process(TrueHdrInput{.image = in.view()}, out.view(), stream));
    CHECK(out.pixels[1] == packed(200, 400, 50));
    CHECK(probe.evaluations() == before + 2);
    CHECK(stream.events == std::vector<std::string>{"hand", "take", "hand", "take"});

    // L'API ne prend pas de pas de ligne : un tampon à marge est refusé, le flux rendu quand même.
    TrueHdrImage padded = out.view();
    padded.pitch += 64;
    CHECK(hdr.process(TrueHdrInput{.image = in.view()}, padded, stream).error().code ==
          RefusalCode::InputMismatch);
    CHECK(stream.events.back() == "take");
  }
  CHECK_FALSE(probe.created());  // shutdown appelé à la destruction
}

TEST_CASE("RTX Video : création refusée par le SDK", "[loader]") {
  const RtxProbe probe;
  probe.refuse_create(true);
  auto opened = detail::open_rtx_video_at(AJP_FAKE_RTX_PATH, small_config());
  probe.refuse_create(false);
  CHECK(refusal_of(opened, RefusalCode::VendorCallFailed) ==
        "TrueHDR RTX Video : création refusée (GPU, pilote ou SDK incompatible)");
  CHECK_FALSE(probe.created());
}
