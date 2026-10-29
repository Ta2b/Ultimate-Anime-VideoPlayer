// La dérivation native du paquet « .span8 » (app/play/span_derive.h) : découpe des images, écriture par
// renommage, reçu relu par le lecteur, refus sans effet de bord. Le backend et la vidéo sont des doublures :
// rien ici ne touche au GPU.
#include "app/play/span_derive.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/auto_bench.h"
#include "app/play/settings_system.h"
#include "app/play/span_int8.h"
#include "gpu/span_fp8/package.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

namespace fs = std::filesystem;
using gpu::span_fp8::Package;

// Un dossier « modeles » et un faux SPAN dedans (le reçu s'attache au SHA-256 du fichier).
struct Folder {
  // Un dossier par essai : ctest lance les cas en parallèle, un nom fixe ferait se marcher dessus.
  Folder()
      : root(fs::temp_directory_path() /
             std::format("ajp-span-derive-test-{}",
                         std::chrono::steady_clock::now().time_since_epoch().count())) {
    fs::remove_all(root);
    fs::create_directories(models());
    model = root / "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx";
    std::ofstream(model, std::ios::binary) << "un SPAN pour les essais";
  }
  ~Folder() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  Folder(const Folder&) = delete;
  Folder& operator=(const Folder&) = delete;

  fs::path models() const { return root / "modeles"; }
  fs::path package() const {
    return span_variant_path(SpanVariantKind::Fp8Cutlass, model, models(),
                             prepare::sha256_file(model).value());
  }
  fs::path receipt() const {
    fs::path path = package();
    path.replace_extension(".toml");
    return path;
  }
  SpanDerivePlan plan() const {
    return {.model = model, .models_dir = models(), .load_weights = [] {
              return std::expected<Package, Refusal>(gpu::span_fp8::synthetic());
            }};
  }
  std::vector<std::string> files() const {
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(models()))
      names.push_back(entry.path().filename().string());
    std::ranges::sort(names);
    return names;
  }

  fs::path root;
  fs::path model;
};

// Douze images 128×4 dont le premier octet dit leur rang.
std::vector<media::Still> twelve() {
  std::vector<media::Still> stills;
  for (int i = 0; i < 12; ++i) {
    media::Still still;
    still.width = 128;
    still.height = 4;
    still.pts_us = 1'000'000 * i;
    still.rgb.assign(std::size_t{3} * 128 * 4, static_cast<std::uint8_t>(i));
    stills.push_back(std::move(still));
  }
  return stills;
}

SpanStillsSource source_of(std::vector<media::Still> stills) {
  return [stills = std::move(stills)](const std::stop_token&, const std::function<void(float)>& progress) {
    if (progress) progress(1.0F);
    return std::expected<std::vector<media::Still>, Refusal>(stills);
  };
}

// Un « backend » qui note ce qu'on lui demande et rend des mesures données.
struct FakeBackend {
  double psnr = 50.0;
  double ssim = 0.995;
  bool refuse = false;
  mutable gpu::SpanFp8Request seen;
  SpanDeriver deriver() const {
    return [this](const gpu::SpanFp8Request& request, gpu::Progress& progress) {
      seen = request;
      if (progress.report) progress.report(0.5F, "calibration");
      if (refuse)
        return std::expected<gpu::SpanFp8Derivation, Refusal>(
            std::unexpected(Refusal{RefusalCode::GpuMemoryExhausted, "plus de mémoire vidéo"}));
      gpu::SpanFp8Derivation derivation;
      derivation.scales.fill(0.05F);
      derivation.peaks.fill(22.4F);
      derivation.scores.assign(request.validation.size(), {.psnr_db = psnr, .ssim = ssim});
      derivation.kernels_ms = 9.2;
      derivation.reference = "cutlass-fp16";
      return std::expected<gpu::SpanFp8Derivation, Refusal>(derivation);
    };
  }
};

}  // namespace

TEST_CASE("douze images : huit calibrent, quatre sont mises de côté", "[app][span_derive]") {
  const StillsSplit twelve_split = split_stills(12);
  CHECK(twelve_split.calibration.size() == 8);
  CHECK(twelve_split.validation == std::vector<std::size_t>{2, 5, 8, 11});
  const StillsSplit eight = split_stills(8);
  CHECK(eight.calibration.size() == 6);
  CHECK(eight.validation == std::vector<std::size_t>{2, 5});
  CHECK(split_stills(2).validation.empty());
}

TEST_CASE("une dérivation qui passe écrit le paquet et son reçu, que le lecteur relit",
          "[app][span_derive]") {
  const Folder folder;
  const FakeBackend backend;
  SpanDeriveProgress progress;
  const auto report =
      derive_span_fp8(folder.plan(), source_of(twelve()), backend.deriver(), progress, std::stop_token{});
  INFO((report ? std::string() : report.error().detail));
  REQUIRE(report);
  CHECK(report->passed);
  CHECK(report->calibration_images == 8);
  CHECK(report->validation_images == 4);
  CHECK(report->size == "128x4");
  CHECK(progress.get().first == 1.0F);

  // Les images de validation n'ont jamais servi à calibrer : ni les mêmes rangs, ni les mêmes octets.
  std::set<int> calibration, validation;
  for (const gpu::SpanStill& still : backend.seen.calibration) calibration.insert(still.rgb.front());
  for (const gpu::SpanStill& still : backend.seen.validation) validation.insert(still.rgb.front());
  CHECK(calibration == std::set<int>{0, 1, 3, 4, 6, 7, 9, 10});
  CHECK(validation == std::set<int>{2, 5, 8, 11});
  CHECK(gpu::span_fp8::parse(backend.seen.package));  // les poids partent sous forme de paquet valide

  CHECK(folder.files() == std::vector<std::string>{folder.package().filename().string(),
                                                   folder.receipt().filename().string()});  // aucun .part
  const auto bytes = [&] {
    std::ifstream in(folder.package(), std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
  }();
  const auto parsed = gpu::span_fp8::parse(std::as_bytes(std::span(bytes)));
  REQUIRE(parsed);
  CHECK(parsed->scales[0] == 0.05F);  // les échelles de la calibration, pas celles du modèle d'origine

  const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models());
  INFO(found.note);
  REQUIRE(found.variant);
  CHECK(found.variant->psnr_mean_db == 50.0);
  CHECK(span_tiers(folder.model, folder.models(), false, true).performance);
  CHECK_FALSE(span_tiers(folder.model, folder.models(), false, false).performance);  // sans le moteur CUTLASS

  // Le reçu dit comment il a été fait.
  std::ifstream in(folder.receipt());
  std::ostringstream receipt;
  receipt << in.rdbuf();
  CHECK(receipt.str().find("format = \"ajp-span-fp8c-v1\"") != std::string::npos);
  CHECK(receipt.str().find("reference = \"cutlass-fp16\"") != std::string::npos);
  CHECK(receipt.str().find("validation_size = \"128x4\"") != std::string::npos);
}

TEST_CASE("des garde-fous manqués écrivent le reçu refusé et retirent le paquet d'avant",
          "[app][span_derive]") {
  const Folder folder;
  SpanDeriveProgress progress;
  FakeBackend good;
  REQUIRE(derive_span_fp8(folder.plan(), source_of(twelve()), good.deriver(), progress, std::stop_token{}));
  REQUIRE(fs::exists(folder.package()));

  FakeBackend bad;
  bad.psnr = 36.0;
  const auto report =
      derive_span_fp8(folder.plan(), source_of(twelve()), bad.deriver(), progress, std::stop_token{});
  REQUIRE(report);
  CHECK_FALSE(report->passed);
  CHECK(report->text.find("refusé") != std::string::npos);
  CHECK(report->package.empty());
  CHECK_FALSE(fs::exists(folder.package()));
  CHECK(fs::exists(folder.receipt()));
  const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models());
  CHECK_FALSE(found.variant);
  CHECK(found.note.find("sous les garde-fous") != std::string::npos);
  // Le cran reste offert (la lecture retombe sur FP16) ; la préparation automatique ne se relance pas dessus.
  CHECK(span_tiers(folder.model, folder.models(), false, true).performance);
  CHECK(make_span_prepare(nullptr, gpu::BackendInit{}, true, folder.model, folder.models(),
                          folder.root / "v.mkv")
            .refused);
}

TEST_CASE("un échec technique n'écrit rien et laisse l'état d'avant", "[app][span_derive]") {
  const Folder folder;
  SpanDeriveProgress progress;
  FakeBackend good;
  REQUIRE(derive_span_fp8(folder.plan(), source_of(twelve()), good.deriver(), progress, std::stop_token{}));
  const std::vector<std::string> before = folder.files();
  const auto content = [&](const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
  };
  const std::string package_before = content(folder.package());

  FakeBackend broken;
  broken.refuse = true;
  const auto memory =
      derive_span_fp8(folder.plan(), source_of(twelve()), broken.deriver(), progress, std::stop_token{});
  REQUIRE_FALSE(memory);
  CHECK(memory.error().code == RefusalCode::GpuMemoryExhausted);

  const SpanStillsSource hdr = [](const std::stop_token&, const std::function<void(float)>&) {
    return std::expected<std::vector<media::Still>, Refusal>(std::unexpected(
        Refusal{RefusalCode::FormatUnsupported, "vidéo HDR : la préparation se fait sur une vidéo SDR"}));
  };
  const auto refused = derive_span_fp8(folder.plan(), hdr, good.deriver(), progress, std::stop_token{});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().detail.starts_with("vidéo HDR"));

  std::stop_source stop;
  stop.request_stop();
  const auto cancelled =
      derive_span_fp8(folder.plan(), source_of(twelve()), good.deriver(), progress, stop.get_token());
  REQUIRE_FALSE(cancelled);
  CHECK(cancelled.error().code == RefusalCode::Cancelled);

  SpanDerivePlan not_b8f64 = folder.plan();
  not_b8f64.load_weights = [] {
    return std::expected<Package, Refusal>(
        std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                "conv_1 de forme (48, 12, 3, 3) : seul le SPAN à 64 canaux est pris"}));
  };
  const auto narrow =
      derive_span_fp8(not_b8f64, source_of(twelve()), good.deriver(), progress, std::stop_token{});
  REQUIRE_FALSE(narrow);
  CHECK(narrow.error().detail.find("64 canaux") != std::string::npos);

  CHECK(folder.files() == before);
  CHECK(content(folder.package()) == package_before);
  CHECK(find_span_variant(SpanVariantKind::Fp8Cutlass, folder.model, folder.models()).variant);

  SpanDerivePlan missing = folder.plan();
  missing.model = folder.root / "absent.onnx";
  missing.load_weights = {};
  CHECK_FALSE(derive_span_fp8(missing, source_of(twelve()), good.deriver(), progress, std::stop_token{}));
}

TEST_CASE("une écriture qui échoue ne laisse aucun fichier à moitié écrit", "[app][span_derive]") {
  const Folder folder;
  const std::vector<std::byte> bytes(64, std::byte{7});
  // Un état d'avant, entier.
  REQUIRE(commit_derivation(folder.package(), folder.receipt(), bytes, "passed = true\n", true));
  const std::vector<std::string> before = folder.files();
  REQUIRE(before.size() == 2);

  // Le reçu provisoire ne peut pas s'écrire : son nom est celui d'un dossier non vide.
  fs::path blocked = folder.receipt();
  blocked += ".part";
  fs::create_directories(blocked / "occupé");
  std::ofstream(blocked / "occupé" / "fichier") << "x";
  const std::vector<std::byte> other(64, std::byte{9});
  const auto committed =
      commit_derivation(folder.package(), folder.receipt(), other, "passed = false\n", true);
  REQUIRE_FALSE(committed);
  CHECK(committed.error().code == RefusalCode::IoFailed);
  // Le paquet d'avant est resté, avec son contenu ; aucun provisoire de plus.
  std::ifstream in(folder.package(), std::ios::binary);
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  CHECK(content == std::string(64, '\7'));
  for (const std::string& name : folder.files()) CHECK_FALSE(name.ends_with(".old"));
  CHECK(folder.files().size() == before.size() + 1);  // + le dossier qui bloquait
  CHECK(fs::exists(folder.package()));
  CHECK(fs::exists(folder.receipt()));

  // Un dossier qui n'existe pas et ne peut pas être créé (un fichier à sa place).
  const fs::path blocker = folder.root / "bloqueur";
  std::ofstream(blocker) << "x";
  CHECK_FALSE(
      commit_derivation(blocker / "modeles" / "a.span8", blocker / "modeles" / "a.toml", bytes, "x", true));
}

TEST_CASE("la préparation n'est offerte qu'avec un moteur qui la calcule et une vidéo ouverte",
          "[app][span_derive]") {
  const Folder folder;
  const fs::path video = folder.root / "episode.mkv";
  const gpu::BackendInit init;

  const SpanPrepare unsupported =
      make_span_prepare(nullptr, init, false, folder.model, folder.models(), video);
  CHECK_FALSE(unsupported.offered);
  CHECK(unsupported.note.find("RTX 40") != std::string::npos);
  CHECK_FALSE(unsupported.run);

  const SpanPrepare home = make_span_prepare(nullptr, init, true, folder.model, folder.models(), {});
  CHECK_FALSE(home.offered);
  CHECK(home.note.find("Ouvrez une vidéo") != std::string::npos);

  const SpanPrepare b5f48 = make_span_prepare(
      nullptr, init, true, folder.root / "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx",
      folder.models(), video);
  CHECK_FALSE(b5f48.offered);
  CHECK(b5f48.note.find("48 canaux") != std::string::npos);

  const SpanPrepare fresh = make_span_prepare(nullptr, init, true, folder.model, folder.models(), video);
  CHECK(fresh.offered);
  CHECK_FALSE(fresh.ready);
  CHECK(fresh.note.find("Pas encore préparé") != std::string::npos);
  CHECK(fresh.run);

  SpanDeriveProgress progress;
  const FakeBackend backend;
  REQUIRE(
      derive_span_fp8(folder.plan(), source_of(twelve()), backend.deriver(), progress, std::stop_token{}));
  const SpanPrepare done = make_span_prepare(nullptr, init, true, folder.model, folder.models(), video);
  CHECK(done.ready);
  CHECK(done.note.starts_with("Prêt : "));
}

TEST_CASE("le paquet et le reçu de l'outil Python sont ceux que le lecteur prend",
          "[.span_compat][app][span_derive]") {
  const fs::path model(AJP_SPAN_FP8_MODEL), package(AJP_SPAN_FP8_PYTHON_PACKAGE);
  if (!fs::is_regular_file(model) || !fs::is_regular_file(package))
    SKIP(
        "AJP_SPAN_FP8_MODEL et AJP_SPAN_FP8_PYTHON_PACKAGE (CMake) : le SPAN et le paquet de "
        "derive_span_fp8.py");
  fs::path receipt = package;
  receipt.replace_extension(".toml");
  REQUIRE(fs::is_regular_file(receipt));
  const fs::path dir = fs::temp_directory_path() / "ajp-span-python-compat";
  fs::remove_all(dir);
  fs::create_directories(dir);
  fs::copy_file(package, dir / package.filename());
  fs::copy_file(receipt, dir / receipt.filename());
  const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, model, dir);
  INFO(found.note);
  REQUIRE(found.variant);
  CHECK(found.variant->onnx == dir / package.filename());
  CHECK(found.variant->psnr_mean_db >= 42.0);
  // Et l'inverse : le nom, le format et les clés que l'écriture native produit sont ceux de l'outil.
  CHECK(span_variant_path(SpanVariantKind::Fp8Cutlass, model, dir, prepare::sha256_file(model).value())
            .filename() == package.filename());
  fs::remove_all(dir);
}

TEST_CASE("la préparation de bout en bout, par le vrai backend, sur une vraie vidéo",
          "[.span_prepare][app][span_derive]") {
  const fs::path backend_path(AJP_BACKEND_TENSORRT), model(AJP_SPAN_FP8_MODEL), video(AJP_SPAN_DERIVE_VIDEO);
  if (backend_path.empty() || !fs::is_regular_file(backend_path) || !fs::is_regular_file(model) ||
      !fs::is_regular_file(video))
    SKIP("AJP_SPAN_FP8_MODEL, AJP_SPAN_DERIVE_VIDEO (CMake) et le backend TensorRT");
  const fs::path candidates[] = {backend_path};
  auto library = BackendLibrary::open(candidates);
  INFO((library ? std::string() : library.error().detail));
  REQUIRE(library);
  const fs::path work = fs::temp_directory_path() / "ajp-span-prepare-test";
  fs::create_directories(work / "engines");
  const gpu::BackendInit init{.cache_dir = work / "engines"};
  auto probe = BackendLibrary::create(*library, init);
  REQUIRE(probe);
  if (!(*probe)->capabilities().span_fp8_cutlass) SKIP("appareil sans les noyaux FP8 CUTLASS (sm 8.9)");
  probe->reset();
  const fs::path out =
      std::string_view(AJP_SPAN_DERIVE_OUT).empty() ? work / "modeles" : fs::path(AJP_SPAN_DERIVE_OUT);
  const SpanPrepare prepare = make_span_prepare(*library, init, true, model, out, video);
  REQUIRE(prepare.offered);
  SpanDeriveProgress progress;
  const auto begin = std::chrono::steady_clock::now();
  const auto result = prepare.run(progress, std::stop_token{});
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  INFO((result ? result.value() : result.error().detail));
  WARN((result ? result.value() : result.error().detail) << " (" << seconds << " s)");
  REQUIRE(result);
  CHECK(progress.get().first == 1.0F);
  const SpanInt8Lookup found = find_span_variant(SpanVariantKind::Fp8Cutlass, model, out);
  INFO(found.note);
  CHECK(found.variant);
  CHECK(span_tiers(model, out, false, true).performance);
}

TEST_CASE("la préparation à la main ne part qu'une fois à la fois, et seulement si elle est offerte",
          "[app][span_derive]") {
  PrepareNote note;
  int started = 0;
  SpanPrepare prepare;
  prepare.offered = true;
  prepare.run = [&started](SpanDeriveProgress&,
                           const std::stop_token&) -> std::expected<std::string, Refusal> {
    ++started;
    return std::string("fait");
  };
  SpanPrepare not_offered = prepare;
  not_offered.offered = false;
  CHECK_FALSE(start_span_prepare(not_offered, note));  // pas de vidéo ou de moteur
  CHECK(started == 0);

  REQUIRE(start_span_prepare(prepare, note));
  REQUIRE(note.pending);
  CHECK_FALSE(start_span_prepare(prepare, note));  // une seule à la fois
  CHECK(note.pending->get().value() == "fait");
  CHECK(started == 1);
}

}  // namespace ajp::app::play
