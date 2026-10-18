// Parité des clés avec le fork : chaque valeur attendue a été calculée par les
// en-têtes de animejanai-inference-rtx@118c272 (content_identity.h,
// program_identity.h) et par engine_name de model_identity.cpp:139-155,
// compilés tels quels. Un écart invaliderait les moteurs déjà construits.
#include "prepare/identity.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "prepare/sha256.h"
#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;

namespace {

const std::string kRuntime = "11.3.0-headers-build99-cuda13000-driver13000";
const std::string kDevice = "NVIDIA GeForce RTX 4070 Ti-sm8.9-00112233445566778899aabbccddeeff";

}  // namespace

TEST_CASE("Sha256 : vecteurs FIPS 180-4", "[prepare][identity]") {
  CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(sha256_hex(std::string(1000000, 'a')) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("Sha256 : par morceaux comme d'un bloc, hex() ne consomme pas l'état", "[prepare][identity]") {
  const std::string text(1000, 'x');
  Sha256 pieces;
  for (std::size_t i = 0; i < text.size(); i += 7) pieces.update(std::string_view(text).substr(i, 7));
  CHECK(pieces.hex() == sha256_hex(text));
  CHECK(pieces.hex() == sha256_hex(text));
}

TEST_CASE("Sha256 : encadrement text et integer du fork", "[prepare][identity]") {
  Sha256 hash;
  hash.text("onnx-source-v2");
  hash.integer(42);
  CHECK(hash.hex() == "8e80489266fcf148e064fc5af8cbde939a362595c1cffb5e0cc8dbf3c0e56f6f");
}

TEST_CASE("sha256_file et is_sha256_hex", "[prepare][identity]") {
  const test::TempDir dir;
  test::write_file(dir / "f.bin", "abc");
  CHECK(sha256_file(dir / "f.bin").value() == sha256_hex("abc"));
  CHECK(test::refusal(sha256_file(dir / "absent")) == RefusalCode::CacheIo);
  CHECK(is_sha256_hex(sha256_hex("abc")));
  CHECK_FALSE(is_sha256_hex("ABC"));
  CHECK_FALSE(is_sha256_hex(std::string(64, 'A')));
}

TEST_CASE("files_digest : noms logiques triés, contenu et taille encadrés", "[prepare][identity]") {
  const test::TempDir dir;
  test::write_file(dir / "weights/a.bin", "poids A");
  test::write_file(dir / "b.bin", "poids B, plus long");
  const auto digest = files_digest("domaine", {{"z", dir / "b.bin"}, {"a", dir / "weights/a.bin"}});
  CHECK(digest.value() == "1b3981abf8872d42f4798c4935a445ef524d2590cf18e8dd7c236024b65052e0");
  CHECK(test::refusal(files_digest("domaine", {{"a", dir / "b.bin"}, {"a", dir / "b.bin"}})) ==
        RefusalCode::FieldOutOfRange);
  CHECK(test::refusal(files_digest("domaine", {{"a", dir / "absent"}})) == RefusalCode::ModelUnreadable);
}

TEST_CASE("onnx_source_key : graphe et poids externes, comme model_identity.cpp:20-27",
          "[prepare][identity]") {
  const test::TempDir dir;
  test::write_file(dir / "weights/a.bin", "poids A");
  test::write_file(dir / "b.bin", "poids B, plus long");
  const auto graph = test::as_bytes("graphe ONNX factice");
  CHECK(onnx_source_key(graph, dir.path(), {}).value() ==
        "b26c0113f89bb0cad44f709bb3821c7fc7b25959f0c41be8e947cf9e54ebd34c");
  const std::vector<std::string> locations{"weights/a.bin", "b.bin"};
  CHECK(onnx_source_key(graph, dir.path(), locations).value() ==
        "28c16f64c24f15b28e4dd80b36a6540b803b81b9c2e91aeac4322b05752d4ab2");
}

TEST_CASE("engine_name : nom du fork pour SPAN et RIFE", "[prepare][identity]") {
  const auto span =
      engine_name({.content = std::string(64, 'a'),
                   .settings = "--builderOptimizationLevel=5 --optShapes=input:1x3x1080x1920 --skipInference",
                   .runtime = kRuntime,
                   .device = kDevice});
  CHECK(span.value() == "aji-v3-91b1d7d88200f111d2f0705397f1d7ee071edd02e4302dc249e0709460c43c47.engine");
  const auto rife = engine_name(
      {.content = std::string(64, '0'),
       .settings = "--stronglyTyped --optShapes=input:1x11x1088x1920 --inputIOFormats=fp16:chw "
                   "--outputIOFormats=fp16:chw --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference",
       .runtime = kRuntime,
       .device = kDevice,
       .build_environment = "LD_LIBRARY_PATH=/opt/trt"});
  CHECK(rife.value() == "aji-v3-ec15ddeababeb8376542b59b9bc0d910b659d6930f56e3065f83cbab9223dfc0.engine");
  CHECK(test::refusal(engine_name({.content = "pas un sha", .runtime = kRuntime, .device = kDevice})) ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("program_key : liaisons spécialisées triées par nom", "[prepare][identity]") {
  Program p{.format = ProgramFormat::OnnxExport,
            .source_scheme = "s",
            .source_digest = std::string(64, 'b'),
            .converter = "c",
            .converter_version = 2,
            .options = "o",
            .io_policy = IoPolicy::Specialized,
            .bindings = {{"z", "f16", {1, -1, 3}}, {"a", "f32", {2}}}};
  CHECK(program_key(p).value() == "a36e8d5e7d4e9a9583a8dcfde9723dbffcff3fba89da503040b6b740b69f0df2");
}

TEST_CASE("program_key : refus des programmes incomplets", "[prepare][identity]") {
  const Program base{.format = ProgramFormat::TensorRt,
                     .source_scheme = "onnx-source-v2",
                     .source_digest = std::string(64, 'c'),
                     .converter = "onnx-to-tensorrt",
                     .converter_version = 1,
                     .compiler = "trtexec",
                     .runtime = "r",
                     .device = "d"};
  REQUIRE(program_key(base));
  auto without_device = base;
  without_device.device.clear();
  CHECK(test::refusal(program_key(without_device)) == RefusalCode::FieldMissing);
  auto portable = base;
  portable.format = ProgramFormat::OnnxExport;
  CHECK(test::refusal(program_key(portable)) == RefusalCode::FieldOutOfRange);
  auto unversioned = base;
  unversioned.converter_version = 0;
  CHECK(test::refusal(program_key(unversioned)) == RefusalCode::FieldMissing);
  auto stray = base;
  stray.bindings = {{"x", "f16", {1}}};
  CHECK(test::refusal(program_key(stray)) == RefusalCode::FieldOutOfRange);
  auto twice = base;
  twice.io_policy = IoPolicy::Specialized;
  twice.bindings = {{"x", "f16", {1}}, {"x", "f16", {1}}};
  CHECK(test::refusal(program_key(twice)) == RefusalCode::FieldOutOfRange);
  auto zero = base;
  zero.io_policy = IoPolicy::Specialized;
  zero.bindings = {{"x", "f16", {0}}};
  CHECK(test::refusal(program_key(zero)) == RefusalCode::FieldOutOfRange);
}

TEST_CASE("Jetons de version TensorRT et d'appareil CUDA (aji_trt.cpp:542-582)", "[prepare][identity]") {
  TrtRuntime runtime = decode_infer_lib_version(110300);
  CHECK(runtime.major == 11);
  CHECK(runtime.minor == 3);
  CHECK(runtime.patch == 0);
  runtime.headers_build = 99;
  runtime.cuda_runtime = 13000;
  runtime.cuda_driver = 13000;
  CHECK(runtime.token() == kRuntime);
  const TrtRuntime legacy = decode_infer_lib_version(8601);
  CHECK((legacy.major == 8 && legacy.minor == 6 && legacy.patch == 1));
  const CudaDevice device{.name = "NVIDIA GeForce RTX 4070 Ti",
                          .sm_major = 8,
                          .sm_minor = 9,
                          .uuid = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
                                   0xcc, 0xdd, 0xee, 0xff}};
  CHECK(device.token() == kDevice);
}
