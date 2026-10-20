#include "core/model_index.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;
namespace fs = std::filesystem;

namespace {

// Dossier de travail propre, sous le dossier de build où CTest lance les tests.
fs::path scratch(const std::string& name) {
  const fs::path dir = fs::current_path() / "test-tmp" / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void write_file(const fs::path& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  out << bytes;
}

const std::string kSha = std::string(64, 'a');

std::string model(const std::string& fields) {
  return "version = 1\n[[model]]\n" + fields + "\n";
}

std::string valid_model_fields() {
  std::string text =
      "id = \"2x_AnimeJaNai_HD_V3.1_Balanced\"\nurl = \"https://models.example/hd.onnx\"\n"
      "sha256 = \"" +
      kSha + "\"\nsize = 1024\nlicense = \"CC-BY-NC-SA-4.0\"\nprofiles = [1, 2]\n";
  return text;
}

RefusalCode lock_refusal(const std::string& text) {
  auto lock = parse_models_lock(text);
  REQUIRE_FALSE(lock);
  return lock.error().code;
}

RefusalCode index_refusal(const std::string& text) {
  auto index = parse_engine_index(text);
  REQUIRE_FALSE(index);
  return index.error().code;
}

}  // namespace

TEST_CASE("SHA-256 : vecteurs officiels de FIPS 180-2 et du NIST", "[model_index]") {
  CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(
      sha256_hex(
          "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmn"
          "opqrstnopqrstu") == "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
  CHECK(sha256_hex(std::string(1'000'000, 'a')) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("SHA-256 : le découpage des données ne change pas le hachage", "[model_index]") {
  std::string data;
  for (int i = 0; i < 200; ++i) data += static_cast<char>(i * 7);
  const std::string whole = sha256_hex(data);
  for (std::size_t cut = 0; cut <= data.size(); cut += 13) {
    Sha256 hash;
    hash.update(std::string_view(data).substr(0, cut));
    hash.update(std::string_view(data).substr(cut));
    CHECK(to_hex(hash.finish()) == whole);
  }
  // Autour des bornes de bourrage : 55, 56, 63 et 64 octets.
  for (const std::size_t size : {55U, 56U, 63U, 64U, 65U}) {
    Sha256 bytewise;
    for (std::size_t i = 0; i < size; ++i) bytewise.update(std::string_view(data).substr(i, 1));
    CHECK(to_hex(bytewise.finish()) == sha256_hex(std::string_view(data).substr(0, size)));
  }
  Sha256 reused;
  reused.update("x");
  (void)reused.finish();
  reused.update("abc");
  CHECK(to_hex(reused.finish()) == sha256_hex("abc"));
}

TEST_CASE("Téléchargement : un fichier se vérifie par sa taille et son SHA-256", "[model_index]") {
  const fs::path dir = scratch("verify");
  const std::string content(100'000, 'z');
  write_file(dir / "model.onnx", content);
  const std::string sha = sha256_hex(content);

  auto file_sha = sha256_file(dir / "model.onnx");
  REQUIRE(file_sha);
  CHECK(*file_sha == sha);
  CHECK(verify_file(dir / "model.onnx", content.size(), sha));

  auto wrong_size = verify_file(dir / "model.onnx", content.size() + 1, sha);
  REQUIRE_FALSE(wrong_size);
  CHECK(wrong_size.error().code == RefusalCode::HashMismatch);
  auto wrong_hash = verify_file(dir / "model.onnx", content.size(), sha256_hex("autre"));
  REQUIRE_FALSE(wrong_hash);
  CHECK(wrong_hash.error().code == RefusalCode::HashMismatch);
  auto missing = verify_file(dir / "absent.onnx", 1, sha);
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::FileUnreadable);
}

TEST_CASE("models.lock : chaque modèle porte url, hachage, taille, licence et profils", "[model_index]") {
  auto lock = parse_models_lock(model(valid_model_fields()) +
                                "[[model]]\nid = \"rife-4.26\"\nurl = \"https://models.example/rife.onnx\"\n"
                                "sha256 = \"" +
                                std::string(64, 'b') + "\"\nsize = 9\nlicense = \"MIT\"\nprofiles = [2]\n");
  REQUIRE(lock);
  REQUIRE(lock->size() == 2);
  const ModelEntry& first = lock->front();
  CHECK(first.id == "2x_AnimeJaNai_HD_V3.1_Balanced");
  CHECK(first.size == 1024);
  CHECK(first.license == "CC-BY-NC-SA-4.0");
  CHECK(first.profiles == std::vector<int>{1, 2});

  CHECK(find_model(*lock, "rife-4.26") == &(*lock)[1]);
  CHECK(find_model(*lock, "absent") == nullptr);
  CHECK(models_for_profile(*lock, 2).size() == 2);
  CHECK(models_for_profile(*lock, 1).size() == 1);
  CHECK(models_for_profile(*lock, 3).empty());
}

TEST_CASE("models.lock : une entrée fausse est un refus nommé", "[model_index]") {
  CHECK(lock_refusal("version = 2\n") == RefusalCode::FormatVersion);
  CHECK(lock_refusal("[[model]]\n") == RefusalCode::FieldMissing);
  CHECK(lock_refusal("version = 1\n[[model]\n") == RefusalCode::FileUnreadable);
  CHECK(lock_refusal("version = 1\nmodel = 3\n") == RefusalCode::FieldType);

  const auto with = [](const std::string& key, const std::string& value) {
    std::string fields = valid_model_fields();
    const std::size_t start = fields.find(key + " = ");
    const std::size_t end = fields.find('\n', start);
    return model(fields.replace(start, end - start, key + " = " + value));
  };
  CHECK(lock_refusal(with("url", "\"http://models.example/hd.onnx\"")) == RefusalCode::FieldOutOfRange);
  CHECK(lock_refusal(with("sha256", "\"ABC\"")) == RefusalCode::FieldType);
  CHECK(lock_refusal(with("sha256", "\"" + std::string(64, 'G') + "\"")) == RefusalCode::FieldType);
  CHECK(lock_refusal(with("size", "0")) == RefusalCode::FieldOutOfRange);
  CHECK(lock_refusal(with("size", "\"1024\"")) == RefusalCode::FieldType);
  CHECK(lock_refusal(with("id", "\"../evil\"")) == RefusalCode::FieldType);
  CHECK(lock_refusal(with("id", "\"a/b\"")) == RefusalCode::FieldType);
  CHECK(lock_refusal(with("license", "\"\"")) == RefusalCode::FieldMissing);
  CHECK(lock_refusal(with("profiles", "[10]")) == RefusalCode::FieldOutOfRange);
  CHECK(lock_refusal(with("profiles", "[1, 1]")) == RefusalCode::FieldOutOfRange);
  CHECK(lock_refusal(with("profiles", "[\"1\"]")) == RefusalCode::FieldType);
  CHECK(lock_refusal(model(valid_model_fields()) + "[[model]]\n" + valid_model_fields()) ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("Index des moteurs : un paquet par moteur et par plateforme", "[model_index]") {
  const std::string package = "url = \"https://engines.example/trt.zip\"\nsha256 = \"" + kSha +
                              "\"\nsize = 2048\nversion = \"11.3.0\"\n";
  auto index =
      parse_engine_index("version = 1\n[[engine]]\nid = \"tensorrt\"\nplatform = \"windows\"\n" + package +
                         "[[engine]]\nid = \"tensorrt\"\nplatform = \"linux\"\n" + package);
  REQUIRE(index);
  REQUIRE(index->size() == 2);
  CHECK((*index)[0].engine == EngineId::TensorRt);
  CHECK((*index)[0].version == "11.3.0");
  CHECK(find_package(*index, EngineId::TensorRt, Os::Linux) == &(*index)[1]);
  CHECK(find_package(*index, EngineId::TensorRt, Os::MacOs) == nullptr);

  CHECK(index_refusal("version = 1\n[[engine]]\nid = \"cuda\"\nplatform = \"linux\"\n" + package) ==
        RefusalCode::FieldOutOfRange);
  CHECK(index_refusal("version = 1\n[[engine]]\nid = \"migraphx\"\nplatform = \"beos\"\n" + package) ==
        RefusalCode::FieldOutOfRange);
  CHECK(index_refusal("version = 1\n[[engine]]\nid = \"migraphx\"\nplatform = \"linux\"\n" + package +
                      "[[engine]]\nid = \"migraphx\"\nplatform = \"linux\"\n" + package) ==
        RefusalCode::FieldOutOfRange);
  CHECK(index_refusal("version = 1\n[[engine]]\nid = \"migraphx\"\nplatform = \"linux\"\n" +
                      std::string(package).replace(package.find("11.3.0"), 6, "../11")) ==
        RefusalCode::FieldType);
  // Windows retire le point final d'un nom de dossier : « 11.3. » y deviendrait « 11.3 ».
  CHECK(index_refusal("version = 1\n[[engine]]\nid = \"migraphx\"\nplatform = \"linux\"\n" +
                      std::string(package).replace(package.find("11.3.0"), 6, "11.3.")) ==
        RefusalCode::FieldType);
}

TEST_CASE("Index des moteurs : seuls les moteurs téléchargés y figurent", "[model_index]") {
  // Vulkan et Metal sont livrés ; les SDK RTX Video et Maxine restent fournis par l'utilisateur.
  const std::string package = "platform = \"windows\"\nurl = \"https://engines.example/x.zip\"\nsha256 = \"" +
                              kSha + "\"\nsize = 1\nversion = \"1\"\n";
  for (const char* id : {"vulkan", "metal", "rtx_video_sdk", "maxine_vfx"}) {
    INFO(id);
    CHECK(index_refusal(std::format("version = 1\n[[engine]]\nid = \"{}\"\n{}", id, package)) ==
          RefusalCode::FieldOutOfRange);
  }
  for (const char* id : {"tensorrt", "migraphx", "openvino", "directml"}) {
    INFO(id);
    CHECK(parse_engine_index(std::format("version = 1\n[[engine]]\nid = \"{}\"\n{}", id, package)));
  }
}

TEST_CASE("Noms des moteurs et des OS : stables et réciproques", "[model_index]") {
  for (const EngineId id : {EngineId::Vulkan, EngineId::TensorRt, EngineId::RtxVideoSdk, EngineId::MaxineVfx,
                            EngineId::MiGraphX, EngineId::OpenVino, EngineId::DirectMl, EngineId::Metal}) {
    CHECK(parse_engine(engine_name(id)) == id);
  }
  CHECK(engine_name(EngineId::TensorRt) == "tensorrt");
  CHECK_FALSE(parse_engine("TensorRT"));
  for (const Os os : {Os::Windows, Os::Linux, Os::MacOs}) CHECK(parse_os(os_name(os)) == os);
}

TEST_CASE("Installation : chemins sous le dossier de données passé par app", "[model_index]") {
  const fs::path data = fs::path("donnees") / "ajp";
  const EnginePackage trt{.engine = EngineId::TensorRt, .version = "11.3.0"};
  const ModelEntry hd{.id = "2x_HD"};

  const InstallLayout windows = install_layout(Os::Windows, data);
  CHECK(model_path(windows, hd) == data / "models" / "2x_HD.onnx");
  CHECK(engine_dir(windows, trt) == data / "backends" / "tensorrt" / "11.3.0");
  CHECK(backend_library(windows, trt).filename() == "ajp_backend_tensorrt.dll");
  CHECK(backend_library(install_layout(Os::Linux, data), trt).filename() == "ajp_backend_tensorrt.so");
  CHECK(backend_library(install_layout(Os::MacOs, data), trt).filename() == "ajp_backend_tensorrt.so");
  CHECK(download_path(windows, kSha) == data / "downloads" / (kSha + ".part"));
}

TEST_CASE("[maxine_vfx] : la ressource NGC du SDK de RTX HDR", "[model_index]") {
  const std::string table =
      "version = 1\n[maxine_vfx]\norg = \"nvidia\"\nteam = \"maxine\"\nresource = \"vfx_sdk_core\"\n"
      "version = \"1.3.0.0_linux\"\nfile = \"VFXSDK_linux_1.3.0.0.tgz\"\nfeature = \"nvvfxtruehdr\"\n"
      "license = \"https://example.org/eula\"\n";
  auto source = parse_maxine_source(table);
  REQUIRE(source);
  REQUIRE(source->has_value());
  CHECK((*source)->file == "VFXSDK_linux_1.3.0.0.tgz");
  CHECK((*source)->feature == "nvvfxtruehdr");

  auto none = parse_maxine_source("version = 1\n");
  REQUIRE(none);
  CHECK_FALSE(none->has_value());

  std::string bad = table;
  bad.replace(bad.find("vfx_sdk_core"), 12, "../etc");
  CHECK_FALSE(parse_maxine_source(bad));
  std::string http = table;
  http.replace(http.find("https://"), 8, "http://");
  CHECK_FALSE(parse_maxine_source(http));
}
