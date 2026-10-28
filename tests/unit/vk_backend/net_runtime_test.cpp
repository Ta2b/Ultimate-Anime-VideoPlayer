// Artefacts des runtimes confiés (net_runtime.h) : publication par renommage depuis un dossier de
// travail, noms d'E/S en dernier, cache compilé en dossier remplacé, effacement des voisins.
// Rien ici ne demande d'appareil : seulement le système de fichiers.
#include "gpu/vulkan/backend/net_runtime.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::gpu::vk;

namespace {

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

TEST_CASE("Artefacts : publiés depuis le dossier de travail, voisins compris", "[vk_backend][runtime]") {
  const auto root = std::filesystem::temp_directory_path() / "ajp-net-runtime-test";
  std::filesystem::remove_all(root);
  const auto work = root / ".construction-1-2";
  const auto staged = work / "span-0123.ovblob";
  const auto final_path = root / "span-0123.ovblob";
  const std::array<std::string, 1> inputs{"input"};
  const std::array<std::string, 2> outputs{"output", "flow"};

  write_text(staged, "réseau");
  write_text(work / "span-0123.ovblob.device", "GPU.0");
  write_text(work / "span-0123.ovblob.mlcache/compiled/model.bin", "neuf");
  write_text(work / "span-0123.ovblobx", "autre réseau");  // même début, autre nom : reste là
  REQUIRE(net::write_io_names(staged, inputs, outputs));
  // Un ancien cache compilé, plein, sous le nom final : remplacé, pas fusionné.
  write_text(root / "span-0123.ovblob.mlcache/compiled/ancien.bin", "ancien");

  REQUIRE(net::publish_artifact(staged, final_path));
  CHECK(read_text(final_path) == "réseau");
  CHECK(read_text(root / "span-0123.ovblob.device") == "GPU.0");
  CHECK(read_text(root / "span-0123.ovblob.mlcache/compiled/model.bin") == "neuf");
  CHECK_FALSE(std::filesystem::exists(root / "span-0123.ovblob.mlcache/compiled/ancien.bin"));
  CHECK(std::filesystem::exists(work / "span-0123.ovblobx"));
  const auto names = net::read_io_names(final_path);
  REQUIRE(names);
  CHECK(names->inputs == std::vector<std::string>{"input"});
  CHECK(names->outputs == std::vector<std::string>{"output", "flow"});

  net::discard_artifact(final_path);
  CHECK_FALSE(std::filesystem::exists(final_path));
  CHECK_FALSE(std::filesystem::exists(root / "span-0123.ovblob.device"));
  CHECK_FALSE(std::filesystem::exists(root / "span-0123.ovblob.mlcache"));
  CHECK_FALSE(net::read_io_names(final_path));
  std::filesystem::remove_all(root);
}

TEST_CASE("Artefacts : rien n'est publié sans l'artefact construit", "[vk_backend][runtime]") {
  const auto root = std::filesystem::temp_directory_path() / "ajp-net-runtime-absent";
  std::filesystem::remove_all(root);
  const auto staged = root / "work" / "rife.mxr";
  const std::array<std::string, 1> names{"x"};
  std::filesystem::create_directories(staged.parent_path());
  REQUIRE(net::write_io_names(staged, names, names));
  const auto published = net::publish_artifact(staged, root / "rife.mxr");
  REQUIRE_FALSE(published);
  CHECK(published.error().code == ajp::RefusalCode::CacheIo);
  CHECK_FALSE(std::filesystem::exists(root / "rife.mxr.io"));
  std::filesystem::remove_all(root);
}

}  // namespace
