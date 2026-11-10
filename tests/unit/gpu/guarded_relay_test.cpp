// A21-04 : le mandataire GuardedKernelOps relaie toutes les virtuelles de KernelOps. Une virtuelle ajoutée à
// KernelOps a une version par défaut qui refuse ; oubliée dans le mandataire, elle compilerait et rendrait
// toujours ce refus, sous TensorRT comme sous Vulkan. Le test lit les deux en-têtes et compare les noms.
#include <cstddef>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace {

std::string read_source(const std::string& relative) {
  std::ifstream in(std::string(AJP_GPU_SOURCE_DIR) + "/" + relative, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

// Le texte entre `from` et `to` (exclus) ; vide si l'un manque.
std::string between(const std::string& text, const std::string& from, const std::string& to) {
  const std::size_t start = text.find(from);
  if (start == std::string::npos) return {};
  const std::size_t end = text.find(to, start + from.size());
  if (end == std::string::npos) return {};
  return text.substr(start, end - start);
}

std::set<std::string> matches(const std::string& text, const std::regex& pattern) {
  std::set<std::string> out;
  for (auto it = std::sregex_iterator(text.begin(), text.end(), pattern); it != std::sregex_iterator(); ++it)
    out.insert((*it)[1].str());
  return out;
}

}  // namespace

TEST_CASE("GuardedKernelOps relaie chaque virtuelle de KernelOps (A21-04)", "[gpu][guarded]") {
  const std::string interface = between(read_source("kernel_ops.h"), "class KernelOps {", "\n};");
  const std::string relay =
      between(read_source("guarded_backend.h"), "class GuardedKernelOps", "class GuardedBackend");
  REQUIRE_FALSE(interface.empty());
  REQUIRE_FALSE(relay.empty());
  std::set<std::string> virtuals = matches(interface, std::regex(R"(virtual\s+[^;{(~]*?\b(\w+)\s*\()"));
  virtuals.erase("KernelOps");  // le destructeur
  const std::set<std::string> relayed =
      matches(relay, std::regex(R"(\b(\w+)\s*\([^()]*\)\s*(?:const\s*)?override)"));
  // Les 27 d'aujourd'hui ; le compte borne la lecture des en-têtes (une expression qui ne trouverait rien
  // passerait sinon).
  CHECK(virtuals.size() >= 27);
  CHECK(relayed == virtuals);
}
