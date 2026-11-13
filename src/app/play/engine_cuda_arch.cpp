// L'architecture CUDA du GPU (« sm89 ») : lue de nvidia-smi, sinon déduite du nom de la carte. Elle choisit
// les fichiers tiers de NVIDIA à tirer (engine_install.cpp) et dit si le FP8 est présumé.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "app/play/engine_install.h"
#include "app/play/engine_tools.h"
#include "core/text.h"

namespace ajp::app::play {

namespace {

// Un mot entier : « a100 » n'est pas dans « rtx a1000 » (une RTX Ampere sm86, pas une A100 sm80 : B06-06).
bool contains_word(std::string_view text, std::string_view word) {
  for (std::size_t at = text.find(word); at != std::string_view::npos; at = text.find(word, at + 1)) {
    const bool before = at == 0 || std::isalnum(static_cast<unsigned char>(text[at - 1])) == 0;
    const std::size_t end = at + word.size();
    const bool after = end >= text.size() || std::isalnum(static_cast<unsigned char>(text[end])) == 0;
    if (before && after) return true;
  }
  return false;
}

}  // namespace

std::string cuda_arch_from_name(std::string_view gpu_name) {
  std::string name(gpu_name);
  name = core::ascii_lower(name);
  const auto has = [&](std::string_view part) { return name.find(part) != std::string::npos; };
  const auto has_word = [&](std::string_view word) { return contains_word(name, word); };
  const auto series = [&](char digit) {  // « rtx 4050 » : la série à deux chiffres, ici 40
    const std::size_t at = name.find("rtx ");
    return at != std::string::npos && at + 5 < name.size() && name[at + 4] == digit && name[at + 5] >= '0' &&
           name[at + 5] <= '9';
  };
  if (has("h100") || has("h200")) return "sm90";
  // Deux exceptions à la règle par série (A37-03) : la RTX 2050 portable est une Ampere (GA107), la
  // RTX 3050 A Laptop une Ada (AD106).
  if (has("rtx 2050")) return "sm86";
  if (has("3050 a ")) return "sm89";
  if (has_word("a100")) return "sm80";
  if (has("blackwell")) return "sm120";
  // Les cartes pro : « RTX 5000 Ada Generation » est une Ada, « Quadro RTX 4000 » une Turing (P3.16).
  if (has(" ada")) return "sm89";
  if (has("quadro rtx")) return "sm75";
  if (series('5')) return "sm120";
  if (has("l40") || series('4')) return "sm89";
  if (series('3') || has("rtx a")) return "sm86";
  if (series('2') || has("gtx 16") || has("titan rtx")) return "sm75";
  return {};
}

bool fp8_presumed(std::string_view gpu_name) {
  // gpu::fp8_allowed : Ada et Hopper ; une RTX 50 (sm120) n'a pas de FP8 tant qu'aucune mesure ne le
  // justifie.
  const std::string arch = cuda_arch_from_name(gpu_name);
  return arch == "sm89" || arch == "sm90";
}

bool span_fp8_presumed(std::string_view gpu_name) {
  return cuda_arch_from_name(gpu_name) == "sm89";
}

bool span_fp8_default(std::string_view gpu_name, bool tensorrt) {
#if defined(__linux__)
  return tensorrt && span_fp8_presumed(gpu_name);
#else
  static_cast<void>(gpu_name);
  static_cast<void>(tensorrt);
  return false;
#endif
}

std::string effective_cuda_arch(std::string_view detected, std::string_view gpu_name) {
  if (!detected.empty()) return std::string(detected);
  if (std::string by_name = cuda_arch_from_name(gpu_name); !by_name.empty()) return by_name;
  return "ptx";  // inconnue : jamais « toutes les architectures » (2 Go tirés, 3,3 Go écrits)
}

namespace {

// Chaque carte : son nom et son architecture (« NVIDIA GeForce RTX 4070, 8.9 » : sm89).
std::vector<std::pair<std::string, std::string>> run_nvidia_smi() {
  std::vector<std::pair<std::string, std::string>> cards;
  const auto smi = system_tool("nvidia-smi");
  if (!smi) return cards;
  auto done = run_tool(
      {.program = *smi,
       .args = {"--query-gpu=name,compute_cap", "--format=csv,noheader"},
       .timeout = std::chrono::seconds(10)},
      [&](std::string_view line) {
        const std::size_t comma = line.rfind(',');
        if (comma == std::string_view::npos) return;
        std::string_view cap = line.substr(comma + 1);
        while (!cap.empty() && (cap.front() == ' ' || cap.back() == ' ' || cap.back() == '\r'))
          cap = cap.front() == ' ' ? cap.substr(1) : cap.substr(0, cap.size() - 1);
        const std::size_t dot = cap.find('.');
        if (dot == std::string_view::npos || dot == 0 || dot + 1 >= cap.size()) return;
        const auto digit = [](char c) { return c >= '0' && c <= '9'; };
        if (!digit(cap[dot - 1]) || !digit(cap[dot + 1])) return;
        // sm100 et plus : « 10.0 » donne sm100 ; la majeure a deux chiffres.
        std::string arch = std::format("sm{}{}", cap.substr(0, dot), cap[dot + 1]);
        cards.emplace_back(std::string(line.substr(0, comma)), std::move(arch));
      },
      {}, "nvidia-smi");
  if (!done) cards.clear();
  return cards;
}

}  // namespace

std::string detect_cuda_arch(std::string_view gpu_name) {
  // Une fois par processus (P2.12) : la carte ne change pas, et nvidia-smi prend jusqu'à deux secondes. Un
  // résultat vide (échec, délai de 10 s juste après l'ouverture de session, pilote pas encore prêt) n'est pas
  // retenu pour toujours : il se retente, deux essais au plus (B06-07).
  static std::mutex mutex;
  static std::optional<std::vector<std::pair<std::string, std::string>>> known;
  static int attempts = 0;
  const std::scoped_lock lock(mutex);
  if (!known || (known->empty() && attempts < 2)) {
    known = run_nvidia_smi();
    ++attempts;
  }
  // La carte de rendu (P3.16) : sur une machine à deux cartes NVIDIA, la première n'est pas forcément elle.
  for (const auto& [name, arch] : *known)
    if (!gpu_name.empty() && name == gpu_name) return arch;
  // Un nom demandé que nvidia-smi n'a pas écrit pareil (espace, suffixe, carte pro) : avec plusieurs cartes,
  // on ne devine pas, l'architecture se déduit du nom (effective_cuda_arch) ; avec une seule, c'est elle
  // (B37-05).
  if (!gpu_name.empty() && known->size() > 1) return {};
  return known->empty() ? std::string() : known->front().second;
}

}  // namespace ajp::app::play
