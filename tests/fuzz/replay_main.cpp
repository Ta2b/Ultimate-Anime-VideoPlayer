// Pilote de rejeu des cibles de fuzz, sans libFuzzer : chaque fichier du corpus tel quel, puis des
// mutations déterministes (graine fixe) de ces fichiers, passés à la cible nommée. Sous les
// préréglages ASan/UBSan, c'est un fuzz court à chaque exécution de ctest ; partout ailleurs, la
// vérification des propriétés de chaque cible. Une entrée qui arrête le processus (rapport d'un
// sanitizer, propriété fausse, signal) est nommée par son numéro, que --only rejoue seul (et que
// --save écrit dans un fichier au lieu de la passer à la cible, pour l'ajouter au corpus).
//
//   ajp_fuzz_replay <cible> <corpus> [mutations] [--only <numéro> [--save <fichier>]]
#include <algorithm>
#include <array>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include "fuzz_targets.h"

namespace {

using Target = int (*)(const std::uint8_t*, std::size_t);
struct Named {
  std::string_view name;
  Target run;
};
constexpr std::array kTargets{
    Named{"settings", &ajp::fuzz::settings}, Named{"play_settings", &ajp::fuzz::play_settings},
    Named{"edid", &ajp::fuzz::edid},         Named{"build_log", &ajp::fuzz::build_log},
    Named{"resume", &ajp::fuzz::resume},     Named{"osd_script", &ajp::fuzz::osd_script},
};

using Bytes = std::vector<std::uint8_t>;

constexpr std::uint32_t kSeed = 20261021;
constexpr std::size_t kDefaultMutations = 4000;
constexpr std::size_t kMaxBytes = 64U << 10U;
// Octets qui comptent pour les formats lus : séparateurs, guillemets, échappements, chiffres.
constexpr char kInterestingBytes[] = "\0\xff\x7f\x80\n\r\t =\"'[]{}\\@.,:-+x0129eEnai#";
constexpr std::string_view kInteresting{kInterestingBytes, sizeof(kInterestingBytes) - 1};

// Le numéro de l'entrée en cours, pour le gestionnaire de signal (rien d'autre n'y est sûr).
volatile std::sig_atomic_t g_current = -1;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void say_current(int signal) {
  std::array<char, 64> text{};
  std::size_t n = 0;
  for (const char c : std::string_view("fuzz : arrêt sur l'entrée ")) text[n++] = c;
  long value = g_current;
  std::array<char, 24> digits{};
  std::size_t d = 0;
  if (value < 0) {
    digits[d++] = '?';
  } else {
    do {
      digits[d++] = static_cast<char>('0' + (value % 10));
      value /= 10;
    } while (value > 0 && d < digits.size());
  }
  while (d > 0 && n < text.size() - 1) text[n++] = digits[--d];
  text[n++] = '\n';
  static_cast<void>(write(2, text.data(), n));
  std::signal(signal, SIG_DFL);
  std::raise(signal);
}

Bytes read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::size_t below(std::mt19937& rng, std::size_t n) {
  return n == 0 ? 0 : std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
}

std::uint8_t any_byte(std::mt19937& rng) {
  if (below(rng, 2) == 0) return static_cast<std::uint8_t>(kInteresting[below(rng, kInteresting.size())]);
  return static_cast<std::uint8_t>(below(rng, 256));
}

// Une à quatre retouches d'une entrée du corpus : bit inversé, octet remplacé, octets insérés,
// plage effacée ou recopiée, troncature, greffe d'une autre entrée, ou octets quelconques.
Bytes mutate(const std::vector<Bytes>& seeds, std::mt19937& rng) {
  Bytes b = seeds[below(rng, seeds.size())];
  const std::size_t edits = 1 + below(rng, 4);
  for (std::size_t e = 0; e < edits; ++e) {
    switch (below(rng, 8)) {
      case 0:
        if (!b.empty()) b[below(rng, b.size())] ^= static_cast<std::uint8_t>(1U << below(rng, 8));
        break;
      case 1:
        if (!b.empty()) b[below(rng, b.size())] = any_byte(rng);
        break;
      case 2: {
        const std::size_t at = below(rng, b.size() + 1);
        const std::size_t count = 1 + below(rng, 8);
        Bytes add(count);
        for (std::uint8_t& c : add) c = any_byte(rng);
        b.insert(b.begin() + static_cast<std::ptrdiff_t>(at), add.begin(), add.end());
        break;
      }
      case 3:
        if (!b.empty()) {
          const std::size_t at = below(rng, b.size());
          const std::size_t count = 1 + below(rng, std::min<std::size_t>(b.size() - at, 32));
          b.erase(b.begin() + static_cast<std::ptrdiff_t>(at),
                  b.begin() + static_cast<std::ptrdiff_t>(at + count));
        }
        break;
      case 4:
        if (!b.empty()) {
          const std::size_t at = below(rng, b.size());
          const std::size_t count = 1 + below(rng, std::min<std::size_t>(b.size() - at, 64));
          const Bytes copy(b.begin() + static_cast<std::ptrdiff_t>(at),
                           b.begin() + static_cast<std::ptrdiff_t>(at + count));
          const std::size_t to = below(rng, b.size() + 1);
          b.insert(b.begin() + static_cast<std::ptrdiff_t>(to), copy.begin(), copy.end());
        }
        break;
      case 5:
        b.resize(below(rng, b.size() + 1));
        break;
      case 6: {
        const Bytes& other = seeds[below(rng, seeds.size())];
        const std::size_t from = below(rng, other.size() + 1);
        b.resize(below(rng, b.size() + 1));
        b.insert(b.end(), other.begin() + static_cast<std::ptrdiff_t>(from), other.end());
        break;
      }
      default: {
        b.resize(below(rng, 257));
        for (std::uint8_t& c : b) c = any_byte(rng);
        break;
      }
    }
    if (b.size() > kMaxBytes) b.resize(kMaxBytes);
  }
  return b;
}

// save : le fichier où écrire l'entrée au lieu de l'exécuter (--save), vide sinon.
void run(Target target, const Bytes& input, long number, const std::filesystem::path& save) {
  if (!save.empty()) {
    std::ofstream(save, std::ios::binary)
        .write(reinterpret_cast<const char*>(input.data()), static_cast<std::streamsize>(input.size()));
    return;
  }
  g_current = static_cast<std::sig_atomic_t>(number);
  static_cast<void>(target(input.data(), input.size()));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage : %s <cible> <corpus> [mutations] [--only <numéro> [--save <fichier>]]\n",
                 argv[0]);
    return 2;
  }
  const auto named = std::ranges::find(kTargets, std::string_view(argv[1]), &Named::name);
  if (named == kTargets.end()) {
    std::fprintf(stderr, "fuzz : cible inconnue %s\n", argv[1]);
    return 2;
  }
  const Target target = named->run;
  const std::filesystem::path corpus = argv[2];
  std::size_t mutations = kDefaultMutations;
  long only = -1;
  std::filesystem::path save;
  for (int i = 3; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--only" && i + 1 < argc) {
      only = std::strtol(argv[++i], nullptr, 10);
    } else if (arg == "--save" && i + 1 < argc) {
      save = argv[++i];
    } else {
      mutations = static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10));
    }
  }
  for (const int s : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) std::signal(s, say_current);

  std::vector<std::filesystem::path> files;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(corpus, error))
    if (entry.is_regular_file()) files.push_back(entry.path());
  if (error) {
    std::fprintf(stderr, "fuzz : corpus %s illisible\n", corpus.string().c_str());
    return 2;
  }
  std::ranges::sort(files);
  std::vector<Bytes> seeds;
  seeds.reserve(files.size() + 1);
  for (const std::filesystem::path& file : files) seeds.push_back(read_file(file));
  seeds.emplace_back();  // l'entrée vide

  // Numéros : le corpus d'abord, dans l'ordre des noms, puis les mutations.
  long number = 0;
  for (const Bytes& seed : seeds) {
    if (only < 0 || only == number) run(target, seed, number, save);
    ++number;
  }
  std::mt19937 rng(kSeed);
  for (std::size_t i = 0; i < mutations; ++i, ++number) {
    const Bytes input = mutate(seeds, rng);
    if (only < 0 || only == number) run(target, input, number, save);
  }
  std::printf("fuzz : %s, %zu entrées du corpus, %zu mutations\n", argv[1], seeds.size(), mutations);
  return 0;
}
