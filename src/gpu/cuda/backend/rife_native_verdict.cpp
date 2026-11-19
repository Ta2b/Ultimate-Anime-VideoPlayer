#include "gpu/cuda/backend/rife_native_verdict.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <vector>

namespace kirari::gpu::cuda::rife_verdict {
namespace {

// L'en-tête du fichier : un autre format (une version qui garde autre chose) ne se lit pas.
constexpr std::string_view kHeader = "kirari-rife-natif-verdict 2";

// La ligne suivante de text à partir de at, sans son saut de ligne ; at passe après.
std::string_view next_line(std::string_view text, std::size_t& at) {
  if (at >= text.size()) return {};
  const std::size_t end = std::min(text.find('\n', at), text.size());
  const std::string_view line = text.substr(at, end - at);
  at = end + 1;
  return line;
}

// La valeur de « clé valeur » si la ligne commence par la clé, rien sinon.
std::optional<std::string_view> value_of(std::string_view line, std::string_view key) {
  if (!line.starts_with(key) || line.size() <= key.size() || line[key.size()] != ' ') return std::nullopt;
  return line.substr(key.size() + 1);
}

template <typename T>
std::optional<T> number(std::string_view text) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  return value;
}

// Des entiers séparés d'une espace, chacun de low à high ; « - » : aucun (allow_none). Rien sur un champ vide
// ou hors bornes.
std::optional<std::vector<int>> integers(std::string_view text, int low, int high, bool allow_none) {
  std::vector<int> out;
  if (allow_none && text == "-") return out;
  std::size_t from = 0;
  while (from <= text.size()) {
    const std::size_t end = std::min(text.find(' ', from), text.size());
    const auto v = number<int>(text.substr(from, end - from));
    if (!v || *v < low || *v > high) return std::nullopt;
    out.push_back(*v);
    from = end + 1;
  }
  return out;
}

std::string joined(const std::vector<int>& values) {
  std::string out;
  for (const int v : values) out += std::format("{}{}", out.empty() ? "" : " ", v);
  return out.empty() ? "-" : out;
}

}  // namespace

bool all_trt(const Verdict& verdict) {
  return std::ranges::all_of(verdict.options, [](int o) { return o == 0; });
}

std::string text_of(const Verdict& verdict) {
  return std::format("{}\noptions {}\nfused_end {}\ntrt_db {:.3f}\napart {}\n", kHeader,
                     joined(verdict.options), verdict.fused_end ? 1 : 0, verdict.trt_db,
                     joined(verdict.apart));
}

std::optional<Verdict> parse(std::string_view text, std::size_t pieces, int options_max) {
  std::size_t at = 0;
  if (next_line(text, at) != kHeader || pieces == 0) return std::nullopt;
  const auto options = value_of(next_line(text, at), "options");
  const auto fused = value_of(next_line(text, at), "fused_end");
  const auto db = value_of(next_line(text, at), "trt_db");
  const auto apart = value_of(next_line(text, at), "apart");
  if (!options || !fused || !db || !apart) return std::nullopt;
  auto chosen = integers(*options, 0, options_max, false);
  auto blocks = integers(*apart, 0, static_cast<int>(pieces) - 2, true);
  const auto fused_end = number<int>(*fused);
  const auto trt_db = number<double>(*db);
  if (!chosen || chosen->size() != pieces || !blocks || !fused_end || (*fused_end != 0 && *fused_end != 1) ||
      !trt_db || !std::isfinite(*trt_db))
    return std::nullopt;
  // Un bloc à part : croissants, sans doublon, et natif (la colle à part d'un bloc de TensorRT ne veut rien
  // dire).
  for (std::size_t i = 0; i < blocks->size(); ++i)
    if ((i > 0 && (*blocks)[i] <= (*blocks)[i - 1]) ||
        (*chosen)[static_cast<std::size_t>((*blocks)[i]) + 1] == 0)
      return std::nullopt;
  return Verdict{.options = std::move(*chosen),
                 .fused_end = *fused_end == 1,
                 .trt_db = *trt_db,
                 .apart = std::move(*blocks)};
}

std::string file_name(std::string_view identity) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  const auto mix = [&hash](std::string_view text) {
    for (const char c : text) {
      hash ^= static_cast<std::uint8_t>(c);
      hash *= 0x100000001b3ULL;
    }
  };
  mix(kHeader);
  mix("\n");
  mix(identity);
  return std::format("rife-natif-{:016x}.verdict", hash);
}

bool still_holds(const Verdict& verdict, double measured_db) {
  return std::isfinite(measured_db) && measured_db >= verdict.trt_db - kTolerance;
}

}  // namespace kirari::gpu::cuda::rife_verdict
