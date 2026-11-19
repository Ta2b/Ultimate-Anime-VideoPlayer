#include "gpu/cuda/backend/rife_native_verdict.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>

namespace kirari::gpu::cuda::rife_verdict {
namespace {

// L'en-tête du fichier : un autre format (une version qui garde autre chose) ne se lit pas.
constexpr std::string_view kHeader = "kirari-rife-natif-verdict 1";

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

}  // namespace

bool all_trt(const Verdict& verdict) {
  return std::ranges::all_of(verdict.options, [](int o) { return o == 0; });
}

std::string text_of(const Verdict& verdict) {
  std::string options;
  for (const int o : verdict.options) options += std::format("{}{}", options.empty() ? "" : " ", o);
  return std::format("{}\noptions {}\nfused_end {}\ntrt_db {:.3f}\n", kHeader, options,
                     verdict.fused_end ? 1 : 0, verdict.trt_db);
}

std::optional<Verdict> parse(std::string_view text, std::size_t pieces, int options_max) {
  std::size_t at = 0;
  if (next_line(text, at) != kHeader) return std::nullopt;
  const auto options = value_of(next_line(text, at), "options");
  const auto fused = value_of(next_line(text, at), "fused_end");
  const auto db = value_of(next_line(text, at), "trt_db");
  if (!options || !fused || !db) return std::nullopt;
  Verdict verdict;
  std::size_t from = 0;
  while (from <= options->size()) {
    const std::size_t end = std::min(options->find(' ', from), options->size());
    const auto o = number<int>(options->substr(from, end - from));
    if (!o || *o < 0 || *o > options_max) return std::nullopt;
    verdict.options.push_back(*o);
    from = end + 1;
  }
  const auto fused_end = number<int>(*fused);
  const auto trt_db = number<double>(*db);
  if (verdict.options.size() != pieces || !fused_end || (*fused_end != 0 && *fused_end != 1) || !trt_db ||
      !std::isfinite(*trt_db))
    return std::nullopt;
  verdict.fused_end = *fused_end == 1;
  verdict.trt_db = *trt_db;
  return verdict;
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
