// Banc des noyaux Vulkan du socle (src/gpu/vulkan/compute) : chaque cas du catalogue est
// préparé, échauffé, puis chronométré passe par passe par requêtes timestamp (GpuTimer), une
// exécution à la fois. Sortie : un tableau Markdown sur stdout ; refus et sauts sur stderr.
// Sur la RTX du PC, toujours sous le verrou GPU :  ajp-gpu ajp_vk_kernels_bench --peak-gbps 504
//
//   ajp_vk_kernels_bench [--passes N] [--warmup N] [--only TEXTE] [--flush-mio N]
//                        [--peak-gbps X] [--validation]
#include <charconv>
#include <cmath>
#include <cstdio>
#include <exception>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cases.h"
#include "gpu/vulkan/compute/caps.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu_timer.h"
#include "stats.h"

namespace {

using namespace ajp;
using namespace ajp::bench;

constexpr std::uint32_t kMinPasses = 50;

constexpr std::string_view kUsage =
    "usage : ajp_vk_kernels_bench [options]\n"
    "  --passes N      passes mesurées par cas, de 50 à 100000 (100)\n"
    "  --warmup N      passes d'échauffement écartées, de 1 à 10000 (20)\n"
    "  --only TEXTE    seulement les cas dont « noyau taille précision » contient TEXTE\n"
    "  --flush-mio N   Mio écrits avant chaque passe pour vider le L2, de 0 (cache chaud) à 4096 ;\n"
    "                  plus que le L2 de la carte (48 Mo sur la 4070 Ti) (64)\n"
    "  --peak-gbps X   débit crête de la mémoire, pour la colonne « % crête »\n"
    "  --validation    couche VK_LAYER_KHRONOS_validation\n";

struct Options {
  std::uint32_t passes = 100;
  std::uint32_t warmup = 20;
  std::string only;
  std::uint32_t flush_mib = 64;
  double peak_gbps = 0.0;
  bool validation = false;
  bool help = false;
};

std::unexpected<Refusal> refused(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(detail)});
}

std::string describe(const Refusal& refusal) {
  return std::format("{} : {}", name(refusal.code), refusal.detail);
}

void write(std::FILE* out, std::string_view text) {
  std::fwrite(text.data(), 1, text.size(), out);
}

std::expected<std::uint32_t, Refusal> count(std::string_view option, std::string_view text, std::uint32_t lo,
                                            std::uint32_t hi) {
  std::uint32_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value < lo || value > hi) {
    return refused(RefusalCode::FieldOutOfRange,
                   std::format("{} {} : entier de {} à {} attendu", option, text, lo, hi));
  }
  return value;
}

std::expected<Options, Refusal> parse(std::span<char* const> args) {
  Options o;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--help" || arg == "-h") {
      o.help = true;
      continue;
    }
    if (arg == "--validation") {
      o.validation = true;
      continue;
    }
    if (i + 1 >= args.size())
      return refused(RefusalCode::OptionUnknown, std::format("{} : option inconnue ou sans valeur", arg));
    const std::string_view value = args[++i];
    if (arg == "--passes") {
      auto n = count(arg, value, kMinPasses, 100000);
      if (!n) return std::unexpected(n.error());
      o.passes = *n;
    } else if (arg == "--warmup") {
      auto n = count(arg, value, 1, 10000);
      if (!n) return std::unexpected(n.error());
      o.warmup = *n;
    } else if (arg == "--only") {
      o.only = value;
    } else if (arg == "--flush-mio") {
      auto n = count(arg, value, 0, 4096);
      if (!n) return std::unexpected(n.error());
      o.flush_mib = *n;
    } else if (arg == "--peak-gbps") {
      double peak = 0.0;
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), peak);
      if (error != std::errc{} || end != value.data() + value.size() || !std::isfinite(peak) || peak <= 0.0)
        return refused(RefusalCode::FieldOutOfRange,
                       std::format("--peak-gbps {} : débit positif attendu", value));
      o.peak_gbps = peak;
    } else {
      return refused(RefusalCode::OptionUnknown, std::format("{} : option inconnue", arg));
    }
  }
  return o;
}

// Échauffement écarté, puis options.passes durées : une ligne du tableau, ou le refus.
std::expected<std::string, Refusal> measure(const Options& options, GpuTimer& timer, const Case& c,
                                            Prepared& prepared, std::vector<double>& samples) {
  for (std::uint32_t i = 0; i < options.warmup; ++i) {
    if (auto t = timer.time(*prepared.recording); !t) return std::unexpected(t.error());
  }
  samples.clear();
  for (std::uint32_t i = 0; i < options.passes; ++i) {
    auto t = timer.time(*prepared.recording);
    if (!t) return std::unexpected(t.error());
    samples.push_back(*t);
  }
  auto s = summarize(samples);
  if (!s) return std::unexpected(s.error());
  auto gbps = gigabytes_per_second(prepared.bytes, s->median);
  if (!gbps) return std::unexpected(gbps.error());
  const std::string peak =
      options.peak_gbps > 0.0 ? std::format("{:.0f} %", 100.0 * *gbps / options.peak_gbps) : "-";
  return std::format("| {} | {} | {} | {:.1f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {:.1f} | {} |\n",
                     c.kernel, c.size, c.precision, static_cast<double>(prepared.bytes) / 1.0e6, s->median,
                     s->p95, s->stddev, s->min, s->slow, *gbps, peak);
}

int run(const Options& options) {
  auto context = gpu::vk::VkContext::create({.validation = options.validation});
  if (!context) {
    write(stderr, std::format("banc : pas d'appareil Vulkan ({})\n", describe(context.error())));
    return 1;
  }
  auto memory = gpu::vk::MemoryAllocator::create(**context, {.budget = VkDeviceSize{3} << 30U});
  if (!memory) {
    write(stderr, std::format("banc : {}\n", describe(memory.error())));
    return 1;
  }
  auto pipelines = gpu::vk::PipelineCache::create(**context);
  if (!pipelines) {
    write(stderr, std::format("banc : {}\n", describe(pipelines.error())));
    return 1;
  }
  gpu::vk::KernelRunner kernels(**context, **pipelines);
  Rig rig{.context = **context, .memory = **memory, .kernels = kernels};
  auto timer = GpuTimer::create(**context, **memory, VkDeviceSize{options.flush_mib} << 20U);
  if (!timer) {
    write(stderr, std::format("banc : {}\n", describe(timer.error())));
    return 1;
  }

  write(stdout, std::format("Appareil : {}\n", gpu::vk::describe((*context)->caps())));
  write(stdout, std::format("Passes : {} mesurées après {} d'échauffement, une exécution à la fois ; L2 {} ; "
                            "timestamps de {} bits, {} ns par tic.\n\n",
                            options.passes, options.warmup,
                            options.flush_mib > 0
                                ? std::format("vidé avant chaque passe ({} Mio écrits)", options.flush_mib)
                                : std::string("chaud"),
                            (*timer)->valid_bits(), (*timer)->period_ns()));
  write(
      stdout,
      "| Noyau | Taille | Précision | Mo lus+écrits | Médiane (ms) | p95 (ms) | Écart type (ms) | Min (ms) "
      "| > 1,5 × méd. | Go/s (médiane) | % crête |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  std::fflush(stdout);

  std::vector<double> samples;
  samples.reserve(options.passes);
  int failures = 0;
  for (const Case& c : catalog()) {
    if (!options.only.empty() && c.label().find(options.only) == std::string::npos) continue;
    auto prepared = c.prepare(rig);
    if (!prepared) {
      const bool skip = prepared.error().code == RefusalCode::FeatureUnavailable;
      write(stderr,
            std::format("{} {} : {}\n", skip ? "sauté" : "refusé", c.label(), describe(prepared.error())));
      if (!skip) ++failures;
      continue;
    }
    auto row = measure(options, **timer, c, *prepared, samples);
    if (!row) {
      write(stderr, std::format("refusé {} : {}\n", c.label(), describe(row.error())));
      ++failures;
      continue;
    }
    write(stdout, *row);
    std::fflush(stdout);
  }
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::span<char* const> args(argv, static_cast<std::size_t>(argc));
    auto options = parse(args.empty() ? args : args.subspan(1));
    if (!options) {
      write(stderr, std::format("banc : {}\n{}", describe(options.error()), kUsage));
      return 2;
    }
    if (options->help) {
      write(stdout, kUsage);
      return 0;
    }
    return run(*options);
  } catch (const std::exception& error) {
    write(stderr, std::format("banc : exception {}\n", error.what()));
    return 3;
  }
}
