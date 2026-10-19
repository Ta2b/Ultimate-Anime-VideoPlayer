// Banc d'un moteur TensorRT par le runtime du lecteur : enqueueV3 sur un flux
// non bloquant, temps GPU de chaque exécution entre deux événements CUDA.
//
//   ajp_trt_bench <moteur> [--shape <entrée>=1x3x720x1280]... [--warmup 50] [--iterations 200] [--runs 5]
//                 [--input noise|rife]
//
// Les entrées sont écrites une fois (bench_input.h) ; aucune copie ne tombe
// dans la mesure, comme trtexec --noDataTransfers. Le résultat donne
// chaque passe, puis médiane, moyenne et écart type de toutes les exécutions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/trt/bench_input.h"
#include "gpu/cuda/trt/cuda_objects.h"
#include "gpu/cuda/trt/log.h"
#include "gpu/cuda/trt/runtime.h"
#include "gpu/cuda/trt/tool_args.h"

namespace {

using namespace ajp::gpu::trt;
namespace fs = std::filesystem;

struct Options {
  fs::path engine;
  std::vector<std::pair<std::string, Dims>> shapes;
  int warmup = 50;
  int iterations = 200;
  int runs = 5;
  bool rife_input = false;  // --input rife : entrée v1 de RIFE comme la lecture
};

void out(const std::string& text) {
  std::fputs(text.c_str(), stdout);
}

std::expected<int, std::string> parse_count(std::string_view text, std::string_view flag) {
  const auto value = tool::parse_number<int>(text);
  if (!value || *value < 1) {
    return std::unexpected(std::format("{} attend un entier positif, reçu « {} »", flag, text));
  }
  return *value;
}

// « input=1x3x720x1280 »
std::expected<std::pair<std::string, Dims>, std::string> parse_shape(std::string_view text) {
  const std::size_t equal = text.find('=');
  auto dims = equal == std::string_view::npos ? std::nullopt : tool::parse_dims(text.substr(equal + 1));
  if (equal == 0 || !dims) {
    return std::unexpected(std::format("--shape attend <entrée>=AxBxC, reçu « {} »", text));
  }
  return std::pair{std::string(text.substr(0, equal)), std::move(*dims)};
}

std::expected<Options, std::string> parse(int argc, const char* const* argv) {
  Options options;
  for (int k = 1; k < argc; ++k) {
    const std::string_view arg = argv[k];
    const bool has_value = k + 1 < argc;
    if (arg == "--shape" && has_value) {
      auto shape = parse_shape(argv[++k]);
      if (!shape) return std::unexpected(shape.error());
      options.shapes.push_back(std::move(*shape));
    } else if ((arg == "--warmup" || arg == "--iterations" || arg == "--runs") && has_value) {
      auto count = parse_count(argv[++k], arg);
      if (!count) return std::unexpected(count.error());
      (arg == "--warmup"       ? options.warmup
       : arg == "--iterations" ? options.iterations
                               : options.runs) = *count;
    } else if (arg == "--input" && has_value &&
               (std::string_view(argv[k + 1]) == "rife" || std::string_view(argv[k + 1]) == "noise")) {
      options.rife_input = std::string_view(argv[++k]) == "rife";
    } else if (!arg.starts_with("--") && options.engine.empty()) {
      options.engine = fs::path(arg);
    } else {
      return std::unexpected(std::format("argument inattendu « {} »", arg));
    }
  }
  if (options.engine.empty()) return std::unexpected("chemin du moteur manquant");
  return options;
}

struct Stats {
  double median = 0;
  double mean = 0;
  double stddev = 0;
  double min = 0;
  double max = 0;
};

Stats stats_of(std::vector<double> values) {
  std::ranges::sort(values);
  const auto n = static_cast<double>(values.size());
  Stats s{.min = values.front(), .max = values.back()};
  const std::size_t mid = values.size() / 2;
  s.median = values.size() % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
  double sum = 0.0;
  for (const double v : values) sum += (v - s.mean) * (v - s.mean);
  s.stddev = values.size() > 1 ? std::sqrt(sum / (n - 1.0)) : 0.0;
  return s;
}

std::expected<void, ajp::Refusal> run_bench(const Options& options) {
  auto runtime = TrtRuntime::create(make_logger());
  if (!runtime) return std::unexpected(runtime.error());
  auto engine = runtime->load_file(options.engine);
  if (!engine) return std::unexpected(engine.error());
  auto stream = CudaStream::create();
  if (!stream) return std::unexpected(stream.error());
  for (const auto& [name, dims] : options.shapes) {
    if (auto ok = engine->set_input_shape(name, dims); !ok) return ok;
  }
  out(std::format("TensorRT {}, moteur {}, activations {} Mio\n", runtime->version().text(),
                  options.engine.filename().string(),
                  engine->device_memory_bytes() / (std::int64_t{1024} * 1024)));
  std::vector<DeviceBuffer> buffers;
  for (const TensorDesc& desc : engine->tensors()) {
    auto dims = engine->shape(desc.name);
    if (!dims) return std::unexpected(dims.error());
    auto bytes = byte_count(desc.type, *dims);
    if (!bytes) return std::unexpected(bytes.error());
    auto buffer = DeviceBuffer::allocate(*bytes);
    if (!buffer) return std::unexpected(buffer.error());
    if (desc.io == TensorIo::Input) {
      auto filled =
          options.rife_input
              ? bench::rife_input(desc.type, *dims)
              : std::expected<std::vector<std::byte>, std::string>(bench::noise_input(desc.type, *bytes));
      if (!filled) return std::unexpected(ajp::Refusal{ajp::RefusalCode::FieldOutOfRange, filled.error()});
      const std::vector<std::byte>& pattern = *filled;
      if (auto ok = cuda_check(cudaMemcpy(buffer->data(), pattern.data(), *bytes, cudaMemcpyHostToDevice),
                               "cudaMemcpy");
          !ok) {
        return ok;
      }
    }
    if (auto ok = engine->bind(desc.name, buffer->data()); !ok) return ok;
    out(std::format("  {} {} {} {}\n", desc.io == TensorIo::Input ? "entrée" : "sortie", desc.name,
                    name(desc.type), dims_text(*dims)));
    buffers.push_back(std::move(*buffer));
  }
  const auto iterations = static_cast<std::size_t>(options.iterations);
  std::vector<CudaEvent> starts;
  std::vector<CudaEvent> ends;
  for (std::size_t k = 0; k < iterations; ++k) {
    auto start = CudaEvent::create(EventTiming::Enabled);
    auto end = CudaEvent::create(EventTiming::Enabled);
    if (!start || !end) return std::unexpected(!start ? start.error() : end.error());
    starts.push_back(std::move(*start));
    ends.push_back(std::move(*end));
  }
  std::vector<double> all;
  for (int run = 0; run < options.runs; ++run) {
    for (int k = 0; k < options.warmup; ++k) {
      if (auto ok = engine->enqueue(stream->get()); !ok) return ok;
    }
    for (std::size_t k = 0; k < iterations; ++k) {
      if (auto ok = starts[k].record(stream->get()); !ok) return ok;
      if (auto ok = engine->enqueue(stream->get(), {.done = ends[k].get()}); !ok) return ok;
    }
    if (auto ok = stream->synchronize(); !ok) return ok;
    std::vector<double> times;
    for (std::size_t k = 0; k < iterations; ++k) {
      auto ms = elapsed_ms(starts[k], ends[k]);
      if (!ms) return std::unexpected(ms.error());
      times.push_back(*ms);
    }
    all.insert(all.end(), times.begin(), times.end());
    const Stats s = stats_of(std::move(times));
    out(std::format(
        "  passe {} : médiane {:.3f} ms, moyenne {:.3f}, écart type {:.3f}, min {:.3f}, max {:.3f}\n",
        run + 1, s.median, s.mean, s.stddev, s.min, s.max));
  }
  const Stats s = stats_of(std::move(all));
  out(
      std::format("total ({} passes x {} exécutions) : médiane {:.3f} ms, moyenne {:.3f}, écart type {:.3f}, "
                  "min {:.3f}, max {:.3f}\n",
                  options.runs, options.iterations, s.median, s.mean, s.stddev, s.min, s.max));
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  auto options = parse(argc, argv);
  if (!options) {
    std::fputs(std::format("ajp_trt_bench : {}\n", options.error()).c_str(), stderr);
    return 2;
  }
  if (auto ok = run_bench(*options); !ok) {
    std::fputs(
        std::format("ajp_trt_bench : {} : {}\n", ajp::name(ok.error().code), ok.error().detail).c_str(),
        stderr);
    return 1;
  }
  return 0;
}
