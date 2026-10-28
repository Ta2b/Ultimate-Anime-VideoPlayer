#include "gpu/cuda/backend/span_fp8_engine.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <system_error>
#include <vector>

#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {
namespace {

constexpr std::string_view kKeyPrefix = "span-fp8:";

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Libère les tampons d'essai quoi qu'il arrive.
struct Scratch {
  void* input = nullptr;
  void* output = nullptr;
  cudaStream_t stream = nullptr;
  ~Scratch() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (input != nullptr) cudaFree(input);
    if (output != nullptr) cudaFree(output);
  }
};

// Une image de zéros par les noyaux, sur un flux à part, et un coup d'œil à la sortie : finie, ou le
// chargement est refusé et la chaîne reprend TensorRT FP16.
std::expected<void, Refusal> smoke_test(SpanFp8Net& net, int height, int width) {
  const std::size_t in_bytes =
      std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 2;
  const std::size_t out_bytes = in_bytes * 4;
  Scratch scratch;
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return refuse(RefusalCode::VendorCallFailed,
                  std::format("SPAN FP8 : essai au chargement ({}) : {}", step, cudaGetErrorString(error)));
  };
  if (cudaError_t e = cudaMalloc(&scratch.input, in_bytes); e != cudaSuccess) return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch.output, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemset(scratch.input, 0, in_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaMemset(scratch.output, 0xFF, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  if (std::string why; !net.launch(scratch.stream, scratch.input, scratch.output, why))
    return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : essai au chargement : " + why);
  if (cudaError_t e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess) return failed(e, "exécution");
  // La dernière ligne de la sortie : 0xFFFF (NaN) tant que rien n'y a écrit, des demi-flottants finis
  // ensuite.
  constexpr std::size_t kProbe = 4096;
  std::vector<std::uint16_t> probe(kProbe);
  if (cudaError_t e =
          cudaMemcpy(probe.data(), static_cast<const std::uint8_t*>(scratch.output) + out_bytes - kProbe * 2,
                     kProbe * 2, cudaMemcpyDeviceToHost);
      e != cudaSuccess)
    return failed(e, "cudaMemcpy");
  for (const std::uint16_t bits : probe)
    if ((bits & 0x7C00U) == 0x7C00U)
      return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : essai au chargement : sortie non finie");
  return {};
}

// Les pointeurs du paquet, pour create().
SpanFp8Weights view_of(const span_fp8::Package& package) {
  const auto view = [](const span_fp8::Conv& conv) {
    return SpanFp8ConvView{.weights = conv.weights.data(), .bias = conv.bias.data()};
  };
  SpanFp8Weights weights;
  weights.scales = package.scales.data();
  weights.first = view(package.first);
  for (std::size_t b = 0; b < package.blocks.size(); ++b) {
    for (std::size_t i = 0; i < 3; ++i) weights.conv[b][i] = view(package.blocks[b].conv[i]);
    weights.attention[b] = view(package.blocks[b].attention);
  }
  weights.concat = view(package.concat);
  weights.pixel = view(package.pixel);
  return weights;
}

}  // namespace

std::expected<std::vector<std::byte>, Refusal> read_span_fp8_package(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size =
      std::filesystem::is_regular_file(path, error) ? std::filesystem::file_size(path, error) : 0;
  if (error || size == 0 || size > span_fp8::kMaxPackageBytes)
    return refuse(RefusalCode::FileUnreadable, std::format("paquet SPAN FP8 {} illisible", path.string()));
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  if (!file || file.gcount() != static_cast<std::streamsize>(size))
    return refuse(RefusalCode::FileUnreadable,
                  std::format("paquet SPAN FP8 {} tronqué pendant sa lecture", path.string()));
  return bytes;
}

std::string span_fp8_artifact_key(std::uint32_t crc, int height, int width) {
  return std::format("{}{:08x}:{}x{}", kKeyPrefix, crc, height, width);
}

std::optional<Shape> span_fp8_artifact_shape(std::string_view key) {
  if (!key.starts_with(kKeyPrefix)) return std::nullopt;
  key.remove_prefix(kKeyPrefix.size());
  const std::size_t colon = key.find(':');
  const std::size_t cross = key.find('x', colon == std::string_view::npos ? 0 : colon);
  if (colon == std::string_view::npos || cross == std::string_view::npos) return std::nullopt;
  int height = 0, width = 0;
  const std::string_view h = key.substr(colon + 1, cross - colon - 1), w = key.substr(cross + 1);
  if (std::from_chars(h.data(), h.data() + h.size(), height).ec != std::errc{} ||
      std::from_chars(w.data(), w.data() + w.size(), width).ec != std::errc{})
    return std::nullopt;
  return Shape{.dims = {1, 3, height, width}};
}

bool is_span_fp8_package(const std::filesystem::path& path) {
  return path.extension() == ".span8";
}

SpanFp8Engine::~SpanFp8Engine() = default;

std::expected<std::unique_ptr<SpanFp8Engine>, Refusal> SpanFp8Engine::load(
    const std::filesystem::path& package, int height, int width,
    const std::shared_ptr<SharedActivations>& shared) {
  auto bytes = read_span_fp8_package(package);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto parsed = span_fp8::parse(*bytes);
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  bytes->clear();
  if (const std::string why = span_fp8::size_refusal(height, width); !why.empty())
    return refuse(RefusalCode::FormatUnsupported, "SPAN FP8 : " + why);
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string why;
  std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(view_of(*parsed), height, width, status, why);
  if (!net)
    return refuse(status == SpanFp8Status::OutOfMemory ? RefusalCode::GpuMemoryExhausted
                                                       : RefusalCode::VendorCallFailed,
                  "SPAN FP8 : envoi des poids : " + why);
  std::unique_ptr<SpanFp8Engine> engine(new SpanFp8Engine());
  engine->net_ = std::move(net);
  engine->input_ = Shape{.dims = {1, 3, height, width}};
  engine->output_ = Shape{.dims = {1, 3, 2 * std::int64_t{height}, 2 * std::int64_t{width}}};
  if (!shared) return refuse(RefusalCode::GraphInvalid, "SPAN FP8 : zone d'activation absente");
  if (auto ok = engine->use_zone(shared); !ok) return std::unexpected(std::move(ok.error()));
  if (auto ok = smoke_test(*engine->net_, height, width); !ok) return std::unexpected(std::move(ok.error()));
  return engine;
}

std::expected<void, Refusal> SpanFp8Engine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  if (auto ok = zone->reserve(net_->workspace_bytes()); !ok) return ok;
  // Une zone agrandie (generation change) ou une autre voie : les paramètres des noyaux portent des adresses.
  if (zone != zone_ || zone->generation() != generation_) {
    net_->bind(zone->data());
    zone_ = zone;
    generation_ = zone->generation();
  }
  return {};
}

std::expected<void, Refusal> SpanFp8Engine::enqueue(cudaStream_t stream,
                                                    const std::shared_ptr<SharedActivations>& arena,
                                                    std::span<const Tensor> inputs,
                                                    std::span<const Tensor> outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return refuse(RefusalCode::ShapeInvalid, "SPAN FP8 : une entrée et une sortie attendues");
  const Tensor& in = inputs[0];
  const Tensor& out = outputs[0];
  if (in.type != TensorType::F16 || !(in.shape == input_))
    return refuse(RefusalCode::ShapeInvalid, "SPAN FP8 : entrée de forme ou de type différents du moteur");
  if (out.type != TensorType::F16 || !(out.shape == output_))
    return refuse(RefusalCode::ShapeInvalid, "SPAN FP8 : sortie de forme ou de type différents du moteur");
  if (!arena) return refuse(RefusalCode::GraphInvalid, "SPAN FP8 : zone d'activation absente");
  if (auto ok = use_zone(arena); !ok) return ok;
  // NOLINTBEGIN(performance-no-int-to-ptr) : un tenseur porte son adresse CUDA en entier
  const void* source = reinterpret_cast<const void*>(in.handle + in.offset);
  void* target = reinterpret_cast<void*>(out.handle + out.offset);
  // NOLINTEND(performance-no-int-to-ptr)
  if (std::string why; !net_->launch(stream, source, target, why))
    return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : " + why);
  return {};
}

}  // namespace ajp::gpu::cuda
