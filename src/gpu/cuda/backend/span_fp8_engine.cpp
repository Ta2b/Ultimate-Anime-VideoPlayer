#include "gpu/cuda/backend/span_fp8_engine.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"
#include "gpu/cuda/backend/shared_activations.h"
#include "gpu/cuda/backend/span_fp8_check.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {
namespace {

constexpr std::string_view kKeyPrefix = "span-fp8:";

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
  // Bornée à la moitié de la sortie : une petite image (moins de 342 pixels) en a moins de 8 192 octets
  // (A02-04).
  const std::size_t probe_count = std::min<std::size_t>(4096, out_bytes / 4);
  std::vector<std::uint16_t> probe(probe_count);
  if (cudaError_t e = cudaMemcpy(
          probe.data(), static_cast<const std::uint8_t*>(scratch.output) + out_bytes - probe_count * 2,
          probe_count * 2, cudaMemcpyDeviceToHost);
      e != cudaSuccess)
    return failed(e, "cudaMemcpy");
  for (const std::uint16_t bits : probe)
    if ((bits & 0x7C00U) == 0x7C00U)
      return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : essai au chargement : sortie non finie");
  return {};
}

// L'essai de zéros, puis, pour le SPAN FP16 natif, le jugement contre le calcul du processeur
// (span_fp8_check.h) : ses noyaux tournent aussi sur des architectures où personne ne les a vus.
std::expected<std::optional<double>, Refusal> try_kernels(SpanFp8Net& net, const span_fp8::Package& package,
                                                          int height, int width) {
  if (auto ok = smoke_test(net, height, width); !ok) return std::unexpected(std::move(ok.error()));
  if (!package.fp16) return std::nullopt;
  return check_native(net, package, height, width);
}

}  // namespace

std::expected<std::vector<std::byte>, Refusal> read_span_fp8_package(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size =
      std::filesystem::is_regular_file(path, error) ? std::filesystem::file_size(path, error) : 0;
  if (error || size == 0 || size > span_fp8::kMaxPackageBytes)
    return refuse(RefusalCode::FileUnreadable,
                  std::format("paquet SPAN FP8 {} illisible", core::to_utf8(path)));
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  if (!file || file.gcount() != static_cast<std::streamsize>(size))
    return refuse(RefusalCode::FileUnreadable,
                  std::format("paquet SPAN FP8 {} tronqué pendant sa lecture", core::to_utf8(path)));
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

std::optional<std::uint32_t> span_fp8_artifact_crc(std::string_view key) {
  if (!key.starts_with(kKeyPrefix)) return std::nullopt;
  key.remove_prefix(kKeyPrefix.size());
  const std::string_view hex = key.substr(0, key.find(':'));
  std::uint32_t crc = 0;
  const auto [end, error] = std::from_chars(hex.data(), hex.data() + hex.size(), crc, 16);
  if (error != std::errc{} || end != hex.data() + hex.size() || hex.size() != 8) return std::nullopt;
  return crc;
}

bool is_span_fp8_package(const std::filesystem::path& path) {
  return path.extension() == ".span8";
}

SpanFp8Engine::~SpanFp8Engine() {
  drop_graphs();
}

void SpanFp8Engine::drop_graphs() {
  for (cudaGraphExec_t& exec : execs_) {
    if (exec != nullptr) cudaGraphExecDestroy(exec);
    exec = nullptr;
  }
  execs_.clear();
  book_.clear();
}

std::expected<std::unique_ptr<SpanFp8Engine>, Refusal> SpanFp8Engine::load(
    const std::filesystem::path& package, std::uint32_t crc, int height, int width,
    const std::shared_ptr<SharedActivations>& shared) {
  auto bytes = read_span_fp8_package(package);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  // Une préparation de fond a pu réécrire le paquet depuis la construction : des poids qu'aucun reçu ne
  // couvre à cette taille ne se chargent pas sous l'ancienne clé, la chaîne refait la recherche (B02-06).
  if (span_fp8::crc32(*bytes) != crc)
    return refuse(RefusalCode::GraphInvalid, "SPAN FP8 : paquet réécrit depuis sa construction");
  auto parsed = span_fp8::parse(*bytes);
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  bytes->clear();
  if (const std::string why = span_fp8::size_refusal(height, width); !why.empty())
    return refuse(RefusalCode::FormatUnsupported, "SPAN FP8 : " + why);
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string why;
  std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(span_fp8_view(*parsed), height, width, status, why);
  if (!net)
    return refuse(status == SpanFp8Status::OutOfMemory ? RefusalCode::GpuMemoryExhausted
                                                       : RefusalCode::VendorCallFailed,
                  "SPAN FP8 : envoi des poids : " + why);
  std::unique_ptr<SpanFp8Engine> engine(new SpanFp8Engine());
  engine->net_ = std::move(net);
  engine->input_ = Shape{.dims = {1, 3, height, width}};
  engine->output_ = Shape{.dims = {1, 3, 2 * std::int64_t{height}, 2 * std::int64_t{width}}};
  if (!shared) return refuse(RefusalCode::GraphInvalid, "SPAN FP8 : zone d'activation absente");
  const std::int64_t need = engine->net_->workspace_bytes();
  if (shared->bytes() < need) {
    // La zone partagée devrait grandir : l'essai se fait d'abord sur une zone à lui, et la zone partagée ne
    // grandit que s'il passe. Un refus ne la laisse pas à la taille du FP8 pour toute la séance (B02-05).
    void* own = nullptr;
    if (cudaError_t e = cudaMalloc(&own, static_cast<std::size_t>(need)); e != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      return refuse(
          e == cudaErrorMemoryAllocation ? RefusalCode::GpuMemoryExhausted : RefusalCode::VendorCallFailed,
          std::format("SPAN FP8 : zone d'essai de {} Mo : {}", need >> 20, cudaGetErrorString(e)));
    }
    engine->net_->bind(own);
    auto tried = try_kernels(*engine->net_, *parsed, height, width);
    cuda_warn(cudaFree(own), "cudaFree");  // attend la fin de l'essai
    if (!tried) return std::unexpected(std::move(tried.error()));
    engine->native_check_db_ = *tried;
    if (auto ok = engine->use_zone(shared); !ok) return std::unexpected(std::move(ok.error()));
    return engine;
  }
  if (auto ok = engine->use_zone(shared); !ok) return std::unexpected(std::move(ok.error()));
  // L'essai tourne dans la zone d'activation partagée de la voie, sur un flux à lui : le travail encore
  // en vol sur les voies finit avant, et la zone est marquée à ce moteur, pour que DistilDRBA fusionné
  // (qui y garde ses tampons d'une image à l'autre) les refasse au lieu de lire ceux que l'essai a
  // écrasés (A02-03).
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return refuse(
        RefusalCode::VendorCallFailed,
        std::format("SPAN FP8 : essai au chargement (cudaDeviceSynchronize) : {}", cudaGetErrorString(e)));
  }
  shared->used(engine.get(), engine->net_->workspace_bytes());
  auto tried = try_kernels(*engine->net_, *parsed, height, width);
  if (!tried) return std::unexpected(std::move(tried.error()));
  engine->native_check_db_ = *tried;
  return engine;
}

std::expected<void, Refusal> SpanFp8Engine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  if (auto ok = zone->reserve(net_->workspace_bytes()); !ok) return ok;
  // Une zone agrandie (generation change) ou une autre voie : les paramètres des noyaux portent des adresses.
  if (zone != zone_ || zone->generation() != generation_) {
    drop_graphs();  // les graphes portent les adresses de l'ancienne zone
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
  return enqueue_body(stream, arena, inputs, outputs).and_then([&] {
    return enqueue_tail(stream, outputs, {});
  });
}

std::expected<void, Refusal> SpanFp8Engine::enqueue_body(cudaStream_t stream,
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
  arena->used(this, net_->workspace_bytes());  // le bas de la zone, comme un moteur entier (C03-01)
  // NOLINTBEGIN(performance-no-int-to-ptr) : un tenseur porte son adresse CUDA en entier
  const void* source = reinterpret_cast<const void*>(in.handle + in.offset);
  // NOLINTEND(performance-no-int-to-ptr)
  return body(stream, source);
}

std::expected<void, Refusal> SpanFp8Engine::enqueue_tail(cudaStream_t stream, std::span<const Tensor> outputs,
                                                         const SpanFp8Ring& ring) {
  if (outputs.size() != 1) return refuse(RefusalCode::ShapeInvalid, "SPAN FP8 : une sortie attendue");
  const Tensor& out = outputs[0];
  // NOLINTNEXTLINE(performance-no-int-to-ptr) : un tenseur porte son adresse CUDA en entier
  void* target = reinterpret_cast<void*>(out.handle + out.offset);
  if (std::string why; !net_->launch_tail(stream, target, ring, why))
    return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : " + why);
  return {};
}

std::expected<void, Refusal> SpanFp8Engine::body(cudaStream_t stream, const void* source) {
  const auto direct = [&]() -> std::expected<void, Refusal> {
    if (std::string why; !net_->launch_body(stream, source, why))
      return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : " + why);
    return {};
  };
  const GraphBook<std::uintptr_t>::Pick pick = book_.next(reinterpret_cast<std::uintptr_t>(source));
  if (pick.step == GraphStep::Direct || pick.step == GraphStep::Overflow || pick.step == GraphStep::Warm)
    return direct();  // Warm : le premier lancement d'une adresse, hors capture
  if (execs_.size() <= pick.index) execs_.resize(pick.index + 1, nullptr);
  cudaGraphExec_t& exec = execs_[pick.index];
  if (pick.step == GraphStep::Capture) {
    // Un emplacement repris par un autre jeu d'adresses (registre plein, GraphBook) garde l'ancien graphe.
    if (exec != nullptr) {
      cudaGraphExecDestroy(exec);
      exec = nullptr;
    }
    cudaGraph_t captured = nullptr;
    std::string why;
    bool queued = false;
    cudaError_t error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (error == cudaSuccess) {
      queued = net_->launch_body(stream, source, why);
      error = cudaStreamEndCapture(stream, &captured);  // termine aussi une capture invalidée
    }
    if (error == cudaSuccess && queued && captured != nullptr)
      error = cudaGraphInstantiate(&exec, captured, 0);
    if (captured != nullptr) cudaGraphDestroy(captured);
    const bool ok = error == cudaSuccess && queued && captured != nullptr;
    book_.captured(
        ok, !queued ? why : std::string(error != cudaSuccess ? cudaGetErrorString(error) : "graphe vide"));
    if (!ok) {
      // Capture refusée : rien n'a tourné (la capture enregistre sans exécuter), le lancement direct fait le
      // travail.
      exec = nullptr;
      if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
        return std::unexpected(lost_refusal(lost, "SPAN FP8 : capture du graphe"));
      return direct();
    }
  }
  // Rejeu. Refusé sans erreur collante : graphe détruit, lancement direct, recapture au passage suivant
  // (après GraphBook::kMaxLaunchFailures refus, graphes coupés) ; une erreur collante : contexte perdu.
  const cudaError_t launched = cudaGraphLaunch(exec, stream);
  if (launched == cudaSuccess) return {};
  if (sticky(launched)) return std::unexpected(cuda_refusal(launched, "SPAN FP8 : cudaGraphLaunch"));
  cudaGraphExecDestroy(exec);
  exec = nullptr;
  book_.launch_failed(pick.index);
  if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
    return std::unexpected(lost_refusal(lost, "SPAN FP8 : cudaGraphLaunch"));
  return direct();
}

}  // namespace ajp::gpu::cuda
