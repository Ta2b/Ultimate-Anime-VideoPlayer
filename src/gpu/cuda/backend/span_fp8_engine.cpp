#include "gpu/cuda/backend/span_fp8_engine.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"
#include "gpu/cuda/backend/shared_activations.h"
#include "gpu/cuda/backend/span_fp8_check.h"
#include "gpu/span_fp8/package.h"
#include "kernels/ref/half.h"

namespace kirari::gpu::cuda {
namespace {

constexpr std::string_view kKeyPrefix = "span-fp8:";

// Libère les tampons d'essai quoi qu'il arrive.
struct Scratch {
  void* input = nullptr;
  void* output = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  ~Scratch() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
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

constexpr double kKernelTrialDb = 50.0;
// CRC du paquet FP16 dérivé de 2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.
constexpr std::uint32_t kPerformanceFp16Crc = 0x30469c4eU;

struct PreciseTrial {
  const SpanFp8Weights& weights;
  void* workspace;
};
struct KernelTrial {
  double db = 0.0;
  std::optional<double> fp32_gain_db;
  float reference_ms = 0.F, native_ms = 0.F;
};

// Trois motifs dans la même image : contours tramés, couleurs saturées et détails des ombres.
std::vector<std::uint16_t> trial_image(int height, int width) {
  constexpr int kTile = 64;
  const auto plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  std::vector<std::uint16_t> image(3 * plane);
  for (int c = 0; c < 3; ++c) {
    std::array<std::array<std::uint16_t, kTile * kTile>, 3> motifs{};
    for (int y = 0; y < kTile; ++y)
      for (int x = 0; x < kTile; ++x) {
        const std::size_t i = static_cast<std::size_t>(y * kTile + x);
        const float m = native_check_pattern(c, y, x);
        motifs[0][i] = kernels::ref::float_to_half(m);
        motifs[1][i] = ((x / 9 + y / 7) % 3 == c) ? 0x3c00U : 0U;
        motifs[2][i] = kernels::ref::float_to_half(m * 0.08F);
      }
    for (int y = 0; y < height; ++y) {
      auto* dst = image.data() + static_cast<std::size_t>(c) * plane +
                  static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
      const auto* row = motifs[static_cast<std::size_t>(3 * y / height)].data() + (y % kTile) * kTile;
      for (int x = 0; x < width; x += kTile) std::copy_n(row, std::min(kTile, width - x), dst + x);
    }
  }
  return image;
}

// Trois petites bandes couvrent la largeur complète, les trois canaux et les deux bords verticaux.
std::expected<std::vector<std::uint16_t>, Refusal> trial_bands(const void* output, int h, int w,
                                                               std::string_view label) {
  const int rows = std::min(16, h);
  const std::size_t row = static_cast<std::size_t>(w), stride = row * static_cast<std::size_t>(h);
  const std::size_t band = row * static_cast<std::size_t>(rows);
  std::vector<std::uint16_t> result(9 * band);
  auto* dst = result.data();
  for (int y : {0, (h - rows) / 2, h - rows})
    for (std::size_t c = 0; c < 3; ++c) {
      const auto* src =
          static_cast<const std::uint16_t*>(output) + c * stride + row * static_cast<std::size_t>(y);
      if (auto ok = cuda_check(cudaMemcpy(dst, src, band * sizeof(*dst), cudaMemcpyDeviceToHost),
                               std::format("SPAN : lecture de l'essai {}", label));
          !ok)
        return std::unexpected(std::move(ok.error()));
      dst += band;
    }
  return result;
}

double trial_db(const std::vector<std::uint16_t>& reference, const std::vector<std::uint16_t>& candidate) {
  const std::size_t per = reference.size() / 3;
  double worst = 200.;
  for (std::size_t b = 0; b < 3; ++b) {
    double squared = 0.;
    for (std::size_t i = b * per; i < (b + 1) * per; ++i) {
      const double diff = double{kernels::ref::half_to_float(candidate[i])} -
                          double{kernels::ref::half_to_float(reference[i])};
      if (!std::isfinite(diff)) return std::numeric_limits<double>::quiet_NaN();
      squared += diff * diff;
    }
    if (squared > 0.) worst = std::min(worst, 10. * std::log10(static_cast<double>(per) / squared));
  }
  return worst;
}

// Le C48 Performance amplifie les erreurs d'arrondi près des couleurs saturées. Comparer les deux
// accumulations f16 entre elles mesure leur désaccord, pas leur écart au calcul avec sommes f32.
std::optional<double> precision_gain(std::span<const std::uint16_t> fp32,
                                     std::span<const std::uint16_t> direct,
                                     std::span<const std::uint16_t> winograd) {
  const std::size_t per = fp32.size() / 3;
  const std::size_t tail = per - 1 - per / 1000;
  std::vector<float> direct_error(per), wino_error(per);
  double gain = 200.0;
  for (std::size_t b = 0; b < 3; ++b) {
    double direct_square = 0., wino_square = 0.;
    for (std::size_t i = 0; i < per; ++i) {
      const std::size_t pos = b * per + i;
      const float want = kernels::ref::half_to_float(fp32[pos]);
      const float da = std::abs(kernels::ref::half_to_float(direct[pos]) - want);
      const float wa = std::abs(kernels::ref::half_to_float(winograd[pos]) - want);
      if (!std::isfinite(da) || !std::isfinite(wa)) return std::nullopt;
      direct_error[i] = da;
      wino_error[i] = wa;
      direct_square += double{da} * da;
      wino_square += double{wa} * wa;
    }
    if (wino_square > direct_square) return std::nullopt;
    std::ranges::nth_element(direct_error, direct_error.begin() + static_cast<std::ptrdiff_t>(tail));
    std::ranges::nth_element(wino_error, wino_error.begin() + static_cast<std::ptrdiff_t>(tail));
    if (wino_error[tail] > direct_error[tail]) return std::nullopt;
    if (wino_square > 0.) gain = std::min(gain, 10. * std::log10(direct_square / wino_square));
  }
  return gain;
}

std::expected<void, Refusal> trial_step(cudaError_t error, const char* step, std::string_view label) {
  return cuda_check(error, std::format("SPAN : essai {} ({})", label, step));
}

std::expected<void, Refusal> launch_trial(Scratch& scratch, SpanFp8Net& net, std::string_view label) {
  std::string why;
  if (net.launch_body(scratch.stream, scratch.input, why) &&
      net.launch_tail(scratch.stream, scratch.output, {}, why))
    return {};
  if (auto ok = trial_step(cudaStreamSynchronize(scratch.stream), "lancement", label); !ok) return ok;
  return refuse(RefusalCode::VendorCallFailed, std::format("essai {} : {}", label, why));
}

std::expected<float, Refusal> measure_trial(Scratch& scratch, SpanFp8Net& net, std::string_view label) {
  if (auto ok = trial_step(cudaEventRecord(scratch.start, scratch.stream), "début", label); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = launch_trial(scratch, net, label); !ok) return std::unexpected(std::move(ok.error()));
  if (auto ok = trial_step(cudaEventRecord(scratch.stop, scratch.stream), "fin", label); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = trial_step(cudaEventSynchronize(scratch.stop), "chronométrage", label); !ok)
    return std::unexpected(std::move(ok.error()));
  float ms = 0.F;
  if (auto ok = trial_step(cudaEventElapsedTime(&ms, scratch.start, scratch.stop), "temps", label); !ok)
    return std::unexpected(std::move(ok.error()));
  if (!(ms > 0.F) || !std::isfinite(ms))
    return refuse(RefusalCode::VendorCallFailed, std::format("essai {} : temps CUDA invalide", label));
  return ms;
}

std::expected<KernelTrial, Refusal> compare_kernels(SpanFp8Net& reference, SpanFp8Net& candidate, int height,
                                                    int width, std::string_view label,
                                                    const PreciseTrial* precise = nullptr) {
  Scratch scratch;
  const auto image = trial_image(height, width);
  const std::size_t in_bytes = image.size() * sizeof(image[0]), out_bytes = 4 * in_bytes;
  const auto step = [label](cudaError_t e, const char* what) { return trial_step(e, what, label); };
  if (auto ok = step(cudaMalloc(&scratch.input, in_bytes), "entrée"); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = step(cudaMalloc(&scratch.output, out_bytes), "sortie"); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = step(cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking), "flux"); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok =
          step(cudaMemcpyAsync(scratch.input, image.data(), in_bytes, cudaMemcpyHostToDevice, scratch.stream),
               "motif");
      !ok)
    return std::unexpected(std::move(ok.error()));
  const auto fetch = [&](SpanFp8Net& net) -> std::expected<std::vector<std::uint16_t>, Refusal> {
    if (auto ok = step(cudaMemsetAsync(scratch.output, 0xff, out_bytes, scratch.stream), "initialisation");
        !ok)
      return std::unexpected(std::move(ok.error()));
    if (auto ok = launch_trial(scratch, net, label); !ok) return std::unexpected(std::move(ok.error()));
    if (auto ok = step(cudaStreamSynchronize(scratch.stream), "chauffe"); !ok)
      return std::unexpected(std::move(ok.error()));
    return trial_bands(scratch.output, 2 * height, 2 * width, label);
  };
  const auto want = fetch(reference);
  if (!want) return std::unexpected(want.error());
  const auto got = fetch(candidate);
  if (!got) return std::unexpected(got.error());
  const double db = trial_db(*want, *got);
  std::optional<double> fp32_gain;
  if (!(db >= kKernelTrialDb)) {
    if (precise != nullptr && std::isfinite(db)) {
      auto weights = precise->weights;
      weights.sums = SpanNativeSums::F32;
      weights.winograd = SpanWinograd::Off;
      SpanFp8Status status = SpanFp8Status::Ok;
      std::string why;
      auto judge = SpanFp8Net::create(weights, height, width, status, why);
      if (!judge || judge->workspace_bytes() != reference.workspace_bytes())
        return refuse(RefusalCode::VendorCallFailed, "essai Winograd : référence FP32 indisponible");
      judge->bind(precise->workspace);
      const auto fp32 = fetch(*judge);
      if (!fp32) return std::unexpected(fp32.error());
      fp32_gain = precision_gain(*fp32, *want, *got);
    }
    if (!fp32_gain)
      return refuse(RefusalCode::VendorCallFailed,
                    std::format("essai {} à {:.1f} dB contre la voie actuelle, seuil {:.0f}", label, db,
                                kKernelTrialDb));
  }

  for (auto* event : {&scratch.start, &scratch.stop})
    if (auto ok = step(cudaEventCreate(event), "événement"); !ok)
      return std::unexpected(std::move(ok.error()));
  std::array<std::array<float, 3>, 2> times{};
  SpanFp8Net* const nets[]{&reference, &candidate};
  for (std::size_t round = 0; round < times[0].size(); ++round)
    for (std::size_t i = 0; i < times.size(); ++i) {
      const std::size_t choice = round % 2 == 0 ? i : 1 - i;
      auto ms = measure_trial(scratch, *nets[choice], label);
      if (!ms) return std::unexpected(std::move(ms.error()));
      times[choice][round] = *ms;
    }
  for (auto& samples : times) std::ranges::sort(samples);
  return KernelTrial{
      .db = db, .fp32_gain_db = fp32_gain, .reference_ms = times[0][1], .native_ms = times[1][1]};
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
  release_tiles();
}

void SpanFp8Engine::drop_graphs() {
  for (BodyGraphs* graphs : {&full_graphs_, &tiled_graphs_}) {
    for (cudaGraphExec_t exec : graphs->execs) {
      if (exec != nullptr) cudaGraphExecDestroy(exec);
    }
    graphs->execs.clear();
    graphs->book.clear();
  }
}

std::expected<std::unique_ptr<SpanFp8Engine>, Refusal> SpanFp8Engine::load(
    const std::filesystem::path& package, std::uint32_t crc, int height, int width,
    const std::shared_ptr<SharedActivations>& shared, bool native_edges, bool native_blocks,
    bool vertical_winograd) {
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
  SpanFp8Weights weights = span_fp8_view(*parsed);
  weights.native_edges = native_edges && !parsed->fp16;
  std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(weights, height, width, status, why);
  if (!net)
    return refuse(status == SpanFp8Status::OutOfMemory ? RefusalCode::GpuMemoryExhausted
                                                       : RefusalCode::VendorCallFailed,
                  "SPAN FP8 : envoi des poids : " + why);
  std::unique_ptr<SpanFp8Engine> engine(new SpanFp8Engine());
  engine->net_ = std::move(net);
  engine->fp16_ = parsed->fp16;
  engine->performance_fp16_ =
      crc == kPerformanceFp16Crc && weights.fp16 && weights.channels == 48 && weights.blocks == 5;
  engine->sums_ = weights.sums;
  engine->native_edges_ = weights.native_edges;
  engine->input_ = Shape{.dims = {1, 3, height, width}};
  engine->output_ = Shape{.dims = {1, 3, 2 * std::int64_t{height}, 2 * std::int64_t{width}}};
  if (!shared) return refuse(RefusalCode::GraphInvalid, "SPAN FP8 : zone d'activation absente");
  bool allocation_refused = false;
  auto placed =
      engine->place_for_load(*parsed, weights, shared, native_blocks, vertical_winograd, allocation_refused);
  if (placed) return engine;
  // La bande de 32 et son halo tiennent dans 64 lignes pour 2 à 8 blocs.
  if (!allocation_refused || !parsed->fp16 || parsed->real_channels != 64 ||
      weights.sums != SpanNativeSums::F16 || height / 2 <= 64)
    return std::unexpected(std::move(placed.error()));

  const auto full_bytes = engine->net_->workspace_bytes();
  engine->net_.reset();  // ne pas garder deux jeux de poids en mémoire sous pression
  weights.band = 32;
  weights.compact_workspace = true;
  engine->net_ = SpanFp8Net::create(weights, height, width, status, why);
  if (!engine->net_)
    return refuse(status == SpanFp8Status::OutOfMemory ? RefusalCode::GpuMemoryExhausted
                                                       : RefusalCode::VendorCallFailed,
                  "SPAN FP16 : préparation de la zone compacte : " + why);
  if (engine->net_->workspace_bytes() >= full_bytes) return std::unexpected(std::move(placed.error()));
  if (auto compact = engine->place_for_load(*parsed, weights, shared, native_blocks, vertical_winograd,
                                            allocation_refused);
      !compact)
    return std::unexpected(std::move(compact.error()));
  engine->runtime_notice_ =
      std::format("SPAN FP16 : zone compacte après refus de mémoire ({} Mio contre {} Mio)",
                  engine->net_->workspace_bytes() >> 20, full_bytes >> 20);
  return engine;
}

std::expected<bool, Refusal> SpanFp8Engine::try_vertical_net(const span_fp8::Package& package,
                                                             SpanFp8Weights weights, void* workspace) {
  const auto reject = [&](std::string detail) -> std::expected<bool, Refusal> {
    if (const auto lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "SPAN FP16 : essai Winograd vertical"));
    winograd_notice_ = "SPAN FP16 : Winograd vertical écarté, " + detail;
    return false;
  };
  const int height = static_cast<int>(input_.dims[2]), width = static_cast<int>(input_.dims[3]);
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string why;
  weights.winograd = SpanWinograd::VerticalUnrolled;
  auto candidate = SpanFp8Net::create(weights, height, width, status, why);
  if (!candidate) return reject("poids : " + why);
  if (candidate->workspace_bytes() != net_->workspace_bytes()) return reject("taille de la zone différente");
  candidate->bind(workspace);
  const PreciseTrial precise{weights, workspace};
  auto compared = compare_kernels(*net_, *candidate, height, width, "Winograd vertical",
                                  performance_fp16_ ? &precise : nullptr);
  if (!compared) {
    if (compared.error().code == RefusalCode::DeviceLost) return std::unexpected(std::move(compared.error()));
    return reject(compared.error().detail);
  }
  // La petite marge évite de choisir sur le bruit de trois chronométrages au chargement.
  const bool faster = compared->native_ms < 0.97F * compared->reference_ms;
  auto result = std::format("{:.2f} ms contre {:.2f} ms, parité {:.1f} dB", compared->native_ms,
                            compared->reference_ms, compared->db);
  if (compared->fp32_gain_db)
    result += std::format(", gain de {:.2f} dB ou plus face aux sommes FP32", *compared->fp32_gain_db);
  if (!faster) return reject(result);

  auto checked = try_kernels(*candidate, package, height, width);
  if (!checked) {
    if (checked.error().code == RefusalCode::DeviceLost) return std::unexpected(std::move(checked.error()));
    return reject(checked.error().detail);
  }
  winograd_notice_ = "SPAN FP16 : Winograd vertical retenu, " + result;
  native_check_db_ = *checked;
  net_ = std::move(candidate);
  return true;
}

std::expected<void, Refusal> SpanFp8Engine::try_relu_net(SpanFp8Weights weights, void* workspace) {
  if (weights.fp16) {
    native_blocks_notice_ = "SPAN : l'option des ReLU natives demande un paquet FP8";
    return {};
  }
  const int height = static_cast<int>(input_.dims[2]), width = static_cast<int>(input_.dims[3]);
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string why;
  weights.native_relu = true;
  auto candidate = SpanFp8Net::create(weights, height, width, status, why);
  if (!candidate) {
    if (const auto lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "SPAN FP8 : poids des ReLU natives"));
    native_blocks_notice_ = "SPAN FP8 : ReLU natives écartées, poids : " + why;
    return {};
  }
  if (candidate->workspace_bytes() != net_->workspace_bytes()) {
    native_blocks_notice_ = "SPAN FP8 : ReLU natives écartées, taille de la zone différente";
    return {};
  }
  candidate->bind(workspace);
  auto compared = compare_kernels(*net_, *candidate, height, width, "ReLU");
  if (!compared) {
    if (compared.error().code == RefusalCode::DeviceLost) return std::unexpected(std::move(compared.error()));
    if (const auto lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "SPAN FP8 : essai ReLU"));
    native_blocks_notice_ = "SPAN FP8 : ReLU natives écartées, " + compared.error().detail;
    return {};
  }
  const bool keep = compared->db >= kKernelTrialDb && compared->native_ms < compared->reference_ms;
  native_blocks_notice_ = std::format(
      "SPAN FP8 : ReLU natives {} (parité {:.1f} dB, seuil {:.0f} ; {:.2f} ms contre {:.2f} ms CUTLASS)",
      keep ? "retenues" : "écartées", compared->db, kKernelTrialDb, compared->native_ms,
      compared->reference_ms);
  if (keep) {
    net_ = std::move(candidate);
    native_blocks_ = true;
  }
  return {};
}

std::expected<void, Refusal> SpanFp8Engine::validate_nets(const span_fp8::Package& package,
                                                          const SpanFp8Weights& weights, void* workspace,
                                                          bool native_blocks, bool vertical_winograd) {
  const int height = static_cast<int>(input_.dims[2]), width = static_cast<int>(input_.dims[3]);
  const bool balanced = weights.channels == span_fp8::kChannels && weights.blocks == span_fp8::kBlocks;
  if (vertical_winograd && weights.fp16 && weights.sums == SpanNativeSums::F16 &&
      (balanced || performance_fp16_) && !weights.compact_workspace && height == 1080 && width == 1920) {
    auto selected = try_vertical_net(package, weights, workspace);
    if (!selected) return std::unexpected(std::move(selected.error()));
    if (*selected) return {};
  }
  auto tried = try_kernels(*net_, package, height, width);
  if (!tried) return std::unexpected(std::move(tried.error()));
  native_check_db_ = *tried;
  if (native_blocks) return try_relu_net(weights, workspace);
  return {};
}

std::expected<void, Refusal> SpanFp8Engine::place_for_load(const span_fp8::Package& package,
                                                           const SpanFp8Weights& weights,
                                                           const std::shared_ptr<SharedActivations>& shared,
                                                           bool native_blocks, bool vertical_winograd,
                                                           bool& allocation_refused) {
  const std::int64_t need = net_->workspace_bytes();
  const auto bind_shared = [&]() -> std::expected<void, Refusal> {
    auto ok = use_zone(shared);
    allocation_refused = !ok && ok.error().code == RefusalCode::GpuMemoryExhausted;
    return ok;
  };
  if (shared->bytes() < need) {
    // Essayer sur une zone privée avant d'agrandir la zone de toute la voie (B02-05).
    void* own = nullptr;
    if (cudaError_t e = cudaMalloc(&own, static_cast<std::size_t>(need)); e != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      allocation_refused = e == cudaErrorMemoryAllocation;
      return refuse(allocation_refused ? RefusalCode::GpuMemoryExhausted : RefusalCode::VendorCallFailed,
                    std::format("SPAN FP8 : zone d'essai de {} Mo : {}", need >> 20, cudaGetErrorString(e)));
    }
    net_->bind(own);
    std::unique_ptr<void, void (*)(void*)> temporary(own,
                                                     [](void* ptr) { cuda_warn(cudaFree(ptr), "cudaFree"); });
    auto tried = validate_nets(package, weights, own, native_blocks, vertical_winograd);
    temporary.reset();  // libérer la zone d'essai avant d'agrandir l'arène partagée
    if (!tried) return tried;
    return bind_shared();
  }
  if (auto ok = bind_shared(); !ok) return ok;
  // Les voies finissent avant l'essai sur son flux privé. Signaler l'usage de la zone aux moteurs qui
  // y gardent des données entre deux images, dont DistilDRBA (A02-03).
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return refuse(
        RefusalCode::VendorCallFailed,
        std::format("SPAN FP8 : essai au chargement (cudaDeviceSynchronize) : {}", cudaGetErrorString(e)));
  }
  shared->used(this, need);
  return validate_nets(package, weights, shared->data(), native_blocks, vertical_winograd);
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
                                                         std::span<const Tensor> outputs, bool ring) {
  end_tiles();  // un appel d'avant arrêté entre le corps et la queue ne laisse pas ses masques
  if (ring) tile_job_.reset();
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
  auto tiled = start_tiles(stream, out.handle + out.offset);
  if (!tiled) return std::unexpected(std::move(tiled.error()));
  // Le contenu du masque vient d'être enfilé sur stream, avant le corps. Le graphe masqué
  // pointe toujours vers le même masque d'appareil et lit son nouveau contenu à chaque rejeu.
  auto ok = body(stream, source, *tiled);
  if (!ok && *tiled) end_tiles();
  return ok;
}

std::expected<void, Refusal> SpanFp8Engine::enqueue_tail(cudaStream_t stream, std::span<const Tensor> outputs,
                                                         const SpanFp8Ring& ring) {
  if (outputs.size() != 1) return refuse(RefusalCode::ShapeInvalid, "SPAN FP8 : une sortie attendue");
  const Tensor& out = outputs[0];
  // NOLINTNEXTLINE(performance-no-int-to-ptr) : un tenseur porte son adresse CUDA en entier
  void* target = reinterpret_cast<void*>(out.handle + out.offset);
  std::string why;
  const bool launched = net_->launch_tail(stream, target, ring, why);
  end_tiles();
  if (!launched) return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : " + why);
  return {};
}

std::expected<void, Refusal> SpanFp8Engine::body(cudaStream_t stream, const void* source, bool tiled) {
  const auto direct = [&]() -> std::expected<void, Refusal> {
    if (std::string why; !net_->launch_body(stream, source, why))
      return refuse(RefusalCode::VendorCallFailed, "SPAN FP8 : " + why);
    return {};
  };
  // Accumulateur f16 : des images débordées rapprochées (sommes bornées) passent le réseau au f32 pour un
  // temps, puis il revient au f16 ; les graphes capturés portent les anciens lancements.
  if (const SpanFp8AccChange change = net_->poll_accumulator(); change != SpanFp8AccChange::None) {
    drop_graphs();
    runtime_notice_ =
        change == SpanFp8AccChange::ToF32
            ? std::format(
                  "SPAN FP8 : {} images débordées en f16 depuis le chargement, dont plusieurs à la suite "
                  "(sommes bornées) ; le réseau accumule en f32 pour un temps",
                  net_->overflow_frames())
            : std::string("SPAN FP8 : retour à l'accumulateur f16 après le passage en f32");
  }
  BodyGraphs& graphs = tiled ? tiled_graphs_ : full_graphs_;
  const GraphBook<std::uintptr_t>::Pick pick = graphs.book.next(reinterpret_cast<std::uintptr_t>(source));
  if (pick.step == GraphStep::Direct || pick.step == GraphStep::Overflow || pick.step == GraphStep::Warm)
    return direct();  // Warm : le premier lancement d'une adresse, hors capture
  if (graphs.execs.size() <= pick.index) graphs.execs.resize(pick.index + 1, nullptr);
  cudaGraphExec_t& exec = graphs.execs[pick.index];
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
    graphs.book.captured(
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
  graphs.book.launch_failed(pick.index);
  if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
    return std::unexpected(lost_refusal(lost, "SPAN FP8 : cudaGraphLaunch"));
  return direct();
}

}  // namespace kirari::gpu::cuda
