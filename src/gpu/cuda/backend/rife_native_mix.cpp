// L'essai au chargement du RIFE natif (RifeFusedEngine::try_native, docs/rife-natif.md). Quatre réseaux
// natifs, du plus rapide au plus juste (sommes f16 ; f16 et tête f32 ; mixtes et tête f32 ; f32), chacun
// réglé couche par couche sur cette carte. Une paire de test passe par la chaîne avec les moteurs TensorRT ;
// chaque morceau (l'encodeur, chaque bloc) est ensuite jugé et chronométré seul, par TensorRT et par chaque
// réseau (rife_native_trial.h). Chaque morceau prend son option la plus rapide ; tant que la sortie de la
// chaîne ne tient pas le seuil contre celle de TensorRT, le morceau natif le moins juste monte d'un cran
// (au-delà du dernier, ou d'un cran plus lent que TensorRT pour lui : TensorRT). Le mélange n'est gardé que
// s'il bat TensorRT d'au moins kMinGain sur une passe de lecture chronométrée (en deçà, l'écart tient de la
// mesure). La paire : un dessin lisse (ondes, un carré aux bords francs) et le même décalé de quelques
// pixels, instant 0,5 ; le flot de RIFE y a du travail sans sortir de ce que le réseau voit en lecture.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <format>
#include <iterator>
#include <limits>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_fused_engine.h"
#include "gpu/cuda/backend/rife_native_trial.h"

namespace kirari::gpu::cuda {
namespace {

constexpr double kMinGain = 0.02;

// L'entrée v1 de RIFE (11 plans) pour la paire de test : A, B (A décalé de (5, 3) pixels), l'instant, la
// grille normalisée et ses pas.
std::vector<__half> test_pair(int h, int w) {
  const auto plane = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  std::vector<__half> input(plane * kernels::kFusedInputPlanes);
  const auto drawing = [](float x, float y, int channel) {
    const auto c = static_cast<float>(channel);
    const float wave =
        0.5F + 0.22F * std::sin(0.071F * x + 0.043F * y + c) + 0.12F * std::sin(0.017F * x - 0.029F * y);
    const bool square = std::fmod(std::abs(x), 160.F) < 64.F && std::fmod(std::abs(y), 128.F) < 48.F;
    return std::clamp(square ? 0.9F - 0.2F * c : wave, 0.F, 1.F);
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t at =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
      const auto fx = static_cast<float>(x);
      const auto fy = static_cast<float>(y);
      for (int c = 0; c < 3; ++c) {
        input[static_cast<std::size_t>(c) * plane + at] = __float2half(drawing(fx, fy, c));
        input[static_cast<std::size_t>(3 + c) * plane + at] = __float2half(drawing(fx - 5.F, fy - 3.F, c));
      }
      input[6 * plane + at] = __float2half(0.5F);
      input[7 * plane + at] = __float2half(2.F * fx / static_cast<float>(w - 1) - 1.F);
      input[8 * plane + at] = __float2half(2.F * fy / static_cast<float>(h - 1) - 1.F);
      input[9 * plane + at] = __float2half(2.F / static_cast<float>(w - 1));
      input[10 * plane + at] = __float2half(2.F / static_cast<float>(h - 1));
    }
  return input;
}

// Le PSNR de got contre want, plan par plan de taille plane, au pic max(1, |want| le plus grand du plan
// groupé) : l'image va de 0 à 1, le flot en pixels peut dépasser 1. Le plus bas des groupes (images : 3
// plans ; mouvement : flot 4 plans, masque 1 plan). Une valeur non finie, des deux côtés : -infini (une
// référence fautive ne laisse rien passer).
double worst_psnr(const std::vector<float>& want, const std::vector<float>& got, std::size_t plane,
                  bool motion) {
  const std::vector<std::pair<std::size_t, std::size_t>> groups =
      motion ? std::vector<std::pair<std::size_t, std::size_t>>{{0, 4}, {4, 5}}
             : std::vector<std::pair<std::size_t, std::size_t>>{{0, 3}};
  double worst = std::numeric_limits<double>::infinity();
  for (const auto& [first, end] : groups) {
    double peak = 1.0;
    double sum = 0.0;
    for (std::size_t i = first * plane; i < end * plane; ++i) {
      if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return -std::numeric_limits<double>::infinity();
      peak = std::max(peak, static_cast<double>(std::abs(want[i])));
      const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
      sum += d * d;
    }
    const double mse = sum / static_cast<double>((end - first) * plane);
    worst = std::min(worst, mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse));
  }
  return worst;
}

// Les crans de précision d'un morceau natif, du plus rapide au plus juste ; TensorRT au-dessus.
constexpr std::array<RifeNativePrecision, 4> kLadder{{
    {.body = RifeNativeSums::kF16, .head = RifeNativeSums::kF16},
    {.body = RifeNativeSums::kF16, .head = RifeNativeSums::kF32},
    {.body = RifeNativeSums::kMixed, .head = RifeNativeSums::kF32},
    {.body = RifeNativeSums::kF32, .head = RifeNativeSums::kF32},
}};
constexpr std::array<const char*, 4> kLadderNames{"f16", "f16 et tête f32", "mixtes et tête f32", "f32"};
constexpr std::size_t kF32 = 3;

// L'option de chaque morceau : 0 TensorRT, 1 + n le réseau n de kLadder.
using Options = std::vector<int>;

std::vector<int> route_of(const Options& options) {
  std::vector<int> route(options.size());
  std::ranges::transform(options, route.begin(), [](int o) { return o - 1; });
  return route;
}

// Le cran au-dessus de l'option o d'un morceau : le réseau suivant plus rapide que TensorRT pour lui, sinon
// TensorRT.
int next_option(const std::vector<double>& times, int o) {
  for (auto n = static_cast<std::size_t>(o) + 1; n < times.size(); ++n)
    if (times[n] < times[0]) return static_cast<int>(n);
  return 0;
}

// Le morceau natif le moins juste (quality : [morceau][réseau]), -1 s'il n'y en a plus.
int least_exact(const Options& options, const std::vector<std::vector<double>>& quality) {
  const auto db = [&](std::size_t p) { return quality[p][static_cast<std::size_t>(options[p] - 1)]; };
  int worst = -1;
  for (std::size_t p = 0; p < options.size(); ++p)
    if (options[p] > 0 && (worst < 0 || db(p) < db(static_cast<std::size_t>(worst))))
      worst = static_cast<int>(p);
  return worst;
}

std::string option_name(int o) {
  return o == 0 ? std::string("TensorRT") : std::string(kLadderNames[static_cast<std::size_t>(o - 1)]);
}

// Une ligne par morceau : le temps seul de chaque option, la justesse de chaque réseau.
std::vector<std::string> piece_lines(const std::vector<std::vector<double>>& times,
                                     const std::vector<std::vector<double>>& quality) {
  std::vector<std::string> lines;
  for (std::size_t p = 0; p < times.size(); ++p) {
    std::string line = std::format("RIFE natif, {} seul : TensorRT {:.3f} ms",
                                   RifeNativeTrial::piece_name(static_cast<int>(p)), times[p][0]);
    for (std::size_t n = 0; n < quality[p].size(); ++n)
      line += std::format(" ; {} {:.3f} ms ({:.1f} dB)", kLadderNames[n], times[p][n + 1], quality[p][n]);
    lines.push_back(std::move(line));
  }
  return lines;
}

using Grid = std::vector<std::vector<double>>;

bool all_trt(const Options& options) {
  return std::ranges::all_of(options, [](int o) { return o == 0; });
}

// L'option la plus rapide de chaque morceau (times : [morceau][option]).
Options fastest_options(const Grid& times) {
  Options options(times.size());
  for (std::size_t p = 0; p < times.size(); ++p)
    options[p] = static_cast<int>(std::ranges::min_element(times[p]) - times[p].begin());
  return options;
}

std::string mix_name(const Options& options) {
  std::string name;
  for (std::size_t p = 0; p < options.size(); ++p)
    name += std::format("{}{} {}", p == 0 ? "" : ", ", RifeNativeTrial::piece_name(static_cast<int>(p)),
                        option_name(options[p]));
  return name;
}

// La paire de test sur l'appareil, et les sorties de la chaîne d'un seul tenant : flot (4 plans) puis
// masque, ou l'image (3 plans).
struct Pair {
  FusedScratch scratch;
  Tensor input;
  std::vector<Tensor> outputs;
  std::size_t elements = 0;
};

std::expected<void, Refusal> make_pair(Pair& pair, int h, int w, bool motion) {
  // Une erreur collante (contexte perdu) le dit comme telle : l'appelant ne garde pas la chaîne pour autant.
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return std::unexpected(cuda_refusal(error, std::format("RIFE natif : essai ({})", step)));
  };
  const auto plane = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  pair.elements = (motion ? 5 : 3) * plane;
  const std::vector<__half> host = test_pair(h, w);
  FusedScratch& scratch = pair.scratch;
  if (cudaError_t e = cudaMalloc(&scratch.buffers[0], host.size() * 2); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch.buffers[1], pair.elements * 2); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemcpy(scratch.buffers[0], host.data(), host.size() * 2, cudaMemcpyHostToDevice);
      e != cudaSuccess)
    return failed(e, "cudaMemcpy");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée (comme load).
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return failed(e, "cudaDeviceSynchronize");
  const auto in = reinterpret_cast<std::uintptr_t>(scratch.buffers[0]);
  const auto out = reinterpret_cast<std::uintptr_t>(scratch.buffers[1]);
  pair.input = fused_tensor(in, 0, kernels::kFusedInputPlanes, h, w);
  if (motion)
    pair.outputs = {fused_tensor(out, 0, 4, h, w), fused_tensor(out, 4U * plane * 2U, 1, h, w)};
  else
    pair.outputs = {fused_tensor(out, 0, 3, h, w)};
  return {};
}

// Un réseau par cran, réglé sur stream ; une couche de mêmes sommes reprend la forme déjà réglée pour un cran
// d'avant.
std::expected<std::vector<std::unique_ptr<RifeNativeNet>>, Refusal> ladder_nets(
    const std::vector<RifeNativePiece>& pieces, const std::vector<int>& reduce, int h, int w,
    cudaStream_t stream) {
  std::vector<RifeNativeTuned> tuned;
  std::vector<std::unique_ptr<RifeNativeNet>> nets;
  for (const RifeNativePrecision& precision : kLadder) {
    std::string error;
    auto net = RifeNativeNet::create(pieces, reduce, h, w, precision, stream, tuned, error);
    if (!net) {
      if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
        return std::unexpected(lost_refusal(lost, "RIFE natif : réglage"));
      return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "RIFE natif : " + error});
    }
    nets.push_back(std::move(net));
  }
  return nets;
}

// Le mélange : l'option de chaque morceau, la sortie de la chaîne et ses dB contre TensorRT.
struct Mix {
  Options options;
  std::vector<float> out;
  double db = 0.0;
};

// Depuis options, tant que la chaîne (chain : sa sortie pour des options) ne tient pas min_db (judge : ses dB
// contre TensorRT), le morceau natif le moins juste monte d'un cran. Chaque pas monte un morceau sans jamais
// redescendre : la boucle finit, au pire tout à TensorRT.
template <class Chain, class Judge>
std::expected<Mix, Refusal> choose_mix(const Grid& times, const Grid& quality, Options options, double min_db,
                                       Chain&& chain, Judge&& judge) {
  Mix mix{.options = std::move(options)};
  while (!all_trt(mix.options)) {
    auto got = chain(mix.options);
    if (!got) return std::unexpected(std::move(got.error()));
    mix.db = judge(*got);
    if (mix.db >= min_db) {
      mix.out = std::move(*got);
      return mix;
    }
    const auto p = static_cast<std::size_t>(least_exact(mix.options, quality));
    mix.options[p] = next_option(times[p], mix.options[p]);
  }
  return mix;
}

}  // namespace

void RifeFusedEngine::keep_route(std::vector<int> route, std::int64_t trt_activation) {
  std::vector<std::unique_ptr<RifeNativeNet>> kept;
  std::vector<int> renamed(natives_.size(), -1);
  for (int& r : route) {
    if (r < 0) continue;
    int& slot = renamed[static_cast<std::size_t>(r)];
    if (slot < 0) {
      slot = static_cast<int>(kept.size());
      kept.push_back(std::move(natives_[static_cast<std::size_t>(r)]));
    }
    r = slot;
  }
  natives_ = std::move(kept);
  set_route(std::move(route));
  own_activation_ = std::max(trt_activation, native_workspace_);
  forget_passes();
}

std::expected<std::vector<std::string>, Refusal> RifeFusedEngine::try_native(
    const std::vector<RifeNativePiece>& pieces, const std::shared_ptr<SharedActivations>& shared,
    double min_db) {
  if (!hwc8_)
    return refuse(RefusalCode::FormatUnsupported,
                  "RIFE natif : blocs TensorRT en plans, le natif lit et écrit X et Y en HWC8");
  const int h = height_;
  const int w = width_;
  const auto plane = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  Pair pair;
  if (auto ok = make_pair(pair, h, w, motion_); !ok) return std::unexpected(std::move(ok.error()));
  cudaStream_t stream = pair.scratch.stream;
  // Quoi qu'il arrive, l'essai rend la chaîne à TensorRT ou à l'aiguillage retenu (sans les réseaux qu'il
  // n'emploie pas), sans graphe de l'essai (ils visent les tampons de l'essai, libérés au retour), et laisse
  // les comptes d'encodage de la lecture.
  const std::int64_t trt_activation = own_activation_;
  const auto saved_encodes = encodes_;
  const auto settle = [&](std::vector<int> route) {
    keep_route(std::move(route), trt_activation);
    encodes_ = saved_encodes;
  };
  const auto abandon = [&](Refusal refusal) {
    settle({});
    return std::unexpected(std::move(refusal));
  };
  settle({});
  const auto started = std::chrono::steady_clock::now();
  const auto seconds = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  };
  auto nets = ladder_nets(pieces, std::vector<int>(plan_.reduce.begin(), plan_.reduce.begin() + plan_.blocks),
                          h, w, stream);
  if (!nets) return abandon(std::move(nets.error()));
  natives_ = std::move(*nets);
  const double tuning = seconds();
  // Tous les réseaux sous le même plancher : la zone ne bouge plus d'un essai à l'autre, et les X_k de la
  // passe de TensorRT restent en place pour les morceaux seuls.
  for (const auto& net : natives_) own_activation_ = std::max(own_activation_, net->workspace_bytes());
  const auto run_chain = [&](const std::vector<int>& route) {
    set_route(route);
    return trial_output(stream, shared, pair.input, pair.outputs, pair.elements);
  };
  auto reference = run_chain({});
  if (!reference) return abandon(std::move(reference.error()));
  std::vector<const RifeNativeNet*> view(natives_.size());
  std::ranges::transform(natives_, view.begin(), [](const auto& net) { return net.get(); });
  RifeNativeTrial trial(*this, stream, shared, pair.input);
  auto quality = trial.quality(view);
  if (!quality) return abandon(std::move(quality.error()));
  auto times = trial.times(view);
  if (!times) return abandon(std::move(times.error()));
  auto profile = trial.profile(*natives_[0], kLadderNames[0]);
  if (!profile) return abandon(std::move(profile.error()));
  std::vector<std::string> lines{
      std::format("RIFE natif : {} réseaux réglés couche par couche en {:.2f} s", natives_.size(), tuning)};
  std::ranges::move(piece_lines(*times, *quality), std::back_inserter(lines));
  lines.insert(lines.end(), profile->begin(), profile->end());
  lines.insert(lines.end(), trial.notes().begin(), trial.notes().end());

  const Options fastest = fastest_options(*times);
  const auto judge = [&](const std::vector<float>& got) {
    return worst_psnr(*reference, got, plane, motion_);
  };
  auto mix = choose_mix(
      *times, *quality, fastest, min_db, [&](const Options& o) { return run_chain(route_of(o)); }, judge);
  if (!mix) return abandon(std::move(mix.error()));
  if (all_trt(mix->options)) {
    settle({});
    lines.push_back(std::format("RIFE natif écarté, TensorRT garde la chaîne ({}×{} : {})", w, h,
                                all_trt(fastest) ? "TensorRT plus rapide sur chaque morceau"
                                                 : std::format("aucun mélange ne tient {:.0f} dB", min_db)));
    return lines;
  }
  // L'écart au calcul le plus juste : le natif f32 partout, contre le mélange et contre TensorRT.
  auto exact = run_chain(std::vector<int>(mix->options.size(), static_cast<int>(kF32)));
  if (!exact) return abandon(std::move(exact.error()));
  forget_passes();
  const std::vector<std::vector<int>> routes{{}, route_of(mix->options)};
  auto chain = trial_times(stream, shared, pair.input, pair.outputs, routes);
  if (!chain) return abandon(std::move(chain.error()));
  const bool faster = (*chain)[1] <= (*chain)[0] * (1.0 - kMinGain);
  const std::string verdict = faster ? "RIFE natif retenu" : "RIFE natif écarté, TensorRT plus rapide";
  lines.push_back(std::format(
      "{} ({}×{}, {} ; chaîne contre TensorRT {:.1f} dB, seuil {:.0f} dB ; contre le "
      "natif f32 : chaîne {:.1f} dB, TensorRT {:.1f} dB ; passe de lecture ×2 : TensorRT "
      "{:.2f} ms, mélange {:.2f} ms ; essai {:.1f} s)",
      verdict, w, h, mix_name(mix->options), mix->db, min_db, worst_psnr(*exact, mix->out, plane, motion_),
      worst_psnr(*exact, *reference, plane, motion_), (*chain)[0], (*chain)[1], seconds()));
  settle(faster ? route_of(mix->options) : std::vector<int>{});
  return lines;
}

}  // namespace kirari::gpu::cuda
