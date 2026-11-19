// L'essai au chargement du RIFE natif (RifeFusedEngine::try_native, docs/rife-natif.md). Quatre réseaux
// natifs (sommes f16 ; f16 et tête f32 ; mixtes et tête f32 ; f32), chacun réglé couche par couche sur cette
// carte. Une paire de test passe par la chaîne avec les moteurs TensorRT ; chaque morceau (l'encodeur, chaque
// bloc) est ensuite chronométré puis jugé seul, par TensorRT et par chaque réseau (rife_native_trial.h). Un
// morceau peut prendre TensorRT ou un réseau plus rapide que lui. L'effet de chacune de ces options sur la
// chaîne est mesuré seul (ce morceau natif, les autres à TensorRT) : son erreur quadratique sur la sortie,
// celle que les morceaux suivants en font. Les erreurs de morceaux différents s'ajoutent à peu près ; le
// mélange retenu est le plus rapide dont la somme prévue tient le seuil, vérifié sur la chaîne (un écart à
// la prévision relève d'autant la barre des mélanges de plusieurs morceaux natifs). Il n'est gardé que s'il
// bat TensorRT d'au moins kMinGain sur une passe de lecture chronométrée (en deçà, l'écart tient de la
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
#include <optional>
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

// Les groupes de plans jugés à part, en plans de taille plane : l'image (3 plans), ou le flot (4 plans) et le
// masque (1 plan).
std::vector<std::pair<std::size_t, std::size_t>> groups_of(bool motion) {
  if (motion) return {{0, 4}, {4, 5}};
  return {{0, 3}};
}

// Le pic de chaque groupe de want : max(1, |want| le plus grand du groupe) ; l'image va de 0 à 1, le flot en
// pixels peut dépasser 1.
std::vector<double> group_peak(const std::vector<float>& want, std::size_t plane, bool motion) {
  std::vector<double> peak;
  for (const auto& [first, end] : groups_of(motion)) {
    double top = 1.0;
    for (std::size_t i = first * plane; i < end * plane; ++i)
      if (std::isfinite(want[i])) top = std::max(top, static_cast<double>(std::abs(want[i])));
    peak.push_back(top);
  }
  return peak;
}

// L'erreur quadratique moyenne de got contre want, groupe par groupe ; infinie sur une valeur non finie
// d'un côté ou de l'autre (une référence fautive ne laisse rien passer).
std::vector<double> group_mse(const std::vector<float>& want, const std::vector<float>& got,
                              std::size_t plane, bool motion) {
  std::vector<double> mse;
  for (const auto& [first, end] : groups_of(motion)) {
    double sum = 0.0;
    for (std::size_t i = first * plane; i < end * plane && std::isfinite(sum); ++i) {
      const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
      sum = std::isfinite(got[i]) && std::isfinite(want[i]) ? sum + d * d
                                                            : std::numeric_limits<double>::infinity();
    }
    mse.push_back(sum / static_cast<double>((end - first) * plane));
  }
  return mse;
}

// Le PSNR d'un groupe : 200 dB sans erreur, moins l'infini sur une erreur non finie.
double db_of(double peak, double mse) {
  if (!std::isfinite(mse)) return -std::numeric_limits<double>::infinity();
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
}

// Le PSNR du groupe le plus bas.
double worst_db(const std::vector<double>& peak, const std::vector<double>& mse) {
  double worst = std::numeric_limits<double>::infinity();
  for (std::size_t g = 0; g < peak.size(); ++g) worst = std::min(worst, db_of(peak[g], mse[g]));
  return worst;
}

double worst_psnr(const std::vector<float>& want, const std::vector<float>& got, std::size_t plane,
                  bool motion) {
  return worst_db(group_peak(want, plane, motion), group_mse(want, got, plane, motion));
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

// Les options qu'un morceau peut prendre (times : [morceau][option]) : TensorRT, et chaque réseau plus
// rapide que lui (un réseau plus lent ne gagne rien sur TensorRT, la référence).
std::vector<std::vector<int>> candidates(const std::vector<std::vector<double>>& times) {
  std::vector<std::vector<int>> out(times.size());
  for (std::size_t p = 0; p < times.size(); ++p) {
    out[p].push_back(0);
    for (std::size_t o = 1; o < times[p].size(); ++o)
      if (times[p][o] < times[p][0]) out[p].push_back(static_cast<int>(o));
  }
  return out;
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

// Le mélange : l'option de chaque morceau, la sortie de la chaîne et ses dB contre TensorRT, ceux que la
// somme des effets prévoyait ; la ligne de journal des effets.
struct Mix {
  Options options;
  std::vector<float> out;
  double db = 0.0;
  double predicted = 0.0;
  std::string effects;
};

// L'effet de chaque option sur la chaîne : [morceau][option], l'erreur quadratique de chaque groupe (vide
// pour une option que le morceau ne peut pas prendre ; nulle pour TensorRT).
using Effects = std::vector<std::vector<std::vector<double>>>;

// Les erreurs d'un mélange, groupe par groupe, à partir de l'effet de ses morceaux : leur somme (des
// erreurs indépendantes s'ajoutent ainsi) et le carré de la somme de leurs racines (des erreurs de même
// signe aux mêmes pixels).
struct Spread {
  std::vector<double> power;
  std::vector<double> amplitude;
  int natives = 0;
};

Spread spread_of(const Options& options, const Effects& effects, std::size_t groups) {
  Spread s{.power = std::vector<double>(groups, 0.0), .amplitude = std::vector<double>(groups, 0.0)};
  for (std::size_t p = 0; p < options.size(); ++p) {
    const auto o = static_cast<std::size_t>(options[p]);
    s.natives += o == 0 ? 0 : 1;
    for (std::size_t g = 0; g < groups; ++g) {
      s.power[g] += effects[p][o][g];
      s.amplitude[g] += std::sqrt(effects[p][o][g]);
    }
  }
  for (double& a : s.amplitude) a *= a;
  return s;
}

// Ce qu'un mélange doit tenir. L'erreur prévue d'un groupe va de la somme (rho 0, au départ) au carré de la
// somme des racines (rho 1) ; un mélange d'au plus un morceau natif a été mesuré tel quel (son effet) et
// tient min_db ou non. Les autres tiennent multi_db : min_db, plus kSlack après une vérification manquée,
// qui recale aussi rho ; les mélanges manqués sont écartés.
struct Bar {
  double min_db = 0.0;
  double multi_db = 0.0;
  std::vector<double> rho;
  std::vector<Options> failed;
};

std::vector<double> predicted(const Spread& s, const std::vector<double>& rho) {
  std::vector<double> mse = s.power;
  if (s.natives <= 1) return mse;
  for (std::size_t g = 0; g < mse.size(); ++g) {
    const double extra = s.amplitude[g] - s.power[g];
    if (extra > 0.0 && rho[g] > 0.0) mse[g] += rho[g] * extra;
  }
  return mse;
}

bool holds(const Bar& bar, const std::vector<double>& peak, const std::vector<double>& mse, int natives) {
  for (std::size_t g = 0; g < peak.size(); ++g)
    if (db_of(peak[g], mse[g]) < (natives <= 1 ? bar.min_db : bar.multi_db)) return false;
  return true;
}

// Le mélange le plus rapide parmi cands (la somme des temps seuls) dont l'erreur prévue tient bar ; à temps
// égal, le plus juste. Ses options et son erreur prévue ; rien si aucun ne tient.
std::optional<std::pair<Options, std::vector<double>>> fastest_within(
    const Grid& times, const std::vector<std::vector<int>>& cands, const Effects& effects,
    const std::vector<double>& peak, const Bar& bar) {
  const std::size_t pieces = cands.size();
  std::vector<std::size_t> at(pieces, 0);  // compteur : l'indice dans cands de chaque morceau
  std::optional<std::pair<Options, std::vector<double>>> best;
  double best_ms = std::numeric_limits<double>::infinity();
  double best_db = -std::numeric_limits<double>::infinity();
  for (;;) {
    double ms = 0.0;
    Options options(pieces);
    for (std::size_t p = 0; p < pieces; ++p) {
      options[p] = cands[p][at[p]];
      ms += times[p][static_cast<std::size_t>(options[p])];
    }
    if (ms <= best_ms && std::ranges::find(bar.failed, options) == bar.failed.end()) {
      const Spread s = spread_of(options, effects, peak.size());
      std::vector<double> mse = predicted(s, bar.rho);
      const double db = worst_db(peak, mse);
      if ((ms < best_ms || db > best_db) && holds(bar, peak, mse, s.natives)) {
        best_ms = ms;
        best_db = db;
        best = std::pair{std::move(options), std::move(mse)};
      }
    }
    std::size_t p = 0;
    while (p < pieces && ++at[p] == cands[p].size()) at[p++] = 0;
    if (p == pieces) return best;
  }
}

// Après une vérification manquée : rho de chaque groupe recalé sur l'erreur mesurée (sans jamais
// descendre ; infini sur une erreur non finie), la barre des mélanges de plusieurs morceaux natifs à min_db
// plus kSlack, le mélange écarté.
void miss(Bar& bar, const Spread& s, const std::vector<double>& measured, Options options, double slack) {
  for (std::size_t g = 0; g < bar.rho.size(); ++g) {
    const double extra = s.amplitude[g] - s.power[g];
    if (extra <= 0.0) continue;
    const double rho = std::isfinite(measured[g]) ? (measured[g] - s.power[g]) / extra
                                                  : std::numeric_limits<double>::infinity();
    bar.rho[g] = std::max(bar.rho[g], rho);
  }
  bar.multi_db = bar.min_db + slack;
  bar.failed.push_back(std::move(options));
}

// Mesure l'effet de chaque option sur la chaîne (chain : sa sortie pour des options), puis prend le mélange
// le plus rapide qui le tient, vérifié sur la chaîne contre want (la sortie par TensorRT) ; une vérification
// sous min_db le fait manquer (miss). Le dernier essai ne prend qu'un mélange d'un seul morceau natif ; s'il
// n'y en a pas, tout à TensorRT.
template <class Chain>
std::expected<Mix, Refusal> choose_mix(const Grid& times, const std::vector<float>& want, std::size_t plane,
                                       bool motion, double min_db, Chain&& chain) {
  constexpr int kAttempts = 4;
  constexpr double kSlack = 0.5;
  const auto cands = candidates(times);
  const std::vector<double> peak = group_peak(want, plane, motion);
  Effects effects(times.size(), std::vector<std::vector<double>>(times.front().size()));
  Mix mix{.options = Options(times.size(), 0)};
  std::string measured;
  for (std::size_t p = 0; p < times.size(); ++p) {
    effects[p][0].assign(peak.size(), 0.0);
    for (const int o : cands[p]) {
      if (o == 0) continue;
      Options alone(times.size(), 0);
      alone[p] = o;
      auto got = chain(alone);
      if (!got) return std::unexpected(std::move(got.error()));
      auto& effect = effects[p][static_cast<std::size_t>(o)];
      effect = group_mse(want, *got, plane, motion);
      measured += std::format(" ; {} {} {:.1f} dB", RifeNativeTrial::piece_name(static_cast<int>(p)),
                              option_name(o), worst_db(peak, effect));
    }
  }
  if (!measured.empty())
    mix.effects = "RIFE natif, effet de chaque option sur la chaîne (le reste à TensorRT)" + measured;
  Bar bar{.min_db = min_db, .multi_db = min_db, .rho = std::vector<double>(peak.size(), 0.0)};
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    if (attempt + 1 == kAttempts) bar.multi_db = std::numeric_limits<double>::infinity();
    auto pick = fastest_within(times, cands, effects, peak, bar);
    if (!pick || all_trt(pick->first)) break;
    auto got = chain(pick->first);
    if (!got) return std::unexpected(std::move(got.error()));
    const std::vector<double> mse = group_mse(want, *got, plane, motion);
    if (worst_db(peak, mse) >= min_db) {
      mix.options = std::move(pick->first);
      mix.out = std::move(*got);
      mix.db = worst_db(peak, mse);
      mix.predicted = worst_db(peak, pick->second);
      return mix;
    }
    const Spread spread = spread_of(pick->first, effects, peak.size());
    miss(bar, spread, mse, std::move(pick->first), kSlack);
  }
  mix.options.assign(times.size(), 0);
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
  std::vector<RifeNativeNet*> nets_view(natives_.size());
  std::ranges::transform(natives_, nets_view.begin(), [](const auto& net) { return net.get(); });
  const std::vector<const RifeNativeNet*> view(nets_view.begin(), nets_view.end());
  RifeNativeTrial trial(*this, stream, shared, pair.input);
  // Le temps d'abord : il pose la colle que chaque réseau garde (à la volée ou à part), et la justesse des
  // morceaux, puis la chaîne, se mesurent avec elle.
  auto times = trial.times(nets_view, kLadderNames);
  if (!times) return abandon(std::move(times.error()));
  auto quality = trial.quality(view);
  if (!quality) return abandon(std::move(quality.error()));
  auto profile = trial.profile(*natives_[0], kLadderNames[0]);
  if (!profile) return abandon(std::move(profile.error()));
  std::vector<std::string> lines{
      std::format("RIFE natif : {} réseaux réglés couche par couche en {:.2f} s", natives_.size(), tuning)};
  std::ranges::move(piece_lines(*times, *quality), std::back_inserter(lines));
  lines.insert(lines.end(), profile->begin(), profile->end());
  lines.insert(lines.end(), trial.notes().begin(), trial.notes().end());

  const Options fastest = fastest_options(*times);
  auto mix = choose_mix(*times, *reference, plane, motion_, min_db,
                        [&](const Options& o) { return run_chain(route_of(o)); });
  if (!mix) return abandon(std::move(mix.error()));
  if (!mix->effects.empty()) lines.push_back(mix->effects);
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
      "{} ({}×{}, {} ; chaîne contre TensorRT {:.1f} dB (prévus {:.1f}), seuil {:.0f} dB ; contre le "
      "natif f32 : chaîne {:.1f} dB, TensorRT {:.1f} dB ; passe de lecture ×2 : TensorRT "
      "{:.2f} ms, mélange {:.2f} ms ; essai {:.1f} s)",
      verdict, w, h, mix_name(mix->options), mix->db, mix->predicted, min_db,
      worst_psnr(*exact, mix->out, plane, motion_), worst_psnr(*exact, *reference, plane, motion_),
      (*chain)[0], (*chain)[1], seconds()));
  settle(faster ? route_of(mix->options) : std::vector<int>{});
  return lines;
}

}  // namespace kirari::gpu::cuda
