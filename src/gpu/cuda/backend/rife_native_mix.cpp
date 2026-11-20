// L'essai au chargement du RIFE natif (RifeFusedEngine::try_native, docs/rife-natif.md). Quatre réseaux
// natifs (sommes f16 ; f16 et tête f32 ; mixtes et tête f32 ; f32), chacun réglé couche par couche sur cette
// carte. Une paire de test passe par la chaîne avec les moteurs TensorRT ; chaque morceau (l'encodeur, chaque
// bloc) est ensuite chronométré puis jugé seul, par TensorRT et par chaque réseau (rife_native_trial.h). La
// chaîne native f32 sert de référence quand la garde la reconnaît (rife_mix::reference_of) : chaque morceau
// f32 près du sien de TensorRT sur la même entrée, la chaîne de TensorRT pas trop loin ; sinon la chaîne de
// TensorRT, comme avant. Le mélange retenu (rife_mix::choose_mix) est le plus rapide qui tient la barre
// contre elle, vérifié sur la chaîne ; il n'est gardé que s'il bat TensorRT d'au moins kMinGain sur une passe
// de lecture chronométrée (en deçà, l'écart tient de la mesure). La paire : un dessin lisse (ondes, un carré
// aux bords francs) et le même décalé de quelques pixels, instant 0,5 ; le flot de RIFE y a du travail sans
// sortir de ce que le réseau voit en lecture.
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
#include "gpu/cuda/backend/rife_mix_choice.h"
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

// Les crans de précision d'un morceau natif, du plus rapide au plus juste ; TensorRT au-dessus.
constexpr std::array<RifeNativePrecision, 4> kLadder{{
    {.body = RifeNativeSums::kF16, .head = RifeNativeSums::kF16},
    {.body = RifeNativeSums::kF16, .head = RifeNativeSums::kF32},
    {.body = RifeNativeSums::kMixed, .head = RifeNativeSums::kF32},
    {.body = RifeNativeSums::kF32, .head = RifeNativeSums::kF32},
}};
constexpr std::array<const char*, 4> kLadderNames{"f16", "f16 et tête f32", "mixtes et tête f32", "f32"};
constexpr std::size_t kF32 = 3;
static_assert(kLadder.size() == static_cast<std::size_t>(rife_verdict::kOptionsMax));

// L'option de chaque morceau : 0 TensorRT, 1 + n le réseau n de kLadder.
using rife_mix::Options;

std::vector<int> route_of(const Options& options) {
  std::vector<int> route(options.size());
  std::ranges::transform(options, route.begin(), [](int o) { return o - 1; });
  return route;
}

// Les blocs dont le réseau retenu (options, nets : l'échelle entière) garde la colle à part plutôt que son
// entrée à la volée.
std::vector<int> apart_blocks(const Options& options,
                              const std::vector<std::unique_ptr<RifeNativeNet>>& nets) {
  std::vector<int> out;
  for (std::size_t p = 1; p < options.size(); ++p) {
    const int b = static_cast<int>(p) - 1;
    if (const int o = options[p]; o > 0 && nets[static_cast<std::size_t>(o - 1)]->can_glue(b) &&
                                  !nets[static_cast<std::size_t>(o - 1)]->glues(b))
      out.push_back(b);
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

// La ligne de journal des effets de chaque option, seule hors de la référence.
std::string effects_line(const std::vector<rife_mix::Effect>& effects, bool native) {
  std::string line = std::format("RIFE natif, effet de chaque option sur la chaîne (le reste {})",
                                 native ? "en natif f32" : "à TensorRT");
  for (const rife_mix::Effect& e : effects)
    line += std::format(" ; {} {} {:.1f} dB", rife_mix::piece_name(e.piece), option_name(e.option), e.db);
  return line;
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

// Un réseau par cran de rungs (indices de kLadder), réglé sur stream ; une couche de mêmes sommes reprend la
// forme déjà réglée pour un cran d'avant.
std::expected<std::vector<std::unique_ptr<RifeNativeNet>>, Refusal> ladder_nets(
    const std::vector<RifeNativePiece>& pieces, const std::vector<int>& reduce, int h, int w,
    cudaStream_t stream, const std::vector<std::size_t>& rungs) {
  std::vector<RifeNativeTuned> tuned;
  std::vector<std::unique_ptr<RifeNativeNet>> nets;
  for (const std::size_t rung : rungs) {
    std::string error;
    auto net = RifeNativeNet::create(pieces, reduce, h, w, kLadder[rung], stream, tuned, error);
    if (!net) {
      if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
        return std::unexpected(lost_refusal(lost, "RIFE natif : réglage"));
      return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "RIFE natif : " + error});
    }
    nets.push_back(std::move(net));
  }
  return nets;
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

std::expected<RifeFusedEngine::NativeTrial, Refusal> RifeFusedEngine::try_native(
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
  const auto settle = [&](std::vector<int> route, bool fused_end = true) {
    keep_route(std::move(route), trt_activation);
    set_fused_end(fused_end);
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
                          h, w, stream, {0, 1, 2, 3});
  if (!nets) return abandon(std::move(nets.error()));
  natives_ = std::move(*nets);
  const double tuning = seconds();
  // Tous les réseaux sous le même plancher : la zone ne bouge plus d'un essai à l'autre, et les X_k de la
  // passe de TensorRT restent en place pour les morceaux seuls.
  for (const auto& net : natives_) own_activation_ = std::max(own_activation_, net->workspace_bytes());
  const auto run_chain = [&](const std::vector<int>& route) {
    set_route(route);
    set_fused_end(true);
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

  // La référence : la chaîne native f32, la plus proche du calcul exact, si la garde la reconnaît
  // (rife_mix_choice.h) ; sinon TensorRT, à min_db.
  auto exact = run_chain(std::vector<int>(times->size(), static_cast<int>(kF32)));
  if (!exact) return abandon(std::move(exact.error()));
  std::vector<double> pieces_db(quality->size());
  std::ranges::transform(*quality, pieces_db.begin(), [](const std::vector<double>& q) { return q[kF32]; });
  const rife_mix::Reference ref =
      rife_mix::reference_of(pieces_db, rife_mix::group_db(*exact, *reference, plane, motion_), min_db);
  const rife_mix::Judge judge{.want = ref.native ? &*exact : &*reference,
                              .plane = plane,
                              .motion = motion_,
                              .bar_db = ref.bar_db,
                              .base = ref.native ? static_cast<int>(kF32) + 1 : 0};
  lines.push_back(std::format("RIFE natif, référence du mélange : {} ({})",
                              ref.native ? "le natif f32" : "TensorRT", ref.why));
  const Options fastest = rife_mix::fastest_options(*times);
  auto mix = rife_mix::choose_mix(*times, judge, [&](const Options& o) { return run_chain(route_of(o)); });
  if (!mix) return abandon(std::move(mix.error()));
  if (!mix->effects.empty()) lines.push_back(effects_line(mix->effects, ref.native));
  const double bar = std::ranges::min(ref.bar_db);
  if (rife_mix::all_trt(mix->options)) {
    settle({});
    lines.push_back(std::format("RIFE natif écarté, TensorRT garde la chaîne ({}×{} : {})", w, h,
                                rife_mix::all_trt(fastest)
                                    ? "TensorRT plus rapide sur chaque morceau"
                                    : std::format("aucun mélange ne tient {:.1f} dB", bar)));
    return NativeTrial{.lines = std::move(lines), .verdict = {.options = Options(times->size(), 0)}};
  }
  forget_passes();
  auto chain = trial_mix_times(stream, shared, pair.input, pair.outputs, route_of(mix->options));
  if (!chain) return abandon(std::move(chain.error()));
  const bool can_fuse = chain->size() == 3;
  const bool separate = can_fuse && (*chain)[2] < (*chain)[1];
  const double native_ms = separate ? (*chain)[2] : (*chain)[1];
  const bool faster = native_ms <= (*chain)[0] * (1.0 - kMinGain);
  const std::string verdict = faster ? "RIFE natif retenu" : "RIFE natif écarté, TensorRT plus rapide";
  lines.push_back(std::format(
      "{} ({}×{}, {} ; chaîne contre la référence {:.1f} dB (prévus {:.1f}), seuil {:.1f} dB ; contre "
      "TensorRT {:.1f} dB ; contre le natif f32 : chaîne {:.1f} dB, TensorRT {:.1f} dB ; passe de lecture "
      "×2 : TensorRT {:.2f} ms, mélange {:.2f} ms ; essai {:.1f} s)",
      verdict, w, h, mix_name(mix->options), mix->db, mix->predicted, bar,
      rife_mix::worst_psnr(*reference, mix->out, plane, motion_),
      rife_mix::worst_psnr(*exact, mix->out, plane, motion_),
      rife_mix::worst_psnr(*exact, *reference, plane, motion_), (*chain)[0], native_ms, seconds()));
  if (can_fuse)
    lines.push_back(
        std::format("RIFE natif, fin du dernier bloc : fondue {:.3f} ms, à part {:.3f} ms, {} gardée",
                    (*chain)[1], (*chain)[2], separate ? "à part" : "fondue"));
  const double trt_db = rife_mix::worst_psnr(*reference, mix->out, plane, motion_);
  // Les options du verdict indexent le banc complet. settle compacte les réseaux retenus pour la lecture.
  auto recorded =
      faster ? rife_verdict::Verdict{mix->options, !separate, trt_db, apart_blocks(mix->options, natives_)}
             : rife_verdict::Verdict{.options = Options(times->size(), 0)};
  settle(faster ? route_of(mix->options) : std::vector<int>{}, !separate);
  return NativeTrial{.lines = std::move(lines), .verdict = std::move(recorded)};
}

std::expected<RifeFusedEngine::CachedTrial, Refusal> RifeFusedEngine::try_cached(
    const std::vector<RifeNativePiece>& pieces, const std::shared_ptr<SharedActivations>& shared,
    const rife_verdict::Verdict& verdict) {
  if (!hwc8_)
    return refuse(RefusalCode::FormatUnsupported,
                  "RIFE natif : blocs TensorRT en plans, le natif lit et écrit X et Y en HWC8");
  if (verdict.options.size() != static_cast<std::size_t>(plan_.blocks) + 1)
    return refuse(RefusalCode::ShapeInvalid, "RIFE natif : verdict gardé d'un autre nombre de morceaux");
  const int h = height_;
  const int w = width_;
  const std::int64_t trt_activation = own_activation_;
  const auto saved_encodes = encodes_;
  const auto settle = [&](std::vector<int> route, bool fused_end) {
    keep_route(std::move(route), trt_activation);
    set_fused_end(fused_end);
    encodes_ = saved_encodes;
  };
  settle({}, true);
  if (rife_verdict::all_trt(verdict))
    return CachedTrial{.held = true,
                       .lines = {std::format("RIFE natif écarté d'après le verdict gardé ({}×{}), TensorRT "
                                             "garde la chaîne",
                                             w, h)}};
  const auto started = std::chrono::steady_clock::now();
  // Les seuls crans du mélange, dans l'ordre de kLadder ; la route renvoie chaque morceau à son réseau.
  std::vector<std::size_t> rungs;
  for (const int o : verdict.options)
    if (o > 0 && std::ranges::find(rungs, static_cast<std::size_t>(o - 1)) == rungs.end())
      rungs.push_back(static_cast<std::size_t>(o - 1));
  std::ranges::sort(rungs);
  std::vector<int> route(verdict.options.size(), -1);
  for (std::size_t p = 0; p < route.size(); ++p)
    if (const int o = verdict.options[p]; o > 0)
      route[p] = static_cast<int>(std::ranges::find(rungs, static_cast<std::size_t>(o - 1)) - rungs.begin());
  Pair pair;
  if (auto ok = make_pair(pair, h, w, motion_); !ok) return std::unexpected(std::move(ok.error()));
  cudaStream_t stream = pair.scratch.stream;
  const auto abandon = [&](Refusal refusal) {
    settle({}, true);
    return std::unexpected(std::move(refusal));
  };
  auto nets = ladder_nets(pieces, std::vector<int>(plan_.reduce.begin(), plan_.reduce.begin() + plan_.blocks),
                          h, w, stream, rungs);
  if (!nets) return abandon(std::move(nets.error()));
  natives_ = std::move(*nets);
  for (const auto& net : natives_) own_activation_ = std::max(own_activation_, net->workspace_bytes());
  // La colle que l'essai avait gardée à part, bloc par bloc, sur le réseau de ce bloc (parse l'a vérifié
  // natif).
  for (const int b : verdict.apart)
    natives_[static_cast<std::size_t>(route[static_cast<std::size_t>(b) + 1])]->use_glue(b, false);
  set_route({});
  auto reference = trial_output(stream, shared, pair.input, pair.outputs, pair.elements);
  if (!reference) return abandon(std::move(reference.error()));
  set_route(route);
  set_fused_end(verdict.fused_end);
  auto chain = trial_output(stream, shared, pair.input, pair.outputs, pair.elements);
  if (!chain) return abandon(std::move(chain.error()));
  const auto plane = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  const double db = rife_mix::worst_psnr(*reference, *chain, plane, motion_);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const bool held = rife_verdict::still_holds(verdict, db);
  settle(held ? route : std::vector<int>{}, verdict.fused_end);
  return CachedTrial{
      .held = held,
      .lines = {std::format("RIFE natif {} ({}×{}, {}, fin {} ; chaîne contre TensorRT {:.1f} dB, {:.1f} au "
                            "verdict ; {} réseau(x) réglé(s), {:.1f} s)",
                            held ? "repris du verdict gardé" : "du verdict gardé rejeté, essai refait", w, h,
                            mix_name(verdict.options), verdict.fused_end ? "fondue" : "à part", db,
                            verdict.trt_db, rungs.size(), seconds)}};
}

}  // namespace kirari::gpu::cuda
