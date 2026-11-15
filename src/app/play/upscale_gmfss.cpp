// GMFSS Fortuna Union dans l'étage gpu (upscale_stage.h, kernels/spec/gmfss.h). Le graphe « paire » une fois
// par paire, sur a et b à la taille de RIFE ; à chaque image voulue, huit splats (l'image et les
// caractéristiques du niveau 1, les niveaux 2 et 3, deux côtés) puis le graphe « fusion », qui reçoit
// l'instant.
//
// Le mélange (GmfssUse::Middle) : une paire qui bouge fort (mouvement mesuré au seuil strong_px) a ses
// clés, des images GMFSS qui servent de sources à RIFE : une au milieu, deux ou trois aux instants k / (n +
// 1) quand le mouvement est grand (gmfss_keys). Une paire qui bouge moins a sa clé quand le crédit de temps
// déborde (gmfss_.spare). Chaque autre instant t est fait par RIFE mouvement sur son sous-intervalle (a, K1),
// (K1, K2)… (Kn, b), puis le warp de l'étage ; un instant sur une clé est la clé elle-même. Le mouvement y
// est deux à quatre fois plus petit, là où RIFE fait le moins de fantômes, et les formes viennent de GMFSS.
// Mesuré sur Re:Zero (docs/reports/gmfss-union-qualite-2026-11-14.md). Les autres paires restent à RIFE seul.
//
// GMFSS seul (GmfssUse::Only) : la fusion à chaque instant, comme RIFE image, sans RIFE.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <string_view>

#include "app/play/links.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {
namespace {

// Les plans [first, first + count) d'un tenseur F16 1 × c × h × w : les plans se suivent en mémoire, la
// tranche est le même tampon décalé de first plans.
gpu::Tensor channel_slice(const gpu::Tensor& whole, std::int64_t first, std::int64_t count) {
  gpu::Tensor slice = whole;
  const std::int64_t plane = whole.shape.dims[2] * whole.shape.dims[3];
  slice.shape.dims[1] = count;
  slice.offset += static_cast<std::uint64_t>(first * plane) * sizeof(std::uint16_t);
  return slice;
}

// Les plans remplis de GMFSS : la taille de RIFE au multiple de kGmfssAlign (GMFlow divise par 64, le RIFE
// d'une fusion à IFNet remplacé travaille à la demi-taille au 1/64).
kernels::RifeGeometry gmfss_geometry(int width, int height) {
  return kernels::rife_geometry(width, height, kernels::kGmfssAlign);
}

std::expected<void, Refusal> alloc_f16(gpu::Backend& backend, std::unique_ptr<gpu::TensorBuffer>& into,
                                       int channels, int width, int height) {
  auto tensor = backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, channels, height, width}});
  if (!tensor) return std::unexpected(std::move(tensor.error()));
  into = std::move(*tensor);
  return {};
}

// Ce que le mode demande à l'étage : les deux graphes ; le mélange avec RIFE mouvement ; GMFSS seul sans
// autre interpolation, à la taille de travail.
std::expected<void, Refusal> gmfss_fits(const UpscaleSetup& s, bool by_motion, bool drba) {
  if (s.gmfss.pair == nullptr || s.gmfss.fusion == nullptr)
    return refuse(RefusalCode::FieldMissing, "GMFSS : les graphes « paire » et « fusion » manquent");
  if (s.gmfss.use == GmfssUse::Middle && !by_motion)
    return refuse(RefusalCode::FieldMissing, "GMFSS au milieu : RIFE mouvement manque");
  const bool alone = s.rife == nullptr && !by_motion && !drba;
  if (s.gmfss.use == GmfssUse::Only &&
      (!alone || s.rife_width != s.work_width || s.rife_height != s.work_height))
    return refuse(RefusalCode::FieldOutOfRange,
                  "GMFSS seul : sans RIFE ni DistilDRBA, à la taille de travail");
  if (s.gmfss.pair->input_names().size() != 1 || s.gmfss.fusion->output_names().size() != 1)
    return refuse(RefusalCode::GraphInvalid,
                  "GMFSS : une entrée à la paire, une sortie à la fusion attendues");
  return {};
}

// Les tenseurs nommés des deux graphes, aux plans remplis g : par côté, half et feature à 1/2, puis par
// niveau l (1/2, 1/4, 1/8) source (niveaux 2 et 3), motion et warped ; enfin time, 1 × 1 × 1 × 1.
std::expected<void, Refusal> allocate_named(gpu::Backend& backend, GmfssBuffers& gmfss,
                                            const kernels::RifeGeometry& g) {
  std::vector<std::pair<std::string, std::array<int, 2>>> wanted;  // nom, plans et diviseur
  for (int side = 0; side < 2; ++side) {
    wanted.push_back({std::format("half{}", side), {kernels::kGmfssImagePlanes, 2}});
    wanted.push_back({std::format("feature{}_1", side), {kernels::kGmfssFeaturePlanes, 2}});
    for (int level = 1; level <= kernels::kGmfssLevels; ++level) {
      const int channels = kernels::gmfss_warped_channels(level);
      if (level > 1) wanted.push_back({std::format("source{}_{}", side, level), {channels, 1 << level}});
      wanted.push_back(
          {std::format("motion{}_{}", side, level), {kernels::kGmfssMotionChannels, 1 << level}});
      wanted.push_back({std::format("warped{}_{}", side, level), {channels, 1 << level}});
    }
  }
  for (const auto& [name, dims] : wanted) {
    const bool doubled = !name.starts_with("warped");  // une sortie de la paire : un tenseur par jeu
    for (int set = 0; set < (doubled ? 2 : 1); ++set) {
      std::unique_ptr<gpu::TensorBuffer> tensor;
      const int scale = dims[1];
      if (auto ok = alloc_f16(backend, tensor, dims[0], g.padded_width / scale, g.padded_height / scale); !ok)
        return ok;
      gmfss.named.emplace_back(set == 0 ? name : name + "#1", std::move(tensor));
    }
  }
  std::unique_ptr<gpu::TensorBuffer> time;
  if (auto ok = alloc_f16(backend, time, 1, 1, 1); !ok) return ok;
  gmfss.named.emplace_back("time", std::move(time));
  return {};
}

const gpu::Tensor* named_tensor(const GmfssBuffers& gmfss, std::string_view name) {
  const auto found = std::ranges::find(gmfss.named, name, [](const auto& entry) { return entry.first; });
  return found == gmfss.named.end() ? nullptr : &found->second->tensor();
}

// Le tenseur name du jeu set : sa copie « #1 » au second jeu, sinon le commun (warped, time).
const gpu::Tensor* set_tensor(const GmfssBuffers& gmfss, std::string_view name, std::uint32_t set) {
  if (set == 1) {
    if (const gpu::Tensor* own = named_tensor(gmfss, std::string(name) + "#1")) return own;
  }
  return named_tensor(gmfss, name);
}

// Les tenseurs dans l'ordre que dit le moteur ; un nom inconnu vient d'un autre export.
std::expected<void, Refusal> order_tensors(const GmfssBuffers& gmfss, std::uint32_t set,
                                           std::span<const std::string> names, std::vector<gpu::Tensor>& into,
                                           std::string_view graph) {
  for (const std::string& name : names) {
    const gpu::Tensor* found = set_tensor(gmfss, name, set);
    if (found == nullptr)
      return refuse(RefusalCode::GraphInvalid,
                    std::format("GMFSS : tenseur « {} » inconnu dans le graphe {}", name, graph));
    into.push_back(*found);
  }
  return {};
}

// Chaque côté pousse ses sources par son mouvement ; au niveau 1, l'image puis les caractéristiques, chacune
// dans sa tranche de warped{k}_1.
void list_splats(GmfssBuffers& gmfss, std::uint32_t set) {
  const auto at = [&](const std::string& name) { return *set_tensor(gmfss, name, set); };
  std::vector<GmfssBuffers::Splat>& splats = gmfss.splats[set];
  for (int side = 0; side < 2; ++side) {
    for (int level = 1; level <= kernels::kGmfssLevels; ++level) {
      const gpu::Tensor motion = at(std::format("motion{}_{}", side, level));
      const gpu::Tensor warped = at(std::format("warped{}_{}", side, level));
      if (level > 1) {
        splats.push_back({.side = side,
                          .source = at(std::format("source{}_{}", side, level)),
                          .motion = motion,
                          .warped = warped});
        continue;
      }
      splats.push_back({.side = side,
                        .source = at(std::format("half{}", side)),
                        .motion = motion,
                        .warped = channel_slice(warped, 0, kernels::kGmfssImagePlanes)});
      splats.push_back(
          {.side = side,
           .source = at(std::format("feature{}_1", side)),
           .motion = motion,
           .warped = channel_slice(warped, kernels::kGmfssImagePlanes, kernels::kGmfssFeaturePlanes)});
    }
  }
}

}  // namespace

std::expected<void, Refusal> UpscaleStage::allocate_gmfss() {
  if (auto ok = allocate_half(); !ok) return ok;  // appelé à la fin d'allocate, après l'échelle par paire
  gmfss_.early_t.fill(-1.0F);
  const GmfssSetup& setup = setup_.gmfss;
  if (setup.use == GmfssUse::Off) return {};
  if (auto ok = gmfss_fits(setup_, by_motion(), drba()); !ok) return ok;
  gpu::Backend& backend = *setup_.backend;
  const kernels::RifeGeometry g = gmfss_geometry(setup_.rife_width, setup_.rife_height);
  if (auto ok =
          alloc_f16(backend, gmfss_.frames, kernels::kGmfssFramePlanes, g.padded_width, g.padded_height);
      !ok)
    return ok;
  if (auto ok = alloc_f16(backend, gmfss_.out, 3, g.padded_width, g.padded_height); !ok) return ok;
  if (auto ok = allocate_named(backend, gmfss_, g); !ok) return ok;
  gmfss_.time = *named_tensor(gmfss_, "time");
  for (std::uint32_t set = 0; set < 2; ++set) {
    if (auto ok = order_tensors(gmfss_, set, setup.pair->output_names(), gmfss_.pair_outputs[set], "paire");
        !ok)
      return ok;
    if (auto ok =
            order_tensors(gmfss_, set, setup.fusion->input_names(), gmfss_.fusion_inputs[set], "fusion");
        !ok)
      return ok;
    list_splats(gmfss_, set);
  }
  if (setup.use == GmfssUse::Only) return {};
  // Le mélange : chaque clé à la taille de RIFE, puis à celle de travail et après SPAN si le warp les lit ;
  // l'entrée de RIFE des sous-intervalles.
  const int w = setup_.work_width;
  const int h = setup_.work_height;
  const std::uint32_t keys = std::clamp<std::uint32_t>(setup.max_keys, 1, kGmfssMaxKeys);
  for (std::uint32_t k = 0; k < keys; ++k) {
    GmfssKey& key = gmfss_.keys[k];
    if (auto ok = alloc_f16(backend, key.middle, 3, setup_.rife_width, setup_.rife_height); !ok) return ok;
    if (setup_.rife_width != w || setup_.rife_height != h) {
      if (auto ok = alloc_f16(backend, key.work, 3, w, h); !ok) return ok;
    }
    if (warps_spanned()) {
      if (auto ok = alloc_f16(backend, key.spanned, 3, w * kSpanFactor, h * kSpanFactor); !ok) return ok;
    }
  }
  if (setup_.rife_width != w || setup_.rife_height != h) {
    if (auto ok = allocate_compose(); !ok) return ok;
  }
  const kernels::RifeGeometry r =
      kernels::rife_geometry(setup_.rife_width, setup_.rife_height, setup_.rife_align);
  return alloc_f16(backend, gmfss_.input, setup_.rife_planes, r.padded_width, r.padded_height);
}

std::expected<void, Refusal> UpscaleStage::allocate_compose() {
  // Une seule échelle du travail à RIFE (MotionGrid) : les deux axes au même rapport, à 1 % près.
  const double rx = static_cast<double>(setup_.work_width) / setup_.rife_width;
  const double ry = static_cast<double>(setup_.work_height) / setup_.rife_height;
  if (!setup_.gmfss.compose || std::abs(rx - ry) > 0.01 * rx) return {};
  constexpr int kSplatted = kernels::kGmfssImagePlanes + 1;  // et la couverture
  for (std::size_t side = 0; side < 2; ++side) {
    if (auto ok =
            alloc_f16(*setup_.backend, gmfss_.high[side], kSplatted, setup_.work_width, setup_.work_height);
        !ok)
      return ok;
    if (auto ok =
            alloc_f16(*setup_.backend, gmfss_.low[side], kSplatted, setup_.rife_width, setup_.rife_height);
        !ok)
      return ok;
  }
  return {};
}

std::expected<void, Refusal> UpscaleStage::compose_key(const GmfssKey& key, float t) {
  // Les sources de la paire splattées par le flot du niveau 1 à l'instant t de la clé, à la taille de travail
  // et à celle de RIFE, puis K = H + up(F − L) dans key.work.
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const kernels::RifeGeometry g = gmfss_geometry(setup_.rife_width, setup_.rife_height);
  const float ox = static_cast<float>(g.left) / 2.0F;
  const float oy = static_cast<float>(g.top) / 2.0F;
  const float scale = 2.0F * static_cast<float>(setup_.work_width) / static_cast<float>(setup_.rife_width);
  gmfss_.jobs.clear();
  for (std::uint32_t side = 0; side < 2; ++side) {
    const std::uint32_t image = side == 0 ? current_ ^ 1U : current_;
    const gpu::Tensor motion = *set_tensor(gmfss_, std::format("motion{}_1", side), gmfss_.set);
    const float s = kernels::gmfss_side_time(static_cast<int>(side), t);
    gmfss_.jobs.push_back({.source = rgb_[image]->tensor(),
                           .motion = motion,
                           .s = s,
                           .out = gmfss_.high[side]->tensor(),
                           .grid = {.scale = scale, .x = ox, .y = oy},
                           .coverage = true});
    gmfss_.jobs.push_back({.source = rife_source(image),
                           .motion = motion,
                           .s = s,
                           .out = gmfss_.low[side]->tensor(),
                           .grid = {.scale = 2.0F, .x = ox, .y = oy},
                           .coverage = true});
  }
  if (auto ok = ops.soft_splat_fill(gmfss_.jobs, gpu::Tensor{}, t); !ok) return ok;
  return ops.gmfss_compose(
      {.high_a = gmfss_.high[0]->tensor(),
       .high_b = gmfss_.high[1]->tensor(),
       .low_a = gmfss_.low[0]->tensor(),
       .low_b = gmfss_.low[1]->tensor(),
       .fusion = gmfss_.out->tensor(),
       .window = {.x = g.left, .y = g.top, .width = setup_.rife_width, .height = setup_.rife_height},
       .t = t,
       .out = key.work->tensor()});
}

bool UpscaleStage::gmfss_pair() const {
  // Une B lissée n'est pas une source : la paire reste à RIFE. Le mouvement est celui que la paire a décidé
  // (begin_motion_pair : le sien, ou celui de la précédente relu sans attente) ; le mélange n'a pas de paire
  // prise d'emblée à chaque instant (motion_direct), la passe du milieu de RIFE mesure chaque paire. Le
  // seuil est donné à 720 lignes : mis à la hauteur où RIFE mesure.
  // Sous le seuil, la clé ne se pose que si le crédit déborde (gmfss_.spare) et que la paire bouge assez
  // pour que RIFE seul y laisse des fantômes.
  if (gmfss_.behind) return false;
  return gmfss_strong() || (gmfss_.spare && gmfss_moving());
}

bool UpscaleStage::gmfss_moving() const {
  constexpr float kSpareFloorPx = 4.0F;  // à 720 lignes : en dessous, RIFE seul est déjà propre
  if (setup_.gmfss.use != GmfssUse::Middle || smooth_first_ || copy_pair_) return false;
  return pair_motion_ >= kSpareFloorPx * static_cast<float>(setup_.rife_height) / 720.0F;
}

bool UpscaleStage::gmfss_strong() const {
  if (setup_.gmfss.use != GmfssUse::Middle || smooth_first_ || copy_pair_) return false;
  return pair_motion_ >= setup_.gmfss.strong_px * static_cast<float>(setup_.rife_height) / 720.0F;
}

std::expected<void, Refusal> UpscaleStage::run_gmfss(float t, const gpu::Tensor& into) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (gmfss_only()) {  // fusion déjà enfilée à l'arrivée de la paire (run_gmfss_early)
    for (std::size_t k = 0; k < gmfss_.early_t.size(); ++k) {
      if (gmfss_.early_t[k] == t) return ops.rife_crop(gmfss_.early[k]->tensor(), into);
    }
  }
  if (auto ok = fuse_gmfss(t, gmfss_.out->tensor()); !ok) return ok;
  return ops.rife_crop(gmfss_.out->tensor(), into);
}

std::expected<void, Refusal> UpscaleStage::run_gmfss_pair() {
  // Sur sa voie et dans l'autre jeu de tenseurs : la paire tourne pendant les fusions encore en file de la
  // paire d'avant, qui lisent leur jeu (gpu/lanes.h : le backend pose les attentes entre voies, sur les
  // images d'entrée écrites par la mesure et sur le jeu que des fusions lisent encore).
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const gpu::Lane back = lane_;
  gmfss_.set ^= 1U;
  use_lane(gpu::Lane::Pair);
  auto ok = ops.gmfss_frames(rife_source(current_ ^ 1U), rife_source(current_), gmfss_.frames->tensor());
  if (ok) {
    const std::array inputs{gmfss_.frames->tensor()};
    ok = blame_stage(setup_.backend->enqueue(*setup_.gmfss.pair, inputs, gmfss_.pair_outputs[gmfss_.set]),
                     kMotionRefusalKey);
  }
  use_lane(back);
  if (!ok) return ok;
  gmfss_.pair_ready = true;
  ++gmfss_.pairs;
  return {};
}

std::expected<void, Refusal> UpscaleStage::run_gmfss_ahead() {
  // La paire en cours a eu son graphe « paire » (sinon le jeu de l'autre côté pourrait encore lui servir) ;
  // le mélange ne le lance que si celle-ci bougeait fort : la suivante bougera sans doute de même. Rien
  // pendant une tenue lissée, où l'image d'après n'est pas la B de la paire suivante.
  if (setup_.gmfss.use == GmfssUse::Off) return {};
  // La paire en cours a-t-elle eu une fusion depuis le dernier passage ? (ahead_cool)
  if (gmfss_.fusions != gmfss_.fusions_seen) {
    gmfss_.ahead_cool = false;
  } else if (gmfss_.ahead_adopted) {
    gmfss_.ahead_cool = true;
    ++gmfss_.ahead_wasted;
  }
  gmfss_.fusions_seen = gmfss_.fusions;
  gmfss_.ahead_adopted = false;
  if (!gmfss_.pair_ready || gmfss_.ahead_queued || gmfss_.ahead_cool) return {};
  if ((!gmfss_only() && !gmfss_strong()) || smooth_first_ || smooth_second_ || !ahead_.id) return {};
  // Le mélange sans le crédit d'une M à l'image d'après : la paire suivante ira à RIFE, rien à préparer.
  if (!gmfss_only() && gmfss_.credit_us + gmfss_.credit_step_us < setup_.gmfss.cost_us) return {};
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const std::uint32_t set = gmfss_.set ^ 1U;
  const gpu::Tensor& next = ahead_.small ? ahead_.small->tensor() : ahead_.rgb->tensor();
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Pair);
  auto ok = ops.gmfss_frames(rife_source(current_), next, gmfss_.frames->tensor());
  if (ok) {
    const std::array inputs{gmfss_.frames->tensor()};
    ok = blame_stage(setup_.backend->enqueue(*setup_.gmfss.pair, inputs, gmfss_.pair_outputs[set]),
                     kMotionRefusalKey);
  }
  use_lane(back);
  if (!ok) return ok;
  gmfss_.ahead_queued = true;
  gmfss_.ahead_frame = ahead_.id->value;
  gmfss_.ahead_epoch = ahead_.epoch;
  gmfss_.ahead_left = source_serial_[current_];
  ++gmfss_.ahead_pairs;
  return {};
}

void UpscaleStage::adopt_gmfss_ahead(const DecodedFrame& frame, bool taken) {
  // next_source vient de passer : l'image de gauche est current_ ^ 1. La paire faite en avance sert si
  // l'image d'après est bien celle-ci, prise telle qu'ingérée en avance, à droite de la même image.
  const bool same = taken && gmfss_.ahead_queued && gmfss_.ahead_frame == frame.id.value &&
                    gmfss_.ahead_epoch == frame.epoch && gmfss_.ahead_left == source_serial_[current_ ^ 1U];
  gmfss_.ahead_queued = false;
  if (!same) return;
  gmfss_.set ^= 1U;
  gmfss_.pair_ready = true;
  gmfss_.ahead_adopted = true;
  ++gmfss_.ahead_used;
}

std::expected<void, Refusal> UpscaleStage::fuse_gmfss(float t, const gpu::Tensor& padded) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (!gmfss_.pair_ready) {  // le graphe « paire » sert toutes les images de la paire
    if (auto ok = run_gmfss_pair(); !ok) return ok;
  }
  gmfss_.jobs.clear();
  for (const GmfssBuffers::Splat& splat : gmfss_.splats[gmfss_.set]) {
    gmfss_.jobs.push_back({.source = splat.source,
                           .motion = splat.motion,
                           .s = kernels::gmfss_side_time(splat.side, t),
                           .out = splat.warped});
  }
  if (auto ok = ops.soft_splat_fill(gmfss_.jobs, gmfss_.time, t); !ok) return ok;
  std::array results{padded};
  if (auto ok = blame_stage(
          setup_.backend->enqueue(*setup_.gmfss.fusion, gmfss_.fusion_inputs[gmfss_.set], results),
          kMotionRefusalKey);
      !ok)
    return ok;
  ++gmfss_.fusions;
  return {};
}

std::expected<void, Refusal> UpscaleStage::run_gmfss_early() {
  // Comme les passes à l'arrivée de RIFE (upscale_early.cpp) : sans elles, une fusion partait à chaque slot
  // rendu par l'écran et le GPU attendait entre elles. Les splats et la fusion d'un instant se suivent sur la
  // voie d'interpolation, dans l'ordre : warped et time servent à tour de rôle sans se marcher dessus. Le
  // mélange y fait M ; les demi-paires de RIFE restent au slot.
  gmfss_.early_t.fill(-1.0F);
  if (!early_possible()) return {};  // early_.runs et gmfss_.early ont kMaxInstantsPerPair places
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Interpolate);
  std::expected<void, Refusal> ok;
  if (!gmfss_only()) {
    if (gmfss_pair()) ok = make_keys();
    use_lane(back);
    return ok;
  }
  const kernels::RifeGeometry g = gmfss_geometry(setup_.rife_width, setup_.rife_height);
  for (std::uint32_t k = 0; k < outputs_ && ok; ++k) {
    const float t = instants_[k].t;
    if (t <= 0.0F || t >= 1.0F) continue;
    if (!gmfss_.early[k]) {
      if (auto made = alloc_f16(*setup_.backend, gmfss_.early[k], 3, g.padded_width, g.padded_height);
          !made) {
        log_channel("gpu")->warn("GMFSS : tampons des fusions à l'arrivée refusés ({}), fusions au slot",
                                 made.error().detail);
        early_.refused = true;
        break;
      }
    }
    ok = fuse_gmfss(t, gmfss_.early[k]->tensor());
    if (ok) {
      gmfss_.early_t[k] = t;
      ++gmfss_.early_fusions;
    }
  }
  use_lane(back);
  return ok;
}

std::uint32_t UpscaleStage::gmfss_keys() const {
  // Une clé par paire du mélange, une de plus à chaque doublement du mouvement au-delà de kGmfssKeyStepPx
  // (à 720 lignes) : plus le mouvement est grand, plus RIFE a de chemin à inventer. Pas plus de clés que
  // d'instants interpolés, ni que le crédit n'en paie (gmfss_.extra_keys).
  constexpr float kGmfssKeyStepPx = 30.0F;
  if (!gmfss_pair()) return 0;
  const float step = kGmfssKeyStepPx * static_cast<float>(setup_.rife_height) / 720.0F;
  std::uint32_t keys = 1;
  for (float at = step; pair_motion_ >= at && keys < kGmfssMaxKeys; at *= 2.0F) ++keys;
  const std::uint32_t most = std::clamp<std::uint32_t>(setup_.gmfss.max_keys, 1, kGmfssMaxKeys);
  return std::min({keys, most, 1 + gmfss_.extra_keys, std::max<std::uint32_t>(intermediates_, 1)});
}

std::expected<void, Refusal> UpscaleStage::make_keys() {
  const std::uint32_t keys = gmfss_keys();
  while (gmfss_.keys_made < keys) {
    const std::uint32_t k = gmfss_.keys_made;
    const float t = static_cast<float>(k + 1) / static_cast<float>(keys + 1);
    if (auto ok = make_key(k, t); !ok) return ok;
    ++gmfss_.keys_made;
  }
  return {};
}

std::expected<void, Refusal> UpscaleStage::make_key(std::uint32_t k, float t) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  GmfssKey& key = gmfss_.keys[k];
  if (auto ok = run_gmfss(t, key.middle->tensor()); !ok) return ok;
  if (key.work && gmfss_.high[0]) {  // la composition, ou l'agrandissement si le backend la refuse
    auto composed = compose_key(key, t);
    if (!composed && composed.error().code != RefusalCode::FormatUnsupported) return composed;
    if (!composed) {
      log_channel("gpu")->warn("GMFSS : composition refusée ({}), clé agrandie", composed.error().detail);
      gmfss_.high = {};
      gmfss_.low = {};
    } else {
      ++gmfss_.composed;
    }
  }
  if (key.work && !gmfss_.high[0]) {  // RIFE plus petit que le travail (travail 1080p, RIFE 720p) : agrandie
    if (auto ok = ops.resample(key.middle->tensor(), kernels::Filter::Spline36, key.work->tensor()); !ok)
      return ok;
  }
  // La première clé paie la paire, les suivantes leur fusion et leurs splats.
  gmfss_.credit_us -= k == 0 ? setup_.gmfss.cost_us : setup_.gmfss.cost_us - setup_.gmfss.pair_us;
  if (k == 0 && !gmfss_strong()) ++gmfss_.spare_keys;
  if (k > 0) ++gmfss_.more_keys;
  key.serial = ++serials_;  // jamais celui d'une source
  if (!warps_spanned()) return {};
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Upscale);
  const std::array inputs{key.work ? key.work->tensor() : key.middle->tensor()};
  std::array results{key.spanned->tensor()};
  auto ok = blame_stage(setup_.backend->enqueue(*setup_.engine, inputs, results), kSpanRefusalKey);
  // Les sources de la paire sont gradées avant le warp : la clé aussi, sinon l'image mêlerait les deux.
  if (ok && setup_.grader && !setup_.yuv_slots && pair_graded()) {
    auto graded = setup_.grader->grade(ops, key.spanned->tensor(), debanded_);
    if (!graded) {
      log_channel("gpu")->warn("grade avant le warp (GMFSS) : {} ; l'écran reprend le grade",
                               graded.error().detail);
      setup_.grader.reset();
    }
  }
  use_lane(back);
  return ok;
}

std::expected<void, Refusal> UpscaleStage::span_rife(const KeySide& left, const KeySide& right, float u) {
  // Le sous-intervalle à son instant u ; l'entrée de RIFE n'est réécrite que pour un autre sous-intervalle.
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const std::array<std::uint64_t, 2> images{left.serial, right.serial};
  const gpu::Tensor& input = gmfss_.input->tensor();
  if (gmfss_.in_pair && gmfss_.in_images == images && rife_time_) {
    if (gmfss_.in_t != u) {
      if (auto ok = ops.rife_time(input, u); !ok) return ok;
    }
  } else {
    auto ok = ops.rife_input(*left.rife, *right.rife, u, input);
    gmfss_.in_pair = ok.has_value();
    if (!ok) return ok;
    gmfss_.in_images = images;
  }
  gmfss_.in_t = u;
  std::array results{flow_->tensor(), mask_->tensor()};
  // Le moteur garde l'encodage de sa dernière passe : celui de la paire en cours n'y est plus.
  const std::uint32_t buffer = rife_pair_;
  rife_pair_ = copies_;  // un tampon d'entrée qui n'est pas celui de rife_ins_
  auto queued = enqueue_on(*setup_.rife_motion, input, images, rife_encoded_, results);
  rife_pair_ = buffer;
  warm_from_ = nullptr;
  if (!queued) return queued;
  ++rife_runs_;
  ++gmfss_.halves;
  return {};
}

std::expected<void, Refusal> UpscaleStage::interpolate_middle(float t, const gpu::Tensor& a,
                                                              const gpu::Tensor& b, const Target& out) {
  if (auto ok = make_keys(); !ok) return ok;
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const std::uint32_t keys = gmfss_.keys_made;
  const float step = 1.0F / static_cast<float>(keys + 1);
  // Le sous-intervalle de t : entre la clé i − 1 (ou a) et la clé i (ou b).
  const auto i = std::min(static_cast<std::uint32_t>(t / step), keys);
  const float u = (t - static_cast<float>(i) * step) / step;
  const auto shown = [this](std::uint32_t k) -> const gpu::Tensor& {
    const GmfssKey& key = gmfss_.keys[k];
    return warps_spanned() ? key.spanned->tensor() : key.work ? key.work->tensor() : key.middle->tensor();
  };
  constexpr float kOnKey = 1.0e-4F;
  const auto on = [&](std::uint32_t k) -> std::expected<void, Refusal> {  // la clé elle-même
    const gpu::Tensor& m = shown(k);
    if (out.ring != nullptr) return ops.write_ring(m, *out.ring->slot, out.ring->sync);
    return ops.resample(m, kernels::Filter::Area, *out.tensor);  // même taille : copie exacte
  };
  if (i > 0 && u < kOnKey) return on(i - 1);
  if (i < keys && u > 1.0F - kOnKey) return on(i);
  const KeySide left =
      i == 0 ? KeySide{.rife = &rife_source(current_ ^ 1U), .serial = source_serial_[current_ ^ 1U]}
             : KeySide{.rife = &gmfss_.keys[i - 1].middle->tensor(), .serial = gmfss_.keys[i - 1].serial};
  const KeySide right =
      i == keys ? KeySide{.rife = &rife_source(current_), .serial = source_serial_[current_]}
                : KeySide{.rife = &gmfss_.keys[i].middle->tensor(), .serial = gmfss_.keys[i].serial};
  if (auto ok = span_rife(left, right, u); !ok) return ok;
  const gpu::Motion motion{.flow = flow_->tensor(), .mask = mask_->tensor(), .area = mid_motion(mid_).area};
  return blend(i == 0 ? a : shown(i - 1), i == keys ? b : shown(i), motion, 0.5F, out);
}

}  // namespace ajp::app::play
