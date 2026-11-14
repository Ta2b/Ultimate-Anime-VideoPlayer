// GMFSS Fortuna Union dans l'étage gpu (upscale_stage.h, kernels/spec/gmfss.h). Le graphe « paire » une fois
// par paire, sur a et b à la taille de RIFE ; à chaque image voulue, huit splats (l'image et les
// caractéristiques du niveau 1, les niveaux 2 et 3, deux côtés) puis le graphe « fusion », qui reçoit
// l'instant.
//
// Le mélange (GmfssUse::Middle) : une paire qui bouge fort (mouvement mesuré au seuil strong_px) a sa fusion
// une seule fois, au milieu : l'image M. Chaque autre instant t est fait par RIFE mouvement sur la
// demi-paire (a, M) à 2t, ou (M, b) à 2t − 1, puis le warp de l'étage ; t = 1/2 est M elle-même. Le
// mouvement y est deux fois plus petit, là où RIFE fait le moins de fantômes, et les formes viennent de
// GMFSS. Mesuré sur Re:Zero (docs/reports/gmfss-union-qualite-2026-11-14.md). Les autres paires restent à
// RIFE seul.
//
// GMFSS seul (GmfssUse::Only) : la fusion à chaque instant, comme RIFE image, sans RIFE.
#include <algorithm>
#include <array>
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

// Les plans remplis de GMFSS : la taille de RIFE au multiple de 64 (ses graphes divisent par 8, GMFlow par
// 64 au plus).
kernels::RifeGeometry gmfss_geometry(int width, int height) {
  return kernels::rife_geometry(width, height, kernels::kRifeAlign);
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
    std::unique_ptr<gpu::TensorBuffer> tensor;
    const int scale = dims[1];
    if (auto ok = alloc_f16(backend, tensor, dims[0], g.padded_width / scale, g.padded_height / scale); !ok)
      return ok;
    gmfss.named.emplace_back(name, std::move(tensor));
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

// Les tenseurs dans l'ordre que dit le moteur ; un nom inconnu vient d'un autre export.
std::expected<void, Refusal> order_tensors(const GmfssBuffers& gmfss, std::span<const std::string> names,
                                           std::vector<gpu::Tensor>& into, std::string_view graph) {
  for (const std::string& name : names) {
    const gpu::Tensor* found = named_tensor(gmfss, name);
    if (found == nullptr)
      return refuse(RefusalCode::GraphInvalid,
                    std::format("GMFSS : tenseur « {} » inconnu dans le graphe {}", name, graph));
    into.push_back(*found);
  }
  return {};
}

// Chaque côté pousse ses sources par son mouvement ; au niveau 1, l'image puis les caractéristiques, chacune
// dans sa tranche de warped{k}_1.
void list_splats(GmfssBuffers& gmfss) {
  const auto at = [&](const std::string& name) { return *named_tensor(gmfss, name); };
  for (int side = 0; side < 2; ++side) {
    for (int level = 1; level <= kernels::kGmfssLevels; ++level) {
      const gpu::Tensor motion = at(std::format("motion{}_{}", side, level));
      const gpu::Tensor warped = at(std::format("warped{}_{}", side, level));
      if (level > 1) {
        gmfss.splats.push_back({.side = side,
                                .source = at(std::format("source{}_{}", side, level)),
                                .motion = motion,
                                .warped = warped});
        continue;
      }
      gmfss.splats.push_back({.side = side,
                              .source = at(std::format("half{}", side)),
                              .motion = motion,
                              .warped = channel_slice(warped, 0, kernels::kGmfssImagePlanes)});
      gmfss.splats.push_back(
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
  if (auto ok = order_tensors(gmfss_, setup.pair->output_names(), gmfss_.pair_outputs, "paire"); !ok)
    return ok;
  if (auto ok = order_tensors(gmfss_, setup.fusion->input_names(), gmfss_.fusion_inputs, "fusion"); !ok)
    return ok;
  list_splats(gmfss_);
  if (setup.use == GmfssUse::Only) return {};
  // Le mélange : M à la taille de RIFE, puis à celle de travail et après SPAN si le warp les lit ; l'entrée
  // de RIFE des demi-paires.
  const int w = setup_.work_width;
  const int h = setup_.work_height;
  if (auto ok = alloc_f16(backend, gmfss_.middle, 3, setup_.rife_width, setup_.rife_height); !ok) return ok;
  if (setup_.rife_width != w || setup_.rife_height != h) {
    if (auto ok = alloc_f16(backend, gmfss_.work, 3, w, h); !ok) return ok;
  }
  if (warps_spanned()) {
    if (auto ok = alloc_f16(backend, gmfss_.spanned, 3, w * kSpanFactor, h * kSpanFactor); !ok) return ok;
  }
  const kernels::RifeGeometry r =
      kernels::rife_geometry(setup_.rife_width, setup_.rife_height, setup_.rife_align);
  return alloc_f16(backend, gmfss_.input, setup_.rife_planes, r.padded_width, r.padded_height);
}

bool UpscaleStage::gmfss_pair() const {
  // Une B lissée n'est pas une source : la paire reste à RIFE. Le mouvement est celui que la paire a décidé
  // (begin_motion_pair : le sien, ou celui de la précédente relu sans attente) ; le mélange n'a pas de paire
  // prise d'emblée à chaque instant (motion_direct), la passe du milieu de RIFE mesure chaque paire. Le
  // seuil est donné à 720 lignes : mis à la hauteur où RIFE mesure.
  if (setup_.gmfss.use != GmfssUse::Middle || smooth_first_ || copy_pair_) return false;
  return pair_motion_ >= setup_.gmfss.strong_px * static_cast<float>(setup_.rife_height) / 720.0F;
}

std::expected<void, Refusal> UpscaleStage::run_gmfss(float t, const gpu::Tensor& into) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (!gmfss_.pair_ready) {  // le graphe « paire » sert toutes les images de la paire
    if (auto ok =
            ops.gmfss_frames(rife_source(current_ ^ 1U), rife_source(current_), gmfss_.frames->tensor());
        !ok)
      return ok;
    const std::array inputs{gmfss_.frames->tensor()};
    if (auto ok = blame_stage(setup_.backend->enqueue(*setup_.gmfss.pair, inputs, gmfss_.pair_outputs),
                              kMotionRefusalKey);
        !ok)
      return ok;
    gmfss_.pair_ready = true;
    ++gmfss_.pairs;
  }
  gmfss_.jobs.clear();
  for (const GmfssBuffers::Splat& splat : gmfss_.splats) {
    gmfss_.jobs.push_back({.source = splat.source,
                           .motion = splat.motion,
                           .s = kernels::gmfss_side_time(splat.side, t),
                           .out = splat.warped});
  }
  if (auto ok = ops.soft_splat_fill(gmfss_.jobs, gmfss_.time, t); !ok) return ok;
  std::array results{gmfss_.out->tensor()};
  if (auto ok = blame_stage(setup_.backend->enqueue(*setup_.gmfss.fusion, gmfss_.fusion_inputs, results),
                            kMotionRefusalKey);
      !ok)
    return ok;
  ++gmfss_.fusions;
  return ops.rife_crop(gmfss_.out->tensor(), into);
}

std::expected<void, Refusal> UpscaleStage::make_middle() {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (auto ok = run_gmfss(0.5F, gmfss_.middle->tensor()); !ok) return ok;
  if (gmfss_.work) {  // RIFE plus petit que le travail (source 4K) : M agrandie pour le warp
    if (auto ok = ops.resample(gmfss_.middle->tensor(), kernels::Filter::Spline36, gmfss_.work->tensor());
        !ok)
      return ok;
  }
  const auto ready = [this] {
    gmfss_.middle_ready = true;
    gmfss_.middle_serial = ++serials_;  // jamais celui d'une source
  };
  if (!warps_spanned()) {
    ready();
    return {};
  }
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Upscale);
  const std::array inputs{gmfss_.work ? gmfss_.work->tensor() : gmfss_.middle->tensor()};
  std::array results{gmfss_.spanned->tensor()};
  auto ok = blame_stage(setup_.backend->enqueue(*setup_.engine, inputs, results), kSpanRefusalKey);
  // Les sources de la paire sont gradées avant le warp : M aussi, sinon l'image mêlerait les deux.
  if (ok && setup_.grader && !setup_.yuv_slots && pair_graded()) {
    auto graded = setup_.grader->grade(ops, gmfss_.spanned->tensor(), debanded_);
    if (!graded) {
      log_channel("gpu")->warn("grade avant le warp (GMFSS) : {} ; l'écran reprend le grade",
                               graded.error().detail);
      setup_.grader.reset();
    }
  }
  use_lane(back);
  if (ok) ready();
  return ok;
}

std::expected<void, Refusal> UpscaleStage::half_pair_rife(float t) {
  // La demi-paire de t, à son instant ; l'entrée de RIFE n'est réécrite que pour une autre demi-paire.
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const bool first = t < 0.5F;
  const float u = first ? 2.0F * t : 2.0F * t - 1.0F;
  const std::array<std::uint64_t, 2> images =
      first ? std::array{source_serial_[current_ ^ 1U], gmfss_.middle_serial}
            : std::array{gmfss_.middle_serial, source_serial_[current_]};
  const gpu::Tensor& input = gmfss_.input->tensor();
  if (gmfss_.in_pair && gmfss_.in_images == images && rife_time_) {
    if (gmfss_.in_t != u) {
      if (auto ok = ops.rife_time(input, u); !ok) return ok;
    }
  } else {
    const gpu::Tensor& rife_m = gmfss_.middle->tensor();
    auto ok = first ? ops.rife_input(rife_source(current_ ^ 1U), rife_m, u, input)
                    : ops.rife_input(rife_m, rife_source(current_), u, input);
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
  if (!gmfss_.middle_ready) {
    if (auto ok = make_middle(); !ok) return ok;
  }
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const gpu::Tensor& m = warps_spanned() ? gmfss_.spanned->tensor()
                         : gmfss_.work   ? gmfss_.work->tensor()
                                         : gmfss_.middle->tensor();
  if (t == 0.5F) {  // M elle-même
    if (out.ring != nullptr) return ops.write_ring(m, *out.ring->slot, out.ring->sync);
    return ops.resample(m, kernels::Filter::Area, *out.tensor);  // même taille : copie exacte
  }
  if (auto ok = half_pair_rife(t); !ok) return ok;
  const gpu::Motion motion{.flow = flow_->tensor(), .mask = mask_->tensor(), .area = mid_motion(mid_).area};
  return t < 0.5F ? blend(a, m, motion, 0.5F, out) : blend(m, b, motion, 0.5F, out);
}

}  // namespace ajp::app::play
