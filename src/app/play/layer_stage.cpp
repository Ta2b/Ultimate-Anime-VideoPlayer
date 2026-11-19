#include "app/play/layer_stage.h"

#include <array>
#include <cstddef>
#include <format>
#include <string>
#include <utility>

#include "core/log.h"
#include "core/trace.h"
#include "kernels/spec/resample.h"

namespace kirari::app::play {
namespace {

int width_of(const gpu::Tensor& t) {
  return static_cast<int>(t.shape.dims[3]);
}
int height_of(const gpu::Tensor& t) {
  return static_cast<int>(t.shape.dims[2]);
}

}  // namespace

std::expected<void, Refusal> LayerStage::allocate(gpu::Backend& backend, const LayerSetup& setup,
                                                  const gpu::Shape& work_shape, const gpu::Shape& out_shape,
                                                  bool spanned) {
  work_width_ = static_cast<int>(work_shape.dims[3]);
  work_height_ = static_cast<int>(work_shape.dims[2]);
  if (!setup.plate_on() || !spanned) return {};
  auto tensor = backend.alloc_tensor(gpu::TensorType::F16, out_shape);
  if (!tensor) return std::unexpected(std::move(tensor.error()));
  plate_ = std::move(*tensor);
  return {};
}

void LayerStage::forget() {
  pair_.reset();
  plate_track_.forget();
  plate_step_ = {};
  plate_ready_ = false;
  key_pending_ = false;
  neighbours_.forget();
}

void LayerStage::arrive(const PairFacts& facts, const LayerSetup& setup, bool smooth_first,
                        bool smooth_second) {
  const bool continues = facts.same_epoch && !(facts.scores && pair_cut(facts));
  const bool smooth = smooth_first || smooth_second;
  pair_.reset();
  pair_layers_ = setup.layers;
  plate_step_ = {};
  if (refused_ || (!setup.layers && !setup.plate_on())) return;
  if (!continues) forget();
  if (smooth) {
    ++counts_.smooth;
    forget();
    return;
  }
  // L'affinage de la paire en retard : la caméra lk d'une paire d'avant (layer_plan.h, with_neighbour).
  neighbours_.next();
  bool borrowed = false;
  auto planned =
      plan_with_neighbour(facts.scene, neighbours_.for_pair(), work_width_, work_height_, borrowed);
  if (!planned && planned.error() == LayerSkip::Cut) neighbours_.forget();  // jamais d'un autre plan
  if (const auto own = refined_camera(facts.scene)) neighbours_.seen(*own);
  if (planned) {
    ++counts_.pairs;
    counts_.share += planned->share;
    if (borrowed) ++counts_.borrowed;
  } else {
    ++counts_.skipped[static_cast<std::size_t>(planned.error())];
  }
  if (plate_ != nullptr) {
    // Une clé décidée dont SPAN n'a pas tourné (SPAN de b repris d'un dessin tenu, fait en avance, ou image
    // sautée) : la plaque tient encore l'ancienne clé, à la place de laquelle le suivi compte déjà. Il
    // repart, la paire refait une clé.
    if (key_pending_) plate_track_.forget();
    plate_step_ = plate_track_.step(planned, continues);
    key_pending_ = plate_step_.action == PlateAction::Refresh;
    if (plate_step_.action != PlateAction::Use) plate_ready_ = false;
  }
  if (planned && setup.layers) pair_ = std::move(*planned);
  // La plaque relit les tuiles de b : gardées même sans calques sur les images intermédiaires.
  if (planned && !setup.layers && plate_step_.action != PlateAction::None) pair_ = std::move(*planned);
}

std::expected<void, Refusal> LayerStage::compose_pair(gpu::KernelOps& ops, const gpu::Tensor& a,
                                                      const gpu::Tensor& b, const gpu::Tensor& out, float t,
                                                      bool cubic) {
  if (!pair_ || !pair_layers_ || refused_) return {};
  KIRARI_TRACE("calques");
  const auto params = pair_params(*pair_, width_of(out), height_of(out), static_cast<double>(t), cubic);
  if (!params) return {};
  auto ok = ops.compose_layers({.out = out, .a = a, .b = b, .params = *params});
  if (!ok) {
    // Un backend sans le noyau (Vulkan) ou qui le refuse : l'image de RIFE reste, comme sans calques.
    log_channel("gpu")->warn("calques refusés ({}) : interpolation de RIFE seule pour la lecture",
                             ok.error().detail);
    refused_ = true;
    ++counts_.refused;
    return {};
  }
  ++counts_.composed;
  return {};
}

std::expected<void, Refusal> LayerStage::after_span(gpu::Backend& backend, const LayerSetup& setup,
                                                    const gpu::Tensor& rgb, const gpu::Tensor& spanned,
                                                    bool cubic) {
  if (refused_ || plate_ == nullptr || !setup.plate_on() || !pair_ || plate_step_.action == PlateAction::None)
    return {};
  KIRARI_TRACE("plaque de décor");
  const gpu::Tensor& plate = plate_->tensor();
  if (plate_step_.action == PlateAction::Refresh) {
    const std::array inputs{rgb};
    std::array results{plate};
    // La sortie de SPAN copiée telle quelle (même taille : Area ne mélange rien), ou le SPAN de la plaque.
    auto ok = setup.plate_copy ? backend.kernels()->resample(spanned, kernels::Filter::Area, plate)
                               : backend.enqueue(*setup.plate, inputs, results);
    if (!ok) {
      log_channel("gpu")->warn("clé de la plaque refusée ({}) : sans plaque pour la lecture",
                               ok.error().detail);
      refused_ = true;
      ++counts_.refused;
      return {};
    }
    plate_ready_ = true;
    key_pending_ = false;
    ++counts_.keys;
    if (setup.plate_copy) return {};  // la clé est la sortie de SPAN : rien à y composer
  }
  if (!plate_ready_) return {};
  const auto params = plate_params(plate_track_.key(), *pair_, plate_step_.key_to_b, width_of(spanned),
                                   height_of(spanned), cubic);
  if (!params) return {};
  auto ok = backend.kernels()->compose_layers({.out = spanned, .a = plate, .params = *params});
  if (!ok) {
    log_channel("gpu")->warn("plaque de décor refusée ({}) : SPAN seul pour la lecture", ok.error().detail);
    refused_ = true;
    ++counts_.refused;
    return {};
  }
  ++counts_.plated;
  return {};
}

bool LayerStage::rewrites(const LayerSetup& setup) const {
  if (refused_ || plate_ == nullptr || !setup.plate_on() || !pair_) return false;
  if (plate_step_.action == PlateAction::Refresh) return !setup.plate_copy;
  return plate_step_.action == PlateAction::Use && plate_ready_;
}

void LayerStage::log_summary(const LayerSetup& setup) const {
  if (!setup.layers && !setup.plate_on()) return;
  std::string skipped;
  for (std::size_t i = 0; i < kLayerSkipCount; ++i) {
    if (counts_.skipped[i] == 0) continue;
    if (!skipped.empty()) skipped += ", ";
    skipped += std::format("{} {}", name(static_cast<LayerSkip>(i)), counts_.skipped[i]);
  }
  const double share = counts_.pairs > 0 ? 100.0 * counts_.share / static_cast<double>(counts_.pairs) : 0.0;
  log_channel("gpu")->info(
      "calques : {} paires avec calques (dont {} avec la caméra lk d'une voisine, {:.0f} % de décor en "
      "moyenne), {} images composées ; sans : {}{} ; plaque : {} clés{}, {} sources reprises{}",
      counts_.pairs, counts_.borrowed, share, counts_.composed,
      skipped.empty() ? std::string("aucune") : skipped,
      counts_.smooth > 0 ? std::format(", Smooth {}", counts_.smooth) : std::string(), counts_.keys,
      setup.plate_copy ? " (reprises de SPAN)" : (setup.plate != nullptr ? " (SPAN FP16 de plus)" : ""),
      counts_.plated, refused_ ? " ; coupés par un refus du backend" : "");
}

}  // namespace kirari::app::play
