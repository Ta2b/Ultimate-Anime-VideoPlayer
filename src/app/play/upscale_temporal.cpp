// La stabilisation temporelle dans l'étage gpu (temporal_fix.h, temporal_window.h) : chaque image ingérée
// pour la chaîne passe par TemporalFix, qui lit sept images de travail (trois avant, l'image, trois après).
// Les images y entrent brutes (ingérées et débandées comme sans elle) dans des tampons gardés par pts ; la
// sortie du modèle prend la place de l'image ingérée, si bien que SPAN, RIFE et les mesures lisent l'image
// stabilisée sans rien savoir d'elle.
//
// Avance : frames_ahead() attend kTemporalFixRadius images de plus derrière la plus lointaine que la chaîne
// lit ; au début d'une époque et à la fin du flux, la fenêtre se reflète (TemporalWindow::window).
#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <string>
#include <utility>

#include "app/play/links.h"
#include "app/play/temporal_fix_model.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "prepare/onnx_cache.h"
#include "prepare/onnx_frames.h"

namespace kirari::app::play {
namespace {

constexpr auto kRadius = static_cast<std::size_t>(kTemporalFixRadius);
constexpr auto kFrames = static_cast<std::size_t>(kTemporalFixFrames);

gpu::Shape planes3(int width, int height) {
  return gpu::Shape{.dims = {1, 3, height, width}};
}

}  // namespace

std::size_t UpscaleStage::temporal_ahead(std::size_t base) const {
  if (setup_.temporal_fix == nullptr) return base;
  // look_ahead lit la suivante sans que frames_ahead la compte : la fenêtre de celle-ci aussi doit être là.
  const std::size_t farthest = std::max<std::size_t>(base, looks_ahead() || setup_.drba_by_shot ? 1 : 0);
  return farthest + kRadius;
}

std::expected<void, Refusal> UpscaleStage::allocate_temporal() {
  if (setup_.temporal_fix == nullptr) return {};
  // Sans encodeur (--no-tfix-cache) : le moteur en un morceau, sept images, pas de traits gardés.
  const bool cached = setup_.temporal_encoder != nullptr;
  if (cached && setup_.temporal_features <= 0)
    return refuse(RefusalCode::ModelMismatch, "stabilisation temporelle : traits sans canaux");
  // De la tête moins le rayon à la plus lointaine lue en avance, plus un tampon de jeu.
  const std::size_t capacity = kRadius + 1 + frames_ahead() + 1;
  temporal_.window = TemporalWindow(capacity);
  temporal_.rgb.clear();
  temporal_.features.clear();
  const gpu::Shape features{.dims = {1, setup_.temporal_features, setup_.work_height, setup_.work_width}};
  for (std::size_t i = 0; i < capacity; ++i) {
    auto tensor =
        setup_.backend->alloc_tensor(gpu::TensorType::F16, planes3(setup_.work_width, setup_.work_height));
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    temporal_.rgb.push_back(std::move(*tensor));
    if (!cached) continue;
    auto encoded = setup_.backend->alloc_tensor(gpu::TensorType::F16, features);
    if (!encoded) return std::unexpected(std::move(encoded.error()));
    temporal_.features.push_back(std::move(*encoded));
  }
  const std::span<const std::string> names = setup_.temporal_fix->input_names();
  const std::size_t expected = cached ? 2 * kFrames : kFrames;
  if (names.size() != expected) {
    return refuse(RefusalCode::ModelMismatch,
                  std::format("stabilisation temporelle : {} entrées au lieu de {}", names.size(), expected));
  }
  // L'entrée k du moteur (dans l'ordre de input_names) lit l'image, ou les traits, de rang r de la fenêtre.
  for (std::size_t k = 0; k < names.size(); ++k) {
    std::size_t slot = 2 * kFrames;
    for (int r = 0; r < kTemporalFixFrames; ++r) {
      const auto rank = static_cast<std::size_t>(r);
      if (names[k] == prepare::frame_input_name(kTemporalFixInput, r)) slot = rank;
      if (names[k] == prepare::feature_input_name(kTemporalFixFeatures, r)) slot = kFrames + rank;
    }
    if (slot == 2 * kFrames) {
      return refuse(RefusalCode::ModelMismatch,
                    std::format("stabilisation temporelle : entrée inconnue « {} »", names[k]));
    }
    temporal_.inputs[k] = slot;
  }
  log_channel("gpu")->info(
      "stabilisation temporelle : {} tampons de {}x{} (traits à {} canaux), {} images d'avance", capacity,
      setup_.work_width, setup_.work_height, setup_.temporal_features, frames_ahead());
  return {};
}

std::expected<bool, Refusal> UpscaleStage::ingest_source(const DecodedFrame& frame, gpu::IngestSpec spec,
                                                         const gpu::Tensor& rgb) {
  if (setup_.temporal_fix == nullptr || frame.pts_us == kNoPts || !temporal_wanted(frame))
    return ingest_plain(frame, spec, rgb);
  // Le rang de frame derrière la tête : 0 pour la tête, sinon celui où next() la donne.
  std::size_t rank = 0;
  for (std::size_t q = 1; q <= frames_ahead(); ++q) {
    const DecodedFrame* candidate = next(q);
    if (candidate == nullptr) break;
    if (candidate->id == frame.id && candidate->epoch == frame.epoch) {
      rank = q;
      break;
    }
  }
  TemporalWindow& window = temporal_.window;
  const auto raw = [&](const DecodedFrame& source) -> std::expected<std::size_t, Refusal> {
    if (auto found = window.find(source, spec.reduce)) return *found;
    // Le tampon est réservé avant l'ingestion : un refus oublie toute la fenêtre.
    const std::size_t index = window.claim(source, spec.reduce, false);
    auto debanded = ingest_plain(source, spec, temporal_.rgb[index]->tensor());
    if (!debanded) {
      window.clear();
      return std::unexpected(std::move(debanded.error()));
    }
    window.set_debanded(index, *debanded);
    temporal_.fresh.push_back(index);
    ++temporal_.raws;
    return index;
  };
  temporal_.fresh.clear();
  auto center = raw(frame);
  if (!center) return std::unexpected(std::move(center.error()));
  for (std::size_t k = 1; k <= kRadius; ++k) {
    const DecodedFrame* after = next(rank + k);
    // Une image d'une autre taille n'est pas lue : son refus SourceResized viendra quand elle sera en tête.
    if (after == nullptr || after->pts_us == kNoPts || after->width != setup_.source_width ||
        after->height != setup_.source_height)
      break;
    if (auto ok = raw(*after); !ok) return std::unexpected(std::move(ok.error()));
  }
  bool mirrored = false;
  const std::array<std::size_t, kTemporalFixFrames> chosen = window.window(*center, &mirrored);
  std::array<gpu::Tensor, std::size_t{2} * kTemporalFixFrames> slots{};
  const std::span<gpu::Tensor> inputs(slots.data(), setup_.temporal_fix->input_names().size());
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    const std::size_t slot = temporal_.inputs[k];
    inputs[k] = slot < kFrames ? temporal_.rgb[chosen[slot]]->tensor()
                               : temporal_.features[chosen[slot - kFrames]]->tensor();
  }
  std::array outputs{rgb};
  const gpu::Lane before = lane_;
  use_lane(gpu::Lane::Upscale);
  // Les images ingérées pour cette passe sont encodées une fois, sur la voie de la fenêtre qui les lit ; les
  // autres reprennent leurs traits gardés (prepare/onnx_cache.h).
  std::expected<void, Refusal> ok;
  for (const std::size_t index : temporal_.fresh) {
    if (setup_.temporal_encoder == nullptr) break;
    std::array encode_in{temporal_.rgb[index]->tensor()};
    std::array encode_out{temporal_.features[index]->tensor()};
    ok = setup_.backend->enqueue(*setup_.temporal_encoder, encode_in, encode_out);
    if (!ok) break;
  }
  if (ok) ok = setup_.backend->enqueue(*setup_.temporal_fix, inputs, outputs);
  use_lane(before);
  if (!ok) {
    window.clear();  // des traits peut-être pas écrits : rien n'est repris
    return std::unexpected(std::move(blame_stage(std::move(ok), kSpanRefusalKey).error()));
  }
  ++temporal_.runs;
  if (mirrored) ++temporal_.mirrored;
  return window.raw(*center).debanded;
}

void UpscaleStage::log_temporal() const {
  if (setup_.temporal_fix == nullptr) return;
  log_channel("gpu")->info(
      "stabilisation temporelle : {} passes, {} reflétées à un bord, {} ingestions brutes encodées une fois",
      temporal_.runs, temporal_.mirrored, temporal_.raws);
}

}  // namespace kirari::app::play
