// Le grade avant le warp (source_grade.h), côté libplacebo : deux images au protocole du ring (core/ring.h),
// l'entrée que le backend écrit et la sortie qu'il relit, un renderer et un grade à part. Tout tourne sur le
// fil de l'agrandissement ; pl_gpu y est sûr (pl_gpu_limits.thread_safe sous Vulkan).
#include <format>
#include <utility>

#include "app/play/placebo_impl.h"
#include "app/play/source_grade.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t kIn = 0;
constexpr std::uint32_t kOut = 1;

pl_frame frame_of(pl_tex tex, const pl_color_space& color) {
  pl_frame frame{};
  frame.num_planes = 1;
  frame.planes[0].texture = tex;
  frame.planes[0].components = 3;
  frame.planes[0].component_mapping[0] = 0;
  frame.planes[0].component_mapping[1] = 1;
  frame.planes[0].component_mapping[2] = 2;
  frame.repr = pl_color_repr_rgb;
  frame.color = color;
  frame.crop = {0, 0, static_cast<float>(tex->params.w), static_cast<float>(tex->params.h)};
  return frame;
}

class PlaceboSourceGrader final : public SourceGrader {
 public:
  PlaceboSourceGrader(pl_gpu gpu, pl_log log, std::shared_ptr<const SourceGradeShare> share, bool external,
                      double source_fps, std::uint32_t generation)
      : gpu_(gpu),
        log_(log),
        share_(std::move(share)),
        external_(external),
        fps_(source_fps),
        generation_(generation) {}
  PlaceboSourceGrader(const PlaceboSourceGrader&) = delete;
  PlaceboSourceGrader& operator=(const PlaceboSourceGrader&) = delete;
  PlaceboSourceGrader(PlaceboSourceGrader&&) = delete;
  PlaceboSourceGrader& operator=(PlaceboSourceGrader&&) = delete;
  ~PlaceboSourceGrader() override {
    if (graded_ > 0) {
      log_channel("app")->info(
          "grade avant le warp : {} images sources gradées, moyenne GPU {:.3f} ms", graded_,
          timed_ > 0 ? static_cast<double>(gpu_ns_) / 1e6 / static_cast<double>(timed_) : 0.0);
    }
    ring_.clear(gpu_);
    pl_renderer_destroy(&renderer_);
    grade_.reset();
  }

  std::expected<bool, Refusal> grade(gpu::KernelOps& ops, const gpu::Tensor& rgb, bool debanded) override {
    const SourceGradeState state = share_->get();
    if (!state.params) return false;
    if (rgb.shape.dims.size() != 4) return refuse(RefusalCode::ShapeInvalid, "grade avant le warp : tenseur");
    const int width = static_cast<int>(rgb.shape.dims[3]);
    const int height = static_cast<int>(rgb.shape.dims[2]);
    if (auto ok = open(width, height); !ok) return std::unexpected(std::move(ok.error()));
    if (auto ok = follow(state); !ok) return std::unexpected(std::move(ok.error()));

    // Entrée : le backend écrit (attente de sa libération, signal impair), libplacebo la lit puis la rend.
    const std::uint64_t in_ready = in_ready_ == 0 ? 1 : in_ready_ + 2;
    if (auto ok = ops.write_ring(rgb, slots_[kIn], {.wait_value = in_ready - 1, .signal_value = in_ready});
        !ok)
      return std::unexpected(std::move(ok.error()));
    in_ready_ = in_ready;
    const RingFrame in{.slot = kIn, .generation = ring_generation_, .ready_value = in_ready};
    const RingFrame out{.slot = kOut, .generation = ring_generation_, .ready_value = out_free_};
    auto in_tex = ring_.acquire(gpu_, in);
    if (!in_tex) return std::unexpected(std::move(in_tex.error()));
    auto out_tex = ring_.acquire(gpu_, out);
    if (!out_tex) {
      ring_.retire(gpu_, in);
      return std::unexpected(std::move(out_tex.error()));
    }
    const bool drawn = render(*in_tex, *out_tex, state, state.deband && !debanded);
    // Rendus au backend même après un échec : les valeurs des sémaphores restent croissantes.
    ring_.retire(gpu_, in);
    ring_.retire(gpu_, out);
    const std::uint64_t out_ready = out_free_ + 1;
    out_free_ = out_ready + 1;
    if (!drawn) return refuse(RefusalCode::VendorCallFailed, "grade avant le warp : pl_render_image");
    // Sortie : libplacebo l'a écrite (signal impair), le backend la relit dans le tenseur puis la libère.
    if (auto ok = ops.read_ring(slots_[kOut], {.wait_value = out_ready, .signal_value = out_free_}, rgb); !ok)
      return std::unexpected(std::move(ok.error()));
    ++graded_;
    return true;
  }

 private:
  // Les deux images à la taille des slots du ring de l'écran ; refaites si elle change.
  std::expected<void, Refusal> open(int width, int height) {
    if (!slots_.empty() && slots_[kIn].width == width && slots_[kIn].height == height) return {};
    // La sortie est la cible de pl_render_image : sans renderable, libplacebo la refuse (rc.13, TensorRT).
    auto slots = ring_.create(gpu_, width, height, 2, external_, RingFormat::Rgba16Unorm, true);
    if (!slots) return std::unexpected(std::move(slots.error()));
    ring_generation_ = slots->front().generation;
    // Le backend n'a qu'une génération de ring à la fois (la plus récente retire les vues des autres) : ces
    // deux images portent celle du ring de l'écran.
    for (RingSlotDesc& slot : *slots) slot.generation = generation_;
    slots_ = std::move(*slots);
    in_ready_ = 0;
    out_free_ = 0;
    if (renderer_ == nullptr) renderer_ = pl_renderer_create(log_, gpu_);
    if (renderer_ == nullptr) return refuse(RefusalCode::VendorCallFailed, "pl_renderer_create");
    return {};
  }

  // Le grade à part suit celui de l'écran : ses LUT (par identité), ses paramètres, la passe de sortie.
  std::expected<void, Refusal> follow(const SourceGradeState& state) {
    if (!state.params) return {};  // sans grade, rien à suivre (l'appelant l'a déjà écarté)
    const render::GradeParams& wanted = *state.params;
    if (!grade_) {
      auto made = render::Grade::create(gpu_);
      if (!made) return std::unexpected(std::move(made.error()));
      grade_.emplace(std::move(*made));
      looks_.reset();
    }
    render::Grade& grade = *grade_;
    if (!looks_ || *looks_ != state.looks) {
      if (auto ok = grade.set_looks(state.looks); !ok) return ok;
      looks_ = state.looks;
    }
    if (grade.params() != wanted) grade.set_params(wanted);
    grade.set_kwin_match(state.kwin);
    // Une fois par source : les lissages du grade gardent leur durée en secondes.
    grade.set_video_fps(fps_);
    return {};
  }

  bool render(pl_tex in, pl_tex out, const SourceGradeState& state, bool deband) {
    static const pl_deband_params kQuietDeband = [] {
      pl_deband_params quiet = pl_deband_default_params;
      quiet.grain = 0.0F;
      return quiet;
    }();
    pl_render_params params = pl_render_default_params;
    params.dither_params = nullptr;  // RGBA16 : l'écran tramera vers sa sortie
    if (deband) params.deband_params = &kQuietDeband;
    if (!grade_) return false;  // follow() pose le grade avant tout rendu
    const std::span<const pl_hook* const> hooks = grade_->hooks();
    params.hooks = hooks.data();
    params.num_hooks = static_cast<int>(hooks.size());
    params.info_priv = this;
    params.info_callback = [](void* priv, const pl_render_info* info) {
      if (info->stage != PL_RENDER_STAGE_FRAME || info->pass == nullptr) return;
      auto& self = *static_cast<PlaceboSourceGrader*>(priv);
      self.gpu_ns_ += info->pass->last;
      if (info->index == 0) ++self.timed_;
    };
    const pl_frame image = frame_of(in, state.source);
    const pl_frame target = frame_of(out, state.target);
    return pl_render_image(renderer_, &image, &target, &params);
  }

  pl_gpu gpu_;
  pl_log log_;
  std::shared_ptr<const SourceGradeShare> share_;
  bool external_;
  double fps_;
  std::uint32_t generation_;
  PlaceboRing ring_;
  std::uint32_t ring_generation_ = 0;
  std::vector<RingSlotDesc> slots_;
  std::uint64_t in_ready_ = 0;  // dernière valeur prête de l'entrée (impaire), 0 avant la première
  std::uint64_t out_free_ = 0;  // valeur où la sortie est libre (paire), 0 avant le premier rendu
  pl_renderer renderer_ = nullptr;
  std::optional<render::Grade> grade_;
  std::optional<std::array<std::shared_ptr<const render::CubeLut>, render::kLookSlots>> looks_;
  std::uint64_t graded_ = 0;
  std::uint64_t gpu_ns_ = 0;
  std::uint64_t timed_ = 0;
};

}  // namespace

void PlaceboScreen::set_grade_before_warp(bool on) {
  impl_->grade_before_warp = on;
}

std::shared_ptr<SourceGrader> PlaceboScreen::source_grader(bool external, double source_fps,
                                                           std::uint32_t generation) {
  return std::make_shared<PlaceboSourceGrader>(impl_->vk->gpu, impl_->log, impl_->source_grade_share,
                                               external, source_fps, generation);
}

}  // namespace ajp::app::play
