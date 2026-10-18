#include "core/resolve.h"

#include <algorithm>
#include <cmath>
#include <format>

namespace ajp::core {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool has_model(const Capabilities& caps, const ModelId& model) {
  return std::ranges::find(caps.models, model) != caps.models.end();
}

// Les réseaux veulent des dimensions paires.
int even(double value) {
  return 2 * static_cast<int>(std::lround(value / 2.0));
}

std::expected<HdrRoute, Refusal> resolve_hdr(HdrMode mode, const Capabilities& caps,
                                             const DisplayInfo& display) {
  if (mode == HdrMode::Off) return HdrRoute{};
  if (!display.hdr) return refuse(RefusalCode::HdrUnavailable, "l'écran n'est pas en HDR");
  if (mode == HdrMode::TrueHdr && !caps.true_hdr) {
    return refuse(RefusalCode::HdrUnavailable, "TrueHDR n'est pas installé");
  }
  return HdrRoute{.kind = mode};
}

}  // namespace

const Rule* match_rule(const Profile& profile, const SourceInfo& source) {
  const std::int64_t pixels = std::int64_t{source.width} * source.height;
  for (const Rule& rule : profile.rules) {
    if (pixels >= rule.min_pixels && pixels <= rule.max_pixels && source.fps >= rule.min_fps &&
        source.fps <= rule.max_fps) {
      return &rule;
    }
  }
  return nullptr;
}

std::expected<Plan, Refusal> resolve(const Profile& profile, const Capabilities& caps,
                                     const SourceInfo& source, const DisplayInfo& display) {
  Plan plan;
  plan.output_width = source.width;
  plan.output_height = source.height;
  plan.output_fps = source.fps;

  // Sans règle, la vidéo passe telle quelle.
  const Rule* rule = match_rule(profile, source);
  if (rule == nullptr) {
    plan.hash = compute_hash(plan);
    return plan;
  }

  // L'ordre des étapes doit se ramener à : réduction ? interpolations ? agrandissement ? interpolations ?
  bool upscale_closed = false;
  for (std::size_t i = 0; i < rule->steps.size(); ++i) {
    const Step& step = rule->steps[i];
    const std::string where = std::format("étape {}", i + 1);

    switch (step.kind) {
      case StepKind::Downscale:
        if (i != 0) return refuse(RefusalCode::StepOrder, where + " : la réduction vient en premier");
        // Une réduction ne s'applique qu'à une source plus haute que sa cible.
        if (source.height > step.height) {
          plan.downscale = DownscaleStep{.filter = step.filter, .height = step.height};
          plan.output_width = even(source.width * static_cast<double>(step.height) / source.height);
          plan.output_height = step.height;
        }
        break;

      case StepKind::Upscale:
        if (upscale_closed) {
          return refuse(RefusalCode::StepOrder, where + " : un seul agrandissement par source");
        }
        if (!has_model(caps, step.model)) return refuse(RefusalCode::ModelUnavailable, step.model);
        if (!plan.upscale) plan.upscale.emplace();
        plan.upscale->passes.push_back(
            ModelPass{.model = step.model, .factor = static_cast<int>(step.factor.num)});
        plan.output_width *= static_cast<int>(step.factor.num);
        plan.output_height *= static_cast<int>(step.factor.num);
        break;

      case StepKind::Interpolate: {
        if (plan.upscale) upscale_closed = true;
        if (!has_model(caps, step.model)) return refuse(RefusalCode::ModelUnavailable, step.model);
        if (step.policy == Policy::Quality && step.pacing != Pacing::Source) {
          return refuse(RefusalCode::PacingForbidden, where + " : Quality impose le rythme source");
        }
        Schedule schedule = Schedule::OutputRes;
        if (step.policy == Policy::Quality) {
          schedule = plan.upscale ? Schedule::AfterUpscale : Schedule::BeforeUpscale;
        }
        plan.levels.push_back(InterpLevel{.factor = step.factor,
                                          .model = step.model,
                                          .policy = step.policy,
                                          .pacing = step.pacing,
                                          .schedule = schedule});
        plan.output_fps *= step.factor.value();
        break;
      }
    }
  }

  auto hdr = resolve_hdr(profile.hdr, caps, display);
  if (!hdr) return std::unexpected(hdr.error());
  plan.hdr = *hdr;

  plan.hash = compute_hash(plan);
  return plan;
}

}  // namespace ajp::core
