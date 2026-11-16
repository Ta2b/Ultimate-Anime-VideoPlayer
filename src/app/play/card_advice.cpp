#include "app/play/card_advice.h"

#include <algorithm>

#include "app/play/auto_config.h"
#include "app/play/i18n.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

CardAdvice card_advice(const SettingsChoices& choices) {
  return {.tier = static_cast<Tier>(std::clamp(choices.machine_tier, 0, kTierCount - 1)),
          .tensor_cores = choices.tensor_cores};
}

std::string recommended_for(const CardAdvice& card, std::string_view value) {
  return i18n::tr_format("Recommended: {} (level {} of this card).", value, tier_label(card.tier));
}

std::string advice_line(const SettingsChoices& choices, std::string (*value)(const CardAdvice&)) {
  const CardAdvice card = card_advice(choices);
  return recommended_for(card, value(card));
}

std::string upscale_advice(const CardAdvice& card) {
  return card.tier == Tier::Economy ? i18n::tr("turned off") : i18n::tr("turned on");
}

std::string model_advice(const CardAdvice& card) {
  if (card.tier == Tier::Economy) return i18n::tr("upscaling off");
  if (card.tier == Tier::Quality || card.tier == Tier::Ultra)
    return model_label(std::string(kQualitySpanModel));
  return i18n::tr("the default model");
}

std::string work_height_advice(const CardAdvice& card) {
  switch (card.tier) {
    case Tier::Economy:
      return i18n::tr("upscaling off");
    case Tier::Fast:
      return "540p";
    case Tier::Normal:
      return "720p";
    case Tier::Quality:
    case Tier::Ultra:
      return i18n::tr("the source, up to 1080p");
  }
  return "720p";
}

std::string interpolation_level_advice(const CardAdvice& card) {
  switch (card.tier) {
    case Tier::Economy:
      return i18n::tr("interpolation off");
    case Tier::Fast:
    case Tier::Normal:
      return i18n::tr("Fast");
    case Tier::Quality:
      return i18n::tr("Balanced");
    case Tier::Ultra:
      return card.tensor_cores ? i18n::tr("High") : i18n::tr("Balanced");
  }
  return i18n::tr("Fast");
}

std::string light_motion_advice(const CardAdvice& card) {
  return card.tier == Tier::Quality || card.tier == Tier::Ultra ? i18n::tr("turned off")
                                                                : i18n::tr("turned on");
}

}  // namespace ajp::app::play
