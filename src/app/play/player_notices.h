// Les avis montrés sur la vidéo ou à l'accueil après une relance ou un moteur remplacé (C31-04, C31-07).
#pragma once

#include <optional>
#include <string>

#include "app/play/i18n.h"
#include "app/play/options.h"
#include "app/play/relaunch_target.h"
#include "core/engines.h"
#include "core/types.h"

namespace ajp::app::play {

// C31-07 : l'avis d'une relance après une perte du GPU vient de --relaunched=, que le lecteur relancé reçoit
// toujours, même sans relaunch.txt (perte à l'accueil ou à l'écran des moteurs, disque plein) : sur la vidéo
// rouverte, sinon à l'accueil (ajouté à problem).
i18n::Translated relaunch_notice(const PlayOptions& options, const std::optional<RelaunchTarget>& reopened,
                                 bool video_reopens, std::string& problem);

// A31-03, C31-04 : le moteur choisi remplacé à l'ouverture ; un pilote trop ancien se dit comme tel.
i18n::Translated fallback_notice(std::optional<RefusalCode> passed_over, core::EngineId wanted,
                                 core::EngineId in_use);

}  // namespace ajp::app::play
