// Après la boucle d'une ouverture de la chaîne (player_run.cpp) : ce qui la fait rouvrir sans que
// l'utilisateur l'ait demandé. Interne à player_run.cpp.
#pragma once

#include <optional>

#include "app/play/chain.h"
#include "app/play/episodes.h"
#include "app/play/player_engines.h"
#include "app/play/session.h"
#include "core/types.h"

namespace ajp::app::play {

// La source a changé de taille (au-delà de 4 fois par minute : un refus, rangé dans `failure`), le
// fichier grandit encore, ou la vidéo est finie et l'épisode suivant du dossier est demandé : marque
// chain.reopen (et chain.next_file). Rien quand la chaîne se rouvre déjà ou quand la lecture a échoué.
void plan_follow_up(Player& p, Chain& chain, const Session& session, const GrowingFile& opened_file,
                    std::optional<Refusal>& failure);

}  // namespace ajp::app::play
