// Les étages de la chaîne qu'un refus n'arrête plus (décisions 6 et 10) : un SPAN ou une
// interpolation qui ne se construit pas, ne se charge pas ou dont la construction est annulée (« Annuler et
// regarder ») laisse la vidéo jouer sans lui, avec un avis et « Réessayer ».
//
// La mémoire de la séance (StageMemory) retient chaque refus par la signature de l'étage (modèle, cran,
// taille, moteur) : la vidéo suivante, ou une réouverture, ne relance pas une construction de 20 minutes déjà
// refusée (A31-04). Un cran FP8 ou FP32 refusé ne fait retomber que son étage à Équilibre (A08-16) ; une
// annulation ne fait rien retomber du tout, elle retire l'étage annulé seul, pour la vidéo en cours (A08-01,
// A31-05, A31-15).
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/i18n.h"
#include "app/play/settings.h"
#include "core/types.h"

namespace ajp::app::play {

enum class ChainStage : std::uint8_t { Upscale, Motion };

// Un refus de la préparation de la chaîne, avec l'étage qui l'a rendu ; rien : la chaîne entière (backend
// refusé), qui arrête la lecture comme avant.
struct StageFailure {
  Refusal refusal;
  std::optional<ChainStage> stage;
};

// « Annuler et regarder » : jamais un refus du moteur, aucun repli ne doit le prendre pour un échec.
bool watch_now(const Refusal& refusal);

// Ce qui distingue un moteur d'un autre pour cet étage : un refus retenu sous cette signature vaut pour la
// même vidéo rouverte comme pour la suivante de même taille. source : largeur et hauteur de la vidéo ;
// library : le module du moteur (TensorRT, Vulkan).
std::string stage_signature(ChainStage stage, const PlaySettings& settings, int source_width,
                            int source_height, std::string_view library);

// Ce que la préparation a retiré ou ramené pour jouer : montré une fois à l'écran et dans le panneau.
enum class StageChange : std::uint8_t { Cancelled, Dropped, Balanced };
struct StageNote {
  ChainStage stage = ChainStage::Upscale;
  StageChange change = StageChange::Dropped;
  Refusal refusal;
};

class StageMemory {
 public:
  struct Entry {
    ChainStage stage = ChainStage::Upscale;
    std::string signature;
    Refusal refusal;
    bool cancelled = false;      // « Annuler et regarder » : pour cette vidéo seulement
    std::filesystem::path file;  // la vidéo de l'annulation
  };

  // Le refus retenu pour cet étage et cette signature ; une annulation ne vaut que pour sa vidéo.
  const Entry* find(ChainStage stage, std::string_view signature, const std::filesystem::path& file) const;
  void remember(Entry entry);
  // « Réessayer » : tout est oublié, la chaîne se rouvre telle qu'elle est demandée.
  void forget() { entries_.clear(); }
  bool empty() const { return entries_.empty(); }

 private:
  std::vector<Entry> entries_;
};

// Les réglages à essayer : wanted sans les étages que la mémoire refuse, un cran refusé ramené à Équilibre
// d'abord. allow_drop faux (export, banc) : seuls les crans retombent, un étage refusé reste un refus (rien).
// notes reçoit ce qui a changé.
struct StagePlan {
  PlaySettings settings;
  std::vector<StageNote> notes;
};
std::optional<StagePlan> plan_stages(const PlaySettings& wanted, const StageMemory& memory, int source_width,
                                     int source_height, std::string_view library,
                                     const std::filesystem::path& file, bool allow_drop);

// L'avis à l'écran pour ces changements, vide s'il n'y en a pas : « Upscaling unavailable: playing without
// it (Video > Post-processing: Retry) ».
i18n::Translated stage_notice(const std::vector<StageNote>& notes);
// La ligne du panneau, avec la raison du journal.
std::string stage_panel_note(const std::vector<StageNote>& notes);

}  // namespace ajp::app::play
