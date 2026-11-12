#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Assistant et écran des moteurs : (clé anglaise, français).
constexpr std::array kWizardTexts{
    Entry{"Calibrating this machine", "Calibration de la machine"},
    Entry{"New graphics card: calibrate this PC again (Video > Post-processing)",
          "Nouvelle carte graphique : refaites la calibration de ce PC (Vidéo > Post-traitement)"},
    Entry{"Machine calibration failed: settings unchanged (see log)",
          "La calibration de la machine a échoué : réglages inchangés (voir le journal)"},
    Entry{"Machine calibration", "Calibration de la machine"},
    Entry{
        "\"Calibrate now\" tests several settings on a very demanding sequence: picture resolution, SPAN "
        "upscaling and RIFE smoothness, and keeps the best combination this machine sustains. Only a "
        "progress bar is shown; the estimate below is longer with TensorRT, which builds its engines during "
        "the test.",
        "« Calibrer maintenant » essaie plusieurs réglages sur une séquence très éprouvante : résolution de "
        "l'image, agrandissement SPAN et fluidité RIFE, et garde la meilleure combinaison que la machine "
        "tient. Seule une barre de chargement s'affiche ; l'estimation ci-dessous est plus longue avec "
        "TensorRT, qui construit ses moteurs pendant l'essai."},
    Entry{"\"Set it up myself\": the Auto preset estimated for this card, changeable in Video > "
          "Post-processing.",
          "« Régler moi-même » : le préréglage Auto estimé pour cette carte, modifiable dans Vidéo > "
          "Post-traitement."},
    Entry{"Machine calibration: with \"Calibrate now\", the best settings this machine sustains (Video > "
          "Post-processing)",
          "Calibration de la machine : avec « Calibrer maintenant », les meilleurs réglages que la machine "
          "tient (Vidéo > Post-traitement)"},
    Entry{"Default video player: asked at the end of the assistant (General > Application)",
          "Lecteur vidéo par défaut : demandé à la fin de l'assistant (Général > Application)"},
    Entry{"Integration", "Intégration"},
    Entry{"Make it the default video player", "En faire le lecteur vidéo par défaut"},
    Entry{"AJP is added to \"Open with\" in any case.", "AJP s'ajoute à « Ouvrir avec » dans tous les cas."},
    Entry{"Calibrate now", "Calibrer maintenant"},
    Entry{"Estimated for this machine: about {} minutes.", "Estimé pour cette machine : environ {} minutes."},
    Entry{"Ready", "Prêt"},
    Entry{"Vulkan works on every GPU. Checked engines install at the end of the assistant, verified by hash; "
          "Vulkan takes over if one fails.",
          "Vulkan sert tous les GPU. Les moteurs cochés s'installent à la fin de l'assistant, vérifiés par "
          "empreinte ; Vulkan prend le relais si l'un échoue."},
    Entry{"Machine calibration: skipped, Auto preset (Video > Post-processing)",
          "Calibration de la machine : passée, préréglage Auto (Vidéo > Post-traitement)"},
    Entry{" · Tensor cores", " · cœurs Tensor"},
    Entry{" · matrix units (WMMA)", " · unités matricielles (WMMA)"},
    Entry{"Always available", "Toujours présent"},
    Entry{"Back", "Précédent"},
    Entry{"Cancel", "Annuler"},
    Entry{"Checked engines install, then playback resumes at the same spot.",
          "Les moteurs cochés s'installent, puis la lecture reprend au même endroit."},
    Entry{"Compare with Vulkan", "À comparer avec Vulkan"},
    Entry{"Compatibility", "Compatibilité"},
    Entry{"Display {}p · {:.0f} Hz", "Écran {}p · {:.0f} Hz"},
    Entry{"Does not suit this GPU", "Ne convient pas à ce GPU"},
    Entry{"{:.1f} GB", "{:.1f} Go"},
    Entry{"{} GB", "{} Go"},
    Entry{"{} MB", "{} Mo"},
    Entry{"active", "actif"},
    Entry{"inactive", "inactif"},
    Entry{"Engine", "Moteur"},
    Entry{"Estimated level: {}", "Niveau estimé : {}"},
    Entry{"Everything can be changed later in Settings (p). To redo this: ajp --first-run.",
          "Tout se change ensuite dans les paramètres (p). Pour le refaire : ajp --first-run."},
    Entry{"Fastest for this GPU", "Le plus rapide pour ce GPU"},
    Entry{"I accept its licence", "J'accepte sa licence"},
    Entry{"Languages", "Langues"},
    Entry{"Languages: chosen (Audio and playback > Playback)",
          "Langues : choisies (Audio et lecture > Lecture)"},
    Entry{"Languages: skipped (Audio and playback > Playback)",
          "Langues : sauté (Audio et lecture > Lecture)"},
    Entry{"Loading the engine", "Chargement du moteur"},
    Entry{"Loading", "Chargement"},
    Entry{"Loading the engines", "Chargement des moteurs"},
    Entry{"Opening the export", "Ouverture de l'export"},
    Entry{"Next", "Suivant"},
    Entry{"Read the licence", "Lire la licence"},
    Entry{"Not installed until its licence is accepted.",
          "Pas installé tant que sa licence n'est pas acceptée."},
    Entry{"Continuing installs it under this licence; untick the box to skip it.",
          "Continuer l'installe sous cette licence ; décochez la case pour l'éviter."},
    Entry{"{} is not available here: {}", "{} n'est pas disponible ici : {}"},
    Entry{"{}: recommended for this card but not selected. The calibration will measure a slower engine.",
          "{} : recommandé pour cette carte mais pas coché. La calibration mesurera un moteur plus lent."},
    Entry{"{}: licence not accepted, it will not be installed",
          "{} : licence non acceptée, il ne s'installera pas"},
    Entry{"Engines: {} ({} to download)", "Moteurs : {} ({} à télécharger)"},
    Entry{"Engines: {}", "Moteurs : {}"},
    Entry{"NVIDIA driver {}.{:02} is too old: TensorRT needs {}.{:02} or newer",
          "pilote NVIDIA {}.{:02} trop ancien : TensorRT demande le {}.{:02} ou plus récent"},
    Entry{"TensorRT needs the NVIDIA driver (nvidia or nvidia-open)",
          "TensorRT demande le pilote de NVIDIA (nvidia ou nvidia-open)"},
    Entry{"{}: {} replaces it.", "{} : {} le remplace."},
    Entry{"Resume playback", "Continuer la lecture"},
    Entry{"Skip", "Passer"},
    Entry{"Set it up myself", "Régler moi-même"},
    Entry{"Cancel and watch", "Annuler et regarder"},
    Entry{"Start", "Commencer"},
    Entry{"The player chose for this GPU; change anything later in Settings (p).",
          "Le lecteur a choisi pour ce GPU ; tout se change ensuite dans les paramètres (p)."},
    Entry{"The player picks the settings for this machine; everything is in Settings (p) afterwards.",
          "Le lecteur choisit seul les réglages pour cette machine ; tout se retrouve ensuite dans les "
          "paramètres (p)."},
    Entry{"Unknown GPU", "GPU inconnu"},
    Entry{"Updates: {} (General > Application)", "Mises à jour : {} (Général > Application)"},
    Entry{"Vulkan works on every GPU. Checked engines install after \"{}\", verified by hash; Vulkan takes "
          "over if one fails.",
          "Vulkan sert tous les GPU. Les moteurs cochés s'installent après « {} », vérifiés par empreinte ; "
          "Vulkan prend le relais si l'un échoue."},
    Entry{"Welcome", "Bienvenue"},
    Entry{"bundled with the player", "livré avec le lecteur"},
    Entry{"installed", "installé"},
    Entry{"neither TensorRT nor Vulkan chosen: no SPAN or RIFE",
          "ni TensorRT ni Vulkan choisis : ni SPAN ni RIFE"},
    Entry{"never", "jamais"},
    Entry{"not enough disk space: {} needed ({} to download), {} left",
          "espace disque insuffisant : il faut {} (dont {} à télécharger), il reste {}"},
    Entry{"not in this package", "absent de ce paquet"},
    Entry{"not published for this version yet", "pas encore publié pour cette version"},
    Entry{"notify me", "me prévenir"},
    Entry{"to download ({}) · needs {} of space, {} free",
          "à télécharger ({}) · demande {} d'espace, {} libres"},
    Entry{"to download ({})", "à télécharger ({})"},
    Entry{"{} is chosen but does not open on this GPU (see log): {} replaces it.",
          "{} est choisi mais ne s'ouvre pas sur ce GPU (journal) : {} le remplace."},
    Entry{"{} needs a newer graphics driver: update it. {} replaces it until then.",
          "{} demande un pilote graphique plus récent : mettez-le à jour. {} le remplace d'ici là."},
};

}  // namespace

std::span<const Entry> french_wizard() {
  return kWizardTexts;
}

}  // namespace ajp::app::play::i18n
