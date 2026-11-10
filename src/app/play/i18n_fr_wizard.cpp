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
        "progress bar is shown; it takes a few minutes, longer the first time (the engines are built once).",
        "« Calibrer maintenant » essaie plusieurs réglages sur une séquence très éprouvante : résolution de "
        "l'image, agrandissement SPAN et fluidité RIFE, et garde la meilleure combinaison que la machine "
        "tient. Seule une barre de chargement s'affiche ; cela prend quelques minutes, plus la première fois "
        "(les moteurs se construisent une fois)."},
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
    Entry{"Right after the machine calibration, the player shows a few dark scenes, then a few bright "
          "scenes: you tune the adaptive grade for this display.",
          "Juste après la calibration de la machine, le lecteur montre quelques scènes sombres, puis "
          "quelques scènes claires : vous réglez le grade adaptatif pour cet écran."},
    Entry{"SDR chosen: no HDR grade to calibrate.", "SDR choisi : pas de grade HDR à calibrer."},
    Entry{"HDR or SDR", "HDR ou SDR"},
    Entry{"The machine calibration is done", "La calibration de la machine est terminée"},
    Entry{"This display can receive HDR. In HDR, the player then opens the adaptive grade calibration: a few "
          "dark scenes, then a few bright ones, to tune the picture for this display. In SDR, there is "
          "nothing more to set.",
          "Cet écran peut recevoir du HDR. En HDR, le lecteur ouvre ensuite la calibration du grade "
          "adaptatif : quelques scènes sombres, puis quelques claires, pour régler l'image sur cet écran. En "
          "SDR, il n'y a rien de plus à régler."},
    Entry{"HDR: calibrate the grade", "HDR : calibrer le grade"},
    Entry{"Machine calibration: skipped, Auto preset (Video > Post-processing)",
          "Calibration de la machine : passée, préréglage Auto (Vidéo > Post-traitement)"},
    Entry{" · Tensor cores", " · cœurs Tensor"},
    Entry{" · matrix units (WMMA)", " · unités matricielles (WMMA)"},
    Entry{"Adaptive HDR", "HDR adaptatif"},
    Entry{"Always available", "Toujours présent"},
    Entry{"Audio: speakers, faithful sound (Audio > Output)",
          "Audio : enceintes en rendu fidèle (Audio > Sortie)"},
    Entry{"Audio: {} (Audio > Output)", "Audio : {} (Audio > Sortie)"},
    Entry{"Auto picks the settings for your card, display and video.",
          "Auto choisit les réglages selon votre carte, votre écran et la vidéo."},
    Entry{"Back", "Précédent"},
    Entry{"Cancel", "Annuler"},
    Entry{"Checked engines install, then playback resumes at the same spot.",
          "Les moteurs cochés s'installent, puis la lecture reprend au même endroit."},
    Entry{"Compare with Vulkan", "À comparer avec Vulkan"},
    Entry{"Compatibility", "Compatibilité"},
    Entry{"Display {}p · {:.0f} Hz", "Écran {}p · {:.0f} Hz"},
    Entry{"Does not suit this GPU", "Ne convient pas à ce GPU"},
    Entry{"Engine", "Moteur"},
    Entry{"Estimated level: {}", "Niveau estimé : {}"},
    Entry{"Everything can be changed later in Settings (p). To redo this: ajp --first-run.",
          "Tout se change ensuite dans les paramètres (p). Pour le refaire : ajp --first-run."},
    Entry{"Fastest for this GPU", "Le plus rapide pour ce GPU"},
    Entry{"Finish", "Terminer"},
    Entry{"Greyed out: the output device only has 2 channels.",
          "Grisé : l'appareil de sortie n'a que 2 canaux."},
    Entry{"Headphones: crossfeed in stereo, binaural in surround.",
          "Casque : crossfeed en stéréo, binaural en surround."},
    Entry{"How do you listen?", "Comment écoutez-vous ?"},
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
    Entry{"Multichannel, direct surround.", "Multicanal, surround direct."},
    Entry{"Next", "Suivant"},
    Entry{"Not said yet", "Pas encore dit"},
    Entry{"Once the engines are installed, \"Calibrate this machine\" (Settings > Video > "
          "Post-processing) tunes "
          "this for your PC in about a minute.",
          "Une fois les moteurs installés, « Calibrer cette machine » (Paramètres > Général > "
          "Configuration) règle cela pour votre PC en une minute environ."},
    Entry{"Picture and smoothness", "Image et fluidité"},
    Entry{"Picture and smoothness: skipped, Auto preset (Video > Post-processing)",
          "Image et fluidité : sauté, préréglage Auto (Vidéo > Post-traitement)"},
    Entry{"Picture and smoothness: {}, priority {} (Video > Post-processing)",
          "Image et fluidité : {}, priorité {} (Vidéo > Post-traitement)"},
    Entry{"Read the licence", "Lire la licence"},
    Entry{"Not installed until its licence is accepted.",
          "Pas installé tant que sa licence n'est pas acceptée."},
    Entry{"{}: recommended for this card but not selected. The calibration will measure a slower engine.",
          "{} : recommandé pour cette carte mais pas coché. La calibration mesurera un moteur plus lent."},
    Entry{"{}: licence not accepted, it will not be installed",
          "{} : licence non acceptée, il ne s'installera pas"},
    Entry{"Engines: {} ({} to download)", "Moteurs : {} ({} à télécharger)"},
    Entry{"Engines: {}", "Moteurs : {}"},
    Entry{"NVIDIA driver {} or newer required, you have {}.{:02}",
          "pilote NVIDIA {} ou plus récent requis, vous avez le {}.{:02}"},
    Entry{"Resume playback", "Continuer la lecture"},
    Entry{"Skip", "Passer"},
    Entry{"Set it up myself", "Régler moi-même"},
    Entry{"Cancel and watch", "Annuler et regarder"},
    Entry{"Skip: Speakers, faithful sound. Change it later in Audio > Output.",
          "Passer : Enceintes, rendu fidèle. À changer ensuite dans Audio > Sortie."},
    Entry{"Speakers: faithful sound, careful mix.", "Enceintes : rendu fidèle, mixage soigné."},
    Entry{"Start", "Commencer"},
    Entry{"Startup profile", "Profil de démarrage"},
    Entry{"Summary", "Résumé"},
    Entry{"The player chose for this GPU; change anything later in Settings (p).",
          "Le lecteur a choisi pour ce GPU ; tout se change ensuite dans les paramètres (p)."},
    Entry{"The player picks the settings for this machine; everything is in Settings (p) afterwards.",
          "Le lecteur choisit seul les réglages pour cette machine ; tout se retrouve ensuite dans les "
          "paramètres (p)."},
    Entry{"The sound format is detected. Headphones or speakers are not: tell us here.",
          "Le format du son se détecte seul, pas le casque ou les enceintes : dites-le ici."},
    Entry{"Unknown GPU", "GPU inconnu"},
    Entry{"Updates: {} (General > Application)", "Mises à jour : {} (Général > Application)"},
    Entry{"Vulkan works on every GPU. Checked engines install after \"{}\", verified by hash; Vulkan takes "
          "over if one fails.",
          "Vulkan sert tous les GPU. Les moteurs cochés s'installent après « {} », vérifiés par empreinte ; "
          "Vulkan prend le relais si l'un échoue."},
    Entry{"Welcome", "Bienvenue"},
    Entry{"bundled with the player", "livré avec le lecteur"},
    Entry{"cannot be installed in this version yet", "pas encore installable dans cette version"},
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
};

}  // namespace

std::span<const Entry> french_wizard() {
  return kWizardTexts;
}

}  // namespace ajp::app::play::i18n
