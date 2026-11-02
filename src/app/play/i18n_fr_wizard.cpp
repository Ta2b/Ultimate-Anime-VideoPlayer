#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Assistant et écran des moteurs : (clé anglaise, français).
constexpr std::array kWizardTexts{
    Entry{"Calibrating this machine", "Calibration de la machine"},
    Entry{"Machine calibration failed: Auto preset (see log)",
          "La calibration de la machine a échoué : préréglage Auto (voir le journal)"},
    Entry{
        "After \"Finish\", the player tests several settings on a very demanding sequence: picture "
        "resolution, SPAN upscaling and RIFE smoothness. It keeps the best combination this machine "
        "sustains, SPAN first.",
        "Après « Terminer », le lecteur essaie plusieurs réglages sur une séquence très éprouvante : "
        "résolution de l'image, agrandissement SPAN et fluidité RIFE. Il garde la meilleure combinaison que "
        "la machine tient, SPAN en priorité."},
    Entry{"An important step: the HDR picture follows it. Redo it or add other profiles later in HDR and "
          "color > HDR.",
          "Une étape importante : l'image HDR en dépend. Refaites-la ou ajoutez d'autres profils plus tard "
          "dans HDR et couleurs > HDR."},
    Entry{"HDR grade calibration", "Calibration du grade HDR"},
    Entry{"Machine calibration", "Calibration de la machine"},
    Entry{"Only a progress bar is shown; it takes a few minutes, longer the first time on a laptop (the "
          "engines are built once).",
          "Seule une barre de chargement s'affiche ; cela prend quelques minutes, plus la première fois sur "
          "un portable (les moteurs se construisent une fois)."},
    Entry{"Right after the machine calibration, the player shows a few dark scenes, then a few bright "
          "scenes: you tune the adaptive grade for this display.",
          "Juste après la calibration de la machine, le lecteur montre quelques scènes sombres, puis "
          "quelques scènes claires : vous réglez le grade adaptatif pour cet écran."},
    Entry{"SDR chosen: no HDR grade to calibrate.", "SDR choisi : pas de grade HDR à calibrer."},
    Entry{"Skipped: the Auto preset. Redo it later in General > Setup.",
          "Passée : le préréglage Auto. Refaites-la plus tard dans Général > Configuration."},
    Entry{"HDR grade calibration: right after the machine calibration if you pick HDR (HDR and color > HDR)",
          "Calibration du grade HDR : juste après celle de la machine si vous choisissez le HDR (HDR et "
          "couleurs > HDR)"},
    Entry{"HDR or SDR", "HDR ou SDR"},
    Entry{"The machine calibration is done", "La calibration de la machine est terminée"},
    Entry{"This display can receive HDR. In HDR, the player then opens the adaptive grade calibration: a few "
          "dark scenes, then a few bright ones, to tune the picture for this display. In SDR, there is "
          "nothing more to set.",
          "Cet écran peut recevoir du HDR. En HDR, le lecteur ouvre ensuite la calibration du grade "
          "adaptatif : quelques scènes sombres, puis quelques claires, pour régler l'image sur cet écran. En "
          "SDR, il n'y a rien de plus à régler."},
    Entry{"HDR: calibrate the grade", "HDR : calibrer le grade"},
    Entry{"Change it later in HDR and color > HDR.", "Modifiable plus tard dans HDR et couleurs > HDR."},
    Entry{"HDR grade calibration: skipped (HDR and color > HDR)",
          "Calibration du grade HDR : passée (HDR et couleurs > HDR)"},
    Entry{"Machine calibration: after \"Finish\", the best settings this machine sustains (General > Setup)",
          "Calibration de la machine : après « Terminer », les meilleurs réglages que la machine tient "
          "(Général > Configuration)"},
    Entry{"Machine calibration: skipped, Auto preset (General > Setup)",
          "Calibration de la machine : passée, préréglage Auto (Général > Configuration)"},
    Entry{" · Tensor cores", " · cœurs Tensor"},
    Entry{" · matrix units (WMMA)", " · unités matricielles (WMMA)"},
    Entry{"Adaptive HDR", "HDR adaptatif"},
    Entry{"Always available", "Toujours présent"},
    Entry{"Any GPU: the HDR look is emulated per scene.",
          "Tous les GPU : l'aspect HDR est émulé selon la scène."},
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
    Entry{"Downloaded, installed at next launch.", "Téléchargée, installée au prochain lancement."},
    Entry{"Engine", "Moteur"},
    Entry{"Estimated level: {}", "Niveau estimé : {}"},
    Entry{"Everything can be changed later in Settings (p). To redo this: ajp --first-run.",
          "Tout se change ensuite dans les paramètres (p). Pour le refaire : ajp --first-run."},
    Entry{"Fastest for this GPU", "Le plus rapide pour ce GPU"},
    Entry{"Finish", "Terminer"},
    Entry{"Greyed out: the output device only has 2 channels.",
          "Grisé : l'appareil de sortie n'a que 2 canaux."},
    Entry{"HDR: skipped, automatic (HDR and color > HDR)",
          "HDR : sauté, automatique (HDR et couleurs > HDR)"},
    Entry{"HDR: {} (HDR and color > HDR)", "HDR : {} (HDR et couleurs > HDR)"},
    Entry{"Headphones: crossfeed in stereo, binaural in surround.",
          "Casque : crossfeed en stéréo, binaural en surround."},
    Entry{"How do you listen?", "Comment écoutez-vous ?"},
    Entry{"I accept its licence", "J'accepte sa licence"},
    Entry{"Languages", "Langues"},
    Entry{"Languages: chosen (General > Playback)", "Langues : choisies (Général > Lecture)"},
    Entry{"Languages: skipped (General > Playback)", "Langues : sauté (Général > Lecture)"},
    Entry{"Loading the engine", "Chargement du moteur"},
    Entry{"Multichannel, direct surround.", "Multicanal, surround direct."},
    Entry{"Next", "Suivant"},
    Entry{"No HDR conversion.", "Pas de conversion HDR."},
    Entry{"Not said yet", "Pas encore dit"},
    Entry{"Once the engines are installed, \"Optimize for this machine\" (Settings > General > Setup) tunes "
          "this for your PC in about a minute.",
          "Une fois les moteurs installés, « Optimiser pour ma machine » (Paramètres > Général > "
          "Configuration) règle cela pour votre PC en une minute environ."},
    Entry{"Picture and smoothness", "Image et fluidité"},
    Entry{"Picture and smoothness: skipped, Auto preset (General > Setup)",
          "Image et fluidité : sauté, préréglage Auto (Général > Configuration)"},
    Entry{"Picture and smoothness: {}, priority {} (General > Setup)",
          "Image et fluidité : {}, priorité {} (Général > Configuration)"},
    Entry{"Read the licence", "Lire la licence"},
    Entry{"Resume playback", "Continuer la lecture"},
    Entry{"Skip", "Passer"},
    Entry{"Skip: Speakers, faithful sound. Change it later in Audio > Output.",
          "Passer : Enceintes, rendu fidèle. À changer ensuite dans Audio > Sortie."},
    Entry{"Speakers: faithful sound, careful mix.", "Enceintes : rendu fidèle, mixage soigné."},
    Entry{"Start", "Commencer"},
    Entry{"Startup profile", "Profil de démarrage"},
    Entry{"Summary", "Résumé"},
    Entry{"The player chooses for the machine.", "Le lecteur choisit selon la machine."},
    Entry{"The player chose for this GPU; change anything later in Settings (p).",
          "Le lecteur a choisi pour ce GPU ; tout se change ensuite dans les paramètres (p)."},
    Entry{"The player picks the settings for this machine; everything is in Settings (p) afterwards.",
          "Le lecteur choisit seul les réglages pour cette machine ; tout se retrouve ensuite dans les "
          "paramètres (p)."},
    Entry{"The sound format is detected. Headphones or speakers are not: tell us here.",
          "Le format du son se détecte seul, pas le casque ou les enceintes : dites-le ici."},
    Entry{"Unknown GPU", "GPU inconnu"},
    Entry{"Updates: {} (General > Application)", "Mises à jour : {} (Général > Application)"},
    Entry{"Vulkan works on every GPU. Checked engines install after \"Finish\", verified by hash; Vulkan "
          "takes over if one fails.",
          "Vulkan sert tous les GPU. Les moteurs cochés s'installent après « Terminer », vérifiés par "
          "empreinte ; Vulkan prend le relais si l'un échoue."},
    Entry{"Vulkan works on every GPU. Checked engines install after \"{}\", verified by hash; Vulkan takes "
          "over if one fails.",
          "Vulkan sert tous les GPU. Les moteurs cochés s'installent après « {} », vérifiés par empreinte ; "
          "Vulkan prend le relais si l'un échoue."},
    Entry{"Welcome", "Bienvenue"},
    Entry{"Your display receives HDR.", "Votre écran reçoit du HDR."},
    Entry{"automatic", "automatiques"},
    Entry{"bundled with the player", "livré avec le lecteur"},
    Entry{"cannot be installed in this version yet", "pas encore installable dans cette version"},
    Entry{"installed", "installé"},
    Entry{"neither TensorRT nor Vulkan chosen: no SPAN or RIFE",
          "ni TensorRT ni Vulkan choisis : ni SPAN ni RIFE"},
    Entry{"never", "jamais"},
    Entry{"not enough disk space: {} needed ({} to download), {} left",
          "espace disque insuffisant : il faut {} (dont {} à télécharger), il reste {}"},
    Entry{"not in this package", "absent de ce paquet"},
    Entry{"not published (NVIDIA licence): build it on this machine, see docs/nvidia-linux.md",
          "non publié (licence NVIDIA) : à construire sur cette machine, voir docs/nvidia-linux.md"},
    Entry{"not published for this version yet", "pas encore publié pour cette version"},
    Entry{"notify me", "me prévenir"},
    Entry{"to download ({}) · needs {} of space, {} free{}",
          "à télécharger ({}) · demande {} d'espace, {} libres{}"},
    Entry{"to download ({}){}", "à télécharger ({}){}"},
    Entry{"{} is chosen but does not open on this GPU (see log): {} replaces it.",
          "{} est choisi mais ne s'ouvre pas sur ce GPU (journal) : {} le remplace."},
};

}  // namespace

std::span<const Entry> french_wizard() {
  return kWizardTexts;
}

}  // namespace ajp::app::play::i18n
