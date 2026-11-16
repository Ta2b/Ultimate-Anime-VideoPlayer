#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Assistant et écran des moteurs : (clé anglaise, français).
constexpr std::array kWizardTexts{
    Entry{"Calibrating this machine", "Calibration de la machine"},
    Entry{"New graphics card: calibrate this PC again (Video > General)",
          "Nouvelle carte graphique : refaites la calibration de ce PC (Vidéo > Général)"},
    Entry{"Machine calibration failed: settings unchanged (see log)",
          "La calibration de la machine a échoué : réglages inchangés (voir le journal)"},
    Entry{"Default video player: asked at the end of the assistant (General > Application)",
          "Lecteur vidéo par défaut : demandé à la fin de l'assistant (Général > Application)"},
    Entry{"Integration", "Intégration"},
    Entry{"Make it the default video player", "En faire le lecteur vidéo par défaut"},
    Entry{"AJP is added to \"Open with\" in any case.", "AJP s'ajoute à « Ouvrir avec » dans tous les cas."},
    Entry{"Ready", "Prêt"},
    Entry{"Vulkan works on every GPU. Checked engines install at the end of the assistant, verified by hash; "
          "Vulkan takes over if one fails.",
          "Vulkan sert tous les GPU. Les moteurs cochés s'installent à la fin de l'assistant, vérifiés par "
          "empreinte ; Vulkan prend le relais si l'un échoue."},
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
    Entry{"Languages: skipped (Languages > Languages)", "Langues : sauté (Langues > Langues)"},
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
    Entry{"{}: recommended for this card but not selected. Playback will use a slower engine.",
          "{} : recommandé pour cette carte mais pas coché. La lecture prendra un moteur plus lent."},
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
    Entry{"Original (marked by the file)", "Originale (indiquée par le fichier)"},
    Entry{"Original version, subtitled", "VO sous-titrée"},
    Entry{"Dubbed in my language", "Doublée dans ma langue"},
    Entry{"Custom (Advanced)", "Personnalisé (Avancé)"},
    Entry{"1st choice", "1er choix"},
    Entry{"2nd choice", "2e choix"},
    Entry{"3rd choice", "3e choix"},
    Entry{"4th choice", "4e choix"},
    Entry{"5th choice", "5e choix"},
    Entry{"My language", "Ma langue"},
    Entry{"The language of your subtitles, or of the dub. The system language by default.",
          "La langue de vos sous-titres, ou du doublage. Celle du système par défaut."},
    Entry{"I prefer", "Je préfère"},
    Entry{"Original version: the audio the file marks as original, else Japanese, else Chinese, with "
          "subtitles in your language, else Japanese with English subtitles, else your dub.\n"
          "Dubbed: your language without subtitles, else the original version.\n"
          "Recommended: original version.",
          "VO : l'audio que le fichier indique comme original, sinon le japonais, sinon le chinois, avec des "
          "sous-titres dans votre langue, sinon le japonais sous-titré en anglais, sinon votre doublage.\n"
          "Doublée : votre langue sans sous-titres, sinon la VO.\n"
          "Recommandé : VO."},
    Entry{"Tracks chosen one by one in Advanced. Pick a version to go back to the automatic choice.",
          "Pistes choisies une à une dans Avancé. Choisissez une version pour revenir au choix automatique."},
    Entry{"Picked when a file opens: the first choice whose audio and subtitles are both in the file. Change "
          "the current track from the bar.",
          "Pris à l'ouverture d'un fichier : le premier choix dont l'audio et les sous-titres sont tous deux "
          "dans le fichier. La piste en cours se change depuis la barre."},
    Entry{"Show the exact track choices", "Voir le choix exact des pistes"},
    Entry{"Settings for this card", "Réglages pour cette carte"},
    Entry{"The Quality preset in Video > General changes it at any time.",
          "Le préréglage de qualité, dans Vidéo > Général, le change à tout moment."},
    Entry{"Finish", "Terminer"},
    Entry{"Recommended: {} (level {} of this card).", "Recommandé : {} (niveau {} de cette carte)."},
    Entry{"turned on", "activé"},
    Entry{"turned off", "désactivé"},
    Entry{"upscaling off", "agrandissement éteint"},
    Entry{"the default model", "le modèle par défaut"},
    Entry{"the source, up to 1080p", "la source, jusqu'à 1080p"},
    Entry{"interpolation off", "interpolation éteinte"},
    Entry{"HDR grade only: upscaling and interpolation stay off, the card keeps up with any video.",
          "Le grade HDR seul : l'agrandissement et l'interpolation restent éteints, la carte suit toute "
          "vidéo."},
    Entry{"HDR grade, upscaling from 540p and interpolation up to ×2.",
          "Grade HDR, agrandissement depuis 540p et interpolation jusqu'à ×2."},
    Entry{"HDR grade, upscaling from 720p and interpolation up to ×3.",
          "Grade HDR, agrandissement depuis 720p et interpolation jusqu'à ×3."},
    Entry{"HDR grade, upscaling from the source up to 1080p with the finest models, and interpolation up to "
          "×4.",
          "Grade HDR, agrandissement depuis la source jusqu'à 1080p avec les modèles les plus fins, et "
          "interpolation "
          "jusqu'à ×4."},
    Entry{"Languages: {}, dubbed (Languages > Languages)", "Langues : {}, doublé (Langues > Langues)"},
    Entry{"Languages: {}, original version with subtitles (Languages > Languages)",
          "Langues : {}, VO sous-titrée (Langues > Langues)"},
    Entry{"Languages: chosen track by track (Languages > Languages)",
          "Langues : choisies piste par piste (Langues > Langues)"},
    Entry{"An engine did not install: Vulkan is used. Install it again in Settings > Video > General.",
          "Un moteur ne s'est pas installé : Vulkan le remplace. Réinstallez-le dans Paramètres > Vidéo > "
          "Général."},
};

}  // namespace

std::span<const Entry> french_wizard() {
  return kWizardTexts;
}

}  // namespace ajp::app::play::i18n
