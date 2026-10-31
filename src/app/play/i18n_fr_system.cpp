#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Export, moteurs, application : (clé anglaise, français).
constexpr std::array kSystemPages{
    Entry{" (ready)", " (préparée)"},
    Entry{": restart the player", " : relancez le lecteur"},
    Entry{"A notification; you start the update yourself.",
          "Une notification ; vous lancez la mise à jour vous-même."},
    Entry{"AJP is already the default video player.", "AJP est déjà le lecteur par défaut des vidéos."},
    Entry{"AV1 10-bit: smaller at equal quality; needs a GPU AV1 encoder (RTX 40/50, Radeon RX 7000/9000, "
          "Intel Arc).",
          "AV1 10 bits : plus compact à qualité égale ; demande un encodeur AV1 dans le GPU (RTX 40/50, "
          "Radeon RX 7000/9000, Intel Arc)."},
    Entry{
        "Above the playback output, libplacebo scales further (EWA Lanczos): a bigger file, no more detail.",
        "Au-dessus de la sortie de la lecture, libplacebo agrandit encore (EWA Lanczos) : fichier plus "
        "lourd, pas plus de détail."},
    Entry{"Active engine", "Moteur actif"},
    Entry{"Add to \"Open with\"", "Ajouter à « Ouvrir avec »"},
    Entry{"An engine is built once per model and size, then read from the cache.",
          "Un moteur est construit une fois par modèle et par taille, puis relu du cache."},
    Entry{"At every launch", "À chaque lancement"},
    Entry{"At scene cuts and held drawings the previous frame repeats, so the file keeps a constant frame "
          "rate any editor reads.",
          "Aux coupes de scène et dessins tenus, l'image précédente est répétée : le fichier garde une "
          "cadence fixe, lue par tout logiciel de montage."},
    Entry{"At this frame rate MKV rounds frames to the millisecond; .mp4 or .mov keeps exact times but drops "
          "subtitles and fonts.",
          "À cette fréquence, le MKV arrondit les images à la milliseconde ; .mp4 ou .mov garde les instants "
          "exacts, sans sous-titres ni polices."},
    Entry{"Audio, subtitles, fonts and chapters are copied as is. Playback stops during the export; Esc "
          "cancels.",
          "Audio, sous-titres, polices et chapitres sont copiés tels quels. La lecture s'arrête pendant "
          "l'export ; Échap l'annule."},
    Entry{"Automatic", "Automatique"},
    Entry{"Available: {}{}", "Disponible : {}{}"},
    Entry{"Below the playback output, libplacebo scales down with its best filter (EWA Lanczos).",
          "Sous la sortie de la lecture, libplacebo réduit avec son meilleur filtre (EWA Lanczos)."},
    Entry{"Built engines", "Moteurs construits"},
    Entry{"Cancel##bench", "Annuler##bench"},
    Entry{"Cancel##moteur", "Annuler##moteur"},
    Entry{"Cancel##span_prepare", "Annuler##span_prepare"},
    Entry{"Channel", "Canal"},
    Entry{"Channel and frequency", "Canal et fréquence"},
    Entry{"Check now", "Vérifier maintenant"},
    Entry{"Choose the output file.", "Choisissez le fichier exporté."},
    Entry{"Codec", "Codec"},
    Entry{"Constant frame rate", "Cadence fixe"},
    Entry{"Daily", "Chaque jour"},
    Entry{"Default", "Par défaut"},
    Entry{"Default folder", "Dossier par défaut"},
    Entry{"Default player", "Lecteur par défaut"},
    Entry{"Default##moteurs", "Par défaut##moteurs"},
    Entry{"Default: {}. Free here: {}. Applies at next launch; installed engines are not moved.",
          "Par défaut : {}. Libre ici : {}. Vaut au prochain lancement ; les moteurs installés ne sont pas "
          "déplacés."},
    Entry{"Downloaded, verified, installed at next launch.",
          "Téléchargée, vérifiée, installée au prochain lancement."},
    Entry{"Encoding", "Encodage"},
    Entry{"FFV1 does not fit in a MOV: use .mkv or .mp4.",
          "FFV1 ne va pas dans un MOV : choisissez .mkv ou .mp4."},
    Entry{
        "FFV1 lossless, 4:2:0 10-bit, encoded by the CPU: very large files, for editing.",
        "FFV1 sans perte, 4:2:0 10 bits, encodé par le processeur : fichiers très lourds, pour le montage."},
    Entry{"File", "Fichier"},
    Entry{"Frames the chain does not interpolate are left out: variable frame rate, a bit smaller, fine in "
          "players.",
          "Les images non interpolées sont omises : cadence variable, un peu plus léger, sans souci dans les "
          "lecteurs."},
    Entry{"Frequency", "Fréquence"},
    Entry{"Greyed out: no update has been installed yet.",
          "Grisé : aucune mise à jour n'a encore été installée."},
    Entry{"H.264 8-bit: for older players; gradients are coarser.",
          "H.264 8 bits : pour les anciens lecteurs ; les dégradés y sont moins fins."},
    Entry{"HEVC 10-bit: plays almost everywhere, encoded by the GPU.",
          "HEVC 10 bits : lu presque partout, encodé par le GPU."},
    Entry{"Install folder", "Dossier d'installation"},
    Entry{"Install more engines…", "Installer d'autres moteurs…"},
    Entry{"Installed engines", "Moteurs installés"},
    Entry{"Installed version: {}", "Version installée : {}"},
    Entry{"Last check: {}", "Dernière vérification : {}"},
    Entry{"Last error: ", "Dernière erreur : "},
    Entry{"MP4 and MOV keep exact frame times, chapters and compatible audio, but no subtitles or fonts. Use "
          ".mkv to keep everything. Esc cancels.",
          "MP4 et MOV gardent les instants exacts, les chapitres et l'audio compatible, mais ni sous-titres "
          "ni polices. Utilisez .mkv pour tout garder. Échap annule."},
    Entry{"Make this the default player", "Faire de ce lecteur le lecteur par défaut"},
    Entry{"Never", "Jamais"},
    Entry{"No checks.", "Aucune vérification."},
    Entry{"No engine built yet.", "Aucun moteur construit."},
    Entry{"Notify me", "Me prévenir"},
    Entry{"Only one engine is installed: \"Install more engines\" lists those that suit this GPU.",
          "Un seul moteur installé : « Installer d'autres moteurs » propose ceux qui conviennent à ce GPU."},
    Entry{"Open a video: the export uses its settings and is named next to it.",
          "Ouvrez une vidéo : l'export reprend ses réglages et se nomme à côté d'elle."},
    Entry{"Optimization cancelled: nothing changed.", "Optimisation annulée : rien n'a changé."},
    Entry{"Optimize for this machine", "Optimiser pour ma machine"},
    Entry{"Playback output · {}x{}", "Sortie de la lecture · {}x{}"},
    Entry{"Prepare again for this video", "Refaire la préparation pour cette vidéo"},
    Entry{"Previews", "Préversions"},
    Entry{"Previous settings restored.", "Réglages d'avant rétablis."},
    Entry{"Remove from \"Open with\"", "Retirer de « Ouvrir avec »"},
    Entry{"Reopens the engines screen; playback resumes at the same spot.",
          "Rouvre l'écran des moteurs ; la lecture reprend ensuite au même endroit."},
    Entry{"Roll back to the previous version", "Revenir à la version précédente"},
    Entry{"SPAN Performance mode (FP8)", "Mode Performance de SPAN (FP8)"},
    Entry{"SPAN Performance mode (FP8) preparation", "préparation du mode Performance de SPAN (FP8)"},
    Entry{"Same engines and settings as playback, with no skipped frame. No RTX HDR, adaptive HDR or "
          "grading: the chain's picture as is.",
          "Mêmes moteurs et réglages qu'à la lecture, sans image sautée. Ni RTX HDR, ni HDR adaptatif, ni "
          "étalonnage : l'image de la chaîne telle quelle."},
    Entry{"Source · {}x{}", "Source · {}x{}"},
    Entry{"Suggested name", "Nom proposé"},
    Entry{"The output is an MKV, MP4 or MOV: its name ends in .mkv, .mp4 or .mov.",
          "Le fichier exporté est un MKV, un MP4 ou un MOV : son nom finit par .mkv, .mp4 ou .mov."},
    Entry{"The player tries a few seconds of video with its recommended settings, lightest to heaviest, "
          "keeps what your PC holds without stutter and applies it. About a minute (several the first time).",
          "Le lecteur essaie quelques secondes de vidéo avec ses réglages recommandés, du plus léger au plus "
          "lourd, garde ceux que votre PC tient sans saccade et les applique. Environ une minute (plusieurs "
          "la première fois)."},
    Entry{"This file exists: it is replaced only when the export succeeds.",
          "Ce fichier existe : il n'est remplacé qu'à la fin d'un export réussi."},
    Entry{"This file is the video being played.", "Ce fichier est la vidéo lue."},
    Entry{"This folder does not exist.", "Ce dossier n'existe pas."},
    Entry{"This package has no signing key: updates cannot be verified.",
          "Ce paquet n'a pas de clé de signature : les mises à jour ne peuvent pas être vérifiées."},
    Entry{"To apply: playback reopens at the same spot with {}; its engines are built the first time for "
          "each model and size.",
          "À appliquer : la lecture se rouvre au même endroit avec {} ; ses moteurs se construisent la "
          "première fois pour chaque modèle et chaque taille."},
    Entry{"Undo these settings", "Annuler ces réglages"},
    Entry{"Updates", "Mises à jour"},
    Entry{"Use the playback settings", "Reprendre les réglages de la lecture"},
    Entry{"Use this engine", "Utiliser ce moteur"},
    Entry{"Very high aims for a picture identical to playback; High and Standard make smaller files.",
          "Très haute vise une image identique à la lecture ; Haute et Standard donnent des fichiers plus "
          "légers."},
    Entry{"Videos (mkv, mp4, webm, avi, mov…) open in AJP from the file manager.",
          "Les vidéos (mkv, mp4, webm, avi, mov…) s'ouvrent dans AJP depuis le gestionnaire de fichiers."},
    Entry{"Weekly", "Chaque semaine"},
    Entry{"With SPAN's Performance precision, the player prepares this mode by itself for the open video "
          "(about a minute, FP16 meanwhile). Redo it by hand here.",
          "Avec la précision Performance de SPAN, le lecteur prépare ce mode seul pour la vidéo ouverte "
          "(environ une minute, en FP16 entre-temps). Refaites-le à la main ici."},
    Entry{"Working…", "En cours…"},
    Entry{"· cached", "· en cache"},
    Entry{"· loaded", "· chargé"},
};

}  // namespace

std::span<const Entry> french_system() {
  return kSystemPages;
}

}  // namespace ajp::app::play::i18n
