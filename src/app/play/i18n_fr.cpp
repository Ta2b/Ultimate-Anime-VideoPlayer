#include "app/play/i18n_fr.h"

#include <array>
#include <vector>

namespace ajp::app::play::i18n {
namespace {

// Textes de l'interface : (clé anglaise, français). Les messages de la barre de lecture, des pistes et des
// notifications.
constexpr std::array kPlayer{
    Entry{" · forced", " · forcés"},
    Entry{" — warning: audio and subtitles stop early, the source could not be read again ({})",
          " — attention : l'audio et les sous-titres s'arrêtent avant la fin, la source n'a pas pu être "
          "relue ({})"},
    Entry{"Apple interpolation unavailable", "Interpolation d'Apple indisponible"},
    Entry{"Apple interpolation unavailable: using RIFE", "Interpolation d'Apple indisponible : RIFE"},
    Entry{"Arabic", "Arabe"},
    Entry{"Audio", "Audio"},
    Entry{"Audio track (a)", "Piste audio (a)"},
    Entry{"Audio track refused", "Piste audio refusée"},
    Entry{"Audio: {}", "Audio : {}"},
    Entry{"Auto ×{} ({:.0f} fps)", "Auto ×{} ({:.0f} i/s)"},
    Entry{"Auto: Fast (experimental)", "Auto : Rapide (expérimental)"},
    Entry{"Auto: Stable", "Auto : Stable"},
    Entry{"Chapter {}/{}", "Chapitre {}/{}"},
    Entry{"Chinese", "Chinois"},
    Entry{"Credits skipped", "Générique passé"},
    Entry{"DistilDRBA unavailable", "DistilDRBA indisponible"},
    Entry{"DistilDRBA unavailable: using RIFE", "DistilDRBA indisponible : RIFE"},
    Entry{"Downscale: {}", "Réduction : {}"},
    Entry{"Dutch", "Néerlandais"},
    Entry{"Engine unchanged: no other one opens (see log)",
          "Moteur inchangé : aucun autre ne s'ouvre (journal)"},
    Entry{"Engine: {}", "Moteur : {}"},
    Entry{"English", "Anglais"},
    Entry{"Exit fullscreen (f, Esc)", "Quitter le plein écran (f, Échap)"},
    Entry{"Export cancelled", "Export annulé"},
    Entry{"Export done: {} ({} frames, {:.0f} MB, {})", "Export terminé : {} ({} images, {:.0f} Mo, {})"},
    Entry{"Export stopped: {}", "Export arrêté : {}"},
    Entry{"Fast mode unavailable: {}", "Mode Rapide indisponible : {}"},
    Entry{"Fast mode: too many frames lost, back to Stable",
          "Mode Rapide : trop d'images perdues, retour au mode Stable"},
    Entry{"French", "Français"},
    Entry{"Fullscreen (f)", "Plein écran (f)"},
    Entry{"GMFSS unavailable", "GMFSS indisponible"},
    Entry{"GMFSS unavailable: using RIFE", "GMFSS indisponible : RIFE"},
    Entry{"German", "Allemand"},
    Entry{"Grade off", "Grade désactivé"},
    Entry{"Grade: {}", "Grade : {}"},
    Entry{"Hindi", "Hindi"},
    Entry{"Impulse response unreadable: using the default",
          "Réponse impulsionnelle illisible : modèle d'origine"},
    Entry{"Indonesian", "Indonésien"},
    Entry{"Interface language", "Langue de l'interface"},
    Entry{"Interpolation off", "Interpolation désactivée"},
    Entry{"Interpolation {}", "Interpolation {}"},
    Entry{"Interpolation: Auto", "Interpolation : Auto"},
    Entry{"Italian", "Italien"},
    Entry{"Japanese", "Japonais"},
    Entry{"Keep poses: off", "Poses : warp simple"},
    Entry{"Keep poses: on", "Poses gardées (DRBA)"},
    Entry{"Korean", "Coréen"},
    Entry{"Language", "Langue"},
    Entry{"Malay", "Malais"},
    Entry{"Mute (m)", "Couper le son (m)"},
    Entry{"Muted", "Muet"},
    Entry{"Next chapter (Page Up)", "Chapitre suivant (Page haut)"},
    Entry{"No next episode in this folder", "Pas d'épisode suivant dans ce dossier"},
    Entry{"No previous episode in this folder", "Pas d'épisode précédent dans ce dossier"},
    Entry{"Off", "Désactivés"},
    Entry{"Pause (Space)", "Pause (espace)"},
    Entry{"Play (Space)", "Lecture (espace)"},
    Entry{"Polish", "Polonais"},
    Entry{"Portuguese", "Portugais"},
    Entry{"Previous chapter (Page Down)", "Chapitre précédent (Page bas)"},
    Entry{"Profile change refused", "Profil : modification refusée"},
    Entry{"Profile {} · {}", "Profil {} · {}"},
    Entry{"Profile {}: empty", "Profil {} : vide"},
    Entry{"RTX HDR needs TensorRT: using adaptive HDR", "RTX HDR demande le moteur TensorRT : HDR adaptatif"},
    Entry{"RTX HDR unavailable: using adaptive HDR", "RTX HDR indisponible : HDR adaptatif"},
    Entry{"Resumed at {}", "Reprise à {}"},
    Entry{"Russian", "Russe"},
    Entry{"Scaler: {}, deband {}", "Image : {}, débandage {}"},
    Entry{"Settings (p)", "Paramètres (p)"},
    Entry{"Skipped 85 s", "Saut de 85 s"},
    Entry{"Smooth Motion: off at next launch", "Smooth Motion coupé au prochain lancement"},
    Entry{"Smooth Motion: on at next launch", "Smooth Motion : au prochain lancement"},
    Entry{"Sound: original", "Son fidèle (sans traitement)"},
    Entry{"Sound: processed", "Son traité"},
    Entry{"Source is already HDR: RTX HDR not needed", "Source déjà HDR : RTX HDR inutile"},
    Entry{"Spanish", "Espagnol"},
    Entry{"Statistics hidden", "Statistiques masquées"},
    Entry{"Statistics shown", "Statistiques affichées"},
    Entry{"Subtitle size: {} %", "Sous-titres : {} %"},
    Entry{"Subtitle track refused", "Piste de sous-titres refusée"},
    Entry{"Subtitles", "Sous-titres"},
    Entry{"Subtitles (j)", "Sous-titres (j)"},
    Entry{"Subtitles off", "Sous-titres désactivés"},
    Entry{"Subtitles: {}", "Sous-titres : {}"},
    Entry{"Thai", "Thaï"},
    Entry{"Track {}", "Piste {}"},
    Entry{"Turkish", "Turc"},
    Entry{"Ukrainian", "Ukrainien"},
    Entry{"Unmute (m)", "Rétablir le son (m)"},
    Entry{"Update {} available (Settings > General > Application)",
          "Mise à jour {} disponible (Paramètres > Général > Application)"},
    Entry{"Vietnamese", "Vietnamien"},
    Entry{"Volume {} %", "Volume {} %"},
    Entry{"off", "non"},
    Entry{"on", "oui"},
    Entry{"track {}", "piste {}"},
    Entry{"{} does not open on this GPU: {} stays active",
          "{} ne s'ouvre pas sur ce GPU : {} reste en service"},
};

}  // namespace

std::span<const Entry> french_player() {
  return kPlayer;
}

std::span<const Entry> french_entries() {
  static const std::vector<Entry> all = [] {
    std::vector<Entry> merged;
    for (const auto part : {french_player(), french_settings(), french_image()})
      merged.insert(merged.end(), part.begin(), part.end());
    return merged;
  }();
  return all;
}

}  // namespace ajp::app::play::i18n
