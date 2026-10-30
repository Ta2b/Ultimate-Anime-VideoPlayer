#include <array>

#include "app/play/i18n_fr.h"

namespace ajp::app::play::i18n {
namespace {

// Audio : (clé anglaise, français).
constexpr std::array kSoundPage{
    Entry{"A WAV impulse response (HeSuVi 14 ch, 7.1 16 ch or 2 ch HRIR) replaces the headphone model. "
          "Empty: the default. Adds about 11 ms of latency.",
          "Une réponse WAV (HeSuVi 14 can., 7.1 16 can. ou HRIR 2 can.) remplace le modèle du casque. Vide : "
          "le modèle d'origine. Ajoute environ 11 ms de latence."},
    Entry{"Surround mix (5.1 and 7.1 tracks)", "Mixage surround (pistes 5.1 et 7.1)"},
    Entry{"Room, equalizer and night mode", "Pièce, égaliseur et mode nuit"},
    Entry{"Stereo tracks (most anime). 5.1 and 7.1 tracks use the surround mix, in Advanced.",
          "Pistes stéréo (la plupart des animes). Les pistes 5.1 et 7.1 passent par le mixage surround, en "
          "Avancé."},
    Entry{"Bundled response", "Réponse livrée"},
    Entry{
        "Bundled responses come from the HeSuVi collection (MIT licence, data/hrir). A file you pick has its "
        "own licence.",
        "Les réponses livrées viennent de la collection HeSuVi (licence MIT, data/hrir). Un fichier que vous "
        "choisissez a sa propre licence."},
    Entry{"Dolby Atmos", "Dolby Atmos"},
    Entry{"DTS Headphone:X", "DTS Headphone:X"},
    Entry{"Windows Sonic", "Windows Sonic"},
    Entry{"Creative SBX (100 %)", "Creative SBX (100 %)"},
    Entry{"Creative SBX (33 %)", "Creative SBX (33 %)"},
    Entry{"Anime", "Anime"},
    Entry{"Sound style", "Style de son"},
    Entry{"Style", "Style"},
    Entry{"Cinema", "Cinéma"},
    Entry{"The track as mixed, with a light touch.", "La piste telle que mixée, avec une touche légère."},
    Entry{"Crisper voices, wider stereo, controlled loudness: tireless dialogue.",
          "Voix plus nettes, stéréo plus large, sonie maîtrisée : des dialogues sans fatigue."},
    Entry{"Anime, pushed further: voices +3 dB, more bass, firmer compression.",
          "Anime en plus fort : voix +3 dB, plus de graves, compression plus ferme."},
    Entry{"Large virtual room and strong effects, for action.",
          "Grande salle virtuelle et effets marqués, pour l'action."},
    Entry{"Deep bass and a little more treble, for action and music.",
          "Graves profonds et un peu plus d'aigus, pour l'action et la musique."},
    Entry{"Quiet scenes stay audible, explosions stay down.",
          "Les scènes calmes restent audibles, les explosions restent contenues."},
    Entry{
        "One choice sets the mix profile and the Anime sound together. Fine control is in Advanced settings.",
        "Un seul choix règle ensemble le profil de mixage et le son anime. Le réglage fin est dans les "
        "réglages avancés."},
    Entry{"Anime sound", "Son anime"},
    Entry{"Anime voices", "Voix d'anime"},
    Entry{"Another device uses the global profile.", "Un autre appareil reprend le profil global."},
    Entry{"Audio delay (ms)", "Délai audio (ms)"},
    Entry{"Back to Natural", "Revenir à Naturel"},
    Entry{"Bass boost", "Renfort des graves"},
    Entry{"Bass level (LFE)", "Niveau des basses (LFE)"},
    Entry{"Center level (voices)", "Niveau du centre (voix)"},
    Entry{"Changing the device reopens the video at the same spot.",
          "Changer d'appareil rouvre la vidéo au même endroit."},
    Entry{"Clear voices", "Voix claires"},
    Entry{"Compare (hold: Natural)", "Comparer (maintenir : Naturel)"},
    Entry{"Crisper voices, clearer treble, lighter bass: tireless anime dialogue.",
          "Voix plus nettes, aigus plus clairs, graves allégés : dialogues d'anime sans fatigue."},
    Entry{"Crossfeed", "Crossfeed"},
    Entry{"Crossfeed delay (ms)", "Retard du crossfeed (ms)"},
    Entry{"Crossfeed filter (Hz)", "Filtre du crossfeed (Hz)"},
    Entry{"Crossfeed for stereo, binaural for surround.", "Crossfeed en stéréo, binaural en surround."},
    Entry{"Decorrelated ambience", "Ambiance décorrélée"},
    Entry{"Device", "Appareil"},
    Entry{"Display latency (ms)", "Latence de l'écran (ms)"},
    Entry{"Equalizer: bass (dB)", "Égaliseur : graves (dB)"},
    Entry{"Equalizer: treble (dB)", "Égaliseur : aigus (dB)"},
    Entry{"Equalizer: voices (dB)", "Égaliseur : voix (dB)"},
    Entry{"Faithful sound; surround is mixed down to stereo with a lifted center.",
          "Rendu fidèle ; le surround est mixé en stéréo, centre rehaussé."},
    Entry{"Follow the system default device", "Suivre l'appareil par défaut du système"},
    Entry{"Greyed out: this device only has 2 channels.", "Grisé : cet appareil n'a que 2 canaux."},
    Entry{"HeSuVi response (.wav)", "Réponse HeSuVi (.wav)"},
    Entry{"Head shadow", "Ombre de la tête"},
    Entry{"Head size (m)", "Taille de la tête (m)"},
    Entry{"Headphones", "Casque"},
    Entry{"Headphones: a bigger head, stronger rear (5.1 and 7.1).",
          "Casque : tête plus grande, arrière plus présent (5.1 et 7.1)."},
    Entry{"Headphones: like speakers in front of you, far from the head.",
          "Casque : comme des enceintes devant soi, loin de la tête."},
    Entry{"Headphones: nearly intact separation, just softer for long listening.",
          "Casque : séparation presque intacte, plus doux pour les longues écoutes."},
    Entry{"Headphones: speakers in a room, a bit farther, with light reverb.",
          "Casque : enceintes dans une pièce, un peu plus loin, avec une réverbération légère."},
    Entry{"Hold to hear the track without Anime sound. The volume differs a little.",
          "Maintenez pour entendre la piste sans Son anime. Le volume diffère un peu."},
    Entry{"Home theater", "Home cinéma"},
    Entry{"In stereo the profile works through the equalizer and compression; center, rear and bass exist in "
          "5.1 and 7.1 only.",
          "En stéréo, le profil agit par l'égaliseur et la compression ; centre, arrière et basses "
          "n'existent qu'en 5.1 et 7.1."},
    Entry{"Level", "Niveau"},
    Entry{"Limiter ceiling (dBTP)", "Plafond du limiteur (dBTP)"},
    Entry{"Listening", "Écoute"},
    Entry{"Living room", "Salon"},
    Entry{"Loudness normalization (EBU R128)", "Normalisation de sonie (EBU R128)"},
    Entry{"Multichannel, surround sent as is.", "Multicanal, surround envoyé tel quel."},
    Entry{"Night", "Nuit"},
    Entry{"Night compression", "Compression de nuit"},
    Entry{"No processing: the track's sound as is.", "Aucun traitement : le son de la piste tel quel."},
    Entry{"None##hrir", "Aucune##hrir"},
    Entry{"On a fixed-rate display, playback speeds up or slows down by at most 0.125% so each frame fits "
          "whole refreshes: no more doubled frames. Sound follows, inaudibly.",
          "Sur un écran à fréquence fixe, la lecture accélère ou ralentit d'au plus 0,125 % pour que chaque "
          "image tienne des rafraîchissements entiers : plus d'image doublée. Le son suit, sans changement "
          "audible."},
    Entry{"Original (no processing)", "Fidèle (aucun traitement)"},
    Entry{"Output device", "Appareil de sortie"},
    Entry{"Own profile for this device", "Profil propre à cet appareil"},
    Entry{"Positive delays the sound. Ctrl+click the slider to type a value (up to ±10,000 ms).",
          "Positif : le son est retardé. Ctrl+clic sur le curseur pour saisir une valeur (jusqu'à ±10 000 "
          "ms)."},
    Entry{"Profile", "Profil"},
    Entry{"Rear level", "Niveau des arrières"},
    Entry{"Reset to 0", "Remettre à 0"},
    Entry{"Room (reverb)", "Pièce (réverbération)"},
    Entry{"Soft", "Doux"},
    Entry{"Soft compressor", "Compresseur doux"},
    Entry{"Spatialization", "Spatialisation"},
    Entry{"Speaker distance", "Distance des enceintes"},
    Entry{"Speakers", "Enceintes"},
    Entry{"Speakers: compressed dynamics (softer explosions, louder whispers) and voices forward.",
          "Enceintes : dynamique comprimée (explosions réduites, murmures relevés) et voix devant."},
    Entry{"Speakers: lifted center and quieter rear and bass; in stereo, louder voices and lighter bass.",
          "Enceintes : centre rehaussé, arrière et basses discrets ; en stéréo, voix relevées et graves "
          "allégés."},
    Entry{"Stereo mix gain", "Gain du mixage stéréo"},
    Entry{"Strong anime", "Anime fort"},
    Entry{"Sync", "Synchronisation"},
    Entry{"Sync playback to the display", "Caler la lecture sur l'écran"},
    Entry{"Target loudness (LUFS)", "Sonie visée (LUFS)"},
    Entry{"The balanced default.", "Le réglage d'origine, équilibré."},
    Entry{"The profile is mostly heard in surround; in stereo it sets the crossfeed.",
          "Le profil s'entend surtout en surround ; en stéréo, il règle le crossfeed."},
    Entry{"The time between choosing a frame and seeing it. Set it here if the picture seems early or late.",
          "Le temps entre le choix d'une image et son apparition. À régler ici si l'image paraît en avance "
          "ou en retard sur le son."},
    Entry{"This device has more than 2 channels: Anime sound needs a stereo output.",
          "Cet appareil a plus de 2 canaux : le Son anime demande une sortie stéréo."},
    Entry{"Turns all processing off: the raw track, simply mixed for the device.",
          "Coupe tout traitement : le son brut de la piste, simplement mixé pour l'appareil."},
    Entry{"Virtual speakers", "Enceintes virtuelles"},
    Entry{"Voices forward", "Voix en avant"},
    Entry{"Voices: 3 kHz presence (dB)", "Voix : présence 3 kHz (dB)"},
    Entry{"Voices: center boost (dB)", "Voix : centre renforcé (dB)"},
    Entry{"Wide", "Large"},
    Entry{"Wider stereo, voices forward, dialogue presence, bass boost, -20 LUFS loudness, soft compressor "
          "and limiter.",
          "Stéréo élargie, voix devant, présence des dialogues, graves renforcés, sonie à -20 LUFS, "
          "compresseur doux et limiteur."},
    Entry{"Wider, voices +3 dB, more bass, firmer compressor, -18 LUFS loudness.",
          "Plus large, voix +3 dB, plus de graves, compresseur plus ferme, sonie à -18 LUFS."},
    Entry{"estimated", "estimée"},
    Entry{"{} channels on this device.", "{} canaux sur cet appareil."},
    Entry{"Headphones: a light virtual room, nearly intact separation.",
          "Casque : une légère pièce virtuelle, séparation presque intacte."},
    Entry{"Headphones: two speakers in front of you in a room, like HeSuVi: the sound leaves your head.",
          "Casque : deux enceintes devant vous dans une pièce, à la HeSuVi : le son sort de votre tête."},
    Entry{"Headphones: speakers wide apart, a large room, stronger rear (5.1 and 7.1).",
          "Casque : enceintes très écartées, grande pièce, arrière plus présent (5.1 et 7.1)."},
    Entry{
        "Speakers: lifted center and quieter rear and bass; in stereo, much louder voices and lighter bass.",
        "Enceintes : centre rehaussé, arrière et basses discrets ; en stéréo, voix bien plus fortes et "
        "graves allégés."},
    Entry{"Headphones: speakers in a room, a bit farther, with a lot of reverb.",
          "Casque : enceintes dans une pièce, un peu plus loin, avec beaucoup de réverbération."},
    Entry{"Deep, strong bass and a little more treble, for action and music.",
          "Des basses profondes et fortes, un peu plus d'aigus : pour l'action et la musique."},
    Entry{"Anime cinema", "Cinéma anime"},
    Entry{"Headphones: speakers in a room, clear voices, deep bass: the whole anime treatment.",
          "Casque : enceintes dans une pièce, voix claires, basses profondes : le traitement anime complet."},
};

}  // namespace

std::span<const Entry> french_sound() {
  return kSoundPage;
}

}  // namespace ajp::app::play::i18n
