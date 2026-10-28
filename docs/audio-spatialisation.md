# Audio : format de sortie, « Comment écoutez-vous ? » et spatialisation (décision de Tatoub)

Résumé de l'analyse du fil « spatialisation du son stéréo » (fichier du projet
`analyses/spatialisation-son-stereo.md`, « Complément 3 », non lisible sur le PC de développement : ce texte
reprend le résumé transmis par le coordinateur, à recouper avec l'original). Intégré au plan
`docs/ux-parametres.md` (4.8 écran 5 et 4.9).

## Décisions
- **Aucune détection casque ou enceintes** (rarement possible).
- **Automatique**
  - *Source* : format par piste (mono, stéréo, 5.1, 7.1, passthrough), par la disposition de canaux FFmpeg.
  - *Sortie* : nombre de canaux du périphérique par défaut (mono, stéréo, 5.1, 7.1), lu via SDL3, **relu quand le
    périphérique par défaut change**.
  - *Conversion* choisie par le lecteur : mono → stéréo (duplication), 2.0 → mono (somme), 5.1 ou 7.1 → stéréo
    (downmix soigné, centre rehaussé), 5.1 ↔ 7.1, passage direct si identiques, contournement pour le passthrough.
- **Choix de l'utilisateur, une seule fois, au premier lancement** : « Comment écoutez-vous ? »
  - **Casque** : crossfeed pour le 2.0, binaural pour le surround ;
  - **Enceintes** : rendu fidèle, downmix soigné ;
  - **Home cinéma** : sortie multicanale, surround en passage direct.
  Modifiable ensuite dans les réglages. Écran passé : **Enceintes en rendu fidèle**.

## Règles
- Profil global, avec **profil par périphérique** possible (mémorisé par identifiant) ; un périphérique inconnu
  reprend le profil global **sans poser de question**.
- Home cinéma seulement si le périphérique expose **plus de 2 canaux** ; sinon grisé, avec l'explication.
- Une source ou un périphérique **mono** ne déclenche aucune spatialisation.
- Mode **Fidèle** dans tous les profils ; bascule A/B par touche conservée.

## Périmètre v1 et recommandation de l'analyse
- v1 : casque = crossfeed + binaural ; enceintes = downmix soigné ; écran de premier lancement ; détecteur de
  formats (petit travail).
- Recommandation : **option, pas pilier** ; **casque d'abord** ; reporter l'upmix 5.1 « intelligent » et les
  traitements dédiés aux voix (risque de les dégrader).

## Ce que le lecteur sait déjà (vérifié dans le code, `src/audio`)
- `SdlSink::open` ouvre **toujours le périphérique par défaut** (`SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK`) et prend
  son format préféré (fréquence et **nombre de canaux, jusqu'à 8**) ; la conversion passe par libswresample
  (`PcmConverter`, dispositions de canaux FFmpeg, 5.1 en FL FR FC LFE SL SR) : **le multicanal et le remixage de
  base existent**.
- **Manque** : choix d'un autre appareil de sortie (aucune liste des appareils), relecture du format quand le
  périphérique par défaut change (le format est lu à l'ouverture seulement), mémorisation par périphérique,
  réglages de mixage (centre rehaussé, crossfeed, binaural), profil Casque / Enceintes / Home cinéma.
- Donc l'écran audio de l'assistant demande d'abord **un petit travail de la couche audio** : lister les
  appareils (`SDL_GetAudioPlaybackDevices`), ouvrir celui choisi, réagir aux événements d'appareil
  (`SDL_EVENT_AUDIO_DEVICE_ADDED/REMOVED`, changement du défaut), lire les canaux d'un appareil
  (`SDL_GetAudioDeviceFormat`).

## Profils de virtualisation (v0.6)

Sous « Écoute », un menu **Profil** (casque ou enceintes) choisit un réglage nommé ; **Réglages avancés** ajoute le profil
**Personnalisé** et ses curseurs (enregistrés dans `play.toml` : `[audio] profile`, `[audio.custom]`). Les valeurs hors plage sont
ramenées dans leur plage ; un profil qui ne va pas à l'écoute en cours (Voix claires au casque) retombe sur Naturel. Naturel = le mixage
d'avant les profils.

| Profil | Écoute | Effet |
|---|---|---|
| Naturel | casque, enceintes | crossfeed 0,32 à 700 Hz, tête de 8,75 cm, centre +2 dB |
| Doux | casque | crossfeed 0,20 à 800 Hz, ombre de tête 0,25 |
| Enceintes virtuelles | casque | crossfeed 0,50 à 650 Hz, retard 0,3 ms, ombre 0,45, arrière +10 % |
| Large | casque | tête de 10 cm, ombre 0,50, arrière +20 % (5.1 et 7.1) |
| Voix claires | enceintes | centre ×1,6, arrière 0,85, basses 0,2 |
| Nuit | enceintes | centre ×1,5, arrière 0,7, basses 0,12, gain 0,55 |

Réglages du mode Personnalisé : crossfeed (niveau, adoucissement, retard), taille et ombre de la tête, niveau des arrières et
côtés, niveau du centre, gain du mixage stéréo, basses (LFE). Les tests (`[mixer][profils]`) vérifient les sens des écarts (fort > naturel
> doux ; Nuit réduit l'arrière) ; l'écoute subjective reste à faire par Tatoub.

## Profils, suite (v0.6) : distance, pièce, égaliseur, nuit, appareils

Réglages ajoutés au mixage (`audio::MixTuning`, tous à zéro par défaut : rien ne change ni ne coûte tant qu'on n'y touche pas) :

| Réglage | Effet |
|---|---|
| Distance (0 à 1) | enceintes virtuelles plus loin : niveau direct jusqu'à −2 dB, pièce plus présente |
| Pièce (0 à 1) | réverbération légère : 4 filtres en peigne amortis et 2 passe-tout par oreille, oreilles décorrélées |
| Égaliseur graves / voix / aigus | étagère 120 Hz, cloche 2,5 kHz, étagère 8 kHz, ±8 dB (formules de l'Audio EQ Cookbook) |
| Nuit (0 à 1) | compresseur lié des deux canaux : seuil −26 dBFS, rapport jusqu'à 6:1, attaque 10 ms, relâchement 250 ms, rattrapage à moitié ; jamais d'écrêtage (limiteur doux) |

Nouveaux profils : **Salon** (casque : distance et pièce), **Voix d'anime** (graves −1,5 dB, voix +3 dB, aigus +1,5 dB ; casque et enceintes) ; **Nuit** compresse la dynamique. Un appareil qui a son profil propre garde aussi son choix de
profil (`[audio.device_profiles]` de play.toml). Tests : `[mixer][profils]` (égaliseur à la fréquence visée, compression, queue de réverbération, aller-retour).
À écouter par Tatoub : les valeurs sont des points de départ.

## Réponses impulsionnelles « à la HeSuVi » (casque, réglages avancés)

Au casque, le lecteur peut convoluer le son avec les réponses impulsionnelles d'un fichier WAV choisi par l'utilisateur (`audio/hrir.h`). Elles **remplacent** le crossfeed et le binaural paramétrique ; sans fichier (le défaut), rien ne change. Réglages de play.toml : `audio.hrir_path` (chemin ; vide : aucune) et `[audio.device_hrir]` (appareil → chemin, pour un appareil qui a son profil propre). Interface : Réglages avancés → Écoute (« Réponse impulsionnelle (HeSuVi, .wav) », bouton « Aucune »).

**Formats lus** : PCM 16 ou 24 bits, ou float 32, de 8 à 384 kHz, au plus 32 768 échantillons par réponse.

| Canaux du fichier | Contenu |
|---|---|
| 14 | HeSuVi : FL→G, FL→D, SL→G, SL→D, BL→G, BL→D, FC→G, FR→D, FR→G, SR→D, SR→G, BR→D, BR→G, FC→D (G, D : oreille gauche, droite). Pas de LFE : les basses passent directement aux deux oreilles (niveau « basses », comme le binaural paramétrique) |
| 16 | huit haut-parleurs dans l'ordre FL FR FC LFE BL BR SL SR, chacun oreille gauche puis droite |
| 2 | HRIR stéréo : les deux oreilles pour une seule source (l'enceinte gauche) ; la droite en est le miroir, le centre la moyenne, les arrières et côtés reprennent la même direction |

Toute autre disposition est refusée (`FormatUnsupported`), un fichier tronqué ou vide aussi (`FileUnreadable`), une réponse trop longue (`FieldOutOfRange`). Un fichier refusé ne coupe pas le son : le lecteur le dit et garde le modèle paramétrique. Le stéréo est traité comme deux enceintes avant, le 5.1 et le 7.1 par leurs huit positions.

**Traitement** : le fichier est lu une fois, puis ramené au taux de sortie (sinc fenêtré, coupure sous le plus bas des deux taux) et normalisé : le plus fort des filtres est borné à 0,9 par la somme de ses valeurs absolues, si bien que le pire signal plein-échelle sur un haut-parleur ne dépasse pas 0,9 (jamais de gain au-dessus de 1). La convolution est partitionnée uniforme (overlap-save, blocs de 512 trames, FFT radix-2 maison de 1 024 points, un état par canal) : elle tient dans le thread audio et sa **latence est d'un bloc, 10,7 ms à 48 kHz**.

**Licences** : le lecteur ne livre aucune réponse impulsionnelle. Chaque jeu a sa licence (celles des fichiers HeSuVi circulant sur Internet varient, ainsi que celles des bases libres de mesures comme CIPIC, SADIE II ou ARI) : lisez celle du fichier que vous choisissez avant de l'utiliser ou de le redistribuer.

Tests : `[hrir]` (lecture de WAV synthétiques, impulsion, égalité avec la convolution directe, rééchantillonnage, gain de sécurité, refus, mixeur avec et sans réponses).
