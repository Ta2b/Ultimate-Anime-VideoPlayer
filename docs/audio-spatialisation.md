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
