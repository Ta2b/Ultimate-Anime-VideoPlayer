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
