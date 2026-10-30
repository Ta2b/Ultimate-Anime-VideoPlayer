# Son anime : le traitement audio dédié aux pistes stéréo

Exigence de Tatoub : « un traitement à part, surtout spécialisé pour les sorties stéréo : une vraie
spatialisation du stéréo et une amélioration complète du son pour les animes ». Les profils de mixage
(`audio/mixer.h`, `docs/audio-spatialisation.md`) agissent surtout sur le 5.1 et le 7.1 ; or la plupart des
animes sont en AAC 2.0. Le module `audio/anime_sound.{h,cpp}` part donc de la paire gauche/droite.

## Quand il agit
Seulement pour une **piste stéréo sur une sortie stéréo**, en casque ou aux enceintes. Repli propre, sans
aucune dégradation, dans tous les autres cas : piste mono (même doublée en stéréo par le décodeur), 5.1 et 7.1
(chemin des profils de mixage), Home cinéma, mode Fidèle, appareil multicanal. Le niveau **Naturel** ne construit
rien : le son passe **bit à bit** comme avant. Défaut : **Naturel** (le niveau Anime se choisit dans Paramètres > Son).

## Chaîne de traitement
1. **Spatialisation** (milieu/côtés : M = (G+D)/2, S = (G−D)/2).
   - *Enceintes* : les côtés sont élargis par une annulation de diaphonie transaurale réduite aux côtés
     (S' = S + c·passe-bas(retard 0,22 ms(S')), puis gain 1 à 1,45) et complétés par une ambiance décorrélée
     (trois passe-tout de 1 à 3,7 ms, module 1). Le milieu M n'est jamais retiré ni filtré hors voix : la somme
     G+D, c'est-à-dire l'écoute mono, ne perd rien.
   - *Casque* : la paire devient six enceintes virtuelles — avants ±30° (côtés et 20 % du milieu), enceinte centrale
     0° (80 à 92 % du milieu : les voix « devant », sans le peigne vers 2 kHz qu'un milieu porté par deux avants
     creuserait), arrières ±110° (ambiance décorrélée, deux jeux de passe-tout de signes opposés) — rendues par le
     binaural paramétrique du mixeur, ou par les réponses HeSuVi de l'utilisateur si elles sont chargées (latence
     d'un bloc de 512 trames, environ 11 ms de plus).
2. **Voix** : la bande vocale du milieu (passe-bande large autour de 1 kHz, Q 0,6) est portée de +1 à +3 dB (réglage
   0 à 4), plus une cloche de présence à 3 kHz (0 à 4 dB). « Voix en avant » creuse les côtés dans la bande vocale
   (jusqu'à −3 dB) et confie davantage de milieu à l'enceinte centrale.
3. **Graves** : harmoniques douces (3f, 5f) de la bande sous 110 Hz, normalisée en niveau puis saturée par tanh,
   fondamental retiré par un passe-haut d'ordre 4, ajoutées en mono. Pour les petits haut-parleurs et le casque, où
   le fondamental seul ne s'entend pas.
4. **Dynamique** : normalisation de sonie (filtre K de l'ITU-R BS.1770-4, fenêtres de 400 ms et de 3 s, porte à −50
   LUFS ; gain visé = cible − sonie court terme, abaissé par la sonie momentanée si un passage fort arrive, borné
   à −9/+6 dB, lissé lentement : 5 s en montée, 2,5 s en descente, 0,5 s pendant les 2 premières secondes) ; compresseur
   doux lié (seuil −16 à −24 dBFS, rapport 1 à 3,5, genou de 6 dB, attaque 10 ms, relâchement 200 ms, sans gain de
   rattrapage) ; limiteur **vrai-crête** (FIR polyphasé ×4 de 48 coefficients, anticipation de 1,3 ms, minimum puis
   moyenne glissante pour que le gain ne dépasse jamais le requis, relâchement 60 ms, plafond de −3 à −0,5 dBTP,
   −1 par défaut). L'égaliseur, la pièce et la compression « nuit » du profil de mixage sont appliqués **avant**
   cette étape, le limiteur reste donc toujours le dernier maillon.

## Niveaux et réglages
| | Naturel | Anime | Anime fort |
|---|---|---|---|
| Spatialisation / ambiance | — | 0,5 / 0,5 | 0,85 / 0,75 |
| Voix (centre) / présence 3 kHz | — | +2 / +1,5 dB | +3 / +2,5 dB |
| Voix en avant | — | 0 | 0,7 |
| Graves (harmoniques) | — | 0,3 | 0,6 |
| Sonie visée | — | −20 LUFS | −18 LUFS |
| Compresseur | — | 0,3 | 0,55 |
| Plafond | — | −1 dBTP | −1 dBTP |

Le niveau se choisit dans Paramètres > Son > « Son anime » ; avec « Réglages avancés », « Personnalisé » ouvre les
curseurs (spatialisation, ambiance, voix, présence, voix en avant, graves, sonie et sa cible de −23 à −16 LUFS,
compresseur, plafond). Le bouton **Comparer (maintenir : Naturel)** bascule sur Naturel tant qu'on le tient. Le
niveau peut être propre à un appareil (« Profil propre à cet appareil »). Il s'applique en direct (message
`SetListening` : 16 cases `anime`, `anime_set` ; les 16 cases `tuning` des profils de mixage sont inchangées).

`play.toml` :
```toml
[audio.anime]
level = "anime"          # natural | anime | strong | custom
[audio.anime.custom]     # utilisé si level = "custom" (valeurs hors plage refusées en nommant la clé)
width = 0.5   ambience = 0.5   voice_db = 2.0   presence_db = 1.5   voice_forward = 0.0
bass = 0.3    loudness = true  target_lufs = -20.0   compress = 0.3   ceiling_db = -1.0
[audio.device_anime]
"Nom de l'appareil" = "strong"
```

## Ce qui est mesuré (tests `tests/unit/audio/anime_sound_test.cpp`, 48 kHz)
- Sonie BS.1770 : sinus 1 kHz et 997 Hz à −23 dBFS sur deux voies = −23,0 LUFS (±0,1) en intégré, court terme et
  momentané ; une voie seule −26,0.
- Compatibilité mono (somme G+D, sans sonie, signal voix + musique) : **+1,2 dB (Anime) et +1,9 dB (Anime fort)
  aux enceintes, +0,7 et +1,6 dB au casque** par rapport à Naturel, jamais d'annulation (exigence : > −3 dB). Hors
  voix, la somme aux enceintes est inchangée à 0,1 dB près.
- Écrêtage : sinus 12 kHz à +2,9 dBFS (1,4) avec phase pour des crêtes entre échantillons : |x| ≤ 1 toujours ;
  crête vraie mesurée (sur-échantillonnage ×8 indépendant) −0,8 dBTP enceintes, −0,84 casque pour un plafond de −1
  (écart de l'interpolation ×4, < 0,3 dB).
- Latence par impulsion : 1,29 ms (anticipation du limiteur) aux enceintes comme au casque paramétrique ; plus
  environ 11 ms avec une réponse HeSuVi. Limite du test : 20 ms.
- Réponse en fréquence d'un sinus centré, hors égaliseur voulu : enceintes neutres 0,00 dB de 50 Hz à 16 kHz ;
  réglage Anime : de −0,07 à +0,03 dB hors de la bande vocale (graves intacts malgré les harmoniques) ; le
  casque neutre reste dans −1,4 à 0 dB (le crossfeed d'avant : jusqu'à −3,7 dB à 1 kHz).

  | Hz | 50 | 100 | 200 | 500 | 1k | 2k | 3k | 4k | 6k | 8k | 12k | 16k |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | enceintes Anime | +0,0 | −0,1 | +0,6 | +1,2 | +2,2 | +2,2 | +2,2 | +1,5 | +0,7 | +0,3 | +0,1 | +0,0 |
  | casque Anime | +0,0 | −0,1 | +0,6 | +0,9 | +1,2 | +0,7 | +1,9 | +1,0 | −0,2 | −0,5 | −0,9 | −1,0 |
  | casque Naturel (crossfeed) | 0,0 | −0,1 | −0,3 | −1,7 | −3,7 | −2,8 | −1,9 | −2,2 | −2,6 | −2,3 | −2,3 | −2,3 |

- Spatialisation : corrélation G/D d'une musique de synthèse à nappes larges 0,38 (Naturel) → 0,13 aux enceintes
  (Anime) ; une voix exactement au centre reste centrée (corrélation > 0,999, côtés < −40 dB) dans les deux modes.
  Au casque, la corrélation *monte* (0,81) : c'est l'effet des enceintes virtuelles (le son d'une enceinte arrive
  aux deux oreilles), non un défaut ; le crossfeed de Naturel agit dans le même sens, en moins marqué.
- Sonie : une piste trop faible est relevée de +4 à +6 dB au plus ; un passage moyen ou fort est ramené à
  −20 ±1,5 LUFS après 12 s.
- Robustesse : NaN, ±inf et 1e30 en entrée → sortie finie dans [−1, 1] et reprise normale ; silence → silence, gain
  tenu ; blocs de 1 à 4799 trames → même son à 1e-6 près ; Naturel, mono, 5.1, Home cinéma, Fidèle, sortie multicanal
  → bit-exacts.

Rendus hors ligne (test caché `[.anime_echantillons]`, dossier `/tmp/claude-1000/son-anime/`) sur 14 s d'une
synthèse (voix centrée + musique) et d'un extrait de l'épisode 18 de Re:Zero S4 (AAC 2.0) :

| signal / version | RMS dBFS | LUFS | corr. | largeur S/M dB | crête vraie dBTP |
|---|---|---|---|---|---|
| synthèse, Naturel | −20,8 | −18,5 | 0,67 | −7,1 | −5,9 |
| synthèse, enceintes Anime | −23,0 | −20,6 | 0,54 | −5,3 | −4,0 |
| synthèse, enceintes Anime fort | −23,2 | −20,8 | 0,42 | −3,9 | −2,9 |
| synthèse, casque Anime | −23,0 | −20,6 | 0,92 | −13,6 | −6,3 |
| épisode, Naturel | −15,5 | −12,9 | 0,86 | −11,2 | −0,3 |
| épisode, enceintes Anime | −23,0 | −20,2 | 0,82 | −10,0 | −3,0 |
| épisode, enceintes Anime fort | −23,2 | −20,4 | 0,80 | −9,6 | −2,0 |
| épisode, casque Anime | −23,0 | −20,4 | 0,96 | −16,4 | −4,3 |

L'épisode, très fort à l'origine (−12,9 LUFS, crête à −0,3 dBTP), est baissé de ~7 dB vers −20 LUFS : c'est la
normalisation voulue, et c'est pourquoi le bouton Comparer ne compare pas « à volume égal ».

## Limites honnêtes
- **Ce qui ne se valide qu'à l'oreille** : le rendu de la spatialisation (naturel ou artificiel ?), l'ambiance
  décorrélée (trop ou pas assez), le réglage de la voix (+2/+3 dB : clair ou agressif ?), les harmoniques des graves
  (sur quel casque ?), la dureté du compresseur, l'effet « enceintes virtuelles » au casque avec le modèle
  paramétrique (aucune HRTF mesurée n'est livrée : un seul profil de tête, qui convient inégalement) et la
  comparaison avec une vraie écoute sur enceintes. Les valeurs des niveaux sont des points de départ raisonnés, pas
  des réglages validés par des auditeurs.
- Le milieu M contient aussi la musique centrée (basse, caisse claire), pas seulement la voix : « voix en avant »
  agit sur la bande vocale du milieu et des côtés, sans séparation de sources (pas d'extraction par apprentissage).
- Le transaural n'est pas une vraie annulation de diaphonie mesurée sur un placement d'enceintes connu ; c'est une
  version prudente (côtés seulement, milieu intact) dont l'effet est modeste sur un mixage déjà mono-compatible. Aux
  enceintes très rapprochées ou mal placées, l'élargissement peut rester discret.
- Les côtés (S) ne sont pas plats : l'ambiance décorrélée et l'annulation de diaphonie modifient leur spectre
  (jusqu'à environ ±2 dB par peigne) ; la planéité mesurée à ±1 dB vaut pour un contenu centré.
- La sonie normalisée change le volume perçu entre Naturel et Anime ; une scène très faible n'est relevée que de
  6 dB au plus, et le gain suit lentement (secondes), donc un changement brusque de scène passe d'abord par le
  compresseur et le limiteur.
- Le limiteur à anticipation de 1,3 ms ne peut pas garantir un plafond au-delà de la précision de l'interpolation
  ×4 (mesuré : 0,2 dB) ; un dernier écrêtage dur à ±1 reste en filet de sécurité.
- Pas de mesure sur le matériel de sortie réel du PC de test : tout ce qui précède est calculé sur signaux de
  synthèse et un extrait réel hors ligne.

## Réponses de tête livrées
`data/hrir/` contient cinq réponses HeSuVi (Dolby Atmos, DTS Headphone:X, Windows Sonic, Creative SBX 100 % et 33 %),
reprises de Virtual Surround (Berny23, MIT) qui les tient de la collection HeSuVi (licence MIT, Matt Gore :
`data/hrir/LICENSE-hesuvi.txt`). Elles sont embarquées dans le lecteur (`app/play/bundled_hrir.h`) et se choisissent
dans les réglages avancés du casque (« Réponse livrée »), ce qui écrit `hrir_path = "bundled:atmos"`. La sixième du
dépôt d'origine (OpenAL Soft, 7 canaux) n'est pas reprise : le chargeur lit les fichiers à 2, 14 ou 16 canaux.
Une source stéréo passe par les réponses des enceintes avant gauche et droite, sans gain caché (le niveau du fichier est
respecté, comme avec PipeWire, HeSuVi ou JamesDSP).
