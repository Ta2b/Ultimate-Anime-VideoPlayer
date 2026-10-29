# Auto libre (expérimental)

Mode d'Auto supplémentaire, `interpolation.auto_mode = "libre"` dans play.toml (défaut `"recommande"`, la loi
de `core/auto_rate.*` ne change pas). Cahier des charges de Tatoub : « pas de cadence ou palier à proprement
parler, juste un interpolateur qui envoie le plus d'images qu'il peut, et nous on ajuste la vitesse du flux
pour que la vidéo garde une vitesse constante, exactement comme sur les jeux vidéo ».

## 1. Analyse de faisabilité

### Ce qui existe

| Pièce | Rôle aujourd'hui | Ce que le mode libre en fait |
| --- | --- | --- |
| `CadenceAccumulator` (`core/cadence.*`) | grille uniforme de `num/den` fois la cadence source ; instants entiers en 1/num de paire ; `retarget` pour un facteur qui bouge | **inchangé** ; le mode libre ne passe plus par lui |
| `UpscaleStage::outputs()` | à l'arrivée de chaque source `d` : ingestion, grille de la paire (instants `t` et `pts`), saut des instants déjà en retard (`late`), mesures, RIFE/DRBA/GMFSS par instant | seule la source des instants change (`FreeGrid` à la place de `CadenceStep` de l'accumulateur) ; le reste de la chaîne lit `t` et `pts` comme avant |
| `RingStage` | un nombre de sorties par source, un slot du ring par sortie ; bloque quand le ring est plein | **contre-pression naturelle** : on ne soumet pas plus vite que l'écran ne libère de slots |
| `Session::choose` | à chaque tour, `queue.pick(horloge + latence d'affichage)` : l'image dont le pts est le plus proche ; les trop tardives sont jetées (`dropped`) ; présentation à la demande (VRR : quand une image neuve existe) | **inchangé** : des pts exacts à l'instant d'affichage sont ce qu'il attend |
| `FrameLatency` + `ready_media_us` | latence médiane soumission → image écrite ; `ready = max(horloge + période, horloge(hôte + latence))` | donne « l'horloge prédite à la fin du calcul » |
| `LiveSettings::auto_*` + `service_cost_us` | coût de service d'une paire (#123) | remplacé en mode libre par le **temps propre par image** (voir plus bas) |

### La plus petite modification propre

Les instants d'une paire `(g, d)` ne viennent plus d'une grille de facteur rationnel, mais d'un **planificateur
de pts** : `core/free_grid.{h,cpp}`, classe voisine de `CadenceAccumulator` (même contrat : `push(source)`
renvoie un `CadenceStep`, mêmes événements `Start / NewEpoch / InvalidPts / PtsNotIncreasing / Stale`, donc
aucun changement dans l'étage gpu en aval). Une classe voisine plutôt qu'un mode dans l'accumulateur : l'accumulateur
repose sur un invariant entier (`acc` en 1/num, jamais de dérive) qui n'a aucun sens quand l'écart est un temps
mesuré ; deux invariants dans une classe la rendraient illisible. Le seul point partagé est `CadenceStep` (un ami
de plus).

Instants d'une paire, en µs de temps de contenu :

```
t_0     = max(échéance, dernier instant émis + écart minimal, horloge prédite, g + 1)
t_{k+1} = t_k + max(écart minimal, temps de service × 1,10)        tant que t <= d
```

- **Horloge prédite** : `ready_media_us(horloge maître, maintenant, écart minimal, latence mesurée)` = le pts
  qu'aura une image soumise maintenant quand elle sera visible. Un instant plus ancien serait déjà passé :
  il est **abandonné avant tout calcul** (c'est le `late` de `upscale_stage`, mais fait en amont, par
  construction, au lieu d'être compté après coup). Une paire entière déjà passée ne donne que la source.
- **Écart minimal** : période de l'écran (fréquence VRR relue, à 1 % près) ou de `auto_cap`. Jamais plus vite
  que l'écran.
- **Temps de service** : sert à *prédire* où l'image sera prête (donc l'écart entre deux instants), pas à
  fixer un multiplicateur. Aucun palier, aucune cadence visée, aucune bande morte, aucun plafond appris : le
  débit est ce qui sort de `1 / (service × 1,10)`.
- **Échéance** : `t` après la dernière image émise ; elle passe d'une paire à l'autre, c'est la **phase
  continue** (la grille n'est pas recalée sur les sources).
- **Source droite** : un instant à moins d'un quart de pas de `d` devient `d` lui-même (aucune passe
  d'interpolation, décalage de quelques ms). Sinon la source droite n'est pas forcée : une image à `d` n'est
  pas plus utile qu'une autre à `d + ε`.
- Au plus `kMaxInstantsPerPair` (10) instants par paire (limite du ring) ; sans mesure encore, un pas d'une
  demi-paire ; si le GPU ne tient aucune image dans la paire, la source seule sort (comme sans interpolation).

Pas de dérive : chaque instant est un **pts exact** (µs) dans la paire qui le contient ; la vitesse de la
vidéo est celle de l'horloge, quelle que soit la cadence. Le test des dix minutes simulées (GPU 67 → 55 →
120 i/s) vérifie des pts strictement croissants, dans `(g, d]`, et que le film arrive à sa fin au bon pts.

### Temps de service mesuré (pas de modèle prédictif)

`LiveSettings::auto_written` (appelé par `main` quand une image de sortie est vue écrite) additionne le temps
propre de chaque image : `fin vue − max(soumission, fin de l'image d'avant)`. L'attente d'un slot du ring
(avant la soumission) n'y compte pas, donc une file pleine ne gonfle pas le coût (l'écueil qu'avait déjà
rencontré #123). Le coût de la paire (plus l'attente de ses mesures avant soumission) divisé par ses images
donne le temps de service par image `s`, moyenne qui **monte vite** (×0,6 par mesure) et **descend lentement**
(×0,12). Seules les paires de 2 images ou plus comptent. Point fixe : la paire coûte `a + b·n` ; avec le pas
`1,1·(a + b·n)/n`, `n` converge vers le `n` tel que la paire occupe 91 % de sa durée, sans oscillation (dérivée
`a / (a + b·n) < 1`).

### Contre-pression réelle

1. **Le ring** : plein, `RingStage` n'accepte plus de sortie ; le GPU ne tourne pas plus vite que l'écran.
2. **L'horloge** : `ready` repousse `t_0` quand la chaîne a du retard ; les sources déjà passées sont sautées
   par `passed()` comme avant. Le débit baisse de lui-même, sans consigne.
3. **Les pertes à l'écran** : une image sautée (`dropped`, arrivée trop tard) ou manquée (trou dans les pts
   plus grand que le pas) **élargit tout de suite le pas** (×1,12 par image perdue, au plus ×2) ; la pression
   ne se relâche qu'après 1 s sans perte, puis lentement (3 s de constante de temps).
4. **Repli** : 3 % d'images perdues (et au moins 6) sur 3 s → retour à Auto (recommandé), message à l'écran et
   ligne de journal.

### Ce qui reste fixe

La paire source `(g, d)` : `d` est décodée en avance et `g` est la précédente ; le décodeur et le ring ne
changent pas. DRBA (qui lit l'image d'après la paire) et GMFSS / mouvement non-RIFE ne sont pas pris en
charge par le mode libre (repli sur Auto avec message) : seul RIFE a été vérifié instant par instant.

## 2. Intégration

- `interpolation.auto_mode` : `"recommande"` (défaut) ou `"libre"` ; toute autre valeur est refusée
  (`FieldOutOfRange`). En direct (`live_part`).
- Liste « Mode d'Auto » sous le Multiplicateur (visible en mode Auto) ; « Auto libre (expérimental) » grisé
  avec infobulle sans écran VRR armé.
- Repli automatique sur Auto (`apply_auto`, `player_run.cpp`) : VRR absent ou coupé, moteur non RIFE,
  fréquence de l'écran inconnue ; ou trop d'images perdues. Les deux motifs écrivent une ligne de journal et un
  message.
- Statistiques : « Auto libre · 87 i/s · limite GPU 78 % · écart service 11,4 ms » (OSD et journal toutes les
  5 s).

## 3. Vérification réelle

Re:Zero 4e saison ép. 18 à 960 s, 75 s, plein écran (QD-OLED 165 Hz VRR armé), RTX 4070 Ti, 1440p de sortie,
RTX HDR, hauteur de travail 720p, `[auto] preset = 'custom'`, données de test à part (`--data-dir`), sous
verrou GPU. Banc : `/tmp/claude-1000/pbt/cmp3.py`. Auto = la loi actuelle (`origin/auto-mesure`, #123).

| | Auto (recommandé) | Auto libre |
| --- | --- | --- |
| cadence moyenne (images neuves) | 44,5 puis 42,2 i/s | **60,2 puis 61,0 i/s** |
| cadence médiane | 46,7 / 43,6 | 65,5 / 65,8 |
| images sautées / manquées | 4 / 15 puis 3 / 18 | 5 / 3 puis 4 / 0 |
| pertes (sautées + manquées) / (neuves + pertes) | 0,57 % puis 0,63 % | **0,18 % puis 0,09 %** |
| intervalle entre images neuves, σ | 7,5 / 9,5 ms | 7,9 / 7,9 ms |
| intervalle entre images neuves, p95 | 38,8 / 42,6 ms | 28,4 / 27,2 ms |
| changements de palier | 6 / 8 | aucun (pas de palier) |
| temps par image médian (présentation) | 20,7 / 21,9 ms | 15,6 / 15,2 ms |

Statistiques en direct sur la scène lourde : « Auto libre · 59 i/s · limite GPU 87 % · écart service 16,2 ms ».
Les « manquées » du mode libre se comptent contre le pas courant (pas de cadence nominale) ; les « sautées » sont
les mêmes dans les deux modes. Critère de livraison (pertes < 1 %) tenu. Une première version, qui ne clôturait
pas la mesure de service (la dernière image d'une paire n'est plus forcément la source), restait à ×2 fixe
(45 i/s) ; une deuxième se repliait dès la première seconde sur les pertes du démarrage (le repli est muet les
3 premières secondes).

## 4. Ce qui sépare de l'idéal

- Le temps de service par image reste une **estimation** (moyenne glissante, marge 10 %, pression) : la
  boucle n'est pas « dès que le GPU est libre » à la milliseconde mais « au rythme où le GPU a livré ». Le débit
  est donc un résultat, mais lissé sur quelques paires.
- Le planificateur décide les instants d'une paire quand sa source droite arrive (à l'ingestion), pas un par
  un au moment de la soumission : le ring a un nombre de sorties fixé par source (`RingStage`). Un vrai « une
  image à la fois » demanderait un étage producteur qui ne connaît pas à l'avance le nombre de sorties.
- DRBA et GMFSS (qui lisent l'image d'après ou un flot de paire) ne sont pas vérifiés : repli sur Auto.
- Au plus 10 instants par paire (ring) : 120 i/s minimum de plafond pour une source de 12 i/s.
- Les instants à moins d'un quart de pas d'une source tombent sur elle : jusqu'à ~4 ms de décalage.
