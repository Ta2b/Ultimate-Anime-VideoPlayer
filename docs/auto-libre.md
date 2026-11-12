# Mode Instantané d'Auto (anciennement « Auto libre », expérimental)

Noms affichés : « Instantané » (le mode par défaut depuis C6, anciennement « Auto libre » puis « Rapide » : le nom a changé pour ne plus se confondre avec le niveau de traitement Rapide) et « Stable » (anciennement « Auto (recommandé) », gardé au choix ; Instantané s'y replie sans message quand l'écran ou le moteur ne le permettent pas). Seuls les noms ont changé : la clé play.toml `interpolation.auto_mode` garde ses valeurs `"recommande"` (Stable) et `"libre"` (Instantané).

Mode d'Auto « Instantané », `interpolation.auto_mode = "libre"` dans play.toml (c'est le défaut ; `"recommande"` donne Stable, la loi
de `core/auto_rate.*` ne change pas). Cahier des charges de Tatoub : « pas de cadence ou palier à proprement
parler, juste un interpolateur qui envoie le plus d'images qu'il peut, et nous on ajuste la vitesse du flux
pour que la vidéo garde une vitesse constante, exactement comme sur les jeux vidéo ».

## 0. Instantané à pas constant (`--fast-steady`, chantier 7, essai)

La grille libre ci-dessous reste le défaut. `--fast-steady` la remplace, dans le mode Instantané seulement, par
N images par source, N constant dans un plan (décision : [ADR 0010](adr/0010-rapide-a-pas-constant-par-plan.md)).

| Pièce | Rôle |
| --- | --- |
| `core/steady_rate.{h,cpp}` (`SteadyRate`) | Choisit N, le plus grand candidat tel que (a + b·N + d·N)·1,15 ≤ P. En VRR, des demi-pas sous la fréquence de l'écran ; à fréquence fixe, les facteurs qui calent l'écran. À chaque coupe : l'a priori × l'échelle mesurée. Un réajustement par plan. Urgence à 2 pertes en 1 s. Repli à N = 1 seulement. |
| `core/steady_grid.{h,cpp}` (`SteadyGrid`) | Enveloppe `CadenceAccumulator`. Les instants g + j·(d − g)/N sont exacts ; quand N change, `retarget` garde la phase ; à N = 1, la source seule. |
| `UpscaleStage` | Chaque paire est mesurée, à N = 1 aussi. Une coupe appelle `LiveSettings::free_cut`. `Fate::Late` laisse sans calcul un instant passé (`upscale_late.cpp`). |
| `LiveSettings::set_steady` | Configure `SteadyRate` selon l'écran et le fps. Il reçoit les coûts de paire, les images produites, les pertes, les settle et les coupes, par les mêmes chemins que le Instantané libre. |
| `ChainStats` | `output_fps()` = fps × N. La ligne de stats montre N, l'échelle s, le budget et les pertes. |

Bancs (tests `[steady]`) :

| Banc | Résultat |
| --- | --- |
| T1 | Une heure à N = 6,5 : intervalles de 6 416 à 6 417 µs, σ 0,47 µs, dérive < 1 µs. |
| T4 | Coupes vers des plans 1,8 fois plus lourds : 0 perte, N moyen ≥ 75 % de l'optimum. |
| T5 | Pointe de 10 s : 9 pertes avec l'urgence, contre 662 sans. |
| T6 | Changements de N sans trou ni doublon. |
| T7 | À 59,94 Hz : tous les instants à ≤ 2 µs d'un vsync. |
| T8 | Plans tirés au hasard : N ≥ 85 % de l'optimum. |
| T9 | Dix minutes : 47,95 / 35,96 / 95,6 i/s, 0 perte. |
| T10 | d = 1,6 ms fait passer N de 6,5 à 5. |

Restent :
- les mesures sur le PC (7.10) et le coût d'affichage d, encore à 0 ;
- un a priori venu du banc ou du profil (sans lui, N s'amorce sur les 4 premières paires) ;
- Smooth (7.9), en option.

## 1. Analyse de faisabilité

### Ce qui existe

| Pièce | Rôle aujourd'hui | Ce que le mode libre en fait |
| --- | --- | --- |
| `CadenceAccumulator` (`core/cadence.*`) | grille uniforme de `num/den` fois la cadence source ; instants entiers en 1/num de paire ; `retarget` pour un facteur qui bouge | **inchangé** ; le mode libre ne passe plus par lui |
| `UpscaleStage::outputs()` | à l'arrivée de chaque source `d` : ingestion, grille de la paire (instants `t` et `pts`), saut des instants déjà en retard (`late`), mesures, RIFE/DRBA par instant | seule la source des instants change (`FreeGrid` à la place de `CadenceStep` de l'accumulateur) ; le reste de la chaîne lit `t` et `pts` comme avant |
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
(×0,12). Comptent les paires de 2 images ou plus, et la paire d'une seule image quand cette image est
interpolée. La source seule d'une coupe ou d'un dessin tenu ne dit rien du coût d'une image interpolée. Reste
la paire que le pas a réduite à sa source (`FreeGrid::starved` : ni l'horloge prédite ni l'écart minimal
n'empêchaient une image interpolée) : elle ne mesure rien, donc l'estimation redescend de 12 % par paire
(`FreeRate::starved`) jusqu'à ce qu'une image interpolée la mesure de nouveau. Sans cela, un à-coup du GPU
portait `s` au-dessus de l'écart des sources, plus aucune paire n'avait deux images et l'interpolation
s'arrêtait pour la séance (A12-12, `tests/unit/core/free_starved_test.cpp`). Point fixe : la paire coûte `a + b·n` ; avec le pas
`1,1·(a + b·n)/n`, `n` converge vers le `n` tel que la paire occupe 91 % de sa durée, sans oscillation (dérivée
`a / (a + b·n) < 1`).

### Contre-pression réelle

1. **Le ring** : plein, `RingStage` n'accepte plus de sortie ; le GPU ne tourne pas plus vite que l'écran.
2. **L'horloge** : `ready` repousse `t_0` quand la chaîne a du retard ; les sources déjà passées sont sautées
   par `passed()` comme avant. Le débit baisse de lui-même, sans consigne.
3. **Les pertes à l'écran** : une image sautée (`dropped`, arrivée trop tard) ou manquée (un instant planifié
   qui n'a jamais paru : jeté en retard, sauté avant calcul) **élargit tout de suite le pas** (×1,12 par image perdue, au plus ×2) ; la pression
   ne se relâche qu'après 1 s sans perte, puis lentement (3 s de constante de temps).
4. **Pas de repli sur Stable** (retiré en novembre 2026) : une paire que le pas réduit à sa source laisse l'image
   d'origine à l'écran, et l'image interpolée revient dès que le GPU suit. Quand RIFE ne produit plus au moins une
   image interpolée par paire de la source pendant une dizaine de secondes (seeks, pauses, coupes et dessins tenus
   hors du compte), la lecture le coupe pour la vidéo en cours et rend la carte à SPAN, avec un message bref
   (`MotionWatch`, `app/play/motion_cut.h`). L'option « Couper l'interpolation si la carte ne suit pas »
   (`interpolation.auto_off`, cochée par défaut) le décide ; décochée, RIFE continue quoi qu'il arrive.
   L'interpolation revient à la vidéo suivante, avec « Réessayer » (Vidéo > Post-traitement) ou dès qu'un
   réglage de l'interpolation change.

### Ce qui reste fixe

La paire source `(g, d)` : `d` est décodée en avance et `g` est la précédente ; le décodeur et le ring ne
changent pas. Seul Apple (flux optique du système, instants imposés) reste en repli sur Auto, avec message
(`free_instants_limit`, `app/play/free_engines.h`).

### Moteurs

| Moteur | tau arbitraire ? | Instants par paire | Remarque |
| --- | --- | --- | --- |
| RIFE (entier, fusionné, mouvement) | oui (plan de l'entrée, ou facteur du warp) | 10 (ring) | vérifié |
| DistilDRBA (fusionné ou entier) | oui : `drba_call(t)` accepte tout t de ]0, 1[ (fenêtre centrée sur a avant 0,5, sur b après), une passe du graphe par instant sur 3 images | 10 (ring) | lit l'image d'après la paire : la grille ne la touche pas |
| Apple | non | 0 | Auto |

## 2. Intégration

- `interpolation.auto_mode` : `"libre"` (défaut, Instantané) ou `"recommande"` (Stable) ; toute autre valeur est refusée
  (`FieldOutOfRange`). En direct (`live_part`).
- Liste « Mode d'Auto » sous le Multiplicateur (visible en mode Auto) ; « Instantané (expérimental) » grisé
  avec infobulle sans écran VRR armé.
- Repli automatique sur Stable (`apply_auto`, `player_run.cpp`) quand Instantané ne peut pas tourner : moteur
  qui ne calcule pas à un instant libre, fréquence de l'écran inconnue (`free_refusal`, `chain_shape.cpp`). Une
  ligne de journal, pas de message. Les images perdues ne replient plus (voir la contre-pression, point 4).
- Statistiques : « Instantané · 87 i/s · limite GPU 78 % · écart service 11,4 ms » (OSD et journal toutes les
  5 s).

## 3. Vérification réelle (historique, fin octobre 2026)

> **Mesure historique**, prise fin octobre 2026 juste après la loi de #123 (branche `auto-mesure`, fusionnée le
> 29/10 sur `7f11230`), avant l'ADR 0012 : RTX HDR et GMFSS ont été retirés depuis, et la chaîne a changé
> (SPAN, RIFE 4.22 lite, grade). Le script du banc (`cmp3.py`) vivait hors du dépôt et est perdu ; le banc
> du dépôt est `tools/bench/banc.sh`. Ces chiffres ne valent pas pour la chaîne actuelle (A28-21) : ne pas
> calibrer une décision dessus sans les refaire.

> **Historique (A32-01).** Ces chiffres datent de la chaîne avec RTX HDR par image créée, retirée depuis :
> le « service de 13 à 16 ms » et les cadences de 60 à 67 i/s ne décrivent plus le lecteur actuel. Ne pas
> s'en servir pour décider d'une cadence ou d'un moteur. La remesure sur le binaire livré (`linux-release`,
> par moteur, même scène) est demandée dans `docs/audits/A-CONFIRMER.md` (A32-01) ; son tableau remplacera
> celui-ci.

Re:Zero 4e saison ép. 18 à 960 s, 75 s, plein écran (QD-OLED 165 Hz VRR armé), RTX 4070 Ti, 1440p de sortie,
RTX HDR, hauteur de travail 720p, `[auto] preset = 'custom'`, données de test à part (`--data-dir`), sous
verrou GPU. Banc : un script hors du dépôt, perdu. Auto = la loi de l'époque (#123).

| | Stable | Instantané |
| --- | --- | --- |
| cadence moyenne (images neuves) | 44,5 puis 42,2 i/s | **60,2 puis 61,0 i/s** |
| cadence médiane | 46,7 / 43,6 | 65,5 / 65,8 |
| images sautées / manquées | 4 / 15 puis 3 / 18 | 5 / 3 puis 4 / 0 |
| pertes (sautées + manquées) / (neuves + pertes) | 0,57 % puis 0,63 % | **0,18 % puis 0,09 %** |
| intervalle entre images neuves, σ | 7,5 / 9,5 ms | 7,9 / 7,9 ms |
| intervalle entre images neuves, p95 | 38,8 / 42,6 ms | 28,4 / 27,2 ms |
| changements de palier | 6 / 8 | aucun (pas de palier) |
| temps par image médian (présentation) | 20,7 / 21,9 ms | 15,6 / 15,2 ms |

Statistiques en direct sur la scène lourde : « Instantané · 59 i/s · limite GPU 87 % · écart service 16,2 ms ».
Les « manquées » du mode libre se comptent contre le pas courant (pas de cadence nominale) ; les « sautées » sont
les mêmes dans les deux modes. Critère de livraison (pertes < 1 %) tenu. Une première version, qui ne clôturait
pas la mesure de service (la dernière image d'une paire n'est plus forcément la source), restait à ×2 fixe
(45 i/s) ; une deuxième se repliait dès la première seconde sur les pertes du démarrage (le repli est muet les
3 premières secondes).

### Par moteur (75 s, même scène, 720p, libre contre Auto)

| Moteur | Auto : cadence / pertes | Libre : cadence / pertes | Écart σ / p95 libre |
| --- | --- | --- | --- |
| RIFE | 44,5 et 42,2 i/s / 0,57 et 0,63 % | 60,2 et 61,0 i/s / 0,18 et 0,09 % | 7,9 / 28 ms |
| DistilDRBA ×3 fusionné | 40,7 i/s / 0,76 % (22 manquées, 10 changements) | **67,5 i/s / 0,04 %** (2 sautées, 0 manquée) | 7,7 / 25,3 ms (Auto 8,7 / 42,2) |
| GMFSS (retiré des sources depuis) | 24,0 i/s / 3,60 % (65 manquées, 18 changements) | 23,9 et 23,8 i/s / 0,99 % (15 sautées, 3 manquées) | 18,7 / 68 ms (Auto 9,7 / 58,9) |

DRBA : « Instantané · 66 i/s · limite GPU 79 % · écart service 13,1 ms ». GMFSS : « 22 i/s · limite GPU 91 %
· écart service 56 ms ». Un bug trouvé avec DRBA : `FreeRate::settle` (seek) raccourcissait le silence de
démarrage et le repli se déclenchait à 1,5 s ; `settle` ne fait désormais que l'allonger.

## 4. Ce qui sépare de l'idéal

- Le temps de service par image reste une **estimation** (moyenne glissante, marge 10 %, pression) : la
  boucle n'est pas « dès que le GPU est libre » à la milliseconde mais « au rythme où le GPU a livré ». Le débit
  est donc un résultat, mais lissé sur quelques paires.
- Le planificateur décide les instants d'une paire quand sa source droite arrive (à l'ingestion), pas un par
  un au moment de la soumission : le ring a un nombre de sorties fixé par source (`RingStage`). Un vrai « une
  image à la fois » demanderait un étage producteur qui ne connaît pas à l'avance le nombre de sorties.
- GMFSS : à 720p sur cette scène lourde (1440p, RTX HDR), le GPU ne tient que ~24 i/s, soit la cadence de la
  source : le mode libre n'apporte pas d'images en plus (23,9 i/s contre 24,0), avec des écarts d'affichage
  plus irréguliers (une paire a 1 instant, la suivante 2). Il reste disponible et marqué expérimental.
- Au plus 10 instants par paire (ring) : 120 i/s minimum de plafond pour une source de 12 i/s.
- Les instants à moins d'un quart de pas d'une source tombent sur elle : jusqu'à ~4 ms de décalage.

## 5. Sans VRR : le mode Instantané calé sur les vsyncs

Les jeux en cadence libre ne sont pas réservés aux écrans VRR ; le mode Instantané non plus. Sans VRR (écran à
fréquence fixe, ou VRR coupé), le mode reste le même (aucun palier ni multiplicateur) mais les instants de la
grille tombent **sur les vsyncs de l'écran** : une image n'est jamais calculée pour un instant entre deux
vsyncs, donc rien ne vibre. C'est le comportement d'un jeu en vsync : un GPU plus lent laisse l'image
précédente tenue un vsync de plus. Ce n'est pas optimal (sur un écran VRR l'image part quand elle est prête),
mais c'est utilisable partout. VRR armé : le comportement décrit plus haut ne change pas.

- **Grille** (`FreeInputs::vsync_period_us`, `vsync_origin_us`, `core/free_grid.cpp`) : l'instant idéal
  (`base`, comme avant : horloge prédite, dernier instant + écart minimal, échéance) est monté au vsync suivant
  (`pts ≡ origine mod période`). L'idéal avance du pas de service sans être arrondi : la cadence moyenne reste
  celle du GPU (par exemple 45 i/s sur 165 Hz donne des écarts de 2 et 3 vsyncs), seule la position de chaque
  image est quantifiée. Il n'y a pas de rattachement à la source droite hors grille. L'écart minimal vaut la
  période de l'écran (ou du plafond `auto_cap`, s'il est plus lent) ; une image tardive est abandonnée avant
  calcul comme avant, la grille repart au vsync d'après.
- **Phase** (`core/vsync_phase.*`) : la position, modulo la période, des vsyncs vus, lissée en cercle (8 %
  par mesure). Un vsync n'est « vu » que quand l'acquisition d'une image de la swapchain a attendu l'écran
  (`ScreenTiming::vsync_us`) : l'heure où la présentation est soumise suit l'heure choisie par la boucle et
  ne dit rien de l'écran (A12-03). La phase n'est connue qu'après huit mesures qui s'accordent ; sans elle,
  et sans grille mesurée, la grille n'est pas calée. L'origine de la grille est le vsync le plus proche
  plus le délai d'affichage de la sélection (`Session::visible_delay_us`) plus une demi-période : l'image
  choisie une demi-période avant son pts s'affiche alors pile au vsync suivant, au milieu de la marge de la
  boucle (pas de bascule d'un vsync à l'autre à cause de la gigue).
- **Statistiques / repli** : une image « manquée » se compte contre l'écart que la grille a planifié pour elle
  (`RingFrame::planned_gap_us`), pas contre le pas d'une paire calme : à 165 Hz, une paire en mouvement fort
  planifie deux vsyncs par image quand une calme en planifie un, et chacune de ses images comptait une
  manquée ; la contre-pression montait au plafond et le repli tombait sans perte visible (novembre 2026).
  Sans écart planifié (début d'époque), le pas arrondi au vsync au-dessus sert comme avant. (Le repli sur
  Stable a depuis été retiré : voir la contre-pression, point 4.)
- **Menu** : « Instantané » n'est plus grisé sans VRR ; l'infobulle dit « sans VRR, la cadence est calée sur
  le rafraîchissement de l'écran ». Option du banc `--no-vrr` : traite l'écran comme fixe même si le VRR est
  armé.

### Tests

Purs (`tests/unit/core/free_test.cpp`) : instants multiples de la période (à la phase près), pts croissants, au
moins un vsync entre deux images, GPU à 25 i/s sur 60 Hz (22 à 26 i/s, jamais entre deux vsyncs), image tenue
quand la paire est passée, suivi de phase avec gigue, et dix minutes simulées avec une horloge qui dérive de
±0,1 % : toutes les images restent à moins d'1 ms de leur vsync.

### Vérification réelle (75 s, `--no-vrr`, 165 Hz traité comme fixe, 720p, plein écran, même scène)

| | Stable (`--no-vrr`, paliers entiers) | Instantané sur vsyncs |
| --- | --- | --- |
| cadence moyenne | 38,0 i/s | **66,8 i/s** |
| sautées / manquées | 4 / 40 | 5 / 0 |
| pertes | 1,47 % | **0,10 %** |
| écart entre images neuves σ / p95 | 11,0 / 44,7 ms | 7,5 / 25,4 ms |
| changements de palier | 16 | aucun |

Les écarts d'affichage sont des multiples de la période de l'écran (1, 2, 3 vsyncs : 827, 1643 et 1053 des
4 906 images ; au-delà de 4 vsyncs, moins de 3 %). Limite de cet essai : l'écran de la machine est en réalité
un VRR, `--no-vrr` ne change que ce que le lecteur en suppose ; la phase mesurée sur un VRR a plus de gigue
qu'avec un vrai écran fixe (les présentations d'un VRR ne sont pas alignées sur une horloge), donc un écran
60 Hz fixe devrait faire au moins aussi bien. À vérifier sur le portable de Tatoub (RTX 4050, GNOME, écran
fixe).

### Limites

- Sur un écran 60 Hz, la période est de 16,7 ms : un GPU qui tient 45 i/s donne des écarts de 1 et 2 vsyncs
  en alternance (cadence moyenne exacte mais rythme inégal) ; c'est le prix du vsync, un VRR n'a pas ce défaut.
- Un compositeur qui ne présente pas à chaque vsync (fenêtre non plein écran, composition en tampon) décale la
  phase mesurée : le mode reste valide mais l'image peut changer de vsync de temps en temps.

## 6. Mode Stable à fréquence fixe : les paliers calés

Sans VRR, le Stable ne prend que les facteurs où chaque image tient un nombre entier k de rafraîchissements :
`écran / (k × cadence source)` (`core::auto_limits`). Quand ce facteur n'est pas un demi-palier, il est donné
en fraction exacte (`core::lock_factor`, comme le Instantané à pas constant) : 11011/3200 à 165 Hz pour 23,976 i/s,
soit 82,5 i/s, une image tous les deux rafraîchissements. Les facteurs sous ×1,5 sont laissés.

| Écran | 23,976 i/s | 29,97 i/s |
|---|---|---|
| 60 Hz | ×1, ×2,5 | ×1, ×2 |
| 120 Hz | ×1, ×1,67, ×2,5, ×5 | ×1, ×2, ×4 |
| 144 Hz | ×1, ×1,5, ×2, ×3, ×6 | ×1, ×1,60, ×2,40, ×4,80 |
| 165 Hz | ×1, ×1,72, ×2,29, ×3,44, ×6,88 | ×1, ×1,84, ×2,75, ×5,5 |

Le niveau choisi borne toujours (×4 : jusqu'à ×4,5). Le plafond `auto_cap` borne la cadence de sortie mais
ne déplace pas le calage, qui se fait sur l'écran : à 165 Hz plafonné à 120 i/s, restent ×1,72, ×2,29 et
×3,44. Il n'y a plus de plafond sur batterie : le plafond est le maximum de l'écran.

La vitesse de lecture cale ensuite la fréquence (`core/pacing.h`, 0,125 % au plus) et la phase
(`core/phase_trim.h`, 0,01 % au plus) : l'échéance de chaque image neuve est ramenée sur un vblank de la
grille mesurée, ce qui rattrape l'erreur de mesure de la fréquence de l'écran.
