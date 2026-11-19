> Relevé daté (historique) : plan « Usages du flux optique matériel NVIDIA » tel qu'au 19 novembre 2026 au soir. Copie du document de suivi partagé ; l'état du code est dans [`carte-de-la-scene.md`](../carte-de-la-scene.md) et [`tuiles-span.md`](../tuiles-span.md).

# Usages du flux optique matériel NVIDIA

19 novembre 2026, pour Tatoub

## Ce que le matériel sait faire

Le bloc de flux optique (NVOFA) des RTX 4070 Ti et 4050 donne un vecteur de mouvement par bloc de 1×1, 2×2 ou 4×4 pixels, sur un circuit à part qui tourne en même temps que TensorRT. Kirari ne l'utilise pas aujourd'hui : un moteur d'interpolation NVOF a vécu du 19 au 20 octobre (commits 5dff1467 et 96af8c3e), retiré sans mesure notée ; seul reste `tools/nvof_interp` (FlowJaNai).

| Grille | 1080p, paires par seconde (4090, réglage lent à rapide) | Temps par paire |
| --- | --- | --- |
| 4×4 | 536 à 1 296 | 0,8 à 1,9 ms |
| 2×2 | 210 à 711 | 1,4 à 4,8 ms |
| 1×1 | 98 à 222 | 4,5 à 10 ms |

Chiffres de NVIDIA sur une 4090 ([note d'application NVOFA](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-application-note)), à mesurer sur la 4070 Ti. Les deux sens dans le même appel doublent à peu près le temps.

Ce qui sert aux pistes ci-dessous :

- **un coût par bloc** (Ampere et Ada) : plus il est haut, moins le vecteur est sûr. C'est une carte de confiance gratuite ;
- **un mouvement global** de l'image (option `enableGlobalFlow`) ;
- **des indices externes** : on peut lui donner un mouvement de départ, qu'il affine ;
- **des régions d'intérêt** (Ampere et Ada) : ne calculer que certaines zones ;
- **les indices temporels** : il réutilise la paire précédente, à couper sur une coupe de scène ;
- **API** : CUDA, Vulkan (`VK_NV_optical_flow`, Ada et Ampere), D3D11 et D3D12, chargée depuis le pilote. Les en-têtes sont sous licence BSD et redistribuables (déjà vendus une fois dans `third_party/nvidia-optical-flow`).

Tout ce qui suit est réservé aux cartes NVIDIA, comme Tatoub l'a choisi : les autres cartes gardent la chaîne actuelle.

### Retour d'expérience du 17 novembre (fil d'interpolation, 4070 Ti)

Le flux NVOF a déjà été mesuré sur l'anime, et il déçoit dès qu'on lui demande des vecteurs précis. Rapports dans `~/ajp-reports/nvof-1bis` et `nvof-etape2`.

| Mouvement (1080p, NVOF calculé en 720p, grille 4) | Écart moyen à RIFE 4.26 |
| --- | --- |
| Moins de 45 px | 2,1 px |
| 45 à 90 px | 11,5 px |
| Plus de 90 px | 44,8 px |

- **Temps réels** sur la 4070 Ti : 4,1 ms par sens en grille 4 (720p), 9,2 ms en grille 1. Bien plus que le chiffre de NVIDIA sur 4090. Les grilles 2 et 1 sont pires sur les grands mouvements. En 1080p, c'est pire qu'en 720p : la portée de recherche de la puce est dépassée, et les aplats la trompent.
- **RIFE guidé par NVOF a échoué** : NVOF avec le seul dernier bloc de RIFE perd 0,26 dB et lisse l'image ; NVOF ajouté à RIFE complet gagne 0,49 dB pour le même coût, gain sans doute dû à l'entraînement sur l'anime. Ce qui a marché : RIFE 4.26 réentraîné sur l'anime, sans NVOF.
- **Erreurs à ne pas refaire** : mesurer le coût TensorRT avant d'entraîner ; warp et coordonnées en FP32 (en FP16, jusqu'à 28 niveaux sur 255 d'écart) ; ne pas juger le mouvement à la différence de luminance (fondus, flashs) ; garder une part de génériques (texte défilant en surimpression) ; au plus deux bancs NVOF à la fois sur 12 Go.

**Ce que ça change.** Les usages qui demandent des vecteurs précis tombent ou deviennent conditionnels : RIFE guidé, TemporalFix guidé, SPAN récurrent, trajectoires. Les usages qui demandent seulement « ça bouge ou pas », « ça correspond ou pas » et « mouvement d'ensemble » restent valables : coupes, dessins tenus, tuiles qui bougent, tri des paires, éclats HDR. Le mouvement de caméra reste possible, parce qu'il s'ajuste sur des milliers de blocs à la fois, mais seulement s'il tient sur les grands panoramiques. Deux moyens jamais essayés : calculer d'abord en basse résolution (360p, déplacements divisés par deux), puis passer ce mouvement à la puce comme indice externe en 720p, ce qui rallonge sa portée de recherche. C'est la porte 2.1 qui tranche.

## Plan revu : le flux tel quel, sans nouveau modèle

Décision de Tatoub (19 novembre) : on retire tout ce qui demande d'entraîner un modèle. Le flux NVIDIA sert tel quel à décider, trier et repérer, pour mieux utiliser SPAN, RIFE, TemporalFix et le grade qu'on a déjà. Cette section remplace les classements plus bas là où ils diffèrent.

**Retiré** : RIFE guidé par NVOF et FlowJaNai, TemporalFix guidé par le flux, SPAN récurrent, parcimonie 2:4 (tous demandent un entraînement) ; trajectoires d'animateur et ralenti d'export (ils demandent des vecteurs précis que NVOF n'a pas sur l'anime) ; séquences de banque (gain trop étroit).

**Ce qui reste, classé par gain**, chaque fois avec ce que ça améliore dans l'existant :

| Rang | Usage du flux | Ce qu'il améliore | Précision demandée |
| --- | --- | --- | --- |
| 1 | Tri des paires et masque d'artefacts | RIFE : copie exacte sur un dessin tenu, fondu au lieu de fantômes sur une paire impossible (coût haut partout), fondu local là où le flux doute | Faible : « ça correspond ou pas » |
| 2 | Dessins tenus et tuiles qui bougent | SPAN : reprise de son résultat là où rien ne change, sans se laisser tromper par le grain ; aussi « Lisser les dessins tenus », qui reconnaît un dessin tenu même quand la caméra bouge | Faible |
| 3 | Interpolation par calques | RIFE ne travaille que sur les personnages ; le décor suit le mouvement de caméra. Aucun modèle nouveau : RIFE tel quel plus un warp | Mouvement de caméra juste : dépend de la porte 2.1 |
| 4 | Coupes de scène | Le découpeur de plans (PR #847) : grade par plan et interpolation justes aux coupes | Faible |
| 5 | Éclats HDR | Le grade : la marge des éclats sur les seuls effets | Faible |
| 6 | Chaîne réglée par plan | L'Auto : DistilDRBA ou RIFE, TemporalFix ou pas, selon le mouvement de chaque plan lu en avance | Faible |
| 7 | Plaque de décor par plan | SPAN (le modèle le plus lourd qu'on a déjà) sur le décor une seule fois par plan | Mouvement de caméra juste : dépend de la porte 2.1 |
| 8 | Petits usages | Tremblement de pellicule, cadences cassées, désentrelaçage, mesure du scintillement | Faible à moyenne |

Les rangs 1, 2, 4, 5 et 6 ne dépendent que de la carte de la scène et marchent même avec un flux grossier. Les rangs 3 et 7 attendent le verdict du banc NVOF sur les panoramiques rapides. Les lots A, B, C et E ne changent pas : aucun ne crée de modèle.

## Pistes classées par gain

Le plus gros gain attendu est l'interpolation par calques : l'anime est un décor peint que la caméra déplace, avec des personnages posés dessus, et le flux sait séparer les deux. Le classement ne tient compte que de ce qu'on peut gagner, pas de la taille du chantier. Les coûts sont des estimations à partir des chiffres de NVIDIA, rien n'est mesuré.

| Rang | Piste | Ce qu'on gagne | Comment | Coût NVOF par paire | Risque principal |
| --- | --- | --- | --- | --- | --- |
| 1 | Interpolation par calques | La fin des décors qui se tordent autour des personnages et des halos de RIFE, le défaut le plus visible sur l'anime | Le mouvement de caméra (global ou affine, ajusté sur le flux) déplace le décor exactement ; RIFE ne travaille que sur ce qui ne suit pas la caméra, puis on compose | 1 à 2 ms (4×4 suffit pour la caméra) | Les bords des personnages et les décors en plusieurs plans (parallaxe) |
| 2 | Nettoyage temporel avant SPAN | Blocs, escaliers et bruit de compression enlevés avant que SPAN ne les grossisse ; sortie plus stable. Touche presque toutes les sources web | Aligner 2 à 4 voisines au pixel (1×1 en 720p), les moyenner là où le coût est bas. Peut nourrir TemporalFix ou le remplacer sur la 4050 | 5 à 15 ms pour 2 à 4 voisines | Effacer un détail fin qui bouge ; recoupe TemporalFix |
| 3 | Plaque de décor par plan | Un décor plus net que n'importe quel agrandissement image par image, et parfaitement stable | Pendant la pré-analyse, assembler toutes les images d'un panoramique en une seule plaque (super-résolution multi-images), l'agrandir une fois avec un modèle lourd, puis découper chaque image dedans | Pré-analyse seulement | Zooms, parallaxe, décors animés (eau, foule) |
| 4 | Flux NVOF comme départ de RIFE | RIFE ne rate plus les grands mouvements du sakuga, et saute ses premiers blocs | Donner le flux matériel à RIFE à la place de son estimation grossière (FlowJaNai, P6.15) | 1 à 4 ms | Déjà étudié par le fil d'interpolation : résultat du banc `nvof_warp` attendu |
| 5 | Tri des paires et rythme par calque | Copie exacte sur les poses tenues, décalage exact sur les panoramiques et génériques, fondu au lieu de fantômes sur les paires impossibles ; rythme des dessins gardé même quand la caméra bouge | Flux nul et coût bas : pose tenue. Flux uniforme : panoramique. Coût haut partout : paire impossible. Par calque, la cadence du personnage (en deux ou en trois) se lit à part de celle de la caméra | 1 ms | Seuils à régler au banc |
| 6 | Éclats pour le grade HDR | La marge des éclats ne s'applique qu'aux vrais effets (explosions, magie, éclairs), pas à un ciel ou une feuille blanche | Un effet est clair, bouge vite et est mal suivi par le flux (coût haut) ; un fond blanc est suivi | 1 ms | Les effets lents ou fixes |
| 7 | Coupes de scène | Moins de coupes ratées (22 sur 503 aujourd'hui, PR #847) et moins de fausses (68), donc un grade et une interpolation justes aux coupes | Sur une vraie coupe, presque aucun bloc ne retrouve son voisin ; sur un panoramique ou un flash, la plupart le retrouvent | Quasi nul en 360p | Fondus enchaînés |
| 8 | Réparation des cadences cassées | Lecture fluide des rips mal faits : doublons suivis d'un saut, restes de télécinéma | Un doublon suivi d'un mouvement double se voit dans le flux ; l'image manquante est recréée | 1 ms | Le confondre avec une vraie pose tenue |
| 9 | Mesure du scintillement | Chaque choix sur SPAN, TemporalFix ou le grade devient chiffrable au banc, sans test à l'aveugle | Aligner une image sur la suivante et mesurer ce qui reste là où le flux est sûr | Outil hors lecture | Aucun : c'est un outil |
| 10 | Désentrelaçage guidé par le mouvement | Vieux DVD propres | Compléter les trames avec le mouvement au lieu de les interpoler à l'aveugle | 1 ms | Utile pour peu de sources |

Les pistes 1, 3, 5 et 6 reposent sur la même brique : séparer le décor (ce qui suit la caméra) des personnages et des effets (ce qui ne la suit pas). C'est elle qu'il faudrait construire en premier.

## Avec le chantier mpv AnimeJaNai

Deux des quatre lots du chantier « Dépasser mpv AnimeJaNai » (19 novembre) rencontrent le flux matériel. Le lot A se marie avec le tri des paires (piste 5). Le lot B se marie avec le nettoyage temporel (piste 2). Les lots C et D n'ont pas de lien avec le flux.

- **Lot A, images tenues.** libaji décide la reprise par des différences de pixels, avec une vue sur N+1 et N+2 pour voir un mouvement commencer. Le flux fait mieux sur trois points. Il reconnaît un dessin tenu pendant que la caméra bouge : on reprend alors le SPAN de l'ancre, déplacé du mouvement de caméra. Il voit un mouvement lent commencer (nuages, fumée) bien avant qu'un seuil de pixels ne le sente. Et sa carte de coût sépare le bruit de compression d'un vrai changement. C'est ce qui rend sûre la limite de 24 reprises de suite.
- **Lot B, TemporalFix.** Le gros du travail de TemporalFix est d'aligner six images sur le centre : trois MCBlocks du grossier au fin, à 1/8, 1/4 et 1/2. Le flux matériel donne cet alignement presque gratuitement. Si on lui donne le flux NVOF comme départ, le MCBlock à 1/8 peut disparaître et les deux autres n'ont plus qu'à affiner. Il faudra réentraîner ou affiner le modèle, donc c'est une étape après le lot B, pas à la place. Il faut seulement découper le graphe au lot B pour que l'alignement reste un morceau qu'on puisse remplacer.

### Ordre de priorité de tous les chantiers

Classés par gain, sans tenir compte de leur taille. Les gains de vitesse comptent comme de la place libérée pour des réglages plus lourds.

| Rang | Chantier | Ce qu'on gagne |
| --- | --- | --- |
| 1 | Interpolation par calques (flux, piste 1) | Le défaut le plus visible de l'interpolation sur l'anime disparaît |
| 2 | Cadence à 165 Hz (présentation) | Kirari plafonne vers 107 i/s alors que le GPU n'est occupé qu'à 36 % : lever ce blocage donne tout de suite une lecture plus fluide sur ton écran |
| 3 | TemporalFix rapide (lot B), puis guidé par le flux (piste 2) | La stabilisation passe de 27,9 ms à 12 ms ou moins en pleine taille, assez pour rester allumée ; avec le flux, encore moins |
| 4 | SPAN plus rapide (lot C) : d'abord notre FP16 sous leurs 11,9 ms (Balanced) et 6,2 ms (Performance) à 55 dB ou plus, par convolutions fondues, un bloc SPAN entier par noyau (tuile gardée en mémoire partagée), entrée NV12 et sortie fondues dans la première et la dernière conv, sans reformatages ; puis les mêmes optimisations dans le FP8 CUTLASS (choix de Tatoub) | Mesuré le 19 novembre : 17,7 ms contre 11,9 chez AnimeJaNai en Balanced, 13,0 contre 6,2 en Performance. 6 à 7 ms rendus par image pour tout le reste, et les RTX 30 de retour |
| 5 | Flux NVOF comme départ de RIFE (piste 4, fil d'interpolation) | Le sakuga rapide sans ratés, et une part des 6,5 ms fixes par paire de RIFE (encodage et flot) remplacée par le matériel |
| 6 | Plaque de décor par plan (flux, piste 3) | Des décors plus nets et parfaitement stables |
| 7 | Images tenues partout (lot A), puis seules les tuiles qui bougent passent dans SPAN, repérées par le flux (piste 5) | 16 à 57 % de passes SPAN évitées : moins de chauffe, de bruit et de consommation ; avec les tuiles, une grande part de chaque plan de dialogue (décor fixe) n'est plus recalculée, alors qu'AnimeJaNai ne saute que 6,9 % des images sur Re:Zero |
| 8 | Colle CUDA de RIFE (lot E) | En ×2 : 7,9 ms par image créée contre 5,0 chez AnimeJaNai ; environ 6,1 ms après le lot. Peu de gain dès ×3, où Kirari mène déjà |
| 9 | Éclats pour le grade HDR (piste 6) | Les effets brillent, les fonds blancs non |
| 10 | Vulkan pour AMD et Intel (lot D) | Rien pour toi ; beaucoup pour les autres cartes |
| 11 | Le reste du flux : coupes, cadences cassées, mesure du scintillement, désentrelaçage, tremblement de pellicule, SPAN récurrent, trajectoires, séquences de banque, parcimonie 2:4 de SPAN (réentraînement), ralenti d'export (en dernier) | Gains plus étroits ou plus incertains |

Ce classement détaillé est regroupé en huit chantiers dans la section « Les chantiers fusionnés », plus bas, qui fait référence pour l'ordre de travail.

Sur les tuiles qui bougent (pistes au-delà d'AnimeJaNai, 19 novembre), le flux apporte deux choses que la différence de pixels n'a pas. Sa carte de coût ne prend pas le grain ni le bruit de compression pour du mouvement, donc moins de tuiles recalculées pour rien. Et quand il voit un panoramique d'un nombre entier de pixels (multiple du pas du pixel-unshuffle), la sortie précédente de SPAN se décale telle quelle : seule la bande qui entre dans l'image est à calculer. Les panoramiques lents de l'anime ne tombent pas toujours sur un nombre entier, donc c'est à mesurer.

## À tête posée : aller plus loin

En relisant tout, la vraie idée n'est pas une piste de plus. C'est de cesser de traiter l'anime image par image, comme le font SVP et AnimeJaNai, et de le traiter comme ce qu'il est : un décor peint, une caméra, des dessins posés dessus, et un rythme par plan. Le flux matériel rend cette lecture presque gratuite. Cinq enrichissements en découlent.

### 1. Une seule carte de la scène, beaucoup de clients

Aujourd'hui, chaque piste imagine son propre passage de flux. Mieux : un seul étage d'analyse, à l'ingestion et en avance sur la chaîne (l'étage lit déjà jusqu'à 5 images d'avance), qui produit pour chaque paire :

- le mouvement de caméra (translation, zoom, rotation) ajusté sur les blocs sûrs ;
- la carte de confiance (coût) ;
- un masque par tuile : décor (suit la caméra), dessin (ne la suit pas), effet (mal suivi et clair) ;
- les drapeaux : dessin tenu, décor tenu, coupe, fondu.

Coupes, tuiles de SPAN, images tenues, interpolation par calques, TemporalFix et éclats HDR lisent tous cette carte. Un seul coût, environ 1 à 2 ms sur le bloc séparé, et un seul endroit à régler et à tester. Deux réglages du matériel la rendent meilleure : donner le mouvement de caméra de la paire précédente comme indice externe (les grands panoramiques rapides, là où la recherche par blocs échoue) et ne demander le 1×1 que sur les régions des dessins (régions d'intérêt), en 4×4 ailleurs.

### 2. Une chaîne planifiée par plan

La pré-analyse (`shot_probe`) parcourt déjà la vidéo en avance, en 256 × 144. Y ajouter le flux en basse résolution donne, pour chaque plan : la cadence du dessin (en un, en deux, en trois), celle de la caméra, l'intensité de l'action, les dessins tenus. L'Auto choisit alors par plan au lieu de par vidéo : DistilDRBA sur un plan d'animation limitée, RIFE 4.25 sur un plan de sakuga, TemporalFix seulement sur les plans qui scintillent, et le budget de calcul connu avant d'arriver au plan lourd. Le résultat se garde avec le cache des plans (`shot_cache`) pour la lecture suivante.

### 3. Le rendu par calques, au bout du chemin

Les pistes 1 et 3 et les tuiles qui bougent sont trois morceaux d'une même chaîne. Pendant un plan :

- le décor est une plaque agrandie une seule fois, avec le modèle le plus lourd ; chaque image, y compris interpolée, y est découpée à la position exacte de la caméra, donc sans aucun défaut d'interpolation sur le décor ;
- les dessins passent dans SPAN une fois par dessin (pas par image) et dans RIFE seuls ;
- les effets suivent leur propre réglage (moins de SPAN, plus d'éclat HDR).

Sur un plan de dialogue ou un panoramique, le calcul de SPAN par image tombe à la surface des personnages. La qualité monte et le coût baisse en même temps. Le garde-fou : là où la carte de confiance doute (zoom, parallaxe, décor animé), on revient à la chaîne image par image d'aujourd'hui, par tuile ou par plan.

### 4. Le rythme par calque, une façon de jouer que personne n'a

Un personnage animé en deux sur une caméra en un est le cas le plus courant de l'anime télé. Avec les calques, on peut rendre la caméra parfaitement fluide tout en gardant, ou en lissant au choix, le rythme du dessin. C'est « Garder les poses » et « Lisser les dessins tenus » appliqués calque par calque, au lieu de toute l'image.

### 5. Une vérité terrain gratuite pour mesurer

Deux outils découlent de la carte et servent tous les chantiers :

- la mesure du scintillement (piste 9) rend objectifs les choix des lots A, B et C, au lieu de tout passer au test à l'aveugle ;
- sur un panoramique, la plaque donne le vrai décor à n'importe quel instant intermédiaire. On obtient enfin une vérité terrain pour l'interpolation sur de vrais épisodes, sans source au ralenti. Ces mêmes plans fournissent des données d'entraînement pour FlowJaNai et pour un TemporalFix guidé par le flux (`tools/nvof_interp/sequences.py` sait déjà les extraire).

### Ce qui peut faire échouer tout ça

- Les aplats de couleur n'ont pas de texture : le flux y est incertain. La caméra se calcule sur les blocs texturés puis s'étend aux aplats ; c'est le premier point à vérifier au banc.
- Les traits fins en 4×4 : d'où le 1×1 limité aux dessins.
- Sous Windows, NVIDIA signale un flux CUDA plus lent quand la planification matérielle du GPU (HAGS) est coupée : à vérifier sur ton PC avant toute mesure.

L'ordre d'implémentation détaillé est dans la section « Guide d'implémentation », juste après « Les chantiers fusionnés ».

## Les chantiers fusionnés

Les quelque vingt-cinq pistes, lots et idées tiennent en huit chantiers : chacun réunit ce qui partage la même brique et se construit ensemble. Classés par gain ; la colonne « Phase » renvoie au guide d'implémentation, juste en dessous.

| Rang | Chantier | Ce qu'il réunit | Ce qu'on gagne | Phase |
| --- | --- | --- | --- | --- |
| 1 | Rendu par calques | Interpolation par calques (piste 1), plaque de décor et super-résolution multi-images (piste 3), rythme par calque, application des éclats HDR au seul calque des effets (piste 6) | Plus de décors tordus ni de halos, des décors plus nets et stables, SPAN réduit à la surface des personnages, la caméra fluide et le dessin à son rythme | 4 |
| 2 | Cadence à 165 Hz | Seul | De 107 à 165 i/s sur ton écran, le GPU étant à 36 % | 0 |
| 3 | Alignement temporel | TemporalFix rapide (lot B), nettoyage temporel avant SPAN (piste 2), désentrelaçage guidé (piste 10). TemporalFix guidé et SPAN récurrent deviennent de la recherche : le flux NVOF n'est pas assez précis (retour du 17 novembre) | Une image propre et stable avant et après SPAN, assez rapide pour rester allumée | 2 puis 4 |
| 4 | SPAN plus rapide | Lot C (FP16 sous les 11,9 et 6,2 ms d'AnimeJaNai, blocs fondus, entrée et sortie fondues, essai C0, puis FP8), parcimonie 2:4 en recherche | 6 à 7 ms par image rendus à tout le reste, les RTX 30 de retour | 1 puis 2 |
| 5 | RIFE guidé | Colle CUDA de RIFE (lot E) et RIFE réentraîné sur l'anime (sans NVOF, le seul qui a marché le 17 novembre). NVOF comme départ de RIFE a échoué ; reste seulement à essayer NVOF comme simple guide en parallèle de l'encodeur. Ralenti d'export en dernier | Environ 1,8 ms de moins par paire en ×2 et +0,2 dB par le réentraînement ; le reste est incertain | 1 (lot E) puis 4 |
| 6 | Ne jamais recalculer deux fois | Images tenues (lot A), tuiles qui bougent, décalage de la sortie sur un panoramique entier, séquences de banque et cycles d'animation | Une grande part des passes SPAN évitée sur le dialogue et l'animation limitée ; moins de chauffe et de consommation | 1 (lot A) puis 3 |
| 7 | Carte de la scène | L'étage d'analyse unique : coupes (piste 7), tri des paires et masque d'artefacts (piste 5, idée SVP), cadences cassées (piste 8), tremblement de pellicule, chaîne planifiée par plan, mesure du scintillement et vérité terrain (piste 9) | Peu par lui-même : c'est la fondation des chantiers 1, 3, 5 et 6, à construire tôt malgré son rang | 0 (mesure) puis 2 |
| 8 | Vulkan pour AMD et Intel | Lot D, plus les portages Vulkan de ce qui sort des chantiers 3, 4 et 5 | Rien pour toi ; beaucoup pour les autres cartes | 5 |

Trois fusions changent la façon de travailler. Le lot A et les tuiles qui bougent ne font qu'un seul mécanisme de reprise, décidé par la carte de la scène. Le lot B et le nettoyage temporel sont le même réseau, nourri ou non par le flux. Et le flux comme départ de RIFE partage avec la colle CUDA du lot E les mêmes noyaux de warp, donc ils se réécrivent ensemble.

## Guide d'implémentation

Cet ordre n'est pas celui du gain : il suit les dépendances. Une idée peu prioritaire passe devant quand d'autres s'appuient sur elle. Chaque étape a une porte : un critère mesuré qui dit si on continue, et ce qu'on fait sinon. Ne jamais sauter une porte.

### Comment on teste (consigne de Tatoub, 19 novembre)

Le strict minimum, regroupé au maximum. Coder le plus possible sans mesure, puis tout vérifier en peu de passages.

- **Les tests de lecture, c'est Tatoub**, pendant ses séances d'anime. Chaque nouveauté arrive avec un interrupteur (option de ligne de commande ou clé de `play.toml`) et une ligne de journal de fin qui dit ce qu'elle a fait, pour qu'il puisse comparer sans outil.
- **Un seul test Catch2 par module touché**, qui couvre plusieurs choses à la fois. Exemple : un seul test « carte absente » qui fait passer tous les clients de la carte sur la même suite d'images synthétiques et vérifie que la sortie est identique à aujourd'hui. Pas de test par fonction.
- **Une seule séance de banc par passage au PC**, par un script unique qui enchaîne tous les bancs en attente (vitesse, 48 ou 55 dB, scintillement) et écrit un seul rapport. Les portes ci-dessous se lisent dans ce rapport ; aucune ne demande un test à l'aveugle séparé : ceux-là se font pendant les séances d'anime.
- Les règles de la CI restent obligatoires : elles sont automatiques et ne coûtent rien.

**Tout s'écrit à l'aveugle, et on teste par groupe** (consigne de Tatoub, 19 novembre, tout sous CachyOS). Aucun fil ne lance de banc ni de test sur le PC au milieu de son travail. On ne teste qu'un groupe de fonctions fini, toutes ensemble, en une seule séance ; s'il y a un problème, on revient au code puis on relance la séance.

| Groupe | Contenu | Quand on teste |
| --- | --- | --- |
| 1 | 165 Hz, outil de mesure, lot A, lot E, essai C0, cache de TemporalFix | Quand les six fils sont prêts, en une séance |
| 2 | Session NVOF, carte de la scène, lot B complet, lot C | Quand les trois voies de la phase 2 sont prêtes |
| 3 | Tous les clients de la phase 3 | Quand ils sont tous écrits |
| 4 | Phase 4 | Après les entraînements |

**Règle de Tatoub du 19 novembre, 08:23, qui remplace les séances par groupe.** Plus de banc intermédiaire, et jamais d'épisode entier. Chaque fil relit et revoit sa propre PR, vérifie dans le cloud qu'elle construit (nvcc des paquets pip NVIDIA pour le CUDA, avec `--warn-on-spills`) et que les règles de la CI passent, puis fusionne et passe à la suite. Chaque nouveauté garde un interrupteur pour la couper. Les portes ci-dessous deviennent la liste d'une séance finale unique, sur un extrait de 60 secondes, quand tout est fusionné (`/mnt/project-files/seance-pc/groupe-1-flux-optique.md`). Les calques s'écrivent sans attendre la porte 2.1, éteints par défaut : la séance finale décide s'ils s'allument.

### Les trois dépendances qui dictent l'ordre

1. **La mesure avant tout.** Les lots A, B et C et toutes les pistes du flux se jugent sur la stabilité dans le temps et sur la qualité. Sans l'outil de mesure du scintillement, chaque décision retombe sur le test à l'aveugle, lent et discutable. Il passe donc en tête, malgré son rang de gain.
2. **Une seule bibliothèque de warp.** La colle de RIFE (lot E), les warps de TemporalFix (lot B, étape 2), la composition par calques et le décalage des tuiles déplacent tous des images par un flux. On l'écrit une fois, bien, dans le lot E (le plus petit et le mieux connu : RIFE fusionné existe), puis les autres la réutilisent.
3. **La carte de la scène avant ses clients.** Coupes, tuiles, tri des paires, éclats HDR, calques, RIFE et TemporalFix guidés lisent tous la même carte. Elle passe avant eux, et chacun doit se comporter exactement comme aujourd'hui quand elle est absente.

### Phase 0 : les fondations de mesure (quelques jours)

**0.1 Mesure du scintillement** (`tools/bench/qualite`, en Python, hors lecteur).

- Pour chaque paire, déplacer l'image N vers N+1 par un flux, puis mesurer l'écart restant là où le flux est sûr. Le flux vient de `nvof_dump` (déjà écrit) ; sans carte NVIDIA, de RIFE.
- Rapporter par plan, jamais en moyenne sur la vidéo : un plan raté se noie sinon dans les plans calmes.
- Exclure les coupes et les fondus (sinon ce sont eux qui dominent le chiffre).
- Comparer toujours deux sorties du lecteur sur les mêmes images, jamais une sortie à la source : la source scintille aussi (grain, compression).
- **Porte** : sur un extrait connu pour scintiller (SPAN sans TemporalFix) contre le même avec TemporalFix, l'outil doit classer TemporalFix devant sur au moins 9 plans sur 10. Sinon l'outil est faux, pas TemporalFix.

**0.2 Conditions de mesure.**

- Sous Windows, vérifier que la planification matérielle du GPU (HAGS) est activée : NVIDIA signale un flux CUDA plus lent sinon. Noter la version du pilote dans chaque rapport.
- Bancs seulement sur `linux-release` ou `windows-release`, en ABAB sur 3 tours. `linux-prefix` est en Debug : ses temps ne valent rien.
- Sur la 4050 portable, vérifier que le contexte CUDA est sur la carte NVIDIA et non sur la puce intégrée (Optimus).

**0.3 Cadence à 165 Hz.** Indépendante de tout le reste, mais à faire tôt : tant que la présentation plafonne à 107 i/s, les gains de vitesse de RIFE (lots E et RIFE guidé) ne se voient pas au-delà, et on mesurerait le plafond au lieu du calcul. **Porte** : 165 i/s tenus sur un extrait en ×7 avec le GPU sous 60 %.

### Phase 1 : les gains immédiats, déjà tournés vers la suite

**1.1 Lot A, images tenues.** Comme prévu dans CHANTIER.md (mesure à facteur 1, vue sur N+1, image abîmée exclue, statistiques), avec deux exigences en plus :

- **Une interface de décision à part.** Écrire la décision de reprise dans un fichier à elle (par exemple `upscale_reuse.{h,cpp}`) qui rend « reprendre l'image entière » ou « recalculer ». À la phase 3, elle rendra une liste de tuiles et lira la carte de la scène, sans toucher au reste.
- **Piège de taille :** `src/app/play/upscale_stage.h` fait déjà 798 lignes, et la CI échoue au-delà de 800. Rien n'y entre : tout va dans des fichiers nouveaux.
- Ne relever `kMaxSpanReuse` (2) qu'avec la vue sur l'image suivante en place, jamais avant.
- Avec TemporalFix allumé, la décision se prend sur sa sortie (c'est ce que SPAN lit). Avec DistilDRBA, tenir compte de son image de retard.
- Si le drapeau d'image abîmée passe par `src/core`, l'empreinte TensorRT change : nouveau paquet Linux.
- **Porte** : taux de reprise dans 16 à 57 % sur trois épisodes, images reprises à 48 dB ou plus d'un calcul neuf, aucun arrêt visible sur nuages, fumée, fondu de carton-titre et fond flou.

**1.2 Essai C0** (lot C) : entrées et sorties en canaux derniers, essais trtexec. Une journée, sans noyau à écrire.

**1.3 Lot E, la bibliothèque de warp.** Réécrire la colle CUDA de RIFE (fused\_glue, fused\_finish, warp\_blend : 3,3 ms par paire en ×2) comme un ensemble de noyaux réutilisables : un seul relèvement par bloc à son échelle, le warp évalué aux points où on échantillonne, spécialisé par forme.

- L'écrire d'emblée pour qu'il prenne un flux de n'importe quelle source (RIFE aujourd'hui ; NVOF, caméra ou calques demain), à n'importe quelle échelle, avec un masque de confiance optionnel.
- Règle des 0,5 dB de `docs/moteurs.md` contre la colle actuelle.
- Aucune ligne reprise de libaji (sans licence) : seulement la méthode, citée.
- **Porte** : la colle sous 1,6 ms par paire en ×2 sur la 4070 Ti, sortie à 50 dB ou plus de l'actuelle.

### Phase 2 : la brique du flux, et les gros lots en parallèle

Trois voies indépendantes, à mener en même temps : le flux (2.1 à 2.3), le lot B (2.4) et le lot C (2.5).

**2.1 Banc NVOF hors lecteur.** Étendre `nvof_dump` et `tools/bench/qualite/nvof_warp.py` :

- temps réels des grilles 4, 2 et 1 en 540p, 720p et 1080p, sur la 4070 Ti et la 4050 ;
- distribution de la carte de coût sur les aplats, les traits, le grain ;
- ajustement du mouvement de caméra (moindres carrés pondérés par la confiance, avec rejet des aberrants) et masque décor contre dessin ;
- sur des extraits choisis : dialogue, panoramique, sakuga, fondu, flash, zoom, parallaxe, pluie.
- **Porte** : sur les panoramiques, l'écart du décor au modèle de caméra sous 0,5 pixel pour 90 % des paires. Sinon, on garde les usages par paire (coupes, tri, tuiles) et on abandonne les calques (chantier 1).

**2.2 La session NVOF dans le backend CUDA.** Reprendre `nvof_session.{h,cpp}` et les en-têtes BSD de `third_party/nvidia-optical-flow` depuis le commit 5dff1467, en corrigeant ce qui manquait :

- **son propre flux CUDA** (`NvOFSetIOCudaStreams`) et des événements pour se synchroniser. L'ancienne session tournait sur le flux du backend, ce qui la mettait en série avec TensorRT et perdait tout l'intérêt du bloc séparé ;
- **entrée sans marge de ligne** : les surfaces NVDEC ont un pas, NVOF n'en veut pas. Un petit noyau réduit la luma NV12 (ou P010, en gardant les 8 bits hauts) dans le tampon de NVOF, à la taille choisie ;
- **les indices temporels coupés** à chaque coupe, saut, changement d'époque et début d'export ; sinon le résultat dépend de l'historique et l'export ne rejoue plus la lecture ;
- **réinitialiser** la session à chaque changement de taille ;
- **un seul fil** l'utilise : l'API n'est pas sûre entre fils ;
- **lire les capacités** (`NvOFGetCaps`), ne rien supposer. Turing n'a ni carte de coût, ni grille 1×1, ni régions d'intérêt : sans coût, pas de calques ni de masque ;
- bibliothèque absente (AMD, Intel, vieux pilote) : une ligne de journal, puis le lecteur exactement comme aujourd'hui ;
- ajout à `KernelOps` : `kBackendAbi` passe de 46 à 47, avec sa ligne d'historique dans `src/gpu/backend.h`. Nouvelle empreinte TensorRT, nouveau paquet Linux ;
- aucun fichier de NVIDIA dans les archives : les en-têtes sont compilés, la bibliothèque vient du pilote.
- **Porte** : pendant une lecture SPAN + RIFE, le temps par image ne monte pas de plus de 0,3 ms avec NVOF allumé (preuve que le bloc tourne vraiment à côté).

**2.3 L'étage « carte de la scène ».** Un résultat par paire, `SceneMotion`, ajouté à `PairFacts` (`src/app/play/rife_plan.h`) sous la forme `std::optional<SceneMotion>` :

- le mouvement de caméra (6 paramètres affines) et son écart résiduel ;
- une grille de tuiles (par exemple 32 × 32 pixels source) avec, pour chacune, une classe (décor, dessin, effet, incertain) et une confiance ;
- les drapeaux : dessin tenu, décor tenu, coupe probable, fondu.

Comment :

- réduire le flux et le coût par tuile sur le GPU, puis relire seulement la grille (environ 2 000 tuiles en 1080p) par une copie asynchrone, consommée une paire plus tard. Jamais d'attente de l'hôte sur le GPU pour la lire : l'avance de l'étage (`frames_ahead`) donne le temps ;
- ajuster la caméra sur le processeur, sur les tuiles sûres et texturées seulement, puis l'étendre aux aplats ;
- lisser les classes dans le temps (une tuile ne change pas de classe sur une seule paire), sinon tous les clients clignotent ;
- référence processeur dans `src/kernels/ref` et tests sur des flux synthétiques (caméra pure, dessin seul, coupe, fondu), comme le faisait le commit 5dff1467.

**Règle absolue** : `SceneMotion` absent (AMD, Intel, Turing sans coût, NVOF refusé) donne exactement le comportement d'aujourd'hui, image pour image. Un test le vérifie pour chaque client.

**Porte** : sur les extraits du banc 2.1, la classe décor contre dessin juste sur 95 % des tuiles sûres, et stable sur les plans fixes.

**2.4 Lot B, TemporalFix (voie parallèle).** Étapes de CHANTIER.md (cache de l'encodeur, warps à nous, appuis en parallèle), avec ces exigences :

- l'étape 2 utilise la bibliothèque de warp du lot E, pas des noyaux à part ;
- découper le graphe pour que l'alignement (les trois MCBlocks) soit un morceau séparé, qu'on pourra plus tard nourrir par le flux NVOF ;
- vérifier d'abord sur le graphe que l'encodeur ne dépend que de son image : sinon le cache est faux ;
- coupes et bords comme aujourd'hui (reflet `gen_shifts`).
- **Porte** : CHANTIER.md (12 ms ou moins en pleine taille 1080p, 52 dB ou plus) et l'outil 0.1 qui ne voit pas plus de scintillement qu'avec le moteur entier.

**2.5 Lot C, SPAN FP16 (voie parallèle).** Porte du prototype de CHANTIER.md, puis la cible fixée par Tatoub : notre FP16 sous 11,9 ms (Balanced) et 6,2 ms (Performance), à 55 dB ou plus de TensorRT, avant de porter les mêmes optimisations dans le FP8.

- Prévoir dès le départ l'exécution par lot de tuiles avec halo : la phase 3 en a besoin pour ne recalculer que les tuiles qui bougent. C'est bien plus simple dans nos propres noyaux que dans un moteur TensorRT à forme fixe.

### Phase 3 : les premiers clients de la carte

Dans cet ordre, du plus facile à vérifier au plus risqué : chaque client valide la carte pour le suivant.

1. **Coupes.** Le banc des 26 extraits donne un verdict en une heure. La part de tuiles sans correspondance devient un signal de plus pour `shot_cutter`, à côté des mesures de luminance. Un fondu fait monter le coût partout mais garde une luminance qui glisse régulièrement : ce n'est pas une coupe. Les coupes de la pré-analyse (`cut_book`, `known_cut`) gardent la priorité. **Porte** : moins de coupes manquées (22) et moins de fausses (68) que la PR #847, les deux à la fois.
2. **Tuiles qui bougent** (la suite du lot A). Seules les tuiles marquées comme changeant passent dans SPAN, avec un halo égal au champ réceptif calculé sur le graphe (pas estimé). Sur un panoramique d'un nombre entier de pas du pixel-unshuffle, sans zoom ni rotation et avec un résidu minuscule, la sortie précédente se décale ; sinon on recalcule. Une limite de reprises de suite et un rafraîchissement complet périodique empêchent toute dérive. **Porte** : 48 dB ou plus d'un calcul complet, et l'outil 0.1 sans hausse.
3. **Tri des paires et masque d'artefacts** (`rife_plan.h`) : panoramique pur, décalage exact ; tenue, copie ; paire impossible, fondu ; zone douteuse, fondu local. **Porte** : test à l'aveugle gagné sur les passages de `rife-fantomes`.
4. **Chaîne planifiée par plan.** Une seconde session NVOF en basse résolution dans la pré-analyse (`shot_probe`, qui a son propre décodeur). Ne jamais changer de moteur au milieu d'un plan, et précharger les deux modèles : sur la 4050 (6 Go), vérifier que la mémoire tient. Le résultat se garde dans `shot_cache`.
5. **Éclats HDR.** Le grade reçoit un masque des effets (un hook libplacebo de plus). Piège majeur : un masque qui bouge d'une image à l'autre fait pulser la luminosité. Il faut le dilater, l'adoucir et le lisser dans le temps. **Porte** : l'outil 0.1 ne voit aucun scintillement de plus, et le test à l'aveugle est gagné sur des scènes d'explosion et de magie.

### Phase 4 : les calques et les réseaux guidés

1. **Interpolation par calques** (seulement si la porte 2.1 est passée). Le décor est déplacé par le modèle de caméra avec la bibliothèque de warp, RIFE ne voit que les dessins, et la composition se fait avec un bord adouci. Revenir à l'interpolation actuelle par tuile dès que la carte doute.
2. **RIFE guidé par le flux et FlowJaNai**, retiré le 19 novembre (aucun nouveau modèle).
3. **TemporalFix guidé par le flux** : retiré le 19 novembre (aucun nouveau modèle). Le lot B reste, sans flux.
4. **Plaques de décor et rythme par calque.** Plafonner la taille d'une plaque : un long panoramique agrandi peut dépasser 16 000 pixels de large, soit plus de 250 Mo en RGBA16. La plaque donne aussi la vérité terrain pour juger l'interpolation sur de vrais épisodes.

### Phase 5 : la recherche et les autres cartes

Tremblement de pellicule, cadences cassées, désentrelaçage, portages Vulkan (lot D, et les sorties des chantiers 3, 4 et 5). SPAN récurrent, parcimonie 2:4, séquences de banque, trajectoires et ralenti d'export sont retirés (19 novembre).

### Les fils à lancer en parallèle

Six fils peuvent partir tout de suite, car ils ne touchent pas les mêmes fichiers et ne dépendent pas les uns des autres. Le banc NVOF remonte en tête : sa porte décide tôt si les calques valent la peine, avant qu'on construise dessus.

| Fil | Contenu | Touche | Démarre | Ce qui en dépend |
| --- | --- | --- | --- | --- |
| 1. Présentation | Cadence à 165 Hz (0.3) | `src/app/play` (présentation), `src/render` | Tout de suite | Toutes les mesures de vitesse de RIFE |
| 2. Mesure | Outil du scintillement (0.1) | `tools/bench/qualite` | Tout de suite | Toutes les portes de qualité dans le temps |
| 3. Images tenues | Lot A (1.1) avec son interface de décision à part | `src/app/play` (fichiers nouveaux), `src/av` | Tout de suite | Tuiles qui bougent (phase 3) |
| 4. Warp et RIFE | Lot E, bibliothèque de warp (1.3) | `src/gpu/cuda` | Tout de suite | Lot B étape 2, calques, RIFE guidé |
| 5. SPAN | Essai C0, puis lot C (1.2, 2.5) | `src/gpu/cuda` (bibliothèque SPAN), `tools/bench/span_native` | Tout de suite | Tuiles qui bougent (exécution par tuiles) |
| 6. Banc NVOF | Banc hors lecteur (2.1) | `tools/nvof_interp`, `tools/bench` | Tout de suite | Session NVOF, carte de la scène, tous les clients |

**Suivi** (mis à jour par ce fil à chaque nouvelle) :

| Chantier | État à 13:10 (mesures du PC sur main, extrait Re:Zero de 60 s) | PR |
| --- | --- | --- |
| Mesure du scintillement | Réussi : 10 plans sur 10, outil 6 fois plus rapide | #849, #883 |
| Banc NVOF (porte 2.1) | Réussi avec la piste lk\_bords : 99 % des paires (403 sur 407), 100 % sur les caméras lentes (lk : 97 %, NVOF seul : 82 %) ; quatre paires restantes sur Spy 4:53, en cours | #851, #875, #903, #921, #934, #944 |
| Bibliothèque de warp (lot E) | Réussi, tests verts | #852, #873, #874, #878 |
| TemporalFix (cache puis lot B) | Réussi : 10,85 ms (12 visé, 27,9 avant). TemporalFix natif : porte passée (19/11, 16:47), fenêtre retenue par le lecteur à 2,01 ms contre 2,16 ms pour TensorRT (−7 %, 56,2 dB), 7,33 contre 7,71 ms en lecture | #853, #862, #864, #879, #882 |
| SPAN natif | Réussi : défaut au cran Équilibre (66,6 dB, 15,0 ms en lecture) ; cran Qualité en sommes FP32 (78,3 dB, 19,3 ms) ; f16 au banc 10,98 ms (10,64 par bandes ; TensorRT 14,76) ; sommes promues écartées (14,39 ms, −6 dB) ; écart de 4 ms entre banc et lecture à l'étude | #854, #866, #887, #933 |
| Cadence 165 Hz | Réussi : 160,3 i/s en mouvement (118 avant, plafond de l'écran 163,4), GPU 54 %, grade 0,49 ms | #856, #877, #904 |
| Images tenues (lot A) | Raté : 0,8 % repris en dialogue ; dépend des tuiles, en attente | #857, #901, #906, #929 |
| Carte de la scène | Porte 2.2 (temps) réussie : aucun surcoût mesurable, 1420 paires posées sur 1441. Porte 2.3 : caméra réparée (95 à 100 %), mais dessin et effets encore rares ; retravail dans le cloud sur vecteurs exportés (#939, second plan de décor pour la parallaxe du ciel). A/B au PC (19/11, 17:19, 9baf5c9c) réussi : 130,9 à 131,4 i/s avec la carte contre 130,0 à 130,6 sans, p95 égal ou meilleur, affinage 1339 sur 1339 ; reste carte\_lecteur sur les 10 extraits pour la porte 2.3 | #860, #880, #905, #920, #925 |
| Coupes par le flux | Réussi | #861, #884 |
| Éclats HDR sur les effets | Coût nul (1,516 contre 1,518 ms) ; aucun effet encore reconnu par la carte, attend la règle des effets | #858, #867, #876, #926 |
| Tri des paires RIFE | Réussi : 30 paires sur 1186 fondues par endroits (362 sur 368 avant), 47 dessins tenus copiés sans RIFE | #859, #886, #895, #911, #928 |
| Auto plan par plan | Fusionné, aucun défaut remonté | #865, #881, #885 |
| Tuiles qui bougent | NVOF trop bruité pour décider (0 image) ; tuiles jugées sur les pixels (#958, essai --span-tiles) : SPAN de 87 à 51 % de calcul sur Re:Zero, de 95 à 76 % sur Spy ; tolérance du lot A retenue (Tatoub m'a laissé décider) ; scintillement mesuré sur processeur, aucune hausse sur 16 plans (Spy 4:53 0,469 contre 0,516) ; allumées par défaut (#962). Au PC (16:48) : pire bloc 0,75 niveau (réussi) ; SPAN −6,3 % seulement (11 456 contre 12 228 ms sur 60 s), car les tuiles coupent les reprises d'image entière (lot A de 19,8 à 14,9 %) ; retravail (#968, lot A d'abord) : porte RÉUSSIE (17:44), pire bloc 0,79, SPAN −8,2 % (11 241 contre 12 244 ms), lot A intact à 19,8 % | #863, #896, #924 |
| Calques avec plaque de décor | Porte 2.1 passée ; caméra affinée exigée et coordonnées corrigées (#946) ; premier essai : aucune casse mais 34 % des paires (affinage en retard) et retards de la plaque ; retravaillé (#957 : caméra lk d'une voisine, plaque sans second SPAN) ; porte : la moitié des paires et aucun retard en plus ; essai final (19/11, 17:04, 9baf5c9c) : calques RÉUSSIS, 74 % des paires sur le panoramique et 83 % sur Re:Zero, retard p95 inchangé (2,8 contre 3,0 ms ; 2,0 contre 1,8), 0,85 ms par appel ; allumage par défaut en cours. Plaque ratée (+1,7 ms au p95 même sans second SPAN), retravaillée (#970, tuiles de SPAN gardées) : porte RÉUSSIE (17:49), aucun retard en plus sur 2 tours ABAB, 0,48 ms. Sur main sans l'affinage à l'heure, calques à 61 % : attend la fusion de #939 et de l'affinage anticipé | #868, #893 |
| GMFSS union en temps réel (natif sauf GMFlow) | Étapes 1 à 5 fusionnées, GMFSS union de retour au menu sous `--gmfss-native` ; paire et fusion natives retenues (62 dB et plus) : 48,0 → 41,6 ms par paire (−13 %), une paire sur 3 à 4 en 1080p ; portes : ×2 en 720p, puis ×2 en 1080p (environ 28 ms) ; plafonds dans gmfss-plafonds.md ; GMFlow natif repris pour l'attention fondue (voir « Viser les plafonds théoriques ») | #870, #872, #927, #930, #931, #935 |
| RIFE entièrement natif | Retenu en mélange : 6,43 → 5,85 ms par passe (−9 %), 49,1 dB contre TensorRT ; puis 11,5 % plus rapide que TensorRT (#942) ; suite reprise le 19/11 à 16:35 pour viser le plafond (1,58 ms par passe) : entrée à la volée en relecture (#961) | #869, #871, #891, #897, #919 |

## Viser les plafonds théoriques (accord de Tatoub, 19 novembre, 16:35)

SPAN, GMFSS et RIFE visent chacun leur plafond théorique sur la RTX 4070 Ti : max(calcul ÷ débit, octets ÷ bande passante), avec 160 TFLOPS en f16 et 504 Go/s. Le travail se prépare dans le cloud (émulateurs, dB) pendant que le PC sert d'abord aux chantiers dont la porte n'est pas passée. Winograd F(2×2, 3×3) en f16 vient en dernier, après tout le reste, et seulement si son coût en dB, mesuré d'abord dans le cloud, est tenable. Gain réel compté : 1,5× environ, pas les 2,25× théoriques.

| Modèle | Aujourd'hui | Plafond visé | Leviers, dans l'ordre | Ce que ça donne en 1080p |
| --- | --- | --- | --- | --- |
| SPAN b8f64 f16 | 10,64 ms | 6,3 ms, environ 4,2 ms avec Winograd | Poids partagés en mémoire partagée (#963, fusionné éteint, banc PC à venir) ; réglage par couche ; fusion des blocs SPAB seulement en bande de colonnes et si le L2 limite encore (par tuiles, les halos coûtent 1,34× le calcul) ; Winograd. Bilan complet : docs/span-bilan.md (#966). Le f32 est déjà à 85 % de son plafond : on s'y arrête. Le FP8 natif (3,1 ms de plafond) reste en pause | De la place pour TemporalFix et GMFSS dans les 41,7 ms d'une image source |
| GMFSS, fusion (GridNet 97 %) | 21,9 ms | 11,2 ms, environ 7,5 ms avec Winograd | Blocs résiduels de GridNet fondus par tuiles (moitié des accès mémoire en moins) ; réglage par couche (écrit) ; Winograd | Corrigé (bilan GMFSS, 19/11 17:23) : GMFSS crée son image avant SPAN, chaque image ajoutée paie un SPAN (11 ms) et 7 ms de splats ; il reste environ 10,6 ms à la paire et à la fusion. 1080p vers 4K : ×2 seulement au-delà des plafonds ou avec un SPAN à 6 ms ; travail 720p : ×2 aujourd'hui, ×3 aux plafonds, ×4 au-delà ; plus haut, mélange GMFSS + RIFE |
| GMFSS, paire | 19,7 ms | 5,7 ms | Profil nsys (part de GMFlow) ; attention fondue à la manière de FlashAttention (GMFlow 6,6 → 2,3 ms de plafond) ; GMFlow natif (#930) | (compté avec la fusion) |
| RIFE 4.25, une passe | 3,0 ms | 2,28 ms couche par couche, 1,58 ms tout fondu | Reprendre la suite du RIFE natif (entrée à la volée, 78bfe89) ; fondre la colle et la fin ; puis la passe entière en un noyau ; ensuite 4.22 lite et DistilDRBA. Winograd n'aide pas RIFE, limité par la mémoire | ×13 (312 i/s) puis ×18 (430 i/s) ; au-delà des 165 Hz, ça libère surtout la carte pour le reste de la chaîne |

Un plafond se déplace mais ne se bat pas : on s'en approche (70 à 80 % est un bon résultat), et seuls moins de bits, moins de travail ou Winograd le font bouger.

**En pause par décision de Tatoub (19 novembre, 14:44 à 14:47)**, à ne reprendre qu'avec son accord ; ce qui est fusionné reste en place :

- SPAN FP8 en noyaux natifs : convolutions FP8 en cours d'écriture, rien de fusionné, le FP8 CUTLASS reste en service.
- Suite du RIFE natif : mélange retenu et fusionné (#919, #942, 11,5 % plus rapide que TensorRT) ; entrée à la volée et blocs 0 à 2 en attente.
- Perfectionnement du flux NVIDIA : porte 2.1 à 99 % ; les 4 paires de Spy 4:53 en attente.
- Étude du GMFlow natif : squelette fusionné (#930), en attente de la clôture du GMFSS natif.

Points d'arrêt détaillés : `/mnt/project-files/seance-pc/chantiers-en-pause.md`.

**Ajouté le 19 novembre, 08:53 : le natif complet.** Le bon choix entre TensorRT et nos noyaux se fait couche par couche. Un petit réseau limité par la mémoire, ou une opération irrégulière, va en natif ; un gros calcul régulier reste à TensorRT. Deux fils partent tout de suite :

- **GMFSS union en temps réel** (`/mnt/project-files/chantier-natif/gmfss-natif.md`) : tri des paires dans l'étage GMFSS, splats sur la bibliothèque de warp, fusion native (IFNet et GridNet), FeatureNet et MetricNet natifs. GMFlow, un transformeur, reste sous TensorRT FP8. Visé : ×2 tenu partout et ×3 en moyenne, en travail 720p avec SPAN.
- **RIFE entièrement natif** (`/mnt/project-files/chantier-natif/rife-natif.md`) : 23 % d'occupation en RIFE 4.25 ×2 (30 % aujourd'hui), sans construction de moteur au premier lancement.

Tous deux reprennent la famille de convolutions de SPAN natif (#854) et se mesurent à une seconde séance finale.

Les demandes de banc passent par le coordinateur, qui tient la session du PC : un fil ne peut pas écrire directement au fil du PC.

Puis, au fil des portes :

- **Lot B** : son étape 1 (le cache de l'encodeur) peut partir tout de suite comme septième fil ; son étape 2 attend la bibliothèque de warp du fil 4.
- **Session NVOF et carte de la scène** (2.2, 2.3) : dès que la porte du fil 6 est passée. C'est le passage obligé de la phase 3 : un seul fil, le plus soigné.
- **Phase 3** : les coupes reviennent au fil des coupes de scène (PR #847) ; les tuiles au fil 3 après le lot A et l'exécution par tuiles du fil 5 ; le tri des paires et les calques au fil 4 ; les éclats HDR à un fil à part.

Ce qui limite vraiment le parallèle :

- **Un seul PC de mesure.** Deux bancs en même temps sur la 4070 Ti faussent les deux. Les fils préparent, testent sous lavapipe et en CI, puis passent au PC chacun leur tour. Tenir un tableau d'occupation du PC.
- **Les paquets TensorRT.** Les fils 3 (si `src/core`), 4, 5 et la session NVOF changent l'empreinte. Grouper leurs fusions pour ne refaire qu'un paquet Linux par lot de fusions.
- **Les conflits de fichiers** : les fils 4 et 5 partagent `src/gpu/cuda/backend/CMakeLists.txt` et `fused_graphs` ; le fil 3 et les clients de la phase 3 partagent `upscale_stage`. Le second arrivé refait sa branche sur main avant de fusionner.

### Ce que l'implémenteur risque de louper

- **Les règles de la CI** : moins de 800 lignes par fichier (`upscale_stage.h` est à 798), `clang-format-20`, aucun `getenv`, aucun nouveau `path::string()`, construction en `-Werror`. Les lancer en local avant chaque PR (`tools/ci/`), et lancer ci.yml sur la branche pour toute PR CUDA, CMake ou Windows.
- **La langue** : commentaires et journaux en français. Tout texte d'interface passe par `i18n::tr` avec son entrée dans `i18n_fr_*.cpp`, sinon `i18n_test` échoue.
- **Les réglages** : un nouveau réglage va sous « Avancé », s'applique en direct ou en rechargeant la vidéo, jamais « au prochain lancement ». La lecture de `play.toml` ignore les clés inconnues ; ajouter un test qui relit un ancien fichier.
- **L'export rejoue la chaîne** : chaque nouveauté y est soit déterministe (indices temporels de NVOF remis à zéro au début de l'export et à chaque coupe), soit explicitement coupée.
- **Les sauts et les époques** : toute mémoire d'une image à l'autre (ancre du lot A, historique de la carte, indices de NVOF, plaques) se vide au saut.
- **Le chemin sans NVIDIA** : le tester explicitement à chaque client, avec la carte absente, contre une sortie de référence d'aujourd'hui.
- **La 4050 portable** : horloges plus basses, 6 Go de mémoire, chaleur. Rien n'est allumé par défaut sans y avoir été mesuré.
- **L'Auto** : tout nouvel étage qui coûte du temps GPU est compté dans la calibration (`auto_config`, `machine_calibration`). NVOF coûte peu au GPU mais un peu au CPU et à la mémoire.
- **Les licences** : aucune ligne de libaji (sans licence), du fork mpv (GPL) ni de mpv-AnimeJaNai (CC BY-NC-SA). Les méthodes se citent. Aucun fichier de NVIDIA dans nos archives.
- **Les releases** : aucune étiquette ni release sans demande explicite de Tatoub, et un lancement vert de ci.yml sur le commit avant.

## Idées très ambitieuses, jamais essayées

Ces idées ne viennent ni de Kirari ni, à ma connaissance, d'un autre lecteur. Aucune n'est vérifiée : ce sont des paris, rangés du plus prometteur au plus fou.

1. **Réutiliser les séquences de banque.** Les transformations, les lancers d'attaque et les génériques reviennent à l'identique d'un épisode à l'autre. Une empreinte par plan, recalée par le flux, les reconnaît. Kirari rejoue alors une version faite une fois avec le modèle le plus lourd, en avance ou pendant un épisode précédent. Les cycles de marche ou de cheveux dans un même plan sont la version courte de la même idée.
2. **Restaurer l'anime sur pellicule (années 70 à 90).** Deux défauts que le flux sait isoler :
   - le flottement de l'image dans le couloir du projecteur : petit tremblement global au hasard, que le mouvement global corrige sans toucher aux vrais mouvements de caméra ;
   - poussières et rayures : une tache n'est que sur une image, ses voisines alignées donnent ce qu'il y avait dessous. Décision du 19 novembre : on ne garde que le tremblement, facile une fois la session NVOF en place ; les poussières sont abandonnées (trop de risques de confusion avec les étincelles et les reflets).
3. **SPAN récurrent.** Réentraîner SPAN avec, en entrée, sa propre sortie précédente déplacée par le flux. C'est la méthode des meilleurs modèles de super-résolution vidéo (famille BasicVSR), d'habitude ruinée par le coût du flux. Ici le flux est gratuit, il ne reste que le coût de SPAN.
4. **Trajectoires d'animateur.** Calculer le flux entre toutes les paires d'une fenêtre de 4 à 6 images (pas seulement voisines) pour obtenir des trajectoires courbes et l'espacement des dessins. Les images créées suivent alors l'arc et le ralenti voulus par l'animateur, au lieu d'une ligne droite à vitesse constante.
5. **Effacer les incrustations.** Logos de chaîne, bandeaux, sous-titres incrustés : pendant un panoramique, ce qui est caché sur une image est visible sur une autre. Le flux aligne ces images et remplit la zone cachée. Abandonnée le 19 novembre : pas simple, et une fausse détection abîmerait une vidéo sans incrustation.
6. **Relief pour casque ou écran 3D.** Les calques de l'anime ont un ordre de profondeur, et la parallaxe d'un panoramique le révèle. Le flux donne une profondeur relative par calque, assez pour une vue stéréo. Mise de côté le 19 novembre : pas utile pour l'instant.
7. **Ralenti d'export.** À l'export, pas de contrainte de temps réel : flux 1×1 au réglage lent dans les deux sens, plusieurs passes, pour des ralentis ×4 ou ×8 propres d'un passage de sakuga. Gardée, mais en dernier : l'export n'est pas une priorité de Tatoub.

## Pistes écartées

- **NvFRUC** (l'interpolation toute faite de NVIDIA) : sa licence empêche de le livrer, et il est pensé pour le jeu vidéo, ce qui mélange les traits fins ([ADR 0015](https://github.com/Ta2b/kirari-player/blob/main/docs/adr/0015-pistes-ecartees.md)).
- **Mode stéréo (disparité)** : NVIDIA annonce sa disparition, et il demande deux vues déjà alignées. L'idée de relief ci-dessus passe par le flux normal.
- **Netteté selon le mouvement** : gain peu visible sur l'anime, où le flou de mouvement est rare et voulu.
- **Remettre le moteur d'interpolation NVOF seul** (celui d'octobre) : un vecteur par bloc sans réseau correcteur ne tient pas les traits. Il ne revient que guidé (piste 4).

## Vu chez SVP

Une seule idée de SVP 4.7 manquait à ce document : le masque d'artefacts autour de RIFE (dernier point). La liste complète est dans comparaison/svp-idees-absentes-de-kirari.md des fichiers du projet. Relevé par le fil de comparaison sur le PC de Tatoub (`D:\SVP 4\script\generate.js`) :

- **Coupes par NVOF** : SVP calcule les vecteurs sur une image réduite et les passe à RIFE. C'est la piste 7 ; Kirari fait aujourd'hui sa pré-analyse par NVDEC en 256 × 144.
- **Suppression des images en double avant l'interpolation** : en fait, durfix.py (DropOther) jette une image sur deux pour une source à 60 i/s dédoublée, rien à reprendre pour l'anime. L'idée voisine existe déjà dans Kirari sous le nom « Lisser les dessins tenus » (`smooth_drawings`, `docs/interpolation.md`). Le flux l'améliorerait comme le lot A : un dessin tenu reconnu même quand la caméra bouge.
- **Interpolation entièrement par NVOF pour les petites cartes** : c'est le moteur retiré en octobre. Kirari couvre les petites cartes avec RIFE 4.22 lite et DistilDRBA ; ce mode reste écarté pour les traits fins.
- **Réutilisation des images tenues** (mpv AnimeJaNai, pas SVP) : c'est le lot A.
- **Masque d'artefacts autour de RIFE** (`fi_masking`) : SVP passe ses vecteurs à RIFE pour adoucir les zones à risque ; un développeur dit que tout se joue dans RIFE, l'effet réel n'est pas établi. Chez nous, la carte de coût NVOF et le désaccord entre NVOF et le flux de RIFE donneraient ce masque : là où l'un ou l'autre doute, on ramène vers le fondu ou la tenue. C'est une variante légère de la piste 1, à faire passer par le tri des paires (piste 5).

## Suite proposée

Première étape proposée : un banc sur la 4070 Ti qui mesure les vrais temps des trois grilles et sort, pour chaque paire, le mouvement de caméra, la carte de coût et le masque décor contre personnages. Il dira vite si la brique commune des pistes 1, 3, 5 et 6 tient sur l'anime.

- `nvof_dump` (`tools/nvof_interp/nvof_dump`) sait déjà appeler le matériel ; le SDK est déjà sur le PC Windows (`E:\@home\ta2b\nvof-sdk`).
- Piste 4 : le fil Améliorer l'interpolation d'anime mène FlowJaNai et le banc `nvof_warp` (grilles 4, 2 et 1). Son résultat sert aussi à la piste 2.
- Piste 7 : à brancher sur le découpeur de la PR #847, avec le fil Améliorer la détection des coupes de scène et son banc de 26 extraits.

Le profil Nsight de SPAN (19 novembre, chantier-animejanai/profil-nsys-span-kirari-vs-animejanai.md) le confirme : sur Ada, TensorRT laisse séparées les petites convolutions et les opérations élément par élément, et choisit des noyaux d'une génération plus ancienne (sm75). Tout ce qui suit le flux (warps, cartes de coût, composition des calques, masques) s'écrit donc en noyaux CUDA à nous, comme RIFE fusionné, et non dans un graphe TensorRT.

## Sources

- [NVIDIA Optical Flow SDK, note d'application](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-application-note)
- [NVIDIA Optical Flow SDK, guide de programmation](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-programming-guide/index.html)
