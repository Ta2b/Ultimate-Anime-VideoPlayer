# L'Auto plan par plan (`--shot-plan`, essai)

Novembre 2026, étape « Chaîne planifiée par plan » du plan du flux NVIDIA. Aucun nouveau modèle : la chaîne
choisit, plan par plan, entre ce qu'elle a déjà.

## Ce qui est fait

1. **Le rythme de chaque plan** (`src/media/shot_rhythm.h`), lu par la pré-analyse (`shot_probe`) en avance
   sur la lecture. Pour chaque paire d'images consécutives :
   - sur la grille de l'analyse (128 × 72 points, toujours là), par blocs de 8 × 8 points : la part des blocs
     qui changent (écart moyen d'au moins 6 codes) et le grain des autres ;
   - avec le flux matériel de NVIDIA à 256 × 144 (`media/probe_flow.h`, la session du fil des coupes, une
     seule pour les deux) : la translation dominante (la caméra), l'écart des vecteurs à elle (les dessins)
     et la part des blocs sans correspondance. C'est ce qui sépare un panoramique d'un passage de sakuga.

   Un plan garde le nombre de ses paires, de ses paires tenues (sans dessin nouveau) et de ses paires fortes,
   le grain moyen et la part immobile. La première paire du plan (qui enjambe la coupe) n'entre pas. Le
   rythme va au cache des plans (signature `AJPSHOT7` : un cache plus ancien se refait).

2. **Le choix d'un plan** (`src/app/play/shot_plan.h`) :
   - DistilDRBA si le plan est en deux ou plus lent (au moins 1,6 image par dessin) et que moins de 15 % de
     ses paires sont fortes : animation limitée, dialogue ;
   - RIFE sinon : sakuga, animation en un ;
   - TemporalFix seulement si au moins 30 % de l'image est immobile avec un grain d'au moins 1 code ;
   - un plan de moins de 6 paires, ou qui commence sans coupe connue (reprise d'une analyse), garde le choix
     du plan d'avant.

3. **La bascule dans l'étage gpu** (`src/app/play/upscale_shot.cpp`). DistilDRBA est chargé en plus de RIFE
   (`chain_shot_plan.cpp`), les deux en variante mouvement, warp après SPAN, à la même taille d'estimation :
   les tampons et l'ordre du warp sont les mêmes. Le moteur ne change qu'à une coupe connue des plans ou au
   début d'une époque ; ce que le moteur quitté avait lancé en avance est relu et oublié. Un plan pas encore
   analysé garde le moteur en cours, et TemporalFix.

Quand la forme de la chaîne ne s'y prête pas (warp avant SPAN, RIFE image, GMFSS, dessins lissés, sans SPAN,
DistilDRBA absent), DistilDRBA n'est pas chargé ou est rendu, et seul TemporalFix se règle par plan. Le
journal le dit (« plan par plan : … »).

## Ce que dit le journal

- à l'ouverture : « plan par plan : DistilDRBA et RIFE mouvement chargés ensemble », ou la raison du refus ;
- à chaque bascule : le moteur, l'instant et le rythme du plan (« en deux (2,1 images par dessin), 4 % de
  paires fortes, grain 0,8 sur 62 % immobile, flux : … ») ;
- à la fin : les images sous chaque moteur, les bascules, celles gardées faute de coupe connue, et les
  ingestions passées par TemporalFix ou sautées.

## Ce qui reste à mesurer (séance finale)

Tous les seuils sont posés sans mesure (`rhythm::` et `shot_plan::`). À la séance de fin :

- mémoire sur la RTX 4050 (6 Go) avec les deux moteurs chargés ;
- `kirari --debug --shot-plan` contre sans, sur un épisode : images perdues, et le scintillement des plans où
  TemporalFix est sauté (outil de mesure du scintillement) ;
- la répartition DistilDRBA / RIFE lue au journal sur des passages connus (dialogue, sakuga, panoramique).
