> Relevé daté (historique) : journal des chantiers du 19 novembre 2026, tenu au fil de la journée. Copie du document de suivi partagé ; heures de Paris. Les chemins `/mnt/project-files/…` sont ceux de l'espace de travail du projet.

# Chantiers du 19 novembre : journal de suivi

19 novembre 2026, pour Tatoub

## Vue d'ensemble

PC de Tatoub : CachyOS, RTX 4070 Ti. Mesures sur l'extrait Re:Zero de 60 s et un panoramique de 20 s, jamais d'épisode entier. Heures en heure de Paris.

| Chantier | État | Mesure clé | Suite |
| --- | --- | --- | --- |
| Cadence 165 Hz | Porte passée | 160,3 i/s en mouvement (118 avant, plafond 163,4) | Rien |
| Cache TemporalFix (lot B) | Porte passée | 27,9 → 10,85 ms par image | Rien |
| TemporalFix natif | Porte passée (17:47) | Fenêtre 2,01 ms contre 2,16 TensorRT (−7 %), 7,33 ms en lecture | Profil ncu facultatif |
| SPAN natif f16 et f32 | Porte passée | Équilibre 10,64 ms (TensorRT 14,76), Qualité 14,74 ms | Vers le plafond (6,3 ms) |
| RIFE natif (mélange) | Porte passée | 4.25 : 3,00 ms contre 3,39 (−11 %) | « Sous stéroïdes », prioritaire |
| Flux NVIDIA (porte 2.1) | Porte passée | Caméra juste sur 99 % des paires (lk\_bords) | Perfectionnement en pause |
| Carte de la scène, coût (2.2) | Porte passée | Aucun surcoût ; A/B 131 i/s avec contre 130 sans | Rien |
| Carte de la scène, classes (2.3) | **Pas passée** | Faux effets et éclats ratés (capture vue-970) | En pause (19:04) : rc.1 avec la carte allumée et ses effets éteints |
| Éclats HDR | **Pas passée** | Coût nul (1,516 contre 1,518 ms) | En pause (19:04) : #977 a tout plafonné à 360 nits, défaut revenu à 100 % (#979) |
| Calques | Porte passée (17:04) | 74 % des paires (panoramique), 83 % (Re:Zero), sans retard | Allumage par défaut ; 61 % sur main tant que l'affinage à l'heure n'y est pas |
| Plaque de décor | Porte passée (17:49) | Aucun retard ajouté, 0,48 ms | Rien |
| Tuiles de SPAN | Porte passée (17:44) | SPAN −8,2 %, lot A intact à 19,8 % | Fusionné (#968) |
| Images tenues (lot A) | Passée avec les tuiles | 19,8 % des images reprises | Rien |
| Tri des paires RIFE | Porte passée | 30 paires fondues sur 1186, 47 dessins tenus copiés | Rien |
| Coupes et Auto par plan | Portes passées | Aucun défaut | Rien |
| GMFSS natif | **Pas passée** | 48,0 → 41,6 ms (−13 %) | Étape 0 passée au PC, #959 dans la rc.1 ; plafonds en pause (19:10) |

## Chantiers dont la porte est passée

**Cadence 165 Hz.** La lecture plafonnait à 118 i/s. Correctifs #856, #877, #904, puis #940 : 160,3 i/s en mouvement pour un plafond de l'écran à 163,4, GPU à 54 %. Contretemps : un chiffre de 138 i/s annoncé par erreur, corrigé.

**TemporalFix (cache puis lot B).** 27,9 → 10,85 ms par image (12 visés), à 52 dB ou plus. PR #853, #862, #864, #879, #882.

**TemporalFix natif.** Contretemps : un test inversé dans `TfixNativeNet::create` (corrigé par c373815c), puis une fenêtre 23 à 34 % plus lente que TensorRT. Retravail : colle en mots de 16 octets, −12 % par 0bd175e4, tête fondue. Résultat : fenêtre native 2,01 ms contre 2,16 ms pour TensorRT, 56,2 dB, 7,33 ms en lecture contre 7,71. Le message « TensorRT plus rapide » n'était qu'un libellé faux (l'essai exige 2 % d'avance). Reste : un profil ncu, qui demande l'option du pilote `NVreg_RestrictProfilingToAdminUsers=0` (sudo de Tatoub), facultatif.

**SPAN natif.** Équilibre (f16) 10,64 ms au banc contre 14,76 pour TensorRT, 66,6 dB et 15,0 ms en lecture. Qualité (sommes f32) 14,74 ms, 78,3 dB. Sommes promues écartées (14,39 ms, −6 dB). Bilan complet : `docs/span-bilan.md` (#966).

**RIFE natif (mélange).** #919 puis #942 : 4.25 à 3,00 ms contre 3,39 sous TensorRT (−11 %), 4.22 lite à 3,20 contre 3,31. Toujours derrière `--rife-native`. Contretemps : le lite ne gagnait presque rien. Cause trouvée dans le cloud : TensorRT lui-même est à 43,6 dB du calcul exact sur ce modèle, et la porte de 45 dB contre TensorRT refusait donc notre encodeur, pourtant juste. Nouvelle porte contre le calcul exact (avec une garde à 60 dB par morceau), gain attendu : lite vers 2,9 ms.

**Flux NVIDIA, porte 2.1.** Caméra à moins de 0,5 px sur 90 % des paires. NVOF seul 82 %, traits 90 %, lk 97 %, lk\_bords 99 % (403 sur 407). Quatre paires ratées sur Spy 4:53, en pause.

**Carte de la scène, porte 2.2 (coût).** Aucun surcoût, 1420 paires posées sur 1441 (NVOFA a son propre bloc matériel). A/B de 17:19 : 130,9 à 131,4 i/s avec la carte contre 130,0 à 130,6 sans, affinage 1339 sur 1339.

**Calques.** Contretemps : premier essai à 34 % des paires, avec des retards de la plaque. Retravail : caméra affinée exigée (#946), caméra lk d'une voisine (#957), affinage à l'heure (branche de #939). Essai final de 17:04 : 74 % (panoramique) et 83 % (Re:Zero), p95 inchangé, 0,85 ms par appel. Point ouvert : sur main sans l'affinage à l'heure, 61 %. Il faut fusionner #939 et l'affinage anticipé.

**Plaque de décor.** Contretemps : +3,4 à +5,3 ms de retard avec un second SPAN, puis +1,7 ms par la voie de copie. Retravail #970 (les tuiles de SPAN restent, la clé copiée ne se recompose pas). Essai de 17:49 : aucun retard ajouté sur deux tours ABAB, 199 sources sur 200 reprises, 0,48 ms.

**Tuiles de SPAN.** Contretemps : NVOF trop bruité pour décider (aucune image par tuiles), puis gain de 6 % seulement, car les tuiles coupaient le lot A. Retravail : tuiles jugées sur les pixels (tolérance du lot A, décidée pour Tatoub), portée de deux lignes, lot A d'abord (#968). Essai de 17:44 : pire bloc 0,79, SPAN −8,2 % (11 241 contre 12 244 ms), lot A intact à 19,8 %, scintillement sans hausse (16 plans). Le levier suivant (région trop large) ne donne qu'un point, rangé dans #971.

**Tri des paires, coupes, Auto par plan.** Tri : 30 paires sur 1186 fondues par endroits (362 sur 368 avant), 47 dessins tenus copiés sans RIFE. Coupes par le flux et Auto par plan : aucun défaut.

## Chantiers dont la porte n'est pas passée

**Carte de la scène, porte 2.3 (classes).** La caméra est juste (95 à 100 %), mais les dessins et les effets sont rarement reconnus. Contretemps du jour :

- faux effets (trait qui bouge, effet tenu sur image fixe), corrigés par #939 ;
- sur le panoramique 0039, le ciel glisse plus lentement que le sol (parallaxe) : un second plan de décor est prévu ;
- capture vue-970 de Tatoub (17:31) : une tache verte sur le fond sombre prise pour un effet, et de vrais éclats non reconnus.

Corrections fusionnées : #967 (aucune lumière immobile retenue) et #969 (aucune marge retenue, rien de la paire coupée). Mesure PC sur 9fff1e32 (17:57) : le faux vert du menton a disparu, mais plus aucun effet sûr n'est reconnu aux trois moments (bougie, halo, corne), et carte\_lecteur donne « effet 0 » (80 à 99 % de tuiles sûres). La correction est allée trop loin. Retravail dans le cloud sur un jeu étiqueté (trois vrais effets, un faux), en mesurant le rappel et les faux positifs avant tout retour au PC. Fil : « Intégrer NVOF et la carte de la scène ».

**Éclats HDR sur les effets.** Coût nul. Ils dépendent entièrement de la reconnaissance des effets par la carte. En cours sur le PC : trois moments (973/983, 995/1005, 969,5/979,5), avec et sans éclats, et la vue du masque. À vérifier : plus de vert à gauche du menton à 979,5. Porte : au plus +0,3 ms, et une part d'éclats sur les effets au-dessus de 0 %. Mesure par le cloud, pas à l'œil. Résultat de 17:57 : avec et sans éclats, l'image est identique (73,1, 200,5 et 162,3 nits de moyenne des deux côtés), puisqu'aucun effet n'est reconnu. Ensuite #974 (lueurs repérées dans l'image, sans la carte) : au PC (18:36), la corne et le halo ne montaient toujours pas. Cause trouvée sur les PQ : ils étaient déjà au plafond de l'écran (626 à 630 nits de contenu), noyés par les cheveux et le visage éclairés à 450-520 nits. Choix de Tatoub : baisser les surfaces. #977 fusionnée (« Blanc des surfaces » à 60 % par défaut, clé surface\_level), les éclats gardent leur niveau. Plus rien ne bloque la rc.1 ; Tatoub jugera à l'œil plus tard.

**GMFSS natif.** #950 fusionnée : paire (demi-flot s05) 22,44 → 19,70 ms à 64,8 dB, fusion 25,57 → 21,90 ms à 62,3 dB, total 48,0 → 41,6 ms. Portes : ×2 en travail 720p, puis ×2 en 1080p. Contretemps :

- l'ancienne paire pleine taille du PC n'avait pas `feature0_1` ;
- le budget de 28 ms annoncé pour le 1080p était faux. GMFSS crée son image avant SPAN, donc chaque image ajoutée paie un SPAN (11 ms) et 7 ms de splats : il ne reste qu'environ 10,6 ms à la paire et à la fusion.

\#959 (natif par défaut) porte maintenant GMFlow natif à attention fondue et GridNet en double tampon, environ −2,5 ms attendues. Le bloc résiduel fondu est écarté. La séance PC unique (`seance-pc-gmfss.md`) passe en dernier, par choix de Tatoub. Bilan : `bilan-gmfss.md`.

## Perfectionnement vers les plafonds théoriques

Accord de Tatoub à 16:35 : viser le plafond de chaque modèle sur la RTX 4070 Ti (160 TFLOPS en sommes f16, 504 Go/s). Le travail se fait dans le cloud. Le PC ne sert qu'après les chantiers sans porte passée. Chaque fil tient son bilan à jour dans `/mnt/project-files/chantier-natif/bilan-*.md` et `docs/`.

| Modèle | Aujourd'hui | Plafond | Où en est le travail |
| --- | --- | --- | --- |
| RIFE 4.25, une passe 1080p | 3,00 ms | 2,28 ms couche par couche, 1,58 ms fondu | Prioritaire (Tatoub, 17:42). #961 (entrée à la volée) en ci.yml ; fin fondue branchée, identique au bit, en relecture ; ensuite blocs 0 à 2 fondus, puis la passe en un noyau |
| RIFE 4.22 lite | 3,20 ms | environ 1,59 ms | Porte contre le calcul exact écrite et relue (garde rôle par rôle, plancher 42 dB, 8 tests) ; gain attendu vers 2,9 ms |
| DistilDRBA v2 lite | 0,917 ms (fusionné, TensorRT) | 0,37 ms, environ 0,27 fondu | Noyaux natifs à écrire, après RIFE |
| SPAN b8f64 f16 | 10,64 ms | 6,3 ms | Poids partagés (#963) mesurés au PC : utiles à 48 canaux seulement, pris par défaut là (#973) ; à 64 canaux le trafic des poids n'était pas la limite. Prochaine étape : un profil Nsight Compute d'une 3×3 pour choisir entre réglage par couche, fusion et Winograd (il demande l'option du pilote NVreg\_RestrictProfilingToAdminUsers=0, posée par Tatoub). Winograd : porte des dB passée (#976, 65,9 à 67,7 dB au b8f64) ; noyau F(2×2, 3×3) écrit dans #980, éteint par défaut, 72,13 dB contre 71,81 au direct sur l'émulateur, identique au bit avec ou sans bandes ; banc PC envoyé, il n'entre par défaut dans la rc.1 que s'il gagne au PC. Fusion des blocs par tuiles écartée (halos à 1,34× le calcul) |
| GMFSS paire | 19,7 ms | 5,7 à 10 ms | GMFlow natif à attention fondue dans #959 |
| GMFSS fusion | 21,9 ms | 11,2 ms | GridNet en double tampon dans #959 (vers 19,1 à 19,6 ms). Tatoub (18:54) : avancer dans le cloud en attendant la release, dans l'ordre du gain : attention fondue de GMFlow (paire vers 5,7 ms), splats fondus dans GridNet (7,1 → 1,5 ms), Winograd sur GridNet (outil de #976), FP8 de la fusion. Mis en pause par Tatoub à 19:10 : bilan complété et poussé, reprise après la rc.1 |

Après tout le reste : Winograd F(2×2, 3×3) en f16 sur SPAN et GridNet, après une mesure des dB dans le cloud.

Une séance PC groupée pour RIFE partira dès que la fin fondue est fusionnée : ABAB dans le lecteur sur 4.25 et lite, puis la PR « natif par défaut » avec un cache du verdict de l'essai au chargement (3 s).

## Chantiers en pause

Aucun ne reprend sans l'accord de Tatoub. Détail des points d'arrêt : `/mnt/project-files/seance-pc/chantiers-en-pause.md`.

| Depuis | Chantier | Point d'arrêt | Pour reprendre |
| --- | --- | --- | --- |
| 14:44 | SPAN FP8 natif | Convolutions FP8 en cours d'écriture, commit 41a4a2f | Six étapes notées dans `docs/span-native16.md` ; plafond 3,1 ms |
| 14:47 | Flux NVIDIA, perfectionnement | Porte 2.1 passée à 99 % ; 4 paires ratées sur Spy 4:53 | Extraire ces paires au PC, les étudier dans le cloud (#955) |
| 14:54 | Couleurs de l'export | Chroma fausse en FFV1 et HEVC, lecture juste | Tatoub : l'export n'est pas prioritaire |
| 15:37 | Horloges de SPAN (#941) | `nvidia-smi -lgc` demande sudo | Règle sudoers limitée, posée par Tatoub |

Repris à 16:35 : la suite du RIFE natif, le GMFlow natif et Winograd (en dernier).

## Vers la première release stable (demande de Tatoub, 18:22)

Priorité : une release stable qui contient l'essentiel des chantiers. Ordre :

1. finir la carte de la scène (porte 2.3) et les éclats HDR ;
2. fusionner tout ce qui est prêt sur main ; ce qui n'a pas passé sa porte reste éteint par défaut ;
3. un ci.yml complet et vert sur main ;
4. la release (`docs/signature-des-releases.md`), avec le paquet TensorRT Linux.

Gel : plus de nouvelle fonction avant la release. SPAN et RIFE continuent vers leurs plafonds dans le cloud, mais rien n'entre sans sa mesure. GMFSS : seule l'étape 0 de sa séance PC passe avant ; elle a réussi à 18:28 (paire 19,73 ms contre 22,48, fusion 21,63 contre 25,83), #959 entre dans la rc.1. Tatoub (18:41) : le RIFE fondu (fin fondue, porte du lite) et Winograd sur SPAN (#976 : 65,9 à 67,7 dB, aucune perte) entrent dans la rc.1 s'ils passent leur mesure au PC avant l'étiquette ; sinon, release suivante. Carte : effets de la carte éteints par défaut pour la rc.1 (#939 à 0accb9ad, en attente du choix de Tatoub), lueurs vues dans l'image (#974) allumées, captures PC en cours.

**PR ouvertes du jour, à fusionner avant la release (19/11, 18:25)** :

| PR | Chantier | État |
| --- | --- | --- |
| #939 | Carte de la scène : classes et affinage à l'heure (9fff1e32) | Effets de la carte éteints ; fusion dès la CI de 5bf37717 verte |
| #964 | TemporalFix natif : colle 16 octets, tête fondue | Porte passée, à fusionner |
| #961 | RIFE natif : entrée à la volée | Tout regroupé sur e05a3a5 (entrée à la volée, porte du lite, fin fondue) ; séance PC puis ci.yml et fusion |
| #959 | GMFSS natif par défaut | Étape 0 passée ; CI sur 16bfdb11, puis fusion |
| #855 | Vulkan : découpe de K des matrices coopératives | À trier (ouverte depuis ce matin) |

Points à trancher avec Tatoub : la version (main dit 0.11.0, une étiquette v0.12.0 existe déjà ; 0.13.0 proposée) ; préversion ou finale : Tatoub a choisi une préversion d'abord (rc.1, 18:25), la finale suivra.

**Éclats HDR, 19:00 :** échec de #977 au PC. Avec les surfaces à 60 %, toute l'image plafonne à 360 nits ; la corne (p99,9 359) et le halo (358) sont au niveau de la tête. Ce sont des blancs purs de la source, que le grade ne distingue pas d'un mur blanc, et les lueurs de #974 ne se déclenchent pas sur le PC. Le défaut est revenu à 100 % sur main (#979), donc la rc.1 a le rendu d'avant. Tatoub l'a mis en pause à 19:04 : la reprise se fera après la rc.1, pour apprendre au grade à distinguer une lueur d'un blanc de décor.

**Carte de la scène, 19:04 :** Tatoub a confirmé à 19:04. Le fil carte fusionne #939 dès que la CI de 5bf37717 est verte, vers 19:25. La reconnaissance des effets est notée dans chantiers-en-pause.md avec trois pistes de reprise. La carte est allumée par défaut, et la reconnaissance des effets de la carte reste éteinte (essai : `--effect-highlights`).

## Règles de Tatoub du jour et ordre du PC

- **Pas de banc intermédiaire** : chaque fil se relit sévèrement, vérifie la construction et rules.yml dans le cloud, fusionne. Seule la fusion d'une PR risquée (Windows, CUDA, CMake) attend ci.yml. Une mesure PC n'attend jamais la CI.
- **Jamais d'épisode entier** : l'extrait Re:Zero de 60 s et un panoramique de 20 s.
- **Cloud d'abord** (15:30) : le PC exporte les données une fois, les analyses tournent sur processeur dans le cloud. Mesures de temps seules et une par une ; mesures de qualité à plusieurs dans 12 Go de VRAM.
- **Priorité du PC** (15:37) : d'abord les chantiers sans porte passée. Rien n'est clos de force.
- **Pas de test pour l'œil seul** : on mesure, et le verdict est un chiffre. Rien sur l'export.
- **GMFSS en dernier** (17:07), **RIFE natif prioritaire dans le cloud** (17:42).
- **Bilans tenus à jour** (17:36) par les fils SPAN, GMFSS et RIFE/DistilDRBA.
- **Tous les fils sur Opus 5.5 en effort medium** (17:39).
- Aucune release ni étiquette sans demande de Tatoub ; un seul paquet TensorRT Linux à la fin, avec son accord.

**Ordre du PC à 18:54** : captures des éclats avec les surfaces à 60 % (#977, en cours), puis séance RIFE (sur sa branche, sans attendre la CI), puis paquet TensorRT Linux pour la rc.1 ; GMFSS et les bancs de plafonds après la release. Version proposée : 0.13.0 (étiquette v0.13.0-rc.1, avec le feu vert de Tatoub).

**Ordre du PC à 17:57** : captures des éclats HDR et carte\_lecteur (en cours) ; séance RIFE dès qu'elle est prête ; banc SPAN #963 ; GMFSS en dernier.

*Ce document est tenu à jour au fil des résultats.*
