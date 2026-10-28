# Menu des paramètres : arbre actuel et plan de tri (0.4)

État du code à `0b60fda` (v0.3.0). Source de l'inventaire : `src/app/play/settings_panel.cpp`,
`settings_chain_tabs.cpp`, `settings_engines.cpp`, `settings_export.cpp`, `settings_system.cpp`
(extraction automatique des widgets ImGui, puis relecture). Rien n'est encore modifié : ce document est le
plan, à valider avant de toucher au code.

## 1. Ce qu'il y a aujourd'hui

12 onglets, une soixantaine de réglages, **plus de 80 notes qui totalisent plus de 7 300 caractères** (environ
1 200 mots) intercalées entre les réglages : Fluidité seul en porte 17 (3 700 caractères).

| Onglet | Réglages | Notes (caractères) | Contenu |
|---|---|---|---|
| Profils | 2 + 2 boutons | 1 (93) | liste des 9 profils, nom, profil de démarrage, retour au profil d'origine |
| Image | 4 | 3 (305) | SPAN (case, modèle), hauteur de travail, algorithme de réduction |
| Fluidité | 6 + conditionnels | 17 (3 699) | multiplicateur (Auto, ×1,5 à ×6…), moteur (RIFE / GMFSS / GMFSS Union / DistilDRBA / Apple), modèle RIFE, mode GMFSS, poses DRBA, warp et SPAN, Smooth Motion |
| HDR | 4 | 3 (223) | moteur HDR (RTX HDR / adaptatif / aucun), pic, gris moyen, débandage RTX HDR |
| Grade | 15 | 8 (1 524) | grade (case, préréglage, 6 curseurs), gamut P3, filtre vers la fenêtre, débandage, noir absolu OLED, soins OLED, cadence fixe |
| Langues | 4 (2 préférences × audio, sous-titres) | 1 (406) | audio et sous-titres ensemble, « Aucun » |
| Sous-titres | 1 + bouton | 1 (143) | taille |
| Son | 4 + bouton | 4 (826) | délai audio, épisode suivant, calage sur l'écran, latence de l'écran |
| Stats | 8 | 0 | quoi afficher avec Tab |
| Moteurs | 2 listes de choix + 3 boutons | 5 | moteur de SPAN et RIFE, moteurs installés et construits |
| Export | ~8 | 12 | taille, deux sections repliables (agrandissement, interpolation), codec, qualité, cadence fixe, fichier |
| Système | 3 boutons | 3 | lecteur par défaut |

Arbre détaillé (réglage → type) : voir l'annexe A.

## 2. Ce qui ne va pas

**Mise en page**
- 12 onglets ne tiennent pas dans la largeur du menu (760 px au plus) : la barre d'onglets défile avec de
  toutes petites flèches, et la molette de la souris ne la fait pas défiler.
- Une barre de défilement par défaut d'ImGui, épaisse, posée contre le contenu : elle mord sur les réglages.
- Les réglages occupent 58 % de la largeur (`PushItemWidth(-width * 0.42)`), les libellés sont à droite
  du champ : on lit « Multiplicateur » après avoir vu la valeur.

**Texte**
- Des paragraphes de 150 à 400 caractères entre chaque réglage : on ne distingue plus les réglages.
- Du vocabulaire interne : « GMFlow en FP16 », « Warp et SPAN », « splat », « Hauteur de travail »,
  « mouvement », « pas de code », « gain sur R, V et B ».
- Des traductions approximatives : **« 720 lignes », « 1080 lignes »** (personne ne dit « lignes » : 720p,
  1080p), « Réduction avant l'agrandissement » (on veut dire « Résolution de travail »), « Grisés : au-delà de
  l'écran », « i/s » non défini, « Expérimental » sans dire ce que ça risque.
- Des chiffres de mesure dans l'interface (« base 24 ms et union 23 ms par image intermédiaire à 720p sur une
  4070 Ti ») : vrais pour une carte, faux pour les autres.

**Logique**
- Un même sujet éclaté : l'image est dans Image, Fluidité et Grade ; le HDR dans HDR et Grade ; le débandage
  est à deux endroits (HDR « Débandage (RTX HDR) » et Grade « Débander les aplats et dégradés ») ; les
  langues dans deux onglets avant 0.3, les sous-titres dans deux maintenant.
- Des réglages dans le mauvais onglet : « Cadence fixe (VRR sans scintillement) » et « Effacer la barre en
  pause » sont dans Grade ; « Lire l'épisode suivant » et « Caler la lecture sur l'écran » dans Son ;
  « Smooth Motion » dans Fluidité.
- Aucun repère pour choisir : pas de « recommandé », pas de coût (plus rapide / plus lent), pas de résumé de ce
  que la chaîne fait.
- Les réglages avancés (grille de 15 curseurs de Grade, algorithmes de réduction, warp avant ou après SPAN,
  poses, Smooth Motion) sont au même niveau que le choix d'un multiplicateur.

## 3. Proposition

### 3.1 Principes
1. **Un réglage = une ligne** : libellé à gauche, valeur à droite, rien d'autre. Toute explication passe en
   **infobulle** (au survol, 0,5 s), jamais en paragraphe.
2. **Des repères dans les listes** : une pastille colorée après l'option : *Recommandé*, *Le plus rapide*,
   *Le plus fin*, *Le plus lourd*, *Expérimental*. Le texte de la pastille est court ; le détail est dans
   l'infobulle.
3. **Basique d'abord, avancé replié** : en tête de chaque onglet 3 à 6 réglages ; le reste dans une section
   « Avancé » repliée par défaut (se souvient de son état).
4. **Un résumé vivant** en haut de l'onglet Vidéo : « 1080p → 720p → SPAN ×2 → RIFE ×3 → 2560×1440 · RTX HDR »,
   déjà calculé par `ChainEngines` (c'est l'étiquette de l'OSD).
5. **Français courant** : 720p et non 720 lignes, « résolution » et non « hauteur de travail », « images par
   seconde » écrit en toutes lettres la première fois, pas de sigle sans infobulle (FP16, SPAN, RIFE, GMFSS
   restent des noms propres, expliqués au survol).
6. **Aucune donnée mesurée dans les textes** : les temps par carte restent dans la documentation ; l'interface
   dit « plus lourd », pas « +9,8 ms ».

### 3.2 Arborescence cible (6 onglets, plus de défilement horizontal)

Le choix du **profil** quitte les onglets : un sélecteur permanent en haut du menu (« Profil : Qualité 1440p
▾ », renommer, profil de démarrage, retour à l'origine dans son menu). C'est le seul élément commun à tous les
écrans.

| Onglet | Contient | Avancé (replié) |
|---|---|---|
| **Vidéo** | résumé de la chaîne ; **Résolution** : résolution de travail (480p à 1080p, Source), agrandissement SPAN (case, modèle) ; **Fluidité** : multiplicateur (Auto, ×1 à ×6), moteur d'interpolation, mode GMFSS (Performance / Qualité) | algorithme de réduction (Area, Spline36, Hermite…), modèle RIFE, poses DRBA, warp avant/après SPAN, Smooth Motion, RIFE sur copie 720p en 4K |
| **Couleur** | moteur HDR, pic, préréglage (Naturel / Éclatant), grade (case) ; débandage (un seul réglage) ; noir absolu OLED (Auto / Oui / Non) | les 6 curseurs du grade, gamut P3, gris moyen, seuil du noir, soins OLED, cadence fixe, filtre vers la fenêtre |
| **Langues** | deux préférences (audio + sous-titres, « Aucun »), taille des sous-titres | — |
| **Son et lecture** | volume, délai audio, épisode suivant | caler sur l'écran, latence de l'écran |
| **Export** | inchangé dans l'esprit : taille, codec, qualité, fichier, bouton Exporter ; les deux sections repliables deviennent « Avancé » | agrandissement et interpolation de l'export |
| **Avancé** | moteurs (installer, choisir), statistiques à l'écran (Tab), lecteur par défaut, dossier des modèles | — |

Fusion Agrandissement + Interpolation : oui, dans **Vidéo**, car c'est une seule chaîne (le choix de la
résolution de travail décide ce que l'interpolation peut tenir). Fusion HDR + Grade : oui, dans **Couleur**.

### 3.3 Mécanique de l'interface (à corriger dans le code)
- **Barre d'onglets** : `ImGuiTabBarFlags_FittingPolicyResizeDown` (les onglets se serrent) et 6 onglets : plus
  de flèches. Si l'écran est étroit, la molette défile la barre (à implémenter proprement si un jour il y a
  plus de 6 onglets).
- **Barre de défilement du contenu** : fine (8 px), arrondie, translucide, au bord extérieur du panneau avec une
  marge de 6 px (ne mord pas sur les réglages), visible seulement au survol ou pendant le défilement.
- **Largeur des champs** : 55 % pour le libellé, 45 % pour la valeur ; libellé à gauche.
- **Infobulles** : `ImGui::SetItemTooltip` avec délai de survol ; le texte d'aide actuel (les `note()`) est
  déplacé là, raccourci, et relu.
- **Pastilles** : petit composant `badge("Recommandé")` dans `panel_style.h` ; couleurs : vert recommandé, bleu
  rapide, ambre lourd, violet expérimental.
- **Avancé replié** : `CollapsingHeader` qui mémorise son état dans `play.toml` (`[ui] advanced_open`).

### 3.4 Textes à corriger tout de suite (liste non exhaustive)
| Aujourd'hui | Proposé |
|---|---|
| 720 lignes (recommandé), 900 lignes… | 720p (recommandé), 900p, 540p, 480p ; « Source (1080p) » |
| Hauteur de travail | Résolution de travail |
| Réduction avant l'agrandissement | Résolution de travail (la source est réduite avant SPAN) |
| Algorithme | Filtre de réduction |
| Warp et SPAN | Ordre du traitement (infobulle : avant ou après l'agrandissement) |
| Moteur de l'interpolation | Moteur d'interpolation |
| GMFSS Fortuna (le plus fin, le plus lourd) | GMFSS Fortuna + pastille *Le plus fin* |
| Grisés : au-delà de l'écran (165 Hz pour 23.976 i/s) | infobulle du multiplicateur : « L'écran affiche 165 images/s au plus » |
| Cadence fixe (VRR sans scintillement) | Cadence fixe (infobulle : pour les écrans VRR OLED) |
| Doubler les images après RIFE | Doubler les images (pilote NVIDIA) + pastille *Expérimental* |
| Seuil (pas de code) | Seuil du noir |
| Effacer la barre en pause (anti-marquage) | Masquer la barre en pause (protège l'OLED) |

## 4. Les autres points de la 0.4

- **Hermite en réduction** : le filtre de réduction a 6 choix (Area, Spline36, Bilinear, Bicubic, Mitchell,
  Catmull-Rom) ; **Hermite n'y est pas**. Hermite est un cubique à support 2 (B=0, C=0), net sans halo : un bon
  candidat pour l'anime. Comme Mitchell et Catmull-Rom sont déjà des cubiques paramétrés (B, C) dans les noyaux
  d'ingestion CUDA et dans la voie Vulkan, l'ajout est un jeu de paramètres de plus (`DownscaleFilter::Hermite`,
  clé `hermite`, étiquette *Hermite* + pastille *Anime*) à tester contre Spline36 et Mitchell (PSNR et regard)
  avant d'en faire un défaut éventuel.
- **GMFSS, amélioration** : à définir avec toi. Pistes mesurées : (a) le mode qualité coûte 7 à 10 ms ; (b) la
  fusion pèse 15 à 16 ms et les convolutions 82 % de son temps ; (c) sortir les warps et splats du graphe, comme
  pour DistilDRBA (76 % de son graphe à 2160p).
- **Mode auto, amélioration** : à reprendre avec le coordinateur (qui l'a écrit) : réglages, repli, retour sur
  « plus de ×4 demande un écran rapide ».

## 5. Étapes proposées

1. **Textes et composants** (sans changer l'organisation) : 720p, vocabulaire, `badge()`, infobulles, retrait
   des paragraphes. Risque faible, visible tout de suite.
2. **Mécanique** : barre d'onglets, barre de défilement, largeurs, section « Avancé » repliable avec mémoire.
3. **Réorganisation** : sélecteur de profil, 6 onglets, déplacement des réglages. Aucune clé de `play.toml`
   ne change (c'est de la présentation), donc aucune migration.
4. **Passe de relecture** : toutes les infobulles relues d'un trait ; essai à 100 %, 150 % et 200 % d'échelle
   d'interface, sur un écran 4K et sur 1366×768.

Vérifications : les tests existants de `settings_test` et du panneau (pas de changement de comportement),
un essai d'instantané du menu avant et après (`--snapshot`) pour chaque onglet, et un contrôle que chaque
réglage de l'annexe A est encore atteignable.

## 6. Questions pour toi
1. Le profil quitte les onglets (sélecteur permanent en haut) : d'accord ?
2. « Avancé » replié par défaut dans chaque onglet, et un onglet « Avancé » pour moteurs, stats, système : d'accord ?
3. Pastilles *Recommandé / Le plus rapide / Le plus fin / Le plus lourd / Expérimental* : la liste te convient ?
4. Faut-il garder les temps mesurés quelque part (infobulle « environ 24 ms par image à 720p sur une RTX 4070 Ti ») ?
   Je propose non dans l'interface, oui dans `docs/nvidia-linux.md`.
5. Pour GMFSS et le mode auto de la 0.4 : qu'est-ce que tu veux améliorer en priorité ?

## Annexe A : arbre actuel, réglage par réglage

(Extraction automatique ; `note` = paragraphe d'aide entre réglages, avec sa longueur.)

```
## Profils  (settings_panel.cpp) : 2 réglages, 1 notes (93 caractères), 2 boutons
  [Profils (Ctrl+1 à Ctrl+9)]
    · note (93 c.) : Changer d'avis dans les autres onglets modifie le profil actif ; il est gardé dans user.toml.
  [Profil actif]
    - saisie : Nom
    - bouton : Renommer
    - case : Profil de démarrage
    - bouton : Revenir au profil d'origine
## Image  (settings_chain_tabs.cpp) : 4 réglages, 3 notes (305 caractères), 0 boutons
  [Agrandissement]
    - case : SPAN ×2 (AnimeJaNai)
    - liste : Modèle
    · note (90 c.) : Performance : le plus rapide. Balanced : plus fin, plus lent. Sharp1 : contours plus nets.
  [Réduction avant l'agrandissement]
    - liste : Hauteur de travail
    - liste : Algorithme
    · note (157 c.) : Area : moyenne exacte. Spline36 : celui d'AnimeJaNai, net. Catmull-Rom et Mitchell gardent du d…
    · note (58 c.) : Pas de réduction : la source entre telle quelle dans SPAN.
## Fluidité  (settings_chain_tabs.cpp) : 6 réglages, 17 notes (3699 caractères), 0 boutons
  [Interpolation]
    - liste : Multiplicateur
    · note (285 c.) : Auto choisit seul le multiplicateur de ×1 à ×6 (24 à 144 i/s pour 23,976) : il baisse dès que l…
    · note (243 c.) : VRR désactivé ou indisponible : à fréquence fixe, un multiplicateur qui ne tombe pas sur un nom…
    · note (102 c.) : À l'export, aucune image n'est sautée : chaque image créée coûte une passe, le temps de l'expor…
    · note (177 c.) : Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un écran assez rap…
    · note (98 c.) : Ni modèle RIFE (rife_v4.25.onnx, rife_v4.26.onnx, rife_v4.22_lite.onnx…) ni graphes GMFSS trouv…
  [Moteur de l'interpolation]
    - liste : Modèle RIFE
    · note (414 c.) : Expérimental. GMFSS Fortuna (réseau fait pour l'animation) donne les images intermédiaires les …
    - liste : GMFSS
    · note (260 c.) : Performance : le flot de GMFlow se calcule en FP16 (base 24 ms et union 23 ms par image intermé…
    · note (158 c.) : Union mêle à GMFSS l'image d'un RIFE intégré au réseau : moins de défauts sur les mouvements ra…
    · note (271 c.) : DistilDRBA : un RIFE à trois images entraîné à imiter DRBA. Le réseau garde lui-même les poses …
    · note (311 c.) : L'interpolation d'Apple : le réseau de flot optique de macOS (VideoToolbox, macOS 15.4 ou plus)…
    · note (233 c.) : Sortie 4K (source sans réduction agrandie par SPAN) : ×3 et plus demandent un GPU haut de gamme…
    · note (78 c.) : Sortie 4K : RIFE estime le mouvement sur une copie en 720 lignes, plus légère.
    · note (64 c.) : Les coupes de scène et les dessins tenus ne sont pas interpolés.
    - case : Garder les poses (DRBA) avec RIFE
    · note (283 c.) : Un personnage animé en deux sur un fond qui défile garde son rythme (il tient sa pose puis chan…
    - liste : Warp et SPAN
    · note (402 c.) : Après SPAN : SPAN agrandit chaque image source une seule fois, puis RIFE déforme les images agr…
  [Smooth Motion (pilote NVIDIA)]
    - case : Doubler les images après RIFE
    · note (259 c.) : Expérimental. La couche du pilote NVIDIA (RTX 40 et 50) génère une image entre deux images prés…
    · note (61 c.) : Grisé : activez RIFE à ×2 ou plus, sans quoi la lecture cale.
## HDR  (settings_panel.cpp) : 4 réglages, 3 notes (223 caractères), 1 boutons
  [Sortie]
    - liste : Moteur HDR
    · note (99 c.) : RTX HDR demande le moteur TensorRT (écran des moteurs) : sous Vulkan, le HDR adaptatif le rempl…
    · note (71 c.) : RTX HDR : le SDK Maxine VideoFX n'est pas installé (écran des moteurs).
    · note (53 c.) : L'écran ne reçoit pas de HDR10 : la sortie reste SDR.
    - curseur : Pic (nits)
    - curseur : Gris moyen
    - case : Débandage (RTX HDR)
    - bouton : Valeurs par défaut
## Grade  (settings_panel.cpp) : 15 réglages, 8 notes (1524 caractères), 0 boutons
  [Grade]
    - case : Grader la sortie (HDR10 et SDR)
    - liste : Préréglage
    - curseur : Éclats
    - curseur : Aplats lumineux
    - curseur : Couleurs
    - curseur : Couleurs étendues (large gamut)
    - curseur : Noirs profonds
    - curseur : Garde-fou couleurs saturées
    - case : Gamut Display P3 de l'écran
    · note (164 c.) : Sortie SDR : couleurs, noirs, garde-fou, contraste et saturation du mode HDR s'appliquent ; écl…
    · note (56 c.) : Route RTX HDR : TrueHDR, puis le grade sur la sortie PQ.
    · note (85 c.) : HDR adaptatif : libplacebo en PQ, aspect TrueHDR émulé selon la scène, puis le grade.
    · note (81 c.) : Sortie HDR10 sans moteur HDR : pas de grade (moteur à choisir dans l'onglet HDR).
  [Composition finale]
    - liste : Filtre vers la fenêtre
    - case : Débander les aplats et dégradés
    · note (276 c.) : Le filtre étire la sortie de la chaîne (par exemple 1440p) vers la fenêtre ou l'écran 4K ; EWA …
  [Noir absolu OLED]
    - liste : Noir absolu
    - curseur : Seuil (pas de code)
  [Soins OLED]
    - case : Effacer la barre en pause (anti-marquage)
    - case : Cadence fixe (VRR sans scintillement)
    · note (330 c.) : Anti-marquage : sur un écran OLED, la barre s'efface aussi en pause après 20 s sans activité. C…
    · note (282 c.) : Les quasi-noirs deviennent un vrai noir (0) ; le pied rejoint l'image au seuil, compté en pas d…
    · note (250 c.) : Éclatant : éclats et aplats lumineux, couleurs et noirs poussés, teinte gardée (même gain sur R…
## Langues  (settings_panel.cpp) : 0 réglages, 1 notes (406 caractères), 0 boutons
    · note (406 c.) : À l'ouverture d'un fichier : la première préférence dont l'audio ET les sous-titres existent es…
## Sous-titres  (settings_panel.cpp) : 1 réglages, 1 notes (143 caractères), 1 boutons
  [Taille]
    - curseur : Taille (%)
    - bouton : Taille d'origine
    · note (143 c.) : Agrandit ou réduit toutes les polices, panneaux compris (comme la taille de mpv). Changement en…
## Son  (settings_panel.cpp) : 4 réglages, 4 notes (826 caractères), 1 boutons
  [Synchronisation]
    - curseur : Délai audio (ms)
    - bouton : Remettre à 0
    · note (101 c.) : Positif : le son est retardé. Ctrl + clic sur le curseur pour saisir une valeur (jusqu'à ±10 00…
    - case : Lire l'épisode suivant à la fin
    · note (227 c.) : À la fin d'une vidéo ouverte par un fichier, lit la suivante du même dossier (ordre naturel des…
    - case : Caler la lecture sur l'écran
    · note (329 c.) : Sur un écran à fréquence fixe, la lecture accélère ou ralentit d'au plus 0,125 % pour que chaqu…
    - curseur : Latence de l'écran (ms)
    · note (169 c.) : Le temps entre le choix d'une image et son apparition : estimé d'après le rendu et la file de l…
## Stats  (settings_panel.cpp) : 8 réglages, 0 notes (0 caractères), 0 boutons
  [Statistiques à l'écran]
    - case : Toutes les statistiques (Tab)
    - case : Chaîne (modèles, tailles, moteur)
    - case : Images (temps par image, sautées)
    - case : RIFE (interpolées, tenues, coupes)
    - case : Pistes
    - case : Sortie HDR
    - case : Position exacte
    - case : Mémoire vidéo
## Moteurs  (settings_engines.cpp) : choix du moteur (une case par moteur installé), 3 boutons, 5 notes
  [Moteur de SPAN et RIFE]
    - choix : un par moteur (TensorRT, Vulkan, OpenVINO, MIGraphX…)
    - bouton : Utiliser ce moteur / Annuler
  [Moteurs installés]
    - bouton : Installer d'autres moteurs…
  [Moteurs construits]
    · liste des moteurs en cache
## Export  (settings_export.cpp) : ~8 réglages, 12 notes
  [Résolution]
    - liste : Taille
    - repliable : Agrandissement (chaîne de l'export)
    - repliable : Interpolation (chaîne de l'export)
    - bouton : Reprendre les réglages de la lecture
  [Encodage]
    - liste : Codec
    - liste : Qualité
    - case : Cadence fixe
  [Fichier]
    - saisie : Fichier
    - bouton : Nom proposé
    - bouton : Exporter
## Système  (settings_system.cpp) : 0 réglages, 3 notes (104 caractères), 3 boutons
  [Lecteur par défaut]
    · note (93 c.) : Les vidéos (mkv, mp4, webm, avi, mov…) s'ouvrent dans AJP depuis le gestionnaire de fichiers.
    - bouton : Définir par défaut
    - bouton : Ajouter à « Ouvrir avec »
    - bouton : Retirer##inscription
    · note (9 c.) : En cours…
    · note (2 c.) : %s
```
