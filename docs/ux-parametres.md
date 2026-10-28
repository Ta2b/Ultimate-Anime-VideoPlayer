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

## 3. Ce que font les lecteurs connus

Recherche faite sur la documentation publique. Ce qui est **confirmé** par une source est dit tel quel ; ce que
je n'ai pas pu vérifier (IINA, MPC-HC/BE, PotPlayer : peu de détail trouvé) est marqué *non vérifié* et je ne
m'en sers pas pour décider.

| Lecteur | Structure des réglages | Ce qu'on en retient |
|---|---|---|
| VLC | Préférences en mode **Simple** (Interface, Audio, Vidéo, Sous-titres, Entrée/codecs, Raccourcis) ; un bouton bascule vers **Toutes** (arbre complet) | Deux niveaux dans la même fenêtre ; le simple tient en 6 rubriques par thème (image, son, sous-titres) |
| Kodi | Quatre niveaux de réglage (**Basique, Standard, Avancé, Expert**), niveau par défaut Standard, rubriques Lecteur, Média, Système… | Un curseur de niveau global plutôt qu'un onglet « avancé » par page |
| mpv / mpv.net | Éditeur de configuration avec **recherche** ; aucune hiérarchie de pédagogie | La recherche compense le nombre de réglages (idée pour plus tard) |
| Jellyfin / Plex | Lecture (langues préférées audio et sous-titres, « jouer la piste par défaut »), Affichage, Qualité | Les langues vivent dans la lecture, pas dans un onglet à part |
| Firefox (mises à jour) | Trois choix : **installer automatiquement / vérifier mais me laisser choisir / ne jamais vérifier** | Le modèle de choix à trois positions pour l'auto-update |
| Sparkle (macOS) | Vérification automatique (demandée au 2e lancement si non réglée), installation automatique (désactivée par défaut), signatures EdDSA | Demander une fois, signer les mises à jour |
| AppImage / Flatpak | AppImageUpdate (delta zsync) ; Flatpak et les gestionnaires de paquets mettent à jour eux-mêmes | Ne pas se mettre à jour soi-même quand un gestionnaire s'en charge |
| IINA, MPC-HC/BE, PotPlayer | *non vérifié* (pages onglets : Général, Vidéo, Audio, Sous-titres, Raccourcis, Avancé, d'après ma connaissance, sans source relue ici) | — |

Conclusions qui tiennent sur les lecteurs vérifiés :
1. **Peu de rubriques au premier niveau** (5 à 7) et une rubrique « Général » en tête.
2. **Simple d'abord, complet derrière** : bascule ou niveaux, jamais les deux à plat.
3. **Les réglages d'une même chose restent ensemble** (langues, mises à jour, sous-titres), et la **couleur**
   ne contient que de la couleur : filtre de réduction et mise à l'échelle sont de la *vidéo*.
4. **Mises à jour : trois positions** (automatique / prévenir / jamais), une fois demandé, modifiable ensuite.

## 4. Proposition

### 4.1 Principes
1. **Un réglage = une ligne** : libellé à gauche, valeur à droite. Toute explication en **infobulle** (survol
   0,5 s), jamais en paragraphe.
2. **Des repères dans les listes** : pastilles *Recommandé*, *Le plus rapide*, *Le plus fin*, *Le plus lourd*,
   *Expérimental*. Détail dans l'infobulle.
3. **Essentiel d'abord, avancé replié** : 3 à 6 réglages visibles par onglet ; le reste dans « Avancé » replié
   (état mémorisé). Rien de technique dans « Général ».
4. **Un résumé vivant** en haut de « IA » : « 1080p → 720p → SPAN ×2 → RIFE ×3 → 2560×1440 · RTX HDR ».
5. **Français courant** : 720p et non « 720 lignes », « résolution » et non « hauteur de travail ». FP16, SPAN,
   RIFE, GMFSS restent des noms propres, expliqués au survol.
6. **Aucune mesure dans les textes** (« plus lourd », pas « +9,8 ms ») ; les temps vont dans la documentation.

### 4.2 Arborescence cible : 7 onglets, plus de défilement horizontal

| Onglet | Contient (visible) | Avancé (replié) |
|---|---|---|
| **Général** | **bouton « Faire de ce lecteur le lecteur par défaut »** (en haut, voir 4.4) ; volume, épisode suivant automatique, taille des sous-titres, langue audio et sous-titres préférées ; **Mises à jour** : Automatique / Me prévenir / Jamais (voir 4.5) | délai audio, caler la lecture sur l'écran, latence de l'écran, masquer la barre en pause |
| **IA** (amélioration) | résumé de la chaîne ; **Agrandissement** : résolution de travail (480p à 1080p, Source), SPAN (case, modèle) ; **Fluidité** : multiplicateur (Auto, ×1 à ×6), moteur d'interpolation, mode GMFSS (Performance / Qualité) | filtre de réduction (Area, Spline36, **Hermite**…), modèle RIFE, poses DRBA, ordre du traitement (warp avant/après SPAN), Smooth Motion, RIFE sur copie 720p en 4K |
| **HDR** | moteur HDR (RTX HDR / adaptatif / aucun), pic de luminance, débandage | gris moyen, débandage RTX HDR détaillé, noir absolu OLED (Auto / Oui / Non), soins OLED |
| **Couleurs** | grade (case), préréglage (Naturel / Éclatant), 2 à 3 curseurs essentiels (luminosité, contraste, saturation), gamut P3 | les autres curseurs du grade (gain R/V/B, gamma…), seuil du noir. **Sortis** : filtre de réduction (→ IA), débandage (→ HDR), cadence fixe et filtre vers la fenêtre (→ Avancé) |
| **Audio et sous-titres** | préférences de langue (2 listes, « Aucun »), sous-titres : taille, piste ; audio : délai | — |
| **Export** | taille, codec, qualité, fichier, bouton Exporter | agrandissement et interpolation de l'export |
| **Avancé** | moteurs (installer, choisir, reconstruire), statistiques à l'écran (Tab), cadence fixe (VRR), filtre vers la fenêtre, dossier des modèles, **Mises à jour : canal, fréquence, vérifier maintenant, version installée**, **Lecteur par défaut** (si l'utilisateur a refusé, voir 4.4), « Retirer de Ouvrir avec » | — |

- **Onglet Profils supprimé pour l'instant** (demande du 28/10). Les 9 profils et le profil de démarrage restent
  dans `play.toml` et dans le code : seul l'écran disparaît. Le retour au profil d'origine redevient une ligne
  « Réinitialiser les réglages » dans Avancé.
- **Fusion Agrandissement + Interpolation** : oui, onglet **IA** (une seule chaîne ; la résolution de travail
  décide ce que l'interpolation peut tenir). **HDR et Couleurs séparés** (demande du 28/10) : HDR configure la
  conversion et l'écran ; Couleurs ne contient que l'image.

### 4.3 Mécanique de l'interface (à corriger dans le code)
- **Barre d'onglets** : `ImGuiTabBarFlags_FittingPolicyResizeDown`, 7 onglets de ~100 px dans 760 px : plus de
  flèches. Si un jour il y en a plus, la molette défile la barre (`ImGui::SetScrollX` sur la barre).
- **Barre de défilement du contenu** : fine (8 px), arrondie, translucide, au bord extérieur du panneau avec une
  marge de 6 px (ne mord pas sur les réglages), visible au survol ou pendant le défilement.
- **Largeur des champs** : 55 % libellé, 45 % valeur.
- **Infobulles** : `ImGui::SetItemTooltip`, délai de survol ; les `note()` actuelles y sont déplacées, raccourcies.
- **Pastilles** : `badge("Recommandé")` dans `panel_style.h` ; vert recommandé, bleu rapide, ambre lourd, violet
  expérimental.
- **Avancé replié** : `CollapsingHeader`, état dans `play.toml` (`[ui] advanced_open`).

### 4.4 Lecteur par défaut et « Ouvrir avec »
- **Premier lancement** : AJP s'inscrit tout seul dans « Ouvrir avec », **sans confirmation** (l'inscription
  ajoute une entrée, ne prend la place d'aucun autre lecteur) : `register_player()` existe déjà
  (`default_player.cpp`). Appelé une fois, à la fin du premier lancement (`first_run_done`), puis mémorisé
  (`open_with_registered`) pour ne pas se réinscrire si l'utilisateur l'a retiré.
- **Bouton en haut de Général** : « Faire de ce lecteur le lecteur par défaut » + case « Ne plus me le
  demander ». Linux : `xdg-mime` pour chaque type vidéo (existe : `choose_as_default`). Windows : le système ne
  laisse aucun programme se choisir seul, le bouton ouvre la page Paramètres > Applications par défaut. macOS :
  le Finder (Lire les informations > Ouvrir avec).
- **Le bouton disparaît** quand AJP est déjà le lecteur par défaut, ou que l'utilisateur coche « Ne plus me le
  demander ». Il reste alors **tout en bas de l'onglet Avancé**, section « Lecteur par défaut » (avec « Retirer
  de Ouvrir avec »).
- Détecter « déjà par défaut » : `xdg-mime query default video/x-matroska` (Linux) ; Windows : lecture de
  `UserChoice` ; sinon la case suffit.

### 4.5 Mise à jour automatique (configurable)
Réglages, dans Général (le choix) et Avancé (le détail) :
- **Mises à jour** : *Automatique* (télécharge, installe au prochain lancement), *Me prévenir* (notification,
  l'utilisateur lance), *Jamais*.
- Avancé : canal (*Stable* / *Préversions*), fréquence (au lancement, quotidienne, hebdomadaire), « Vérifier
  maintenant », version installée, « Revenir à la version précédente », dernière vérification.

Mécanique :
1. **Vérification** : un fil d'arrière-plan lit la dernière release du dépôt des releases (`AJP_RELEASE_REPO`)
   au plus une fois par jour, jamais pendant une lecture. Seule donnée envoyée : la requête HTTPS (pas de
   télémétrie).
2. **Intégrité** : le lien de la release est confirmé par SHA-256 (déjà dans `engines.lock`) et, en plus, une
   **signature Ed25519** d'un manifeste (clé publique compilée dans le lecteur) : un hachage seul ne protège pas
   d'une release remplacée.
3. **Installation** : téléchargée à côté, appliquée **au prochain lancement** (jamais en pleine lecture) ;
   l'ancienne version est gardée pour le retour arrière. Une mise à jour ratée laisse l'ancienne en place.
4. **Selon l'installation** : tarball Linux et zip Windows/macOS : AJP se met à jour lui-même ; paquet de
   distribution, Flatpak, AppImage : réglage grisé avec « mis à jour par votre gestionnaire de paquets »
   (AppImage : AppImageUpdate à voir plus tard).
5. **Moteurs et modèles** : déjà versionnés par leur index (`engines.lock`) ; la mise à jour du lecteur ne les
   retélécharge que si le hachage change.
6. Défaut proposé : **Automatique**, avec un message à la première fois (Sparkle fait pareil : demande au 2e
   lancement). Question 5 plus bas.

### 4.6 Textes à corriger tout de suite (liste non exhaustive)
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

## 5. Les autres points de la 0.4

- **Hermite en réduction** : le filtre de réduction a 6 choix (Area, Spline36, Bilinear, Bicubic, Mitchell,
  Catmull-Rom) ; **Hermite n'y est pas**. Hermite est un cubique (B=0, C=0), net sans halo : bon candidat pour
  l'anime. Mitchell et Catmull-Rom sont déjà des cubiques paramétrés (B, C) dans les noyaux CUDA et la voie
  Vulkan : un jeu de paramètres de plus (`DownscaleFilter::Hermite`, clé `hermite`, pastille *Anime*), à tester
  contre Spline36 et Mitchell (PSNR et regard) avant d'en faire un défaut.
- **GMFSS, amélioration** : à définir avec toi. Pistes mesurées : (a) le mode qualité coûte 7 à 10 ms ; (b) la
  fusion pèse 15 à 16 ms, dont 82 % de convolutions ; (c) sortir les warps et splats du graphe, comme pour
  DistilDRBA (76 % de son graphe à 2160p).
- **Mode auto, amélioration** : à reprendre avec le coordinateur (qui l'a écrit) : réglages, repli, retour sur
  « plus de ×4 demande un écran rapide ».

## 6. Étapes proposées

1. **Textes et composants** (organisation inchangée) : 720p, vocabulaire, `badge()`, infobulles, retrait des
   paragraphes.
2. **Mécanique** : barre d'onglets, barre de défilement, largeurs, « Avancé » replié avec mémoire.
3. **Réorganisation** : 7 onglets, Profils retiré, déplacement des réglages, bouton lecteur par défaut,
   inscription « Ouvrir avec » au premier lancement. Aucune clé de `play.toml` ne change (sauf `advanced_open`,
   `open_with_registered`, `ask_default_player`), donc aucune migration.
4. **Mise à jour automatique** : vérification + réglages d'abord (sans installation), puis téléchargement signé
   et application au lancement, test sur Linux puis Windows. C'est un chantier à part (signature, clé, format du
   manifeste) : ne pas le mélanger à l'étape 3.
5. **Passe de relecture** : infobulles relues d'un trait ; essai à 100 %, 150 % et 200 % d'échelle, sur 4K et
   1366×768.

Vérifications : tests existants de `settings_test` et du panneau, instantané du menu avant/après (`--snapshot`)
par onglet, contrôle que chaque réglage de l'annexe A reste atteignable.

## 7. Questions pour toi
1. Les 7 onglets (Général, IA, HDR, Couleurs, Audio et sous-titres, Export, Avancé) te conviennent ?
2. Les réglages de langue : dans Général (les deux préférences) et le détail dans « Audio et sous-titres » : ok ?
3. Pastilles *Recommandé / Le plus rapide / Le plus fin / Le plus lourd / Expérimental* : la liste te convient ?
4. Les temps mesurés : jamais dans l'interface, oui dans `docs/nvidia-linux.md` ?
5. Mises à jour : défaut **Automatique** (installe au prochain lancement) ou **Me prévenir** ? Canal
   Préversions proposé seulement dans Avancé.
6. Pour GMFSS et le mode auto de la 0.4 : qu'est-ce que tu veux améliorer en priorité ?

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
