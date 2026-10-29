# Menu des paramètres : arbre actuel et plan de tri (0.4)

État du code à `0b60fda` (v0.3.0). Source de l'inventaire : `src/app/play/settings_panel.cpp`,
`settings_chain_tabs.cpp`, `settings_engines.cpp`, `settings_export.cpp`, `settings_system.cpp`
(extraction automatique des widgets ImGui, puis relecture). Ce document est le plan unifié (arbre, encadrés en ligne,
configuration automatique, lecteur par défaut, mise à jour). L'implémentation avance par petites PR :
(a) détection et tranche, (b) préréglages, (c) benchmark, (d) textes et pastilles, (e) cadre en arbre.

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
| PotPlayer | **Vu sur ta capture** : arbre de catégories et sous-catégories à gauche, pages en groupes encadrés titrés, plusieurs champs sur une même ligne (résolution largeur × hauteur + cadence) | Retenu : arbre + groupes + lignes multiples. Profondeur limitée à 2 niveaux chez nous |
| IINA, MPC-HC/BE | *non vérifié* (pas de source exploitable trouvée) | — |

Conclusions qui tiennent sur les lecteurs vérifiés :
1. **Peu de rubriques au premier niveau** (7 à 9, dans un arbre et non en onglets) et une rubrique « Général » en tête.
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

### 4.2 Arborescence cible : un arbre de catégories à gauche, des groupes à droite

Retour du 28/10 (capture des réglages de PotPlayer) : l'arbre y est profond, mais on s'y retrouve mieux que dans
12 onglets. AJP a bien moins de réglages : **une colonne d'arbre (catégories et sous-catégories) à gauche, le
contenu à droite**, comme PotPlayer, VLC (vue « Toutes ») ou Kodi. La barre d'onglets disparaît donc : plus de
flèches, plus de défilement horizontal à gérer, et le problème de molette n'existe plus.

```
Général
  ├ Configuration            sélecteur de préréglage EN TÊTE (Auto, Économe, Rapide, Normal, Qualité, Ultra,
  │                          Personnalisé), priorité (Fluidité / Équilibre / Image), « Évaluer ma machine »
  │                          (dernier résultat et date), carte et écran détectés (voir 4.7)
  ├ Lecture                  volume, épisode suivant, reprendre là où on s'est arrêté
  ├ Lecteur par défaut       bouton « Faire de ce lecteur le lecteur par défaut », Ouvrir avec  (voir 4.4)
  └ Mises à jour             Automatique / Me prévenir / Jamais, canal, fréquence, version (voir 4.5)
Image IA
  ├ Agrandissement           résolution de travail, SPAN (case, modèle) ; en Réglages avancés : « Préparer le
  │                          mode Performance (SPAN FP8) » et « Préparer en tâche de fond » (docs/span-fp8-cutlass.md)
  ├ Fluidité                 multiplicateur, moteur d'interpolation, mode GMFSS
  └ Avancé                   filtre de réduction (Hermite…), modèle RIFE, poses DRBA, ordre du traitement, Smooth Motion
HDR
  ├ Conversion               moteur HDR, pic de luminance, débandage
  └ Écran                    noir absolu OLED, soins OLED, gris moyen
Couleurs
  ├ Réglages                 grade (case), préréglage, luminosité, contraste, saturation
  └ Avancé                   gain R/V/B, gamma, gamut P3, seuil du noir
Audio
  └ Sortie                   délai audio, caler la lecture sur l'écran, latence de l'écran
Langues et sous-titres
  ├ Langues préférées        audio, sous-titres (« Aucun »)
  └ Sous-titres              taille, piste
Export
  ├ Fichier                  taille, codec, qualité, destination, bouton Exporter
  └ Traitement               agrandissement et interpolation de l'export
Moteurs
  ├ Installés                moteurs et modèles, installer, reconstruire
  └ Choix                    moteur de SPAN, de RIFE
Avancé
  ├ Affichage                statistiques à l'écran (Tab), cadence fixe (VRR), filtre vers la fenêtre, masquer la barre en pause
  ├ Lecteur par défaut       (réapparaît ici si l'utilisateur a refusé, voir 4.4), « Retirer de Ouvrir avec »
  └ Réinitialiser            retour au profil d'origine, réglages d'usine
```

- **9 catégories**, 2 à 4 sous-catégories chacune. Les langues, l'export et les moteurs ont leur catégorie
  (demande du 28/10) ; rien n'est caché dans un onglet fourre-tout.
- **Profils** : l'écran disparaît pour l'instant (demande du 28/10). Les 9 profils et le profil de démarrage
  restent dans `play.toml` et dans le code ; « retour au profil d'origine » passe dans Avancé > Réinitialiser.
- **Image IA** : par défaut (préréglage choisi dans *Général > Configuration*, 4.7) les pages Agrandissement et Fluidité
  sont **en lecture** (valeurs choisies par l'automatique, grisées avec « Auto »), et se débloquent en
  *Personnalisé*. Elle regroupe agrandissement et interpolation : une seule chaîne (la résolution de travail décide ce que
  l'interpolation peut tenir), avec le résumé vivant de la chaîne en haut de la catégorie.
- **HDR et Couleurs séparés** : HDR configure la conversion et l'écran ; Couleurs ne contient que l'image. Le
  filtre de réduction et la mise à l'échelle n'y sont plus.
- **Recherche** (idée mpv.net, à faire si le temps le permet) : un champ en haut de l'arbre qui filtre les
  réglages par nom.

**Disposition d'une page (ce que PotPlayer fait bien)** : des **groupes encadrés avec un titre** (« Agrandissement »,
« Fluidité »…), et dans un groupe, **plusieurs réglages sur une même ligne quand ils vont ensemble** : résolution
de travail + filtre, pic de luminance + gris moyen, taille de l'export largeur × hauteur, audio + sous-titres côte
à côte, curseurs de couleur en deux colonnes. Règle : on met côte à côte ce qui se lit comme une phrase ou une
paire (largeur × hauteur, avant / après), jamais deux réglages sans rapport pour remplir la place. Une ligne
verticale unique reste la norme pour le reste.

### 4.3 Mécanique de l'interface (à corriger dans le code)
- **Fenêtre** : plus large (environ 980 px à 100 %, 220 px d'arbre + contenu) et bornée à l'écran, minimum 640 px ;
  sur 1366×768 l'arbre se replie en une liste déroulante en haut. Plus de `BeginTabBar("##sections")`.
- **Arbre** : `ImGui::BeginChild` à gauche, `TreeNodeEx` pour les catégories (ouvertes par défaut sur la
  catégorie courante), `Selectable` pour les sous-catégories ; navigation au clavier (haut/bas, gauche/droite),
  dernière page mémorisée (`[ui] last_page`).
- **Groupes** : `BeginChild(..., ImGuiChildFlags_Borders)` avec le titre par-dessus la bordure, dans
  `panel_style.h` (`group("Agrandissement")`) ; lignes multiples avec `BeginTable` à 2 ou 3 colonnes sans bordure.
- **Barre de défilement du contenu** : fine (8 px), arrondie, translucide, au bord extérieur avec 6 px de marge
  (ne mord pas sur les réglages), visible au survol ou au défilement.
- **Largeur des champs** : 45 % libellé à gauche, valeur à droite, alignés d'un groupe à l'autre.
- **Infobulles** : `ImGui::SetItemTooltip` avec délai ; les `note()` actuelles y sont déplacées, raccourcies.
- **Pastilles** : `badge("Recommandé")` dans `panel_style.h` ; vert recommandé, bleu rapide, ambre lourd, violet
  expérimental.
- **Avancé** : les réglages avancés ont leur sous-catégorie « Avancé » dans l'arbre (au lieu d'un repli dans la
  page), ce qui évite un état de repli à mémoriser.

### 4.2 bis Variante du coordinateur et choix du niveau « avancé »

Le coordinateur propose 8 catégories au plus : **Général** (Comportement, Langues, Mises à jour, Système),
**Lecture** (Audio, Sous-titres, Affichage), **Image et IA** (Agrandissement, Interpolation, Rendu), **HDR et
couleurs**, **Export**, **Moteurs**, avec une case **« Réglages avancés »** au pied de l'arbre et la **recherche**
en haut. (Je n'ai pas pu relire son fichier, `/mnt/project-files/ux/…` n'existe pas sur ce PC : je ne m'appuie que
sur ce résumé.) Différences avec l'arbre ci-dessus, à trancher avec toi :

| Point | Mon arbre (4.2) | Variante coordinateur |
|---|---|---|
| HDR / Couleurs | deux catégories (ta demande du 28/10 : « un onglet HDR, un onglet couleurs ») | une seule catégorie « HDR et couleurs » |
| Langues | avec les sous-titres, catégorie propre | dans Général |
| Audio, sous-titres, affichage | « Audio », « Langues et sous-titres », « Avancé > Affichage » | une catégorie « Lecture » |
| Avancé | sous-catégorie « Avancé » par catégorie | **une case globale** « Réglages avancés » : coché, chaque page affiche en plus son groupe « Avancé » |

Je recommande de **garder HDR et Couleurs séparés** (tu l'as demandé) et d'**adopter la case globale
« Réglages avancés »** : c'est le niveau Basique/Avancé de Kodi, plus simple qu'une sous-catégorie par page et
ça désencombre l'arbre pour l'utilisateur courant (les sous-catégories « Avancé » disparaissent, leurs
réglages passent dans un groupe « Avancé » de la page, visible seulement case cochée). Reste : Langues dans
Général (le coordinateur) ou avec les sous-titres (moi) : question 2.

### 4.3 bis Ce qui est réaliste avec Dear ImGui, et ce qui ne l'est pas

| Élément | Verdict |
|---|---|
| Arbre à gauche, page à droite | **Faisable** : deux `BeginChild` côte à côte, `TreeNodeEx` / `Selectable`. Largeur d'arbre fixe |
| Encadrés titrés | **Faisable** : `BeginChild` avec `ImGuiChildFlags_Borders`, titre dessiné par-dessus la bordure (`ImDrawList`). La hauteur doit être donnée : `ImGuiChildFlags_AutoResizeY` la calcule |
| Réglages en ligne, plusieurs colonnes | **Faisable** avec `BeginTable` (2 à 3 colonnes, sans bordure). Pas d'alignement automatique entre encadrés : on fixe les largeurs de libellé |
| Pastilles | **Faisable** : texte coloré sur un rectangle arrondi (`ImDrawList::AddRectFilled`) |
| Infobulles | **Faisable** : `ImGui::SetItemTooltip`. Un survol sur un champ **désactivé** a besoin de `ImGuiHoveredFlags_AllowWhenDisabled` |
| Recherche | **Faisable mais à cadrer** : il faut un **registre des réglages** (nom, catégorie, mots-clés) pour filtrer et sauter à la page. Les réglages sont aujourd'hui dessinés en dur dans des fonctions : le registre est un vrai chantier. Version 1 : filtrer les pages de l'arbre par mot-clé, sans filtrer les lignes |
| État mémorisé | **Faisable** : dernière page et « Réglages avancés » dans `play.toml` ; ImGui n'a pas de mémoire d'arbre persistante, donc on l'écrit nous-mêmes |
| Navigation au clavier / manette | Partielle : ImGui navigue au clavier (`ImGuiConfigFlags_NavEnableKeyboard`), mais l'ordre sur un arbre + page demande des tests |
| Accessibilité (lecteur d'écran) | **Non** : ImGui ne l'expose pas ; hors périmètre |
| Défilement horizontal de la molette | Sans objet : plus de barre d'onglets |
| Redimensionnement | Fenêtre fixe (`NoResize`) conservée ; sous 900 px de large l'arbre devient une liste déroulante en haut |

Aucun de ces éléments n'est irréaliste ; seul le **filtrage fin de la recherche** (ligne par ligne) l'est sans
registre. Ce serait une étape 6, après le reste.

### 4.3 ter Parcours du premier lancement (tout ensemble)
1. L'inscription « Ouvrir avec » se fait toute seule (4.4).
2. AJP détecte la carte, l'écran et choisit une tranche (4.7) : le préréglage est **Auto**.
3. Un seul écran d'accueil : résumé « RTX 4070 Ti · 4K 165 Hz → 720p → SPAN ×2 → DistilDRBA lite », bouton
   **« Évaluer ma machine (15 s) »** (ou *Plus tard*), choix de la **mise à jour** (4.5) et, sur Windows,
   « Faire de ce lecteur le lecteur par défaut ».
4. Tout se retrouve ensuite dans les réglages : Général (lecteur par défaut, mises à jour), Général > Configuration
   (préréglage, benchmark).

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
6. Défaut proposé : **Me prévenir** (à confirmer, question 5), choisi sur l'écran du premier lancement (Sparkle
   demande aussi une fois).

### 4.8 Assistant de premier démarrage (« first-run wizard »)

Il fait évoluer l'écran de bienvenue actuel (`first_run.cpp` : GPU vu par Vulkan, moteurs proposés avec badges,
profil de démarrage ; `--first-run` le remontre). Plusieurs **écrans guidés, tous sautables** (« Plus tard »),
**refaisables depuis Général** (bouton « Relancer l'assistant ») et par `ajp --first-run`. Barre de progression,
« Précédent / Suivant / Passer », résumé final. Rien ne s'applique avant « Terminer » (sauf une installation de
moteur lancée explicitement). État mémorisé dans `settings.toml` : `first_run_done` (existe) et
`first_run_step` (le dernier écran vu, pour reprendre si on ferme en route).

| # | Écran | Contenu | Condition d'affichage |
|---|---|---|---|
| 1 | **Bienvenue et détection** | carte, tranche (4.7), VRAM, Resizable BAR, écran (résolution, fréquence, VRR, HDR) ; lecteur par défaut et « Ouvrir avec » sont faits sans question (4.4), l'écran le dit | toujours |
| 2 | **Moteur** | Vulkan (toujours), TensorRT (NVIDIA), MIGraphX (AMD), OpenVINO (Intel), Core ML (Apple), Windows ML : ce que chacun apporte (une ligne + infobulle), ce qui est déjà installé, **recommandé** présélectionné, installation avec progression (téléchargement direct, `engine_install`) | toujours ; les moteurs absents de l'OS ne sont pas listés |
| 3 | **Image et fluidité** | préréglage (Auto par défaut), priorité **Fluidité / Équilibre / Image**, résumé de la chaîne choisie, « Évaluer ma machine (15 s) » ou « Plus tard » | toujours |
| 4 | **HDR** | détection de l'écran (HDR actif ? crête ?) ; choix **RTX HDR / HDR adaptatif / SDR** selon ce qui est possible ; si l'écran n'est pas HDR, l'écran est sauté ou propose SDR seul | HDR : écran HDR détecté ou choix manuel ; RTX HDR : NVIDIA seulement |
| 4b | **RTX HDR pas à pas** (Linux) | n'apparaît que si RTX HDR est choisi : vérifie le moteur TensorRT (`tensorrt_backend_present`), le SDK Maxine / VideoFX (`maxine_available`, importé ou à importer), le pilote ; chaque point en « fait / à faire » avec le geste exact à faire ; « Vérifier de nouveau » ; repli sur HDR adaptatif si l'utilisateur renonce | Linux et NVIDIA ; sous Windows RTX HDR est automatique |
| 5 | **Audio** | **appareil de sortie** (liste des sorties du système, celle par défaut présélectionnée) ; **format** : détecté, affiché (« 5.1 · 6 canaux »), jamais demandé ; puis **« Comment écoutez-vous ? »** Casque / Enceintes / Home cinéma (4.9). Home cinéma grisé avec l'explication si l'appareil n'a que 2 canaux ; ignoré : Enceintes en rendu fidèle. **Aucune détection** casque ou enceinte | toujours |
| 6 | **Langues** | audio et sous-titres préférés (une liste de préréglages, « Aucun »), taille des sous-titres | toujours |
| 7 | **Mises à jour** | Me prévenir / Automatique / Jamais (défaut : Me prévenir, à confirmer), canal Stable | toujours (grisé pour Flatpak ou paquet de distribution, 4.5) |
| 8 | **Résumé** | ce qui a été choisi, ce qui a été sauté, où le changer (« Réglages > Général > Configuration » etc.), « Terminer » | toujours |

Conditions par système : Linux (moteurs Vulkan et TensorRT, RTX HDR guidé, `.desktop`), Windows (Windows ML,
RTX HDR direct, page « Applications par défaut »), macOS (Core ML, pas de moteur à choisir : l'écran 2 saute).
Une carte sans moteur supplémentaire possible n'a pas d'écran 2 ; un écran SDR n'a pas d'écran 4.

**Réserves, dites ici pour ne rien promettre à tort** :
- L'écran 5 demande un **petit travail de la couche audio**, vérifié dans le code (4.9) : le lecteur ouvre
  aujourd'hui toujours l'appareil par défaut, sans liste d'appareils ni relecture quand le défaut change. Le
  multicanal (jusqu'à 8 canaux) et le remixage de base existent déjà. Le **rendu** (crossfeed, binaural, downmix
  soigné) est défini dans `docs/audio-spatialisation.md` ; l'assistant ne fait que **poser le réglage**.
- L'écran 4b s'appuie sur les vérifications du code existant (Maxine, TensorRT) ; la liste exacte des points à
  contrôler et leurs gestes se fixera en lisant `maxine_install` et `vendor_hdr` au moment de l'écrire, pas avant.

### 4.9 Audio : appareil, format et « Comment écoutez-vous ? »

Décision de Tatoub (résumé dans `docs/audio-spatialisation.md`) : **pas de détection casque ou enceintes**.

**Automatique, sans question** : format de la source par piste (mono, stéréo, 5.1, 7.1, passthrough) et format de la
sortie = canaux de l'appareil par défaut, relus quand il change. Le lecteur choisit seul la conversion (mono → stéréo,
2.0 → mono, 5.1 ou 7.1 → stéréo avec centre rehaussé, 5.1 ↔ 7.1, direct si identiques, contournement du passthrough).

**Une seule question, au premier lancement** (écran 5 de l'assistant) : Casque (crossfeed en 2.0, binaural en
surround), Enceintes (rendu fidèle, downmix soigné), Home cinéma (multicanal, surround direct). Passer = Enceintes en
rendu fidèle.

**Dans les réglages** (catégorie *Audio*, sous-catégorie *Sortie*) :
- groupe **Appareil** : liste des sorties, « Suivre l'appareil par défaut du système » (case) ;
- groupe **Écoute** : Casque / Enceintes / Home cinéma (Home cinéma grisé si l'appareil n'a que 2 canaux, avec
  infobulle), **Fidèle** (aucun traitement), « Profil de cet appareil » (global ou propre à l'appareil, mémorisé par
  identifiant ; un appareil inconnu reprend le global sans question) ;
- groupe **Synchronisation** : délai audio, caler la lecture sur l'écran, latence de l'écran (existants, en Avancé) ;
- la touche A/B (traité / fidèle) reste. Source ou appareil mono : aucune spatialisation, les choix sont grisés.

**Ce qui existe et ce qui manque** (code relu) : existent le multicanal jusqu'à 8 canaux et le remixage libswresample
sur l'appareil par défaut. Manquent : liste des appareils et choix d'un autre, relecture au changement du défaut,
profils par appareil, crossfeed, binaural, downmix à centre rehaussé. **Périmètre v1** (recommandation de
l'analyse) : option et non pilier ; casque d'abord (crossfeed + binaural) ; enceintes = downmix soigné ; **reporter**
l'upmix 5.1 « intelligent » et les traitements dédiés aux voix (risque de dégrader les voix).

Étapes : (1) couche audio : lister et choisir l'appareil, événements de changement ; (2) détecteur de formats et
conversion automatique ; (3) profils Casque / Enceintes / Home cinéma (crossfeed, downmix) ; (4) binaural ;
(5) écran de l'assistant et page Audio. Ni (3) ni (4) ne passent avant (1) et (2).

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

### 4.7 Configuration automatique (idée du 28/10, intégrée à la refonte)

**But** : l'utilisateur ne règle rien. AJP reconnaît la machine, choisit la chaîne, et tout reste modifiable en
détail (mode *Personnalisé*, case *Réglages avancés*). Elle remplace l'onglet Profils retiré : les profils
reviennent comme **préréglages** d'une seule ligne en tête de la catégorie *Image IA*.

**Ce qu'AJP sait déjà** (pas à inventer) : `GpuFacts` (nom, fabricant, VRAM, VRAM visible, cœurs Tensor, WMMA),
`apple_model`, `display_edid`, `host_facts`, `ChainEngines` et le mode Auto du multiplicateur (`rife_auto`), le
catalogue RIFE (4.22 lite, 4.25 lite…), les 9 profils actuels, les statistiques d'image par étage (`--stats`).

#### Entrées
- **Carte** : fabricant, génération, gamme, VRAM, iGPU ou dédiée, cœurs matriciels. Ce sont des **règles par
  famille** et non un classement carte par carte (voir « Le fichier de classement »).
- **Écran** : résolution et fréquence (EDID, VRR), HDR ou non.
- **Vidéo en cours** : résolution et cadence de la source. L'automatique se recalcule **par vidéo** : une source
  60 i/s n'est pas interpolée ; une source 480p n'est pas réduite.
- **Priorité** (réglage de l'utilisateur) : *Fluidité* (cadence d'abord), *Équilibre*, *Image* (résolution et
  finesse d'abord).

#### Tranches (à confirmer par mesure, pas figées)
| Tranche | Exemples | Agrandissement | Interpolation (multiplicateur : toujours Auto) |
|---|---|---|---|
| **0 · Économe** | iGPU anciens, cartes d'avant 2016 | aucun : on ne réduit pas, on n'agrandit pas | aucune, ou RIFE 4.22 lite ×2 si la mesure le permet |
| **1 · Léger** | iGPU récents, GTX 10/16, RX 400 à 5000 | SPAN, travail 540p → 1080p | RIFE 4.22 lite |
| **2 · Normal** | RTX 2060 à 3060, RX 6600 à 7600, Arc, Apple M | SPAN, 720p → 1440p ou écran | RIFE 4.25 lite |
| **3 · Rapide** | RTX 3070 à 4070, RX 7800 | SPAN, 720p → 1440p | DistilDRBA lite |
| **4 · Ultra** | RTX 4080 et plus, RX 7900 | SPAN, 720p → 1440p ou 2160p | GMFSS Union (mode Performance) |

Valeurs communes : modèle SPAN **Sharp (performance)** partout (ta remarque : pas de différence nette avec
Balanced, et Sharp convient mieux à l'anime), filtre de réduction **Mitchell** (ou **Hermite** une fois ajouté),
**Bilinear/Area** en tranche 0 si le coût compte, DistilDRBA **poses** activées si la mesure confirme (voir
« Points à vérifier »).

#### Résolution : réduire avant d'agrandir
Ton constat (720p → 2K, et 540p → 1080p plus net que le 1080p natif d'un anime) devient une règle :
la **résolution de travail** est choisie d'après **l'écran** et la **tranche** : écran 1080p et carte moyenne →
travail 540p, SPAN ×2 vers 1080p ; écran 1440p ou 4K → travail 720p, SPAN ×2 vers 1440p, puis mise à
l'échelle finale. Tranche 0 sur un écran 1080p : on ne touche à rien. *Priorité Image* monte d'un cran la
résolution de travail, *Fluidité* la baisse d'un cran. C'est ton observation, pas encore une mesure : avant de
la fixer en défaut, je l'essaie sur 3 ou 4 épisodes avec des instantanés (`--snapshot`) pour t'en montrer
l'effet.

#### Le fichier de classement
- Un fichier de données, `data/gpu_tiers.toml`, **versionné avec le lecteur** et rafraîchi par la mise à jour
  automatique (4.5) sans changer de version du lecteur.
- **Des règles par famille, pas 1 000 lignes** : par exemple `pattern = "RTX 40"`, `base = 3`, ajustée par la
  gamme (`Ti`, `Super`, `XT`, `X`, nombre) ; `Arc A`, `RX 7000`, `Radeon 7x0M`, `Apple M1…M4` (+ Pro / Max /
  Ultra), `Iris Xe`, `UHD`. Une centaine de règles couvrent 2015 à aujourd'hui avec moins d'erreurs que
  des milliers de lignes écrites à la main.
- **Carte inconnue** (nouvelle génération, nom étrange) : on retombe sur une estimation par fabricant, VRAM et
  cœurs matriciels, puis le benchmark corrige.
- **Honnêteté sur les chiffres** : je n'ai mesuré que sur la RTX 4070 Ti. Les autres tranches s'appuient sur
  la puissance publique des cartes et sur les coûts mesurés de chaque moteur : ce sont des **estimations**. C'est
  la raison du benchmark ci-dessous, et de laisser la tranche modifiable.

#### Optimiser pour ma machine (ex-« Évaluer ma machine ») : un bouton, une phrase
- **Un seul bouton**, en tête de Général > Configuration : **« Optimiser pour ma machine »**. Le lecteur joue
  quelques secondes de vidéo (10 s par essai, fenêtre masquée et son coupé) avec ses réglages recommandés, du plus
  léger au plus lourd, garde ceux que le PC tient sans saccade et **les applique tout de suite** (préréglage Auto
  sur la mesure ; la chaîne se rouvre à la même position). Aucun mot « tranche » ni « niveau » à l'écran.
- **Pendant l'essai** : une ligne d'étape lisible (« Étape 2 sur 4 : test des réglages « Rapide » »), une barre
  de progression globale, un bouton **Annuler** ; l'interface ne gèle jamais (les essais sont des processus
  enfants, la barre avance même en pause). Les quatre étapes : 1 préparation des moteurs et première mesure ;
  2 test des réglages ; 3 recherche d'une image plus fine (hauteur de travail, modèle RIFE) ; 4 préparation du
  mode Performance de SPAN (FP8) quand la carte et le moteur le permettent et qu'une vidéo est ouverte.
- **Durée** : environ une minute quand les moteurs sont déjà construits. La toute première fois, le premier essai
  construit les moteurs (un SPAN seul prend 2 min 30 sur une RTX 4070 Ti, plus sur un portable) : il a un délai de
  20 minutes, annulable, avec une note qui le dit ; les essais d'après ont 4 minutes (au-delà, les réglages sont
  « non testés »), la recherche fine 25 s (elle ne construit rien, et elle est remise à la fois suivante si les
  moteurs viennent d'être construits).
- **Résultat en une phrase** : « Votre PC tient : agrandissement ×2 depuis 720p, interpolation jusqu'à ×3, RIFE.
  Performance FP8 pour SPAN : prête / non — <pourquoi et que faire> ». Le rapport technique (essais, pertes,
  retard p95) va au journal et à `ajp bench`, jamais dans l'interface.
- **Annuler ces réglages** : restaure le préréglage et la priorité d'avant et remet `bench.toml` comme il était
  (ou le retire).
- **Un essai qui échoue est toujours expliqué** : « L'essai des réglages « Rapide » n'a rien rendu : <dernière
  ligne d'erreur de l'enfant> — le lecteur de l'essai s'est arrêté (code 3) » ; le journal garde le code de sortie
  et les 12 dernières lignes de l'enfant. Un délai dépassé le dit (construction d'un moteur).
- **Fenêtre des essais** : minuscule (16×9) et sans bordure, sauf sous GNOME / Wayland où le masquage n'est pas
  fiable (le compositeur ne rend plus la main à la fenêtre) : elle y reste visible, petite (320×180), sans plein
  écran.
- **Mode Performance de SPAN (FP8)** : le cran « Performance » de « Précision de SPAN » est offert dès que le moteur
  TensorRT annonce les noyaux CUTLASS (RTX 40, Ada 8.9, portables compris). Choisi, le lecteur fabrique seul le
  paquet de la vidéo (~40 s, « Préparation du mode Performance… n % » à l'écran, lecture en FP16 pendant ce
  temps), rouvre la chaîne à la fin, ou dit en une phrase pourquoi c'est impossible (moteur TensorRT non installé,
  carte non Ada, refus des garde-fous avec les dB). « Refaire la préparation sur cette vidéo » reste en réglages
  avancés.
- Au premier lancement, l'assistant (écran 3) renvoie à ce bouton, qui demande des moteurs installés.
- Hors ligne, aucune donnée envoyée. Un PC chaud ou occupé donne un résultat plus bas : relancez.

#### Préréglages (remplacent l'onglet Profils)
Liste en tête de Général > Configuration : *Auto* (par défaut), *Économe*, *Rapide*, *Normal*, *Qualité*, *Ultra*, *Personnalisé*. Modifier un réglage
passe en *Personnalisé*, à partir des valeurs courantes. Plus de préréglages ne servent à rien tant qu'on n'a
pas de cas d'usage précis : *Sans interpolation* et *HDR* se règlent par catégorie, pas par profil.

#### Quand l'automatique se recalcule
Au premier lancement ; à chaque changement de carte, d'écran ou de pilote ; à chaque nouvelle vidéo (cadence et
résolution) ; jamais en pleine lecture. Un utilisateur en *Personnalisé* n'est jamais écrasé.

#### Points à vérifier avant de les fixer
1. DistilDRBA lite contre RIFE 4.22 et 4.25 : le rendu « meilleur » est ton impression, je la teste (coût
   mesuré, instantanés côte à côte) avant d'en faire le défaut de la tranche 3.
2. DRBA **poses** : activée par défaut seulement si le coût et la fidélité le justifient ; je mesure.
3. Le coût de Mitchell / Hermite sur iGPU ancien : probablement négligeable devant l'IA, à confirmer.
4. GMFSS Union (et pas base) : d'accord, les mesures vont dans ce sens (même coût, meilleurs résultats).
5. Une tranche trop optimiste = saccades : le benchmark et la baisse automatique de la tranche (le mode Auto
   fait déjà descendre le multiplicateur) servent de filet.

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

## 6. Étapes proposées (ordre de l'implémentation, une PR par étape)

1. **(a) Classement des cartes** (4.7) : `data/gpu_tiers.toml`, tranche détectée, sans effet. *PR #22.*
2. **(b) Préréglages et application** : `Preset` / `Priority`, `apply_preset`, clés `[auto]` de `play.toml`,
   appliqué à chaque ouverture de vidéo ; Personnalisé jamais écrasé.
3. **(c) Benchmark** : « Évaluer ma machine », résultat mémorisé par carte, pilote et version du moteur.
4. **(c bis) Assistant de premier démarrage** (4.8) : écrans 1 à 3 d'abord (détection, moteur, image et
   fluidité, benchmark), puis HDR et RTX HDR guidé, audio (après le petit travail de la couche audio, 4.9),
   langues, mises à jour, résumé ; « Relancer l'assistant » dans Général.
5. **(d) Textes, pastilles, infobulles** : 720p, vocabulaire, `badge()`, retrait des paragraphes.
6. **(e) Nouveau cadre en arbre** : arbre à gauche, groupes encadrés et lignes multiples, 9 catégories, Profils
   retiré, case « Réglages avancés », bouton lecteur par défaut. Les clés nouvelles : `last_page`,
   `advanced_shown`, `open_with_registered`, `ask_default_player`, `first_run_step`.
7. **Mise à jour automatique** : vérification et réglages d'abord (sans installation), puis téléchargement signé
   et application au lancement, essai Linux puis Windows. Chantier à part (signature, clé, manifeste).
8. **Recherche** : registre des réglages, filtre par page puis par ligne (4.3 bis).
9. **Passe de relecture** : infobulles relues d'un trait ; essai à 100 %, 150 % et 200 % d'échelle, sur 4K et
   1366×768.

Chaque étape se teste localement (build, tests, snapshots du menu avant/après) puis un seul envoi à la fin, comme
d'habitude. Les étapes 2 et 3 sont indépendantes ; la 4 dépend des deux.

Vérifications : tests existants de `settings_test` et du panneau, instantané du menu avant/après (`--snapshot`)
par onglet, contrôle que chaque réglage de l'annexe A reste atteignable.

## 7. Questions pour toi
1. L'arbre à gauche (9 catégories, 2 à 3 sous-catégories) à la place des onglets : c'est bien ce que tu veux ?
2. HDR et Couleurs séparés (mon arbre, ta demande) ou fusionnés (coordinateur) ? Langues dans Général ou avec les sous-titres ? Case globale « Réglages avancés » (4.2 bis) : ok ?
3. Pastilles *Recommandé / Le plus rapide / Le plus fin / Le plus lourd / Expérimental* : la liste te convient ?
4. Les temps mesurés : jamais dans l'interface, oui dans `docs/nvidia-linux.md` ?
5. Mises à jour : défaut **Me prévenir** (avis du coordinateur, à confirmer) ou **Automatique** (installe au prochain lancement) ?
6. Pour GMFSS et le mode auto de la 0.4 : qu'est-ce que tu veux améliorer en priorité ?
7. Configuration automatique (4.7) : les 5 tranches et les préréglages te conviennent ? (Benchmark : décidé, proposé au début et refaisable dans les réglages.)
8. L'ordre des étapes (section 6) te convient ?
9. Assistant de premier démarrage (4.8) : les 8 écrans te conviennent ? L'écran audio demande d'abord un petit travail de la couche audio (liste des appareils, relecture du défaut) : vérifié dans le code, voir 4.9.

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
    - case : RIFE fusionné (28/10 : six petits moteurs TensorRT et des noyaux CUDA à la place du RIFE entier, `docs/rife-fusionne.md`)
    · note (493 c.) : Coupe RIFE 4.25 ou 4.22 lite en six petits moteurs TensorRT reliés par des noyaux CUDA : une passe prend un tiers du temps…
    - case : DistilDRBA fusionné (28/10 : quatre petits moteurs TensorRT et des noyaux CUDA à la place du DistilDRBA entier, `docs/drba-fusionne.md`)
    · note (559 c.) : Coupe DistilDRBA en quatre petits moteurs TensorRT (les convolutions à basse résolution) reliés par des noyaux CUD…
    - case : GMFSS fusionné (29/10 : la paire en deux moteurs TensorRT, les caractéristiques d'une image gardées pour la paire suivante, `docs/gmfss-fusionne.md`)
    · note (468 c.) : Coupe la paire de GMFSS en deux moteurs TensorRT : les caractéristiques d'une image et le flot. Les caractér…
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

## Audit « Réglages avancés » (v0.6)

Règle posée par Tatoub (28/10) : un réglage que presque personne ne touche (synchro audio, algorithmes, ordre du traitement…)
n'est pas une fonction principale du lecteur : il n'apparaît que la case « Réglages avancés » cochée. Le défaut reste inchangé.

| Page | Visible d'office | Réglages avancés |
|---|---|---|
| Agrandissement | SPAN ×2, modèle, résolution de travail | algorithme de réduction, composition finale (filtre vers la fenêtre, débandage) |
| Fluidité | multiplicateur (dont Auto), moteur d'interpolation | modèle RIFE, GMFSS Performance/Qualité, garder les poses (DRBA), Warp et SPAN, Smooth Motion |
| HDR (Conversion) | moteur HDR, pic (automatique ou en nits) | gris moyen, débandage RTX HDR, valeurs par défaut, noir absolu OLED, soins OLED |
| Couleurs | contraste, saturation, grader la sortie et préréglage | les 6 curseurs du grade et le gamut P3 |
| Audio (Sortie) | appareil, écoute (casque, enceintes, home cinéma), épisode suivant | mode fidèle, profil propre à l'appareil, délai audio, latence de l'écran, calage sur l'écran |
| Langues, Sous-titres | tout | — |
| Général, Moteurs, Export | tout | — |
| Général | configuration, mises à jour, lecteur par défaut, statistiques, réinitialisation | (page lecteur par défaut revient avec les avancés si « Ne plus demander ») |

Mis en œuvre par `panel::advanced()` (`panel_style.h`), posé à chaque image par le panneau. À venir : déplacer composition finale et soins OLED
de Couleurs vers une page avancée « Rendu », OLED en automatique.

## Référence visuelle : GNOME Showtime (capture de Tatoub, 28/10)

La capture montre la fenêtre de Showtime (lecteur vidéo de GNOME) ; l'interface du lecteur s'en inspire pour la 0.6.

- **Fenêtre** : coins arrondis (~14 px), pas de barre de titre séparée : la vidéo occupe toute la fenêtre, les commandes sont superposées.
- **Haut** : à gauche, un bouton « plein écran » (flèches en diagonale) ; à droite, menu (trois traits), réduire, agrandir, fermer, en icônes blanches sans fond, sur un dégradé sombre très doux.
- **Centre** : trois commandes rondes sans cadre : retour de 10 s (flèche circulaire avec « 10 »), lecture/pause plus grande, avance de 10 s.
- **Bas** : le titre du fichier en gras blanc, aligné à gauche (les titres japonais exigent la police CJK, #59) ; à droite le volume et un engrenage (réglages) ; en dessous, une barre de lecture fine avec une poignée ronde blanche, le temps écoulé à gauche et la durée à droite en petits caractères gras ; le tout sur un dégradé sombre du bas.
- **Comportement** : l'interface disparaît à l'inactivité (curseur compris en plein écran) et revient au mouvement ; fond de pause jamais opaque.
- **Couleurs** : blanc sur voiles sombres translucides, un seul accent ; aucune bordure marquée.

À faire : reprendre ces proportions dans `controls_draw.cpp`, avec captures avant/après en fenêtre et en plein écran (voir l'item « interface » du plan v0.6).

Règle de Tatoub (28/10, 09:42) : les réglages avancés étendent ou complètent les onglets existants (masqués par défaut), ils ne créent pas de catégorie « Avancé » : l'arbre n'en a plus.
