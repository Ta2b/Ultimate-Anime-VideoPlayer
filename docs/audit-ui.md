# Audit esthétique et fonctionnel de l'interface (0.6.1)

Demande de Tatoub : « refaire un petit audit esthétique du lecteur, des menus : que aucune écriture ne dépasse, que
tout soit clair et simple, cohérence des catégories et sous-catégories, déplacer ou fusionner des sous-catégories si
nécessaire, vérifier que chaque bouton fonctionne et que le tout soit beau et esthétique ». Aucune fonctionnalité
n'est ajoutée : ce sont des textes, des largeurs, un regroupement de pages et des corrections de défauts.

## 1. Méthode

- **Captures** : `ajp play FICHIER --data-dir=D --window=LxH --keys=… --snapshot=… --snapshot-at=S` (copies des
  `settings.toml`, `user.toml`, `play.toml` de la machine de développement dans `D` : la vraie configuration n'est
  jamais touchée), sous `flock ~/ajp-reports/gpu-bench.lock`. Les pages s'atteignent à la molette sur l'arbre
  (`--keys=…:wheel=-1`, une page par cran) ; la case « Réglages avancés » se coche par `click=`.
- **Tailles** : fenêtre 1366×768 (le portable de Tatoub), 1280×720 et 1920×1080 ; échelle de l'interface 100 % et
  150 % (SDL sous X11 : `SDL_VIDEO_DRIVER=x11 SDL_VIDEO_X11_SCALING_FACTOR=1.5`, l'échelle d'ImGui suit
  `SDL_GetWindowDisplayScale`).
- **Boutons** : une exécution qui clique les contrôles d'une ou plusieurs pages (`--keys=…:click=XxY`), puis lit les
  fichiers de réglages de `D` (un réglage qui change = un effet) et le journal `D/ajp.log`. Les boutons qui touchent
  le système (lecteur par défaut, « Ouvrir avec ») tournent avec `XDG_DATA_HOME` et `XDG_CONFIG_HOME` dans `D` ; les
  sommes de `~/.config/mimeapps.list` et de `mimeinfo.cache` sont les mêmes avant et après.
- Captures « avant » : `/tmp/claude-1000/audit-ui/avant` (arbre à 14 pages), « après » : `…/apres`.

## 2. Défauts trouvés

| # | Défaut | Où | État |
|---|---|---|---|
| 1 | La petite fenêtre « Lecteur par défaut » du lancement recouvre le panneau et lui prend la molette (on ne peut plus changer de page) | écran de lecture, panneau ouvert | corrigé : la fenêtre attend que le panneau soit fermé |
| 2 | Plusieurs catégories ouvertes : l'arbre déborde et défile à 720 et 768 px de haut (les entrées du bas disparaissent) | arbre | corrigé : changer de catégorie replie les autres |
| 3 | Libellés de curseurs coupés à droite (« Couleurs étendues (large ga », « Niveau des arrières et côté », « Distance des enceintes virtu ») : les encadrés sont des fenêtres filles, la largeur des champs repartait de son défaut | Couleurs, Audio > Sortie | corrigé : 44 % de la largeur laissés aux libellés dans chaque encadré, libellés de 30 caractères ou plus raccourcis |
| 4 | Quatorze pages dont deux petites (Langues, Sous-titres) et deux masquables (« Lecteur par défaut » disparaissait avec « Ne plus me le demander ») | arbre | corrigé : douze pages (§ 3) |
| 5 | « Lire l'épisode suivant » rangé dans Audio > Sortie | Audio | corrigé : page Lecture |
| 6 | Page Moteurs : section « Moteurs installés » portant le nom de la page, dossier d'installation avant la liste, nom de moteur long qui sort du cadre | Export et moteurs | corrigé : ordre (en service, installés, construits, dossier), retour à la ligne, champ du dossier avec indication « Dossier par défaut » |
| 7 | Champs « Fichier » (export) et « Dossier » (moteurs) : libellé et bouton collés derrière le champ, débordement possible à grande échelle | Export, Moteurs | corrigé : champ pleine largeur, bouton à droite |
| 8 | Rangée de trois boutons du lecteur par défaut sur une seule ligne : sortait du cadre à 150 % | Application | corrigé : les boutons passent à la ligne (`same_line_if_fits`) |
| 9 | Bouton « Retirer » sans objet | Application | corrigé : « Retirer de « Ouvrir avec » » |
| 10 | Guillemets « » et deux-points qui commençaient une ligne (« plus bas) » après « Préparer le mode Performance ») | toutes les notes | corrigé : espaces insécables (`french_spaces`) |
| 11 | Contraste et saturation sans unité (« 200 ») | Couleurs | corrigé : « 200 % » |
| 12 | « 23.976 i/s » (point décimal) dans une phrase en français, « Grisés » au pluriel sans objet | Fluidité | corrigé |
| 13 | Barre de progression de l'assistant en jaune (couleur par défaut d'ImGui) au lieu de l'accent bleu | assistant | corrigé |
| 14 | Textes périmés : « onglet Grade », « onglet HDR », « dans l'onglet Système », « HDR > Conversion » dans le résumé de l'assistant, « (aucune dégradation) » | notes, assistant, message d'inscription | corrigés (les chemins du résumé sont vérifiés par un test) |
| 15 | Textes trop techniques ou chiffrés : « 11 ms par image sur une RTX 4070 Ti », chemins de scripts d'export, « même gain sur R, V et B », « maximum local 8x8 », note de la page Réinitialiser qui cite `user.toml` | Fluidité, HDR, Réinitialiser | raccourcis |
| 16 | Infobulle de précision de SPAN affichée aussi sur le modèle quand la liste des précisions est absente | Agrandissement | corrigé |
| 17 | Temps de la barre (position / durée · chapitre) pouvant passer sous les boutons de droite avec un titre de chapitre long | barre de lecture | corrigé : le titre est coupé par « … » avant les boutons |
| 18 | Les deux modes d'Auto s'appelaient « Auto (recommandé) » et « Auto libre (expérimental) » | Fluidité, OSD, journaux, docs | renommés : **Stable** et **Rapide (expérimental)** ; clé `play.toml` inchangée |

## 3. Structure

Avant (14 pages) :

```
Général            Configuration · Lecteur par défaut · Mises à jour · Langues · Sous-titres · Statistiques · Réinitialiser
Image IA           Agrandissement · Fluidité
HDR et couleurs    Conversion HDR · Couleurs
Audio              Sortie
Export et moteurs  Fichier · Moteurs installés
```

Après (12 pages, 5 catégories) :

```
Général            Configuration · Lecture · Application · Statistiques · Réinitialiser
Image IA           Agrandissement · Fluidité
HDR et couleurs    Conversion HDR · Couleurs
Audio              Sortie
Export et moteurs  Export · Moteurs
```

- **Lecture** = Langues + Sous-titres + « Lire l'épisode suivant » : tout ce qui règle ce qui se lit et comment.
- **Application** = Lecteur par défaut + Mises à jour : ce qui lie le lecteur au système et à ses versions. La page
  ne disparaît plus : « Ne plus me le demander » ne règle que la fenêtre du lancement.
- **Export** et **Moteurs** : le nom dit ce que la page contient.
- Gardés tels quels : **Statistiques** (ce que Tab affiche) et **Réinitialiser** restent dans Général : ce sont des
  réglages du lecteur, pas du son, de l'image ni du HDR ; les mettre ailleurs ajouterait une catégorie ou une page
  d'un seul réglage. **Export** reste sous « Export et moteurs » : l'export est un usage de la chaîne et des moteurs,
  et une catégorie de plus ferait six.
- Le test `settings_pages_test.cpp` garde : au plus 5 catégories et 12 pages, des pages d'une catégorie qui se
  suivent, des noms uniques, et les renvois « Catégorie > Page » du résumé de l'assistant.

## 4. Contrôles, page par page

Légende : **oui (script)** = cliqué dans une exécution à 1366×768, effet lu dans les réglages ou le journal ;
**oui (code)** = branché sans ambiguïté dans le code et couvert par un test unitaire, non cliqué ici ; **non (ici)** =
impossible sans matériel ou réseau.

### Général > Configuration

Page d'un autre chantier (benchmark simple) : non modifiée, non auditée ici.

### Général > Lecture

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Langue de l'audio / des sous-titres (2 rangs) | change `language_presets` ; vaut au prochain fichier | oui (code, `settings_test`) |
| Taille des sous-titres, « Taille d'origine » | `subtitle_scale`, en direct | oui (code) |
| Lire l'épisode suivant à la fin | `playlist.autoplay_next` | oui (code) |

### Général > Application

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Ne plus me le demander | `default_player.dismissed = true` | oui (script) |
| Faire de ce lecteur le lecteur par défaut | écrit le `.desktop` et `mimeapps.list` | oui (script, dossiers XDG isolés) |
| Ajouter à « Ouvrir avec » | écrit le `.desktop` | oui (script, dossiers XDG isolés) |
| Retirer de « Ouvrir avec » | retire ce que l'inscription écrit ; message « AJP ne figure plus dans « Ouvrir avec » » | oui (script, dossiers XDG isolés) |
| Me prévenir / Automatique / Jamais | `updates.mode` | oui (script : Automatique) |
| Canal, Fréquence | `updates.channel`, `updates.frequency` | oui (code) |
| Vérifier maintenant | interroge le dépôt des versions | non (ici : réseau) |
| Revenir à la version précédente | grisé tant qu'aucune mise à jour n'a été installée (infobulle ajoutée) | oui (code, `updater_test`) |

### Général > Statistiques

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Toutes les statistiques (Tab) | grise les cases suivantes, affiche tout | oui (code) |
| Chaîne, Images, Interpolation, Pistes, Sortie HDR, Position exacte, Mémoire vidéo | `stats.*` | oui (script : Pistes, Chaîne) |

### Général > Réinitialiser

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Revenir au profil d'origine | retire le profil de `user.toml`, rouvre la chaîne avec le profil d'origine (« profil 1 » dans le journal, préparation du moteur) | oui (script) |

### Image IA > Agrandissement

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| SPAN ×2 | `image.upscale` | oui (script : bascule la valeur du fichier) |
| Modèle | `image.model` | oui (code) |
| Précision de SPAN | visible seulement si le poste rend un autre cran | oui (code, `settings_test`) |
| Résolution de travail | `image.work_height`, note de sortie | oui (script : la réinitialisation la rend) |
| Algorithme (avancé) | `image.downscale` | oui (code) |
| Préparer le mode Performance (avancé) | chantier du benchmark simple | non (autre chantier) |
| Filtre vers la fenêtre, Débander (avancé) | `image.final_scaler`, `image.deband` | oui (code) |

### Image IA > Fluidité

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Multiplicateur | `interpolation.factor` ou Auto ; grisé au-delà de l'écran | oui (code, `settings_test`) |
| Plafond d'Auto, Mode d'Auto (Stable / Rapide) | `interpolation.auto_cap`, `interpolation.auto_mode` (`recommande` / `libre`) ; journal « mode d'Auto en direct : rapide (expérimental) » | oui (code, `settings_test`) |
| Moteur (RIFE, GMFSS, GMFSS Union, DistilDRBA) | `interpolation.engine` | oui (code) |
| Avancé : modèle RIFE, poses, « fusionné » ×3, Warp et SPAN, précision de GMFSS, Smooth Motion | champs de `interpolation.*` ; les cases d'un moteur absent sont grisées | oui (code) |

### HDR et couleurs > Conversion HDR

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Moteur HDR | `hdr.route` ; RTX HDR grisé sans SDK | oui (code) |
| Pic automatique | `hdr.peak = 0` | oui (script) |
| Pic (nits) | `hdr.peak` | oui (code) |
| Avancé : gris moyen, débandage RTX HDR, valeurs par défaut | `hdr.gray`, `hdr.debanding` | oui (code) |
| Avancé : noir absolu, seuil, anti-marquage, cadence fixe | `oled.*` | oui (code) |

### HDR et couleurs > Couleurs

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Contraste, Saturation | `hdr.contrast`, `hdr.saturation` (en %) | oui (code) |
| Contraste et saturation par défaut | les deux à 100 | oui (script : 200 → 100) |
| Grader la sortie | `grade.enabled` | oui (script) |
| Préréglage, curseurs du grade (avancé), Gamut P3 | `grade.*` | oui (code) |

### Audio > Sortie

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Suivre l'appareil par défaut, Appareil de sortie | `audio.device` ; rouvre la vidéo au même endroit | oui (code) |
| Casque / Enceintes / Home cinéma | `audio.listening` ; Home cinéma grisé sur 2 canaux (infobulle) | oui (script : Casque) |
| Profil, Niveau du Son anime | `audio.profile`, `audio.anime.level` | oui (code, tests audio) |
| Comparer (maintenir : Naturel) | tant que le bouton est tenu, niveau Naturel | oui (code) |
| Avancé : curseurs du profil et du Son anime, réponse HeSuVi, Fidèle, profil par appareil | `audio.*` | oui (code, tests audio) |
| Avancé : délai audio, Remettre à 0, caler sur l'écran, latence | `audio.delay_ms`, `playback.display_sync`, `video.latency_ms` | oui (code) |

### Export et moteurs > Export

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Taille, Agrandissement / Interpolation (volets), Reprendre les réglages de la lecture | réglages de l'export (pas ceux de la lecture) | oui (code, `export_job_test`) |
| Codec, Qualité, Cadence fixe | `ExportJob` | oui (code) |
| Fichier, Nom proposé | champ pleine largeur ; message si le chemin est refusé | oui (code) |
| Exporter | grisé tant que le chemin est refusé ; sinon ferme la lecture, exporte, reprend | non (ici : écrit un fichier de plusieurs minutes ; couvert par `export_sink_test`) |

### Export et moteurs > Moteurs

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Un moteur par case (si deux ou plus installés), Utiliser ce moteur, Annuler | rouvre la lecture avec ce moteur | oui (code, `engine_switch_test`) |
| Installer d'autres moteurs… | ouvre l'écran des moteurs sans relancer | oui (script, voir § 5) |
| Dossier, Par défaut | `settings.engines_dir` ; vaut au prochain lancement | oui (code) |

### Barre de lecture

| Contrôle | Effet attendu | Vérifié |
|---|---|---|
| Chapitre précédent / suivant (Page bas / haut) | saute au chapitre | oui (script) |
| Lecture / Pause | bascule (espace) | oui (script) |
| Muet, curseur de volume | `ToggleMute`, volume | oui (script) |
| Barre de progression | saut ; bulle du temps et du chapitre | oui (script) |
| Sous-titres, Piste audio | menus des pistes ; « Désactivés » | oui (script, voir § 5) |
| Paramètres | ouvre le panneau | oui (script) |
| Plein écran | bascule (f) | oui (code, `controls_test`) |

## 5. Tailles et échelles

Voir les captures « après ». Résumé des constats :

- 1366×768 et 1280×720 à 100 % : le panneau fait 760 px, la page garde 540 px utiles, tout est lisible ; l'arbre tient
  grâce au repli des catégories.
- 1280×720 à 150 % : le panneau prend toute la largeur utile (`min(760 × échelle, largeur − marges)`), l'arbre se
  réduit à 30 % de la largeur (`min(200 × échelle, 0,3 × largeur)`) ; les notes passent à la ligne, les boutons aussi.
- 1920×1080 : rien de plus à signaler.

## 6. Ce qui reste

- **Configuration** (autre chantier) : « Joue 12 secondes… » tutoie alors que le reste du panneau vouvoie
  (« relancez-le ») ; à harmoniser quand ce chantier aura fusionné.
- Les sous-titres du film passent à travers le bas du panneau (le panneau est translucide) : choix de style, non
  modifié.
- Beaucoup de commentaires du code parlent encore d'« onglets » (`settings_sound.h`, `engine_screen.h`…) : sans effet
  à l'écran, à reprendre dans un nettoyage de commentaires.
- `--seconds` ne termine pas une exécution tant que la lecture est en pause : les scripts de capture doivent borner
  l'exécution (`timeout`).
