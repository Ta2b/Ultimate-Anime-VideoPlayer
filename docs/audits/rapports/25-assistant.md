# Rapport de l'audit 25. Assistant de premier lancement

Branche : `claude/project-thread-xyotbc`. Commit audité : `4802a89`. Date : `2026-11-09`.

## Verdict

L'ordre des écrans, la reprise en route et l'écriture des choix avant la calibration (P3.2) tiennent, mais
l'assistant relancé pendant une lecture voit ses choix et sa calibration écrasés par l'ancienne chaîne, et sur
une RTX la calibration mesure Vulkan sans le dire (TensorRT n'est jamais coché d'office, rien ne signale qu'il
manque) puis n'est jamais refaite quand TensorRT arrive : l'Auto calibré garde alors des bornes de Vulkan, donc
une image plus petite que la carte ne le permet. Les sept défauts connus sont toujours présents (UX-07 revu en
mineur).

## Cadrage retenu

Pas de questions posées : Tatoub a demandé d'auditer au maximum sans cadrage. Retenu :

- cible NVIDIA sous TensorRT, Windows 11 et CachyOS ; macOS et Core ML hors cible (lignes Core ML non creusées) ;
  le VRR est hors sujet ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant. Question suivie : **ce que l'assistant fait ou omet qui
  empêche DistilDRBA d'être bien plus rapide que RIFE** (A25-08) ;
- consigne de Tatoub : toute baisse de qualité d'image faite pour la vitesse est un défaut sauf mesure qui la
  justifie (section « Qualité sacrifiée à la performance »).

Aucun GPU ni dépendance (SDL3, ImGui, libplacebo) dans ce conteneur : rien ne se compile, tout est « Lu ». Les
tests qui échoueraient sont décrits.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A25-01 | Majeur | `src/app/play/engine_screen.cpp:202-311` (appelant : `src/app/play/player.cpp:541`, `:559`, `:585`, écriture `src/app/play/player_run.cpp:334-339`) | **UX-02.** L'assistant relancé pendant une lecture (« Run the wizard again », `settings_panel.cpp:282`) écrit `play.toml` (langues, mises à jour, langue de l'interface, chaîne calibrée) puis rend la main ; la boucle reprend `settings = (*outcome)->settings` (l'ancienne chaîne en mémoire), rouvre la vidéo avec, et pose `player.unsaved = settings`, que `run_chain` réécrit dans `play.toml`. La fusion de `save_settings` (`settings_toml.cpp:609-616`) prend la version de l'assistant pour base mémorisée : chaque champ où l'ancienne chaîne diffère compte comme « notre changement » et l'emporte. | En lecture, Paramètres > Général > « Run the wizard again », choisir sous-titres français et « Never », puis « Optimise now » (≈ 17 min de calibration). La vidéo reprend avec l'ancienne chaîne, sans la chaîne calibrée, et `play.toml` revient aux anciennes langues et au mode de mise à jour d'avant ; seul `calibration.toml` reste. La langue d'interface choisie s'affiche, puis revient au lancement suivant. | Après `handle_engine_change` avec `OpenEngineScreen{.wizard = true}`, relire `play.toml` (`load_settings`) dans `settings`, puis rejouer `apply_preset` comme à l'ouverture d'un fichier (`player.cpp:405-416`) ; ou rendre les réglages depuis `run_engine_screen`. Test : simuler `save_settings(assistant)` puis `save_settings(ancienne)` dans le même processus, et vérifier que les langues de l'assistant restent. | Lu. |
| A25-02 | Majeur | `src/app/play/first_run.cpp:263-265`, `:370-389` ; `src/app/play/engine_screen.cpp:259-267`, `:299-305` | Sur une RTX, la calibration de l'assistant mesure Vulkan sans le dire, et n'est jamais refaite quand TensorRT arrive. TensorRT se télécharge avec la licence de NVIDIA (`package-linux.sh:194-196`) : sa case n'est jamais cochée d'office (`first_run.cpp:265`), et le résumé de Prêt ne liste que les moteurs cochés (`engines_summary`) : rien ne dit qu'un moteur « Recommandé » manque. Une installation ratée finit aussi sur Vulkan (`engine_downloads.cpp:110-113`), puis la calibration part quand même. Plus tard, « Install more engines » ajoute TensorRT (`network_added`), mais rien ne propose de refaire la calibration : `calibrated_chain` ne compare que la carte et la hauteur d'écran (`machine_calibration.cpp:244-246`). | RTX 4070 Ti, premier lancement, « Next » partout puis « Optimise now » : la case TensorRT est restée vide, Prêt dit « Engines: Vulkan », la calibration mesure Vulkan et retient par exemple 360p + RIFE. TensorRT installé ensuite depuis l'onglet Moteurs : l'Auto calibré garde 360p sous TensorRT, qui tiendrait 540p. L'image est plus petite que nécessaire jusqu'à un « Refaire le banc » que rien ne suggère. | À Prêt, une ligne d'alerte quand une ligne `Recommended` utilisable n'est pas retenue (« TensorRT, recommandé pour cette carte, n'est pas coché : la calibration mesurera Vulkan ») ; noter dans `calibration.toml` le moteur mesuré et l'ignorer s'il diffère du préféré ; après un moteur ajouté (`network_added`), proposer la recalibration. Test : `engines_summary` sur une vue RTX non cochée doit rendre une alerte. | Lu (lié à A07-05, qui applique les bornes à tout moteur). |
| A25-03 | Majeur | `src/app/play/engine_screen.cpp:223-227` | **UX-05.** La durée de « Optimise now » est calculée à l'ouverture, avec `tensorrt` = case TensorRT cochée. Elle ne l'est jamais au départ (licence, A25-02) : l'annonce oublie toujours les constructions, et n'est pas recalculée quand on coche TensorRT. Le téléchargement de TensorRT (module, puis les archives de NVIDIA) n'est compté nulle part. | Écran 1080p : 3 paliers, `calibration_minutes(1080, false, false)` = 4 min annoncées, contre 17 min avec les constructions (`machine_calibration.cpp:61-71`) ; 4K : 6 min au lieu de 26 ; portable 1080p : 4 au lieu de 29. | Calculer la durée à l'affichage de Prêt, à partir des lignes retenues (`engines_summary`), et ajouter une ligne « plus le téléchargement de N Go ». Test : `calibration_minutes` appelé avec la vue après choix. | Lu, calcul refait à la main. |
| A25-04 | Majeur | `src/app/play/engine_screen.cpp:286-291` ; `src/app/play/wizard.cpp:103` | **UX-06.** « Make it the default video player » s'applique même quand l'écran Intégration a été passé ; le résumé, lui, le tait (`!skipped(Updates)`). Même logique pour l'écran Moteurs : « Skip » n'y annule rien, `settings_after` lit les cases telles quelles. | Cocher « lecteur par défaut », puis « Skip » : à la fin, `choose_as_default` lance `xdg-mime default` sous Linux (ou ouvre les Paramètres de Windows), alors que Prêt n'en dit rien. Cocher TensorRT et accepter sa licence, puis « Skip » : le téléchargement part quand même. | Faire suivre `make_default` du même `skipped(Updates)` que les mises à jour ; pour Moteurs, revenir à la présélection quand l'écran est sauté (ou ne pas offrir « Skip » sur cet écran). Test : `skip` après `make_default = true`, puis vérifier qu'une fonction pure (à extraire) `wants_default_player(wizard)` rend faux. | Lu. |
| A25-05 | Mineur | `src/app/play/wizard.h:34`, `src/app/play/engine_screen.cpp:214-216`, `src/app/play/first_run.cpp:393-401`, `:445-447` | **UX-07.** `WizardFacts::updates_managed` n'est jamais rempli et l'écran ne le lit pas (`radio_note` a un `enabled` toujours vrai). L'assistant propose « Automatic » là où `update_managed_reason` dit que rien ne s'installera. | CachyOS avec l'AppImage, ou Windows installé dans un dossier non inscriptible : « Automatic » choisi, puis `run_update_check` retombe sans le dire sur « Me prévenir » (`updater.cpp:523`). Gravité revue en mineur : la mise à jour est quand même annoncée. | Remplir `updates_managed` avec `!update_managed_reason(...).empty()` et griser « Automatic » avec la raison en infobulle. | Lu. |
| A25-06 | Mineur | `src/app/play/engine_screen.cpp:110-113`, `:147-149`, `:164-168`, `:188-191` | Renvoi 07 → 25. `calibrate_on_screen` rend toujours `Preset::Auto`, même sur « Annuler et regarder » ou un échec. | « Run the wizard again » ou « Refaire le banc » (`player.cpp:563`) avec une chaîne Personnalisée, puis « Annuler et regarder » : la chaîne est remplacée par l'Auto estimé. Le bouton dit aussi « regarder » quand aucune vidéo n'attend (premier lancement depuis l'accueil). | Sur annulation ou échec, rendre `base` tel quel ; ne passer en Auto que sur une calibration réussie. | Lu. |
| A25-07 | Mineur | `src/app/play/engine_screen.cpp:274-291` | Renvoi 19 → 25. Inscription « Ouvrir avec » et lecteur par défaut sur le thread de la fenêtre, sans pompe. Sous Linux, `register_player` lance `update-desktop-database` (15 s au plus), et `choose_as_default` le relance (il rappelle `register_player`, `default_player.cpp:335`) avant `xdg-mime default` (30 s au plus). | CachyOS, NAS ou disque lent, « lecteur par défaut » coché : jusqu'à une minute de fenêtre figée entre Prêt et la calibration ; KDE propose de tuer l'application. | Passer ces appels par `run_pumped`, et ne pas réinscrire dans `choose_as_default` quand `open_with_registered` vient d'être posé. | Lu. |
| A25-08 | Mineur | `src/app/play/first_run.cpp:461-465` ; `src/app/play/engine_screen.cpp:195-197` | Prêt promet « the best combination this machine sustains », mais la calibration ne connaît qu'une chaîne : RIFE 4.22 lite, ×2, auto Rapide, SPAN en FP8 d'abord, réduction Hermite (`candidate_settings`, `machine_calibration.cpp:138-151`). Ni RIFE 4.25 ni DistilDRBA ne sont essayés, et la priorité (Image, Équilibre, Fluidité) n'est pas demandée : la règle « fluidité d'abord » (P3.11) s'impose. **Pour DistilDRBA** : l'assistant est le seul endroit où la machine est mesurée, et il écrit `motion_engine = Rife` ; la passe moins chère de DistilDRBA n'est jamais réinvestie en taille d'image. | Une machine où SPAN + RIFE ne tient qu'à 360p : l'assistant retient 360p + RIFE plutôt que 540p avec SPAN seul, ou 540p + DistilDRBA, qui tiendrait peut-être. L'utilisateur croit avoir « le meilleur ». | Dire à Prêt ce qui est mesuré (« RIFE 4.22 lite ×2 ; la taille d'image peut baisser pour garder la fluidité ») ; ajouter une question de priorité à l'écran Bienvenue ou Prêt, transmise à la calibration ; renvoi 07 pour essayer DistilDRBA. | Lu (cause côté calibration : renvois 18 → 07, 03 → 07). |
| A25-09 | Mineur | `src/app/play/engine_screen.cpp:218`, `:231-243` ; `src/app/play/wizard.cpp:116-131` | L'assistant relancé (`--first-run`, « Run the wizard again ») part des valeurs par défaut, pas des réglages en place (`resume_choices` n'est appelé que pour une reprise). Avec `--first-run`, chaque « Next » écrit `play.toml` avec les écrans pas encore vus (langues japonais / langue du système, « Notify »). | `ajp --first-run` sur un poste réglé (« Never », audio français), « Next » sur Bienvenue puis fermer : `play.toml` a perdu « Never » et l'audio français. L'infobulle du bouton dit « rien ne change avant Terminer », mais il n'y a pas de bouton Terminer. | Appeler `resume_choices` dans tous les cas où `play.toml` existe ; dans `wizard_play_settings`, n'écrire un écran qu'une fois vu (`seen`). | Lu. |
| A25-10 | Mineur | `src/app/play/engine_screen.cpp:262`, `:248-255` | Fenêtre fermée pendant le téléchargement des moteurs : `settings_after` n'est pas écrit. L'assistant reprend à Prêt (P3.3) avec la case TensorRT décochée et la licence à réaccepter ; Prêt affiche « Engines: Vulkan ». | Fermer pendant les 650 Mo de TensorRT, relancer, cliquer « Optimise now » sans revenir en arrière : calibration sous Vulkan (A25-02). | Écrire `chosen_engines` (licence acceptée comprise) avant `install_chosen`, et relire les cases depuis `settings.toml` à la reprise. | Lu. |
| A25-11 | Mineur | `src/app/play/engine_screen.cpp:125` | **TRT-18.** Dossier de travail de la calibration à nom fixe, `temp_directory_path() / "ajp-calibration"`, effacé par `remove_all` (`machine_calibration.cpp:361`). Si `temp_directory_path` échoue, l'erreur est ignorée et le chemin devient relatif au dossier courant. | CachyOS, deux comptes : le dossier de `/tmp` appartient au premier, l'écriture de `chaine-0.toml` échoue pour le second, la calibration est refusée, on passe en Auto (« Machine calibration failed »). | Un dossier unique par processus sous le dossier de données de l'utilisateur ; ne rien lancer si le chemin est vide. | Lu. |
| A25-12 | Mineur | `src/app/play/first_run.cpp:195-218`, `:123-131` | **UX-09.** « actif », « inactif », « Go » et « Mo » en dur dans la ligne GPU de Bienvenue (`rebar_text`, `gpu_line`), et dans les tailles de l'écran Moteurs (`format_size`, renvoi 06 → 25). Le test i18n ne voit que `tr()`. | Interface anglaise : « NVIDIA GeForce RTX 4070 Ti · 12 Go · Resizable BAR actif, 12 Go » et « to download (900 Mo) ». | Passer les mots et les unités par `tr()` ; ajouter une vérification de la ligne en anglais dans `first_run_test.cpp`. | Lu (le test de `first_run_test.cpp:34-44` fige le français). |
| A25-13 | Mineur | `src/app/play/wizard.cpp:30` ; `src/app/play/engine_screen.cpp:214-216` | **UX-14.** Quand aucun moteur autre que Vulkan n'est utilisable (pilote < 580, place insuffisante, paquet non publié), l'écran Moteurs est sauté et sa raison (`first_run.cpp:129-133`, `:250-251`) n'est jamais montrée. | RTX sous Windows avec un pilote 575 : l'assistant passe de Intégration à Prêt, « Engines: Vulkan », sans dire qu'une mise à jour du pilote donnerait TensorRT. | Garder la raison de la ligne TensorRT dans les lignes de Prêt quand l'écran est sauté. | Lu. |
| A25-14 | Mineur | `src/app/play/engine_screen.cpp:72-76` ; `src/app/play/ui.cpp:443-455` | **UX-23.** Les touches n'arrivent à ImGui qu'en saisie de texte, sans `ImGuiConfigFlags_NavEnableKeyboard` : ni Entrée, ni Tab, ni les flèches dans l'assistant. « q » (`cmd::Quit`) le ferme d'un coup, sans confirmation. Relancé pendant une lecture, l'assistant n'a pas de bouton « Annuler » : fermer ou « q » quitte le lecteur entier (`handle_engine_change` rend faux, `player.cpp:546-547`). | Taper « q » pour une langue dans la liste : le lecteur se ferme. En lecture, ouvrir l'assistant par erreur : la seule sortie sans quitter est d'aller au bout. | Activer la navigation clavier dans l'assistant ; ignorer `Quit` hors Échap confirmé ; ajouter « Annuler » quand `restart_wizard`. | Lu. |
| A25-15 | Mineur | `src/app/play/engine_screen.cpp:43`, `:84-89` | L'assistant redessine toutes les 16 ms même sans aucun événement (`SDL_WaitEventTimeout(nullptr, 16)` puis `show`) : GPU réveillé 60 fois par seconde pendant qu'on lit une licence. | Portable sur batterie laissé sur Bienvenue : GPU dédié tenu éveillé. | Attendre l'événement sans délai, sauf animation en cours. | Lu. |
| A25-16 | Mineur | `src/app/play/engine_screen.cpp:343`, `:391`, `:396`, `:398` | Quatre `path::string()` dans les journaux et la notice de l'onglet Moteurs (règle de la CI ; dette, voir A20-02). | Profil non ASCII sous Windows : texte abîmé dans le journal et la notice. | `core::to_utf8`. | Lu. |
| A25-17 | Mineur | `src/app/play/first_run.cpp:138-147`, `:166`, `:241-243`, `:322-327` ; `src/app/play/first_run.h:1-7`, `:121-122` ; `src/app/play/wizard.h:29` | Code mort et commentaires périmés : MIGraphX, OpenVINO et DirectML encore traités (`prefer_chosen`, `first_run_view`, `within_engines`, `network_added`), un `namespace {}` vide, l'en-tête qui décrit l'ancien écran « Commencer », les exemples de titres « Moteur », « Image et fluidité » ; `first_run.cpp` à 580 lignes (cible 500). | Lecture du code : on croit ces moteurs encore proposés. | Retirer avec A06-11 ; scinder le dessin de l'assistant. | Lu (même cause qu'A06-11). |
| A25-18 | Mineur | `tests/unit/app/wizard_test.cpp`, `tests/unit/app/first_run_test.cpp` | Rien ne teste `run_engine_screen`, `calibrate_on_screen`, ni : `make_default` après « Skip », la durée après le choix de TensorRT, une RTX dont TensorRT n'est pas coché, l'assistant relancé (`resume_choices` absent), `updates_managed`. | Les quatre majeurs passent la CI. | Extraire les décisions de `run_engine_screen` en fonctions pures (ce qui s'écrit, ce qui est demandé) et les tester. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| UX-05 | Toujours présent | `src/app/play/engine_screen.cpp:223-227` : A25-03. Pire que décrit : TensorRT n'est jamais coché à l'ouverture (licence), donc l'annonce n'est jamais la bonne. |
| UX-06 | Toujours présent | `src/app/play/engine_screen.cpp:286-291` : A25-04 ; même chose pour les cases de l'écran Moteurs. |
| UX-07 | Toujours présent, gravité revue en mineur | `src/app/play/wizard.h:34` : A25-05 ; l'updater retombe sur Me prévenir (`updater.cpp:523`). |
| TRT-18 | Toujours présent | `src/app/play/engine_screen.cpp:125` : A25-11. |
| UX-09 | Toujours présent | `src/app/play/first_run.cpp:195-218` : A25-12. |
| UX-14 | Toujours présent | `src/app/play/wizard.cpp:30` : A25-13. |
| UX-23 | Toujours présent | `src/app/play/engine_screen.cpp:74` : A25-14 ; aggravé en lecture (quitte le lecteur). |
| UX-02 (point de la fiche) | Toujours présent | A25-01. |
| Renvoi 06 → 25 (`first_run.cpp:125-131`) | Toujours présent | Dans A25-12. |
| Renvoi 07 → 25 (`engine_screen.cpp:111`) | Toujours présent | A25-06. |
| Renvoi 19 → 25 (`engine_screen.cpp:277-290`) | Toujours présent | A25-07. |
| « Choix écrits avant la calibration » (P3.2) | Tient | `src/app/play/engine_screen.cpp:295-298` : `play.toml` et `settings.toml` écrits avant `calibrate_on_screen` ; la fenêtre fermée ne perd que la chaîne calibrée. |

## Qualité sacrifiée à la performance

Ce que l'assistant fait vivre à l'image, au-delà de son code propre :

| Où | Sacrifice | Gain mesuré | Statut |
|---|---|---|---|
| `src/app/play/first_run.cpp:265` + `engines_summary` | TensorRT non coché sans alerte : calibration et lecture sous Vulkan, taille de travail plus basse | Aucun (aucun gain : c'est un oubli) | A25-02 |
| `machine_calibration.cpp:138-151` (appelé par `engine_screen.cpp:195-197`) | SPAN essayé en FP8 d'abord sur toute carte qui le prend | Aucun chiffre dans l'assistant ; porte de qualité (A02-01 la fait passer sur NaN) | Renvoi 22 → 07 déjà ouvert |
| idem | Réduction Hermite écrite dans la chaîne calibrée | Aucun | A22-14 |
| idem, `best_candidate` (`machine_calibration.cpp:115-135`) | Fluidité d'abord : SPAN + RIFE à 360p préféré à SPAN seul à 540p, sans demander la priorité | Fluidité, pas de mesure de l'image perdue | A25-08, renvoi 07 |
| idem | Interpolation figée sur RIFE 4.22 lite ×2 : ni RIFE 4.25 ni DistilDRBA, même quand DistilDRBA tiendrait plus haut | Aucun | A25-08 (renvois 18 → 07, 03 → 07) |
| `calibrated_chain` (`machine_calibration.cpp:240-252`) | Bornes mesurées sous Vulkan gardées sous TensorRT | Aucun | A25-02, A07-05 |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 24 | `src/app/play/home_screen.cpp:307-316` | « Run the wizard again » depuis l'accueil ignore `OpenEngineScreen::wizard` : seul l'écran des moteurs s'ouvre. |
| 07 | `src/app/play/machine_calibration.cpp:115-135` | `best_candidate` ignore `PlaySettings::priority` : la fluidité passe toujours avant la taille d'image (A25-08). |
| 07 | `src/app/play/machine_calibration.cpp:240-252` | `calibration.toml` ne note pas le moteur mesuré (Vulkan ou TensorRT) : bornes Vulkan gardées sous TensorRT (A25-02). |
| 07 | `src/app/play/machine_calibration.cpp:288-290` | Une écriture refusée de `chaine-N.toml` (dossier partagé, A25-11) arrête toute la calibration au lieu de changer de dossier. |
| 08 | `src/app/play/player.cpp:559`, `:585` | Après `handle_engine_change`, la chaîne rouverte et `player.unsaved` partent des réglages d'avant (cause d'A25-01). |
| 19 | `src/app/play/default_player.cpp:334-336` | `choose_as_default` rappelle `register_player` : `update-desktop-database` lancé deux fois de suite (A25-07). |
| 43 | `src/app/play/settings_panel.cpp:283` | Infobulle « Nothing changes until "Finish" » : aucun bouton Terminer dans l'assistant. |

## Points positifs

- L'ordre des écrans est simple et testé ; un écran revu puis validé n'est plus « passé » (P3.16).
- La reprise en route tient : `first_run_step` est écrit à chaque écran, `resume_choices` relit les choix, et un
  rang d'une version plus ancienne reprend au dernier écran.
- Le pilote NVIDIA est lu avant 650 Mo de téléchargement, la place disque est annoncée avec la place restante, et
  une licence n'est jamais acceptée d'office.
- La calibration tourne sur un thread, avec une barre qui avance même pendant une construction, et une fenêtre
  fermée arrête l'enfant (`jthread`, jeton d'arrêt passé à `run_process`).

## Questions ouvertes

- Faut-il cocher TensorRT d'office sur une RTX, la licence étant acceptée en un clic sur la même ligne, ou garder
  la case vide avec une alerte à Prêt (A25-02) ? La seconde option respecte la règle actuelle (« une licence ne se
  coche jamais d'office »).
- L'assistant doit-il demander la priorité (image ou fluidité) avant « Optimise now » (A25-08) ?
