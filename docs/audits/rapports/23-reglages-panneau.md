# Rapport de l'audit 23. Réglages : le panneau

Branche : `claude/project-thread-8famvb`. Commit audité : `71b43a9`. Date : `2026-11-09`.

## Verdict

Le panneau tient sa promesse principale (le direct s'applique tout de suite, ce qui rouvre la vidéo attend
« Appliquer » sous un encadré qui dit quoi), mais deux cases n'ont aucun effet (« Restaurer les traits », la carte
graphique : démontré), « Warp and SPAN » est grisé pour DistilDRBA alors qu'il en décide la variante, « Auto
(calibré) » choisi dans le panneau applique une chaîne non calibrée, et la page Application crée et efface un
fichier dans le dossier d'installation à chaque image. Les dix défauts connus sont toujours présents.

## Cadrage retenu

Pas de questions posées : Tatoub a demandé d'auditer au maximum sans cadrage. Retenu :

- cible NVIDIA sous TensorRT, Windows 11 et CachyOS ; le VRR est hors sujet (notes VRR de l'onglet Fluidité non
  creusées) ; l'export est hors cible mais sa page vit dans le panneau : ses défauts y sont notés en mineur ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant. Question suivie : **ce que le panneau fait ou cache qui
  empêche DistilDRBA d'être bien plus rapide que RIFE** (A23-02, A23-06, A23-07, A23-08) ;
- consigne ajoutée en cours d'audit : relever tout endroit où la qualité du rendu est sacrifiée à la performance
  (section « Qualité sacrifiée à la performance »).

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A23-01 | Majeur | `src/app/play/settings_chain_tabs.cpp:84`, `src/app/play/settings_system.cpp:82-92` (cause : `src/app/play/settings.cpp:383-433` et `:449-505`) | « Restaurer les traits » et « Carte pour la lecture » ne sont jamais appliqués ni enregistrés : `restore_lines` et `gpu` ne sont ni dans `live_part` ni dans `change_between`. Le panneau calcule `live == running` et un coût `Live` : pas d'envoi, pas de bandeau « Appliquer ». De même `rife_fused` et `drba_fused` cochés quand l'interpolation est coupée (`change_between` ne les regarde que si `rife_factor >= 2`). | Portable à deux cartes : choisir la RTX dans « Carte pour la lecture ». La note dit « au prochain lancement », mais `play.toml` garde l'ancienne valeur et la case revient à Automatique à la prochaine ouverture du panneau. Idem : décocher « Restaurer les traits » sur une source 1080p, rien ne change (SPAN tourne toujours sur l'image réduite), et le choix disparaît à la fermeture. Le réglage ne passe que si un autre changement provoque une réouverture. | Ajouter `gpu` à `live_part` (« au prochain lancement », comme `engines_dir`), `restore_lines` à `change_between` (réouverture : SPAN change d'entrée), `rife_fused`/`drba_fused` à `live_part` quand `rife_factor < 2`. Test ci-dessous. | Démontré : petit programme liant `settings.cpp` (`live_part` puis `change_between` sur chaque champ seul) : `restore_lines: live!=running 0 ; cost Live`, `gpu: live!=running 0 ; cost Live`. Le balayage des 52 champs de `PlaySettings` ne trouve que ces deux-là hors des deux fonctions (plus `hdr_native`, qui n'est pas un réglage, et `audio_device_delays_ms`, porté par `audio_device`). |
| A23-02 | Majeur | `src/app/play/settings_chain_tabs.cpp:375-383` | « Warp and SPAN » est grisé dès que le moteur n'est pas RIFE, alors que `span_order` décide la variante de DistilDRBA (`chain_setup.cpp:376-377` : mouvement = un SPAN par source, image = un SPAN par image créée). | Avec RIFE, choisir « Before SPAN », puis passer en « Very fast » (DistilDRBA) : DistilDRBA tourne en variante image, SPAN passe sur chaque image créée (×3 : trois SPAN par paire au lieu d'un), et le panneau interdit de revenir à Auto ou After. À l'inverse, qui veut la variante image pour DistilDRBA (la variante mouvement fusionnée n'a aucun test, A03-03) ne peut pas la choisir. | Griser seulement pour Apple ; pour DistilDRBA, écrire sous la liste la variante qui en découle (« DistilDRBA : variante mouvement, un SPAN par source »). | Lu. Renvoi 03 → 23 confirmé. |
| A23-03 | Majeur | `src/app/play/settings_panel.cpp:250-255` | Le choix d'un niveau appelle `apply_preset` avec `Machine{tier, display_height, display_hz}` seulement : ni `calibrated`, ni `explored_work_height`, ni `explored_rife_model`. `SettingsChoices::calibrated` n'est qu'un booléen qui sert au libellé. | Machine calibrée par l'assistant : essayer « Qualité », puis revenir à « Auto (calibré) ». Le brouillon reçoit la chaîne de la tranche estimée (Rapide ou Ultra : RIFE 4.25 et SPAN FP16) au lieu de la chaîne mesurée (RIFE 4.22 lite, SPAN FP8, `auto_config.cpp:192-209`). « Appliquer » rouvre avec ces réglages tels quels (`player.cpp:559`, `apply_preset` n'est pas rejoué), et le fichier suivant reprend la chaîne calibrée (`player.cpp:405-416`) : deux chaînes sous le même nom « Auto (calibré) ». | Passer dans `SettingsChoices` la `Machine` complète que `player_run.cpp:293` construit déjà, et l'utiliser telle quelle dans `level_section`. | Lu. |
| A23-04 | Majeur | `src/app/play/settings_system.cpp:107-108` | La page Général > Application relit `update.toml` (lecture et analyse TOML) et appelle `update_managed_reason` **à chaque image** : `getenv`, `exists("/.flatpak-info")`, puis création et effacement de `.ajp-write-test` dans le dossier de l'exécutable (`updater.cpp:490-497`). Tout cela sur le thread de présentation. | Panneau ouvert sur Application pendant une lecture à 165 Hz : 165 fichiers créés puis effacés par seconde à côté de `ajp.exe`. Sous Windows, Defender analyse chaque fichier créé ; un disque lent ou un dossier synchronisé (OneDrive) bloque l'image. | Calculer `managed` et `state` une fois à l'ouverture de la page (ou quand une tâche finit), pas à chaque image. | Lu. À confirmer : sous Windows, ouvrir Application pendant une lecture avec les statistiques Images, comparer les images perdues avec la page Son ; Process Monitor montre les `.ajp-write-test`. |
| A23-05 | Mineur | `src/app/play/settings_panel.cpp:290-301` | « Tout réinitialiser » agit d'un clic, sans confirmation, et rouvre la vidéo. Seul `engines_dir` est gardé : la langue repasse en anglais (`language = "en"`), le mode et le canal des mises à jour, la carte graphique et les délais par appareil sont remis à zéro, alors que la note ne parle que de « chaîne, HDR, grade et couleurs, son, lecture ». | Un utilisateur français clique : la vidéo se rouvre, l'interface est en anglais, ses mises à jour « Jamais » sont repassées à « Me prévenir ». « Annuler » ne rattrape que tant que le panneau reste ouvert, et demande encore « Appliquer ». | Une confirmation en deux temps (« Confirmer ? »), garder `language` (et `gpu`, `update_*`), écrire dans la note ce qui est gardé. | Lu. |
| A23-06 | Mineur | `src/app/play/settings_panel.h:53-54`, `src/app/play/player_run.cpp:272-273` | Aucun contrôle pour `motion_precision` : `rife_tiers` et `drba_tiers` sont calculés à chaque ouverture et jamais lus. Une valeur hors Équilibre venue de `play.toml` (main, ancienne version) coupe sans un mot RIFE fusionné et DistilDRBA fusionné (`rife_fused_choice.cpp:11`, `drba_fused_choice.cpp:9`), pendant que les deux cases restent cochées. | `motion_precision = "performance"` dans `play.toml` : DistilDRBA tourne entier, 2,10 ms par passe au lieu de 0,92 ms (`settings.h:207-211`), RIFE à 6,5 ms au lieu de 2,2 ms. Rien dans le panneau ne le montre ; seul « Tout réinitialiser » remet Équilibre. | Montrer le sélecteur `precision_combo` pour le moteur du mouvement quand un graphe `_fp8`/`_fp32` existe (il ne s'affiche pas sinon), et écrire sous les cases « fusionné » qu'elles demandent Équilibre. Sinon retirer `rife_tiers`/`drba_tiers`. | Lu. |
| A23-07 | Mineur | `src/app/play/settings_chain_tabs.cpp:363-374` | Le panneau ne dit pas ce que DistilDRBA fait vraiment : fusionné ou entier, variante mouvement ou image, ni pourquoi le fusionné a été refusé (Vulkan, cran, structure non reconnue). Ces faits existent dans la chaîne (`chain_setup.cpp:369-400`, `c.drba_fused`, `c.drba_motion`, `fused.declined`) et ne vont qu'au journal. « Fused DistilDRBA » reste coché et actif sous Vulkan. | Tatoub compare DistilDRBA à RIFE : sans lire `ajp.log`, il ne peut pas savoir que DistilDRBA tourne entier et en variante image, ce qui explique à lui seul qu'il ne soit pas plus rapide. | Ajouter à `SettingsChoices` l'état réel (fusionné utilisé, refus, variante) et l'écrire sous la case, comme `engine_note` pour le moteur. | Lu. |
| A23-08 | Mineur | `src/app/play/settings_chain_tabs.cpp:63-64`, `:342`, `:280-284` | Textes faux sur le coût de DistilDRBA : « Very fast is the lightest » et « DistilDRBA: the lightest », alors qu'il fait une passe par image créée et RIFE une par paire (A03-13) : au-delà de ×5, il calcule plus que RIFE 4.25. « Above ×4 each pair needs a RIFE pass per created frame » est faux pour DistilDRBA, qui le fait à tout facteur. | Auto en mode Rapide sur 165 Hz (×6 à ×7 sur 23,976) : l'utilisateur choisit « Very fast » sur la foi du texte et obtient la chaîne la plus lourde des trois. | Écrire le coût vrai : « le plus léger jusqu'à ×4 ; au-delà, une passe par image créée ». Ou corriger A03-13 d'abord. | Lu (A03-13 déjà démontré). |
| A23-09 | Mineur | `src/app/play/settings_chain_tabs.cpp:213`, `:280-284`, `:349-351` | Le plafond d'Auto est caché et dépend du chemin : 8 si Auto est choisi depuis « Off », 3 par défaut, 2 à 6 selon le préréglage. En Stable il borne vraiment la cadence (`chain_setup.cpp:142`), alors que l'infobulle dit « ×1 to ×6 ». Les notes « Above ×4 » et « 4K output: ×3 and above » s'affichent ou non selon ce plafond invisible. | Deux utilisateurs en « Auto · Stable » sur le même écran 144 Hz : l'un plafonne à ×3 (défaut), l'autre à ×6 (Auto repris depuis Off), sans rien dans le panneau qui les distingue. | Montrer le plafond dans « Auto limit », ou le fixer (celui de l'écran) quel que soit le chemin. | Lu. |
| A23-10 | Mineur | `src/app/play/settings_panel.cpp:242`, `src/app/play/settings_chain_tabs.cpp:248-251`, `:296`, `:328` | Une même page porte deux « Niveau » (traitement : Économe à Ultra ; interpolation : Très rapide à Équilibré) et trois « Rapide » de sens différent (préréglage, RIFE 4.22 lite, mode d'Auto). | « Mets le niveau sur Rapide » : trois listes possibles sur la page Post-traitement. | Renommer la liste du mouvement (« Modèle d'interpolation ») et le mode d'Auto (« Sans paliers »). | Lu. |
| A23-11 | Mineur | `src/app/play/settings_sound.cpp:72-77` | L'infobulle du délai audio promet « jusqu'à ±10 000 ms » ; la borne est ±5 000 (`settings.h:300`), la valeur saisie est rabotée sans mot. Le délai audio n'est d'ailleurs visible qu'en réglages avancés, alors que celui des sous-titres est en vue de base. | Casque Bluetooth à 250 ms : le réglage est caché tant que « Réglages avancés » n'est pas coché ; saisir 8000 donne 5000. | Corriger le texte ; sortir « Délai audio » de la vue avancée. | Lu. |
| A23-12 | Mineur | `src/app/play/settings_panel.cpp:199` (effet : `chain.cpp:183-185`) | Le curseur du délai des sous-titres s'applique à chaque image du glissement (pas de garde au relâchement comme l'audio, `settings_panel.cpp:581`) : un `set_subtitle_delay` et une notification « Subtitle delay » par image. | Glisser de 0 à 2 s : une soixantaine de notifications à l'écran, une remise à l'heure des sous-titres par image. | Même garde que le délai audio (`IsItemActive`). | Lu. |
| A23-13 | Mineur | `src/app/play/settings_export.cpp:68`, `:70`, `:198`, `:216` | La page Export fait quatre appels au système de fichiers par image (`is_directory`, `equivalent`, `exists` deux fois) sur le thread de présentation. | Vidéo sur un NAS ou un disque en veille : chaque image attend le réseau ou le réveil du disque, la lecture derrière le panneau saccade. | Ne revérifier qu'à la saisie (quand `path` change) ou au plus quatre fois par seconde. | Lu. À confirmer : page Export ouverte sur une vidéo d'un partage SMB. |
| A23-14 | Mineur | `src/app/play/settings_system.cpp:495-506` | « Signaler un problème » coupe le journal à 48 Kio par octet (milieu de ligne, voire d'un caractère UTF-8) et ne dit rien dans le panneau : la copie dans le presse-papiers n'est annoncée que par l'infobulle. | L'utilisateur clique, le navigateur s'ouvre sur un formulaire vide, il ne sait pas qu'il doit coller. | Commencer à la ligne suivante ; écrire « Journal copié : collez-le dans la page » sous le bouton. | Lu. |
| A23-15 | Mineur | `src/app/play/settings_panel.cpp:310-320`, `:371-373`, `src/app/play/settings_panel.h:177`, `:155` | Dette : le drapeau `advanced` des pages vaut faux partout (le code qui masque une page avancée ne sert plus), l'en-tête parle d'une « catégorie Avancé » qui n'existe plus, `wants_draw` et `grade_tab` ont des paramètres inutilisés, `BenchNote bench_` ne sert qu'à `bench_page`, sans appelant (UX-21). | Sans effet visible ; trompe le lecteur du code. | Retirer avec `bench_page`. | Lu. |

### Tests à écrire (la construction n'est pas possible ici : pas de dépendances)

- **A23-01** (`tests/unit/app/settings_test.cpp`) : pour chaque réglage du panneau, partir de `PlaySettings{}`, changer
  ce seul champ, et vérifier `live_part(running, wanted, true) != running` ou
  `change_between(live, wanted, true).cost == ChangeCost::Reopen`. Échoue aujourd'hui pour `restore_lines` et `gpu`
  (vérifié hors CI par le programme décrit dans la preuve), et pour `rife_fused` avec `rife_factor = 0`.
- **A23-03** : `apply_preset` appelé comme `level_section` contre l'appel de `player.cpp:405` avec une
  `CalibratedChain` : les deux résultats doivent être égaux.

## Qualité sacrifiée à la performance

Relevé demandé par Tatoub : endroits de la zone où le panneau choisit, ou pousse à choisir, une image moins bonne pour
gagner du temps.

| Où | Sacrifice | Avis |
|---|---|---|
| `settings_chain_tabs.cpp:84-88`, défaut `restore_lines = true` (`settings.h:180`) | Une source 1080p est réduite à 720p avant SPAN « pour un quart du travail » : grain et textures fines perdus, l'infobulle le dit. Et la case ne peut pas être décochée (A23-01). | Le plus gros sacrifice de la zone. À trancher par une comparaison à l'œil sur une source 1080p propre ; au minimum, A23-01 doit être corrigé pour que le choix existe. |
| `settings_chain_tabs.cpp:147` | Le badge « Recommandé » est collé à 720p quelle que soit la source et la carte : il pousse une RTX 4090 sur une source 1080p à réduire. | Le badge devrait suivre ce que la carte tient (calibration), pas une hauteur fixe. |
| `settings_chain_tabs.cpp:375-386` (Auto de `span_order`) | Auto choisit « After » au-dessus de 1080p ou dès ×3 : les images créées sont warpées après SPAN au lieu d'être nettoyées par SPAN (« Before also cleans each in-between frame »). | Sacrifice assumé par le texte, choisi par défaut sans mesure de qualité visible. À mesurer (« Before » contre « After » sur une 4070 Ti à ×3) ; si la carte tient « Before », Auto devrait le prendre. |
| `settings_chain_tabs.cpp:353` (note « 4K output ») | En sortie 4K, RIFE estime le mouvement sur une copie 720p. | Hors zone (chaîne, 09) ; le panneau ne propose pas de l'éviter. |
| `settings_panel.cpp:116-130`, défaut `FinalScaler::Default` | La mise à l'échelle finale prend le filtre par défaut de libplacebo, pas EWA Lanczos (« le plus net mais le plus lourd »). | À confirmer : le filtre réel derrière « libplacebo default » (`placebo_screen.cpp:132`). Sur une RTX, EWA Lanczos coûte peu face à SPAN : il devrait être le défaut. |
| `settings_chain_tabs.cpp:363-374`, `drba_fused_choice.cpp:9` | DistilDRBA et RIFE fusionnés exigent le cran Équilibre (FP16) : le Qualité (FP32) les rend entiers et lents. | Raisonnable (FP16 est la précision d'entraînement) ; à dire dans le panneau (A23-06). |
| Préréglages (`settings_panel.cpp:250`, cause `auto_config.cpp:134-147`) | Sur un écran 1080p, le préréglage vise 540p de travail : une source 1080p est réduite à 540p puis remontée par SPAN. Économe coupe SPAN. | Hors zone : renvoi 07. |
| `settings_chain_tabs.cpp:248-262` (mode Rapide d'Auto) | Une image en retard répète la précédente au lieu d'être montrée. | Choix de fluidité, pas de qualité d'image ; VRR hors sujet. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| TRT-07 | Toujours présent | `settings_chain_tabs.cpp:97-101` propose « · FP8 » pour chaque modèle de la liste, ArtCNN compris, et d'après les crans du **seul** modèle en cours (`span_tiers` est calculé pour lui) : rien ne filtre ArtCNN (`grep -i artcnn src` ne trouve aucun refus). La partie « fp16 calcule en FP32 » est hors zone (02). |
| UX-13 | Toujours présent | L'infobulle (`settings_system.cpp:509-511`) promet « sans votre nom ni les dossiers de vos vidéos ». `redact_log_text` ne remplace que le dossier personnel ; `chain.cpp:352` journalise le chemin complet du fichier suivant. Sous CachyOS, un disque externe monté sous `/run/media/<utilisateur>/…` garde nom et dossier ; sous Windows, `D:\Anime\…` reste. Renvoi 39 déjà ouvert. |
| QUA-14 | Toujours présent | `settings_panel.h:129` (`std::array<char, 512>`), `settings_engines.cpp:76` (`snprintf` coupe à 511 octets, au milieu d'un caractère UTF-8). Une seule frappe dans le champ enregistre ensuite le chemin coupé (`:84`). |
| SEC-08 | Toujours présent | `settings_system.cpp:195-217` : avec `staged_version` non vide, « Install now » applique sans vérification ; `apply_staged_update` (`updater.cpp:554`) ne compare pas à la version installée. |
| SEC-09 | Toujours présent | `settings_system.cpp:253-258` (« Offer it again ») lit et réécrit `update.toml` sur le thread de l'interface pendant qu'une tâche « Check now », « Install now » ou « Roll back » (`:181`, `:200`, `:226`) peut le réécrire. |
| UX-10 | Toujours présent | `settings_chain_tabs.cpp:275` : la virgule sans condition de langue. |
| UX-16 | Toujours présent, aggravé | `settings_chain_tabs.cpp:273-277` : la note « Greyed out… » s'affiche dès que la cadence et l'écran sont connus. En lecture, les facteurs fixes sont masqués (`:219`), seuls Auto et Off restent : rien n'est jamais grisé. À l'export, `display_hz` vaut 0 (`settings_export.cpp:118`) : la note n'y apparaît jamais. Elle est donc fausse chaque fois qu'elle s'affiche. |
| UX-18 | Toujours présent | `settings_panel.cpp:199` : curseur à ±5 000 ms, borne et touches à ±30 000 (`settings.h:307`, `chain.cpp:323`). |
| UX-19 | Toujours présent | `settings_engines.cpp:83-84` : chaque frappe passe par `live_part` (`settings.cpp:410`) ; `play.toml` est écrit toutes les `kSettingsSaveEveryUs` et en sortant (`player_run.cpp:644-673`) : quitter pendant la saisie garde le chemin à moitié tapé. |
| UX-21 | Toujours présent | `bench_page` (`settings_system.cpp:369-491`) n'a aucun appelant. Environ 168 entrées de `i18n_fr_*.cpp` sur 854 n'ont aucun littéral correspondant dans `src` (comptage heuristique : littéraux adjacents recollés). |
| Renvoi 08 (UX-08) | Toujours présent | Échap devient `LeaveFullscreen`, qui ferme le panneau (`chain.cpp:306-308`) ; les touches n'arrivent à ImGui qu'en saisie de texte (`ui.cpp:449`), donc une liste ouverte ne capte pas Échap. Fermer par Échap jette aussi sans confirmation les changements qui attendent « Appliquer » (le bandeau le dit, `settings_panel.cpp:591`). |
| Renvoi 09 (`home_screen.cpp:140`) | Toujours présent | À l'accueil, `fp8_engine` vaut vrai pour toute RTX à cœurs tenseurs (RTX 20 et 30 comprises) : si un graphe `_fp8` est présent, la liste propose « · FP8 », que la chaîne remplace par l'Équilibre au chargement. En lecture, le fait vient du moteur (`player_run.cpp:230`) : correct. |
| Renvoi 16 (`settings_sound.cpp:27-33`) | Toujours présent | En « suivre le système », `audio_device` est vide : `switch_device_delay` range tous les appareils sous la clé `""`. Passer du casque Bluetooth (250 ms) aux enceintes par défaut du système garde 250 ms. |
| Renvoi 03 (`settings_chain_tabs.cpp:375-383`) | Toujours présent | Devenu A23-02. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 22 | `src/app/play/settings.cpp:383-433`, `:449-505` | Cause d'A23-01 : `restore_lines` et `gpu` absents de `live_part` et de `change_between` ; `rife_fused`/`drba_fused` perdus quand l'interpolation est coupée. |
| 22 | `src/app/play/settings.h:304-307` | Le commentaire sur le seuil du HDR est posé au-dessus de `kSubtitleDelayMs` au lieu de `kTrueHdrPeak`. |
| 26 | `src/app/play/updater.cpp:490-497` | `update_managed_reason` écrit un fichier d'essai à chaque appel ; appelée à chaque image par le panneau (A23-04). |
| 07 | `src/app/play/auto_config.cpp:134-147` | Écran 1080p : travail visé à 540p, une source 1080p y est réduite avant SPAN (qualité sacrifiée par le préréglage). |
| 08 | `src/app/play/chain.cpp:183-185` | Une notification par changement du délai des sous-titres : sans garde côté panneau, une par image (A23-12). |

## Points positifs

- Le bandeau « Appliquer » nomme ce qui sera reconstruit (`change_between`) et l'application en direct est séparée
  proprement ; le délai audio ne s'applique qu'au relâchement.
- Annuler et rétablir regroupent un glissement en un seul pas.
- `grade_slider` arrondit et borne : plus de grade refusé pour un 0,800000012.
- Les actions lentes (xdg-mime, vérification, préparation FP8) tournent hors du thread de présentation et s'annulent à
  la fermeture ; l'ordre des membres d'`UpdateNote` garantit que le futur ne bloque pas la sortie.
- L'export refuse les chemins impossibles avant d'écrire et ne remplace un fichier que sur un bouton qui le dit.

## Questions ouvertes

Aucune posée. Option retenue pour A23-06 : montrer le cran de précision du mouvement seulement quand un graphe
`_fp8` ou `_fp32` est présent, comme pour SPAN.
