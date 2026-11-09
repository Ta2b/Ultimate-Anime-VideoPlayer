# Rapport de l'audit 24. Interface : rendu, contrôles et messages

Branche : `claude/project-thread-fmlo9w`. Commit audité : `4802a89`. Date : `2026-11-09`.

## Verdict

L'interface ne touche pas à la qualité de l'image (elle se dessine après la composition, sans rien retirer à la
chaîne), mais elle perd les messages qui disent qu'une qualité a été retirée (repli de DistilDRBA sur RIFE,
lecture sans agrandissement), elle peut faire présenter des images hors cadence pendant la lecture, et sur CachyOS
le presse-papiers ne sort pas du lecteur : 3 majeurs, 20 mineurs.

## Cadrage retenu

Pas de questions posées à Tatoub (consigne du coordinateur) : tout est audité au maximum, sur la cible NVIDIA
TensorRT, Windows 11 et CachyOS. Consigne transmise : toute baisse de qualité d'image faite pour la vitesse est un
défaut sauf mesure ; RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ; le VRR est hors sujet.

Sur la qualité : aucun des 31 fichiers ne dégrade l'image pour gagner du temps. Les seeks au point clé pendant un
glissement ou une flèche maintenue sont un aperçu, suivi d'un seek exact au lâcher (`controls.cpp:109-126`,
`input.cpp:72-89`). Deux écarts de fidélité restent, sans motif de vitesse : les sous-titres étirés après un
redimensionnement (A24-20) et les panneaux typographiés déplacés avec la barre (A24-19). Ce qui freine DistilDRBA
n'est pas dans cette zone ; la zone cache seulement à l'utilisateur qu'il ne tourne pas (A24-01).

Le fil n'a ni GPU ni dépendances (`~/ajp-deps` absent) : rien n'a été compilé ni lancé. Les tests sont décrits.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A24-01 | Majeur | `src/app/play/controls.cpp:128-131` | Un seul emplacement de message, 1,6 s, écrasé par le suivant (UX-04). Les avis de qualité perdue disparaissent avant d'être lus ou ne paraissent jamais. | (1) DistilDRBA refusé : `chain_setup.cpp:404` appelle `notify` pendant la préparation de la chaîne, puis RIFE se charge (`chain_setup.cpp:445-518`, plusieurs secondes sous TensorRT, des minutes s'il se construit) : le message a expiré avant la première image, l'utilisateur ne sait jamais qu'il regarde RIFE au lieu de DistilDRBA. (2) Reprise avec un avis : `player_run.cpp:342-348` pose « Repris à 12:34 » puis `p.notice` au même tour, le premier n'est jamais dessiné. (3) « Construction annulée : lecture sans agrandissement ni fluidité » (`player.cpp:532`) est remplacé par « Auto ×2 (48 i/s) » au premier tour du mode Auto (`player_run.cpp:502`). (4) Plusieurs looks illisibles (`chain_hdr.cpp:221-222`) : seul le dernier se lit. | Une file de messages : chacun reste au moins 1,6 s plus environ 60 ms par mot, les suivants attendent. Un `notify` avant la première image passe par `defer`. Un état dégradé durable (DistilDRBA remplacé, SPAN absent, Rapide retombé en Stable) reste visible dans l'OSD tant qu'il dure, pas seulement 1,6 s. Test : deux `notify` au même instant, le second visible après le premier ; un `notify` à t, première image à t + 3 s, le message encore visible. | Lu. |
| A24-02 | Majeur | `src/app/play/controls.cpp:167-174` | Pendant la lecture, l'interface demande une présentation à elle seule : tous les 100 ms quand la barre est visible (pas de la position), à 60 Hz pendant un fondu ou un message. `Session::tick` présente aussitôt (`session.cpp:506`, `ui_changed`), sans attendre l'échéance de l'image suivante. | Écran fixe 144 Hz en FIFO, 23,976 i/s sans cadence fixe : la souris bouge, une présentation de l'interface part à un instant quelconque et prend le vblank où l'image neuve était attendue ; l'image neuve sort un vblank plus tard (6,9 ms). En interpolation à la fréquence de l'écran, chaque présentation en plus peut remplir la file et bloquer le thread de présentation. Saccade tant que la barre est visible. | En lecture, ne pas présenter pour l'interface seule : la recomposer, et la livrer avec la présentation suivante (image neuve ou cadence). Garder la présentation immédiate en pause. Test : `Session` avec un faux écran, image toutes les 41,7 ms, `UiFrame.changed` à 10 Hz : aucune image neuve ne doit glisser d'un vblank. | À confirmer : `--stats`, 24p sur écran fixe, comparer « manquées » et σ de l'intervalle avec la souris immobile, puis qui bouge sans arrêt sur l'image (raison `u` dans `--stats-file`). |
| A24-03 | Majeur | `src/app/play/ui.cpp:325-343` | Le contexte ImGui n'a pas de fonctions de presse-papiers (`Platform_SetClipboardTextFn`, `Platform_GetClipboardTextFn`). ImGui n'utilise le presse-papiers du système que sous Windows ; ailleurs il garde le texte dans un tampon à lui. | Sur CachyOS, « Signaler un problème » (`settings_system.cpp:505`) copie le journal dans ce tampon et ouvre la page GitHub : l'utilisateur colle un presse-papiers vide. Ctrl+V dans le champ du dossier des moteurs ou du fichier d'export ne colle rien venu d'ailleurs. Le seul chemin de support de la cible Linux est muet. | Brancher `SDL_SetClipboardText` et `SDL_GetClipboardText` (libéré par `SDL_free`) dans `ImGui::GetPlatformIO()` à la création. | Lu (`imgui.cpp` 1.93 : implémentation Win32, sinon tampon local). À confirmer : sous CachyOS, « Signaler un problème », puis coller dans un éditeur. |
| A24-04 | Mineur | `src/app/play/input.cpp:18-58` | `key_of` ne traduit ni `-`, `=`, `/`, `\`, `;`, `'`, `[`, `]`, `` ` `` ni Inser, alors que `core/keys.cpp:31-61` les accepte dans `[keys]`. | `step_forward = ";"` dans `user.toml` : lu sans erreur, la touche ne fait jamais rien, et rien ne le dit. | Traduire toutes les touches de `core::Key` (SDLK_MINUS, SDLK_EQUALS, SDLK_SLASH, SDLK_BACKSLASH, SDLK_SEMICOLON, SDLK_APOSTROPHE, SDLK_LEFTBRACKET, SDLK_RIGHTBRACKET, SDLK_GRAVE, SDLK_INSERT). Test : pour chaque `core::Key`, un keycode SDL qui la rend (aller-retour complet). | Lu. |
| A24-05 | Mineur | `src/app/play/input.cpp:76` | AZERTY (UX-11) : la répétition lit `key_of(event.key.key)` sans la traduction de Maj des lignes 119-127 ; Ctrl + « - » (délai audio, ligne 103) attend SDLK_MINUS. | « . » maintenu (Maj + « ; ») : keycode `;`, aucune répétition, une seule image avance. En AZERTY, « - » est sur la touche 6 ; avec l'option `french_numbers` de SDL 3.4 (par défaut), elle rend `6` : Ctrl + « - » n'avance pas le délai audio (seul le pavé numérique marche). | Factoriser la traduction de Maj dans une fonction appelée aussi pour la répétition ; pour le délai audio, lire le caractère tapé (`SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false)`) en plus du keycode. Test : `input_of` avec scancode de « ; » en disposition française, Maj, `repeat = true` : `StepFrame`. | Lu (SDL 3.4 `SDL_keyboard.c:481-505` : keycode sans modificateur, sauf la rangée des chiffres en AZERTY). |
| A24-06 | Mineur | `src/app/play/input.cpp:94-116` | Quinze touches sont en dur avant la carte `[keys]` (q, f, Échap, m, j, z, x, a, p, n, o, Page préc./suiv., haut, bas, Ctrl + -/+), et `parse_keys` ne les refuse pas. | `toggle_osd = "O"` est accepté : « o » passe le générique, l'OSD ne s'affiche jamais. `pause = "Up"` monte le volume. | Soit ces touches deviennent des `core::Action` (remplaçables), soit `parse_keys` refuse un raccourci qui les prend (KeyConflict). Test : `parse_keys("[keys]\ntoggle_osd = \"O\"")` refusé ou `input_of` rend ToggleOsd. | Lu. |
| A24-07 | Mineur | `src/app/play/user_keys.cpp:16-24` | Une seule entrée fautive de `[keys]` (`core/keys.cpp:240`, `248`) remet tous les raccourcis par défaut, avec un avertissement au journal seulement (UX-11). | Une faute de frappe (`seek_foward`) efface aussi les neuf autres redéfinitions ; l'utilisateur voit ses raccourcis revenir sans explication. | Ignorer l'entrée fautive, garder les autres, et le dire à l'écran par `defer` (« Raccourci ignoré : seek_foward »). | Lu. |
| A24-08 | Mineur | `src/app/play/user_keys.cpp:32` | `user_toml.string()` dans le journal : sous Windows, MSVC lève `std::system_error` quand le chemin contient un caractère hors de la page de codes (règle CI : passer par `core::to_utf8`). | Profil Windows au nom japonais sur un Windows français, `user.toml` illisible (verrouillé par un antivirus) : exception au démarrage. | `core::to_utf8(user_toml)`. | Lu ; la levée est à confirmer sur Windows. |
| A24-09 | Mineur | `src/app/play/visible_error.h:44-45` | 56 codes sur 71 tombent dans `default` et s'affichent par leur nom anglais : « erreur BuilderFailed (détails au journal) ». Ce sont justement ceux de la cible : BuilderFailed, BuilderTimeout, BuilderCrashed, ModelUnavailable, ModelMismatch, LibraryUnavailable, VendorCallFailed, SdkRejected, AbiMismatch, DeviceUnavailable, FeatureUnavailable, HashMismatch. Et `visible_error.h:22-28` dit « fichier abîmé ou absent » pour HwDecodeUnavailable et DecoderUnavailable. | Moteur TensorRT dont la construction échoue : « La lecture s'est arrêtée : erreur BuilderFailed (détails au journal). » NVDEC refusé pour de l'AV1 sur une RTX 20 : « le fichier ne se lit pas (format inconnu, abîmé ou absent) », l'utilisateur jette un fichier sain. | Une phrase par famille (construction de moteur, modèle, pilote ou carte, décodeur matériel, réglages, réseau) ; `switch` sans `default` pour que `-Wswitch` force un cas à chaque code ajouté. Test : pour chaque code, `visible_error` ne contient pas `name(code)`. | Lu (71 codes dans `core/types.h`, 15 cas). |
| A24-10 | Mineur | `src/app/play/visible_error.h:28` | UX-17 : la phrase répète celle de l'appelant, et le défaut ajoute « détails au journal » que l'appelant ajoute aussi. | `player.cpp:163` et `367` : « « ep01.mkv » ne se lit pas : le fichier ne se lit pas (format inconnu, abîmé ou absent) ». Boîte fatale (`main.cpp:539-541`) : « … erreur X (détails au journal). Détails dans C:\…\ajp.log ». | `visible_error` rend la cause seule (« format inconnu, abîmé ou absent ») ; le renvoi au journal est l'affaire de l'appelant. | Lu. |
| A24-11 | Mineur | `src/app/play/controls_draw.cpp:354-371` | UX-15 : le libellé de préparation et les lignes de détail sont écrits sans largeur maximale, alors que « Annuler et regarder » est posé à `x1 - frame.x` (ligne 386). Le titre (ligne 104-112) est coupé par le rectangle de découpe, sans « … ». | Calibration (`engine_screen.cpp:177`) : « Calibrage de cette machine · Construction du moteur DistilDRBA 1080p … · 45 % » à 1 280 px, échelle 1,25 : le texte passe sous le bouton et sort de la fenêtre. Un titre japonais long est tranché au milieu d'un caractère. | `ellipsize(text, x1 - frame.x - 12·s - x0)` pour le libellé et chaque détail ; `ellipsize` aussi pour le titre. | Lu. |
| A24-12 | Mineur | `src/app/play/controls_draw.cpp:114-125` | Le message bref n'a ni largeur maximale ni retour à la ligne, et `defer` colle les avis d'avant la lecture sur une seule ligne (`controls.cpp:133-135`) (UX-04). | Mise à jour disponible + licence à accepter + moteur non mis à jour : 30 à 40 mots sur une ligne, `p0.x` devient négatif, le début sort par la gauche, et le tout s'efface en 1,6 s. | Largeur maximale (40 % de la sortie), `AddText` avec `wrap_width`, et les avis différés en file (A24-01), pas concaténés. | Lu. |
| A24-13 | Mineur | `src/app/play/panel_style.h:115-130` | UX-22 : `badge` se met toujours sur la ligne (`SameLine`) sans vérifier la place, et sa marge (5 px), son arrondi et sa hauteur ne suivent pas l'échelle. | Menu à 1 280 px ou à 150 % : « Recommandé » coupé en « Recommenc… » par le bord du cadre (capture `menu.png`). | Reprendre la logique de `same_line_if_fits` ; multiplier marge et arrondi par l'échelle. | Lu. |
| A24-14 | Mineur | `src/app/play/ui.cpp:452` | UX-08 : hors saisie de texte, aucune touche n'arrive à ImGui ; Échap va au lecteur, qui ferme tout le panneau (`chain.cpp:306-311`, `home_screen.cpp:281-287`) et jette les changements en attente (`settings_panel.cpp:591`). | Liste « Modèle » ouverte, Échap pour la refermer : le panneau se ferme et le changement de moteur choisi plus haut est perdu. | Quand un menu déroulant est ouvert (`ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)`), donner Échap à ImGui et le consommer ; sinon fermer le panneau. | Lu. |
| A24-15 | Mineur | `src/app/play/home_screen.cpp:205-213` | Renvoi 19 : `dropped()`, une lecture d'événement, fait passer la fenêtre au premier plan et interroge le disque (`is_regular_file`, ligne 218) sur le thread de l'interface. Seules l'accueil et la lecture l'appellent. | Un épisode ouvert depuis le gestionnaire de fichiers pendant la construction d'un moteur, la calibration ou l'assistant : la boucle d'attente (`player_engines.cpp:54-57`) jette l'événement, la fenêtre ne passe pas devant et le fichier est perdu sans un mot. Un sous-titre glissé sur l'accueil est ignoré sans message (ligne 268). | Séparer la lecture (fonction pure) du passage au premier plan ; garder les fichiers transmis dans une file lue par toutes les boucles. | Lu. |
| A24-16 | Mineur | `src/app/play/home_screen.cpp:244-250` | Renvoi 22 confirmé : à l'accueil, `save_settings` (écriture atomique avec `fsync`) tourne sur le thread de l'interface, comme `core::save_settings` des moteurs (ligne 300), alors que la lecture passe par `writer.post` (`player_run.cpp:641-644`). | Glisser un curseur à l'accueil sur un disque lent ou sous antivirus : l'écran se fige quelques dizaines de millisecondes toutes les `kSettingsSaveEveryUs`. | Le même écrivain de fond que la lecture. | Lu ; durée à confirmer. |
| A24-17 | Mineur | `src/app/play/home_screen.cpp:336-341` | L'accueil recompose ImGui et présente une image toutes les 16 ms, même sans entrée ni animation (`PlayerUi::home` recompose à chaque appel). | Lecteur laissé ouvert sur l'accueil : CPU et GPU actifs en continu (portable : batterie, ventilateurs). | Comme `Controls::needs_frame` : recomposer seulement après une entrée, une tâche du panneau ou un changement d'état ; sinon attendre l'événement suivant. | Lu ; à confirmer : charge GPU sur l'accueil immobile. |
| A24-18 | Mineur | `src/app/play/placebo_imgui.cpp:107-124` | L'atlas des polices est en RGBA32 (format par défaut d'ImGui, `TexDesiredFormat` jamais posé), et chaque mise à jour renvoie l'atlas depuis l'origine jusqu'au coin du dernier glyphe ajouté, sur le thread de présentation. | Titre japonais, puis une infobulle, puis un menu de pistes coréennes : à chaque nouveau glyphe, jusqu'à plusieurs Mo envoyés au milieu de la lecture. | `io.Fonts->TexDesiredFormat = ImTextureFormat_Alpha8` (le shader traite déjà `r8`) : quatre fois moins ; puis vérifier si la version épinglée de libplacebo accepte un rectangle décalé. | Lu ; coût à confirmer (`AJP_TRACE` autour de `update_texture`). |
| A24-19 | Mineur | `src/app/play/placebo_overlays.cpp:63-76` | Quand la barre est visible, tout morceau de sous-titre dont le haut est dans la moitié basse monte, y compris les panneaux typographiés (`\pos`), et un dialogue sur deux lignes dont la première commence au-dessus du milieu ne monte qu'en partie. | Un panneau traduit posé sur une enseigne en bas de l'image se détache de l'enseigne tant que la souris bouge ; une réplique de deux lignes est coupée en deux par la barre. | Ne monter que les événements de dialogue (alignement bas, sans `\pos` ni `\move`), par événement entier : l'information vient de `subs` (renvoi 17 pour la donnée). | Lu. |
| A24-20 | Mineur | `src/app/play/placebo_overlays.cpp:45-47` | Les sous-titres rendus pour l'ancienne taille sont étirés à la nouvelle : bitmaps libass mis à l'échelle, flous ou crénelés jusqu'au rendu suivant. | Passage en plein écran pendant une réplique longue : la réplique reste étirée ×1,5 (1280 → 1920) jusqu'à l'événement suivant. | Demander un nouveau rendu à la nouvelle taille dès le changement (zone 17) et ne garder l'étirement que pour l'image de transition. | Lu ; durée à confirmer. |
| A24-21 | Mineur | `src/app/play/placebo_imgui.cpp:161-207` | L'interface est dessinée par une passe `pl_dispatch_vertex` par commande ImGui, et rejouée à chaque présentation (le `draw_` de `ui.cpp:482` est réutilisé). | Menu des paramètres ouvert pendant la lecture, sortie 4K à 144 Hz : des dizaines de passes de rendu par présentation, sur le budget de la chaîne. | Dessiner l'interface dans une texture RGBA seulement quand ImGui recompose (60 Hz au plus), puis un seul mélange par présentation. | À confirmer : `--stats`, temps de rendu avec et sans le menu ouvert. |
| A24-22 | Mineur | `src/app/play/osd_text.cpp:68` | Renvoi 17 confirmé : l'OSD a sa propre bibliothèque libass (caches de polices et de glyphes en double, police CJK chargée deux fois) et son script est reconstruit puis rendu à chaque changement de texte (toutes les 250 ms, `session.cpp:493`), sur le thread de présentation (`placebo_overlays.cpp:100-109`). Renvoi 11 lié : le relevé NVML (`chain_stats.cpp:184-208`) continue toutes les 500 ms après que l'OSD a été caché. | OSD affiché pendant une lecture à 144 Hz : un analyse-et-rendu libass toutes les 250 ms dans la boucle qui présente ; OSD caché, NVML relevé jusqu'à la fin de l'épisode. | Dessiner l'OSD avec ImGui (le commentaire `osd_text.h:5` le prévoit) ; suspendre l'échantillonneur quand aucune ligne de mémoire n'est demandée depuis 1 s. | Lu ; coût à confirmer (`AJP_TRACE` sur `OsdText::render`). |
| A24-23 | Mineur | `src/app/play/ui.cpp:117-129` | Sous Windows, seule une police japonaise complète Segoe UI ; sous Linux la recherche essaie aussi le chinois et le coréen (lignes 140-142). Sous Linux, si la police principale couvre déjà le japonais, toutes les candidates sont écartées (ligne 144) et le journal annonce à tort « aucune police CJK ». | Piste audio « 한국어 » ou titre chinois sous Windows 11 : des carrés à la place. | Ajouter `malgun.ttf` et `msyh.ttc` en repli ; sous Linux, tester la couverture de la police principale avant de chercher. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| UX-04 | Toujours présent | A24-01 et A24-12 : `controls.cpp:128-135`, `controls_draw.cpp:114-125`. |
| QUA-16 | Toujours présent, en partie hors zone | Dans la zone : statiques mutables `input.cpp:65` (`seek_repeated`, partagé par toutes les boucles), `scripted_input.cpp:39` (essais seulement), `user_keys.cpp:12`. `strtoul` (`display_edid.cpp:358`, `377`) : zone 11. `moving_area` : zones 09 et 45. Traductions orphelines : 43. CHANGELOG : 28. |
| UX-08 | Toujours présent | A24-14. |
| UX-11 | Toujours présent | A24-05 (répétition, Ctrl + « - »), A24-07 (une faute remet tout), A24-04 (touches lues mais jamais traduites). |
| UX-15 | Toujours présent | A24-11. |
| UX-17 | Toujours présent | A24-10. |
| UX-22 | Toujours présent | A24-13. |
| Renvoi 11 → 24 (`chain_stats.cpp:184-208`) | Confirmé | Dans A24-22. |
| Renvoi 17 → 24 (`osd_text.cpp:70`) | Confirmé | A24-22 (ligne 68 au commit audité). |
| Renvoi 19 → 24 (`home_screen.cpp:204-212`) | Confirmé | A24-15. |
| Renvoi 20 → 24 (`visible_error.h:44-45`) | Confirmé, chiffre corrigé | A24-09 : 56 codes sur 71 au `default`, pas 58. |
| Renvoi 22 → 24 (`home_screen.cpp:244-250`) | Confirmé | A24-16. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 23 | `src/app/play/settings_system.cpp:497-509` | L'infobulle promet un rapport « sans votre nom d'utilisateur ni les dossiers de vos vidéos », mais la fin brute de `ajp.log` est copiée telle quelle ; le journal contient des chemins complets (`/home/<nom>/…`, `C:\Users\<nom>\…`, noms de vidéos). |
| 38 | `src/app/play/settings_system.cpp:507-509` | Même promesse non tenue que ci-dessus, vue comme promesse de l'interface. |
| 08 | `src/app/play/player.cpp:163`, `367`, `598` | `file.filename().string()` dans un message : sous Windows, une exception sur un nom de fichier japonais quand la page de codes ne le couvre pas, exactement quand le fichier ne se lit pas. |
| 05 | `src/app/play/player_engines.cpp:135`, `142` | `onnx.stem().string()` et `path.string()` : même risque sous Windows. |
| 03 | `src/app/play/chain_setup.cpp:396-397` | « DistilDRBA fusionné indisponible : DistilDRBA entier » n'est écrit qu'au journal : l'utilisateur ne voit jamais que DistilDRBA tourne sur le chemin lent. À afficher dans l'OSD avec son motif. |
| 12 | `src/app/play/session.cpp:506` | Une présentation pour l'interface seule part hors de l'échéance des images (cause côté session de A24-02). |
| 29 | `src/app/play/player_engines.cpp:54-57` | Les boucles d'attente (construction, calibration, assistant) jettent les fichiers transmis par un second lancement (A24-15). |
| 17 | `src/subs/subtitle_frame.h` | Le rectangle d'un sous-titre ne dit pas s'il vient d'un dialogue ou d'un panneau positionné : il faudrait le savoir pour A24-19. |
| 27 | `cmake/Dependencies.cmake:123` | `imgui_impl_sdl3.cpp` compilé dans `ajp_imgui` sans être utilisé (`ui.cpp` a son propre pont SDL3). |

## Points positifs

- Le dessin de la barre est séparé de son état (`Controls`), testé seul : fondu, OLED, glissement, volume, double
  clic, recomposition à la demande (`tests/unit/app/controls_test.cpp`, 16 cas).
- La recomposition est économe pendant la lecture : 60 Hz au plus, et seulement quand quelque chose change
  (`controls.cpp:167-174`) ; le reste du temps l'image ImGui précédente est réutilisée.
- Le glissement sur la barre fait des seeks au point clé espacés de 120 ms et un seek exact au lâcher ; la flèche
  maintenue fait de même (P3.43) : l'aperçu est rapide sans fausser la position finale.
- La recherche des polices tourne sur un thread pendant l'ouverture de la fenêtre, la police CJK est choisie par
  couverture réelle (あ et 日), FreeType lit les polices CFF2 de Fedora.
- `build_progress.cpp` lit les journaux de trtexec par leurs deux bouts (4 Mo au plus) et refuse `inf` et `nan`.
- Le pont libplacebo d'ImGui gère les textures dynamiques d'ImGui 1.92 et contourne le défaut d'envoi de
  libplacebo, avec un commentaire qui le dit.
- `osd_script` neutralise les accolades et barres obliques d'ASS dans le texte de l'OSD.

## Tests à écrire

Aucun n'a pu être compilé ici (dépendances absentes).

- `visible_error` : pour chacun des 71 codes, le texte rendu ne contient pas `name(code)` (A24-09).
- `input_of` : chaque `core::Key` a un keycode SDL qui la rend (A24-04) ; en disposition française simulée,
  Maj + « ; » répété rend `StepFrame` (A24-05) ; `[keys] toggle_osd = "O"` est refusé ou rend `ToggleOsd`
  (A24-06).
- `Controls` : deux `notify` au même instant restent lisibles l'un après l'autre ; un `notify` suivi d'une
  première image 3 s plus tard est encore visible (A24-01).
- `Session::tick` avec un faux écran : la barre visible à 10 Hz ne décale aucune image neuve (A24-02).

## Questions ouvertes

- A24-02 dépend du mode de présentation réel sur la machine de Tatoub (FIFO, cadence fixe ou non). La manipulation
  `--stats` souris immobile contre souris en mouvement le tranche.
- Faut-il garder l'OSD libass jusqu'au passage annoncé à ImGui, ou avancer ce passage (A24-22) ?
