# Rapport de l'audit 38. Promesses tenues

Branche : `claude/project-thread-t46bqg`. Commit audité : `58ddedf`. Date : `2026-11-09`.

## Verdict

Les textes livrés promettent plus de qualité et plus de discrétion que le code n'en donne : le README annonce
RIFE 4.25 en Qualité et Ultra et des plafonds d'interpolation que le chemin par défaut (assistant, Auto calibré,
mode Rapide) ne respecte pas, et « Signaler un problème » copie dans une issue publique les chemins complets des
vidéos rangées hors du dossier personnel, nom d'utilisateur compris sous Linux.

## Cadrage retenu

Pas de questions de cadrage : consigne du coordinateur, « auditer tout au maximum ». Retenu :

- toutes les promesses écrites que voit un utilisateur de la cible (Windows 11 et CachyOS, carte RTX) : README,
  `ajp --help`, assistant, panneau, notes de version 0.10.0 à 0.10.3, `CHANGELOG.md`, modèle de bug,
  `THIRD_PARTY_NOTICES.md` ;
- la consigne de Tatoub sur la qualité : une promesse qui couvre une baisse d'image faite pour la vitesse est
  relevée, avec le gain quand une mesure existe ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ; ce qui, dans les textes, empêche DistilDRBA d'être
  choisi ou cru plus rapide est regroupé plus bas. Le VRR est hors sujet.

Beaucoup de promesses fausses ont déjà un défaut dans `SYNTHESE.md`. Elles sont dans l'inventaire avec ce
défaut comme cause, sans nouvel ID. Un nouvel ID n'est donné qu'à une cause nouvelle, ou quand la correction est
le texte lui-même.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A38-01 | Majeur | `src/app/play/settings_system.cpp:493-511`, `src/core/log_redact.cpp:52-73`, `src/app/play/chain.cpp:346-353`, `src/app/play/player.cpp:161`, `src/media/demuxer.cpp:161`, `.github/ISSUE_TEMPLATE/bug.yml:1-2` | **UX-13, renvoi 24 → 38.** L'infobulle promet « sans votre nom d'utilisateur ni les dossiers de vos vidéos ». Le nettoyage ne remplace que le dossier personnel (`redact_log_text`), à l'identique, casse comprise. Or chaque fichier ouvert et chaque épisode suivant sont journalisés par leur chemin complet (`chain.cpp:346-347` et `:352-353`), et un fichier illisible aussi (`player.cpp:161`, le `where` du démultiplexeur). Le bouton recopie ensuite la fin brute de `ajp.log` (`settings_system.cpp:498-503`). Le commentaire de `bug.yml` dit aussi « noms de fichiers seuls », ce qui est faux. | Sous CachyOS, une vidéo sur un disque monté par le bureau : `/run/media/<utilisateur>/<disque>/Anime/<série>/ep01.mkv` part tel quel dans l'issue publique, avec le nom d'utilisateur et l'arborescence. Sous Windows : `D:\Anime\<série>\…` en entier. À confirmer : un `%TEMP%` en nom court 8.3 (`C:\Users\JEAN~1\…`, compte avec espace ou accent) n'est pas reconnu comme le dossier personnel. Sous CachyOS le presse-papiers reste vide (A24-03), et le journal est coupé au milieu d'une ligne (A23-14). | Ne journaliser une vidéo que par `filename()` (les deux lignes de `chain.cpp` et le `where` des refus de `Demuxer::open`). Ajouter à `redact_log_text` les racines de montage (`/run/media/<u>/`, `/media/<u>/`, `/mnt/`), et toute autre racine que le dossier personnel, sous Windows sans tenir compte de la casse. Corriger le commentaire de `bug.yml`. Test Catch2 dans `tests/unit/core/log_test.cpp` : `set_log_home("/home/u")`, puis `redact_log_text("lecture de /run/media/u/SSD/a.mkv")` ne doit contenir ni `u/` ni `SSD`. Ce test échoue aujourd'hui. | Lu. Le test n'a pas été lancé : spdlog et les dépendances manquent dans ce conteneur. |
| A38-02 | Majeur | `README.md:32`, `src/app/play/auto_config.cpp:183`, `:203-204`, `src/app/play/settings_chain_tabs.cpp:66-68` | Le README promet « RIFE 4.22 lite, ou 4.25 aux niveaux Qualité et Ultra ». Le chemin par défaut est l'assistant puis « Optimiser maintenant », qui donne l'Auto calibré. Celui-ci sort de la tranche avant de choisir RIFE (`:183`), puis impose `rife_v4.22_lite.onnx` (`:204`), même sur une RTX 4080 à 5090. **C'est une baisse de qualité faite pour la vitesse, sans mesure** : la calibration n'essaie jamais 4.25 (A25-08). Même cause qu'A25-08 et A07-01, mais ici la promesse publique est fausse. | RTX 4070 Ti (tranche Qualité) ou 4090 (Ultra), assistant suivi jusqu'au bout : le panneau montre « Rapide · RIFE 4.22 lite » alors que le README promet 4.25. « Passer » la calibration donne 4.25 : ne pas calibrer donne une meilleure image que calibrer. | Faire essayer 4.25 (et DistilDRBA) par la calibration et garder le meilleur que la machine tient (A25-08). En attendant, ne pas écraser `rife_model` dans la branche calibrée quand la tranche est Qualité ou Ultra. Test : `apply_preset` avec `Machine{tier=Ultra, calibrated={…}}` doit rendre `rife_v4.25.onnx` (échoue aujourd'hui). | Lu. |
| A38-03 | Mineur | `README.md:74-83`, `src/app/play/auto_config.cpp:48-56`, `src/app/play/chain_setup.cpp:135-143`, `data/gpu_tiers.toml` | Le tableau des niveaux est faux sur trois points. (1) « interpolation jusqu'à ×2 / ×3 / ×4 » : en mode Rapide, le défaut (`auto_free = true`), le plafond est l'écran et non `rife_factor` (A22-10) ; ×6 sur un écran 144 Hz. (2) « SPAN depuis 540p / 720p » ne vaut que pour un écran 1440p : sur un écran 1080p, Rapide travaille en 480p, Normal en 540p, Qualité en 720p et Ultra en 900p. (3) « Cartes typiques » contredit la table : GTX 1070 et 1080 sont en Normal, RX 5700 en Normal, RTX 2080 en Qualité, et les portables une tranche plus bas (A07-08). | Un utilisateur de RTX 3060 sur un écran 1080p lit « SPAN depuis 720p » : il joue du 540p. Sur 165 Hz, il lit « jusqu'à ×3 » et obtient ×6, le régime où DistilDRBA perd contre RIFE (A03-13). | Écrire « sur un écran 1440p » au-dessus du tableau. Remplacer « jusqu'à ×N » par « suit l'écran (mode Rapide) ; ×N en mode Stable ». Recopier les familles depuis `gpu_tiers.toml`. Ou faire de `rife_factor` le plafond (A22-10). | Lu. |
| A38-04 | Mineur | `README.md:29-31`, `src/app/play/auto_config.cpp:48`, `:139-146` | « Une source 720p ou 1080p devient 1440p ou 4K. » En Auto, une source 1080p n'entre entière dans SPAN qu'en Ultra sur un écran 4K. Partout ailleurs, elle est réduite à 720p ou moins (`kWorkCap`), puis SPAN sort du 1440p, que libplacebo agrandit. C'est une baisse de détail faite pour la vitesse (A49-02, A22-14). | Blu-ray 1080p sur un écran 4K avec une RTX 4070 Ti (Qualité) : 1080p → 720p (Hermite) → SPAN 1440p → 4K par libplacebo. Le texte annonce une sortie 4K de SPAN. | Écrire la promesse vraie (« SPAN double la résolution de travail, qui suit l'écran et le niveau »), ou corriger A49-02. | Lu. |
| A38-05 | Mineur | `README.md:57-66`, `src/app/play/wizard.cpp:10-24`, `src/app/play/first_run.cpp:456-470` | « Premier lancement » décrit un assistant qui n'existe plus : ordre Moteurs, Calibration, Langues, mises à jour. Le vrai est Bienvenue, Langues, Intégration (mises à jour, lecteur par défaut), Moteur, Prêt. Le bouton « Terminer » n'existe pas : c'est « Optimiser maintenant » ou « Régler moi-même ». « Quelques minutes, un peu plus la première fois » : 17 min à 1080p avec les constructions TensorRT (A25-03), sans le téléchargement. Le même texte est dans l'assistant (`first_run.cpp:462-464`). | Un nouvel utilisateur cherche « Terminer », et prévoit cinq minutes pour une vingtaine. | Réécrire la section d'après `wizard_steps`. Donner l'ordre de grandeur vrai : « 5 à 30 min la première fois, selon l'écran et TensorRT ». | Lu. |
| A38-06 | Mineur | `README.md:140`, `src/prepare/identity.h:35-48`, `src/app/play/machine_calibration.cpp:61-71` | « Le premier lancement de TensorRT construit ses moteurs […] une seule fois. » La clé d'un moteur contient la version de TensorRT, l'appareil et les options (formes comprises). La construction se refait à chaque nouvelle version de TensorRT (un index changé suffit, A06-01) et pour chaque taille de travail : la calibration compte elle-même une construction par palier. | Mise à jour du paquet TensorRT, ou passage d'un écran 1080p à un écran 1440p : nouvelle attente de plusieurs minutes, que le README disait finie. | « Une fois par taille d'image, puis de nouveau après une mise à jour de TensorRT. » | Lu. |
| A38-07 | Mineur | `README.md:129`, `src/core/keys.cpp:180-191`, `src/app/play/input.cpp:94-116` | « Les raccourcis se changent dans `user.toml` (section `[keys]`). » Seules 9 actions se changent. `q`, `f`, Échap, `m`, `j`, `z`, `x`, Ctrl+`-` et Ctrl+`+`, `a`, `p`, `n`, `o`, Page↑ et Page↓, ↑ et ↓ sont en dur et passent avant `[keys]` (A24-06). | `[keys] toggle_osd = "o"` : `o` passe toujours le générique, l'écran des statistiques ne vient pas. | Lister dans le README les 9 actions qui se changent, ou les faire toutes passer par `KeyMap` (A24-06). | Lu. |
| A38-08 | Mineur | `README.md:4`, `:44-50`, `CHANGELOG.md:7-12`, `CMakeLists.txt:4` | Le badge annonce la version 0.10.0, le code est en 0.10.3. `CHANGELOG.md` s'arrête à 0.10.0 alors que les notes 0.10.1 à 0.10.3 existent. Le tableau d'installation ne cite ni l'installeur Windows (`ajp-windows-x64-setup.exe`), ni l'AppImage, ni le `.dmg`, que les notes promettent. | Le lecteur de la page GitHub croit la 0.10.0 à jour, et ne trouve pas l'installeur. | Badge tiré de `project()` (ou vérifié par `tools/ci/check-version.sh`), trois lignes au CHANGELOG, et les noms de fichiers des workflows. | Lu. |
| A38-09 | Mineur | `THIRD_PARTY_NOTICES.md:40-49`, `README.md:183-186`, `tools/upscale/models/ArtCNN-LICENSE.txt` | Le README promet « le détail de chaque composant tiers » dans `THIRD_PARTY_NOTICES.md`. (1) ArtCNN C4F32, livré depuis 0.10.2, n'y est pas. Sa licence MIT demande que la notice accompagne la copie, et les paquets ne livrent que le `.onnx` (`fetch-models.sh:76-78`). (2) « 55 LUT » alors que 4 restent. (3) « Moteurs facultatifs » : OpenVINO, MIGraphX, ONNX Runtime et Windows ML, tous retirés. (4) « Paramètres > HDR » : ce menu n'existe plus. | Une redistribution d'ArtCNN sans sa notice MIT. | Ajouter ArtCNN (MIT, © 2024 João Chrisóstomo) aux notices et au README ; corriger les trois autres paragraphes. | Lu. |
| A38-10 | Mineur | `tools/release/notes/v0.10.3.md:3`, `:9`, `src/app/play/engine_online.cpp:85-106` | Notes 0.10.3. (1) « Le lecteur relit l'index de sa release au lancement, une fois par jour au plus. » Après un 404, la copie gardée est effacée (`:101`) : chaque lancement suivant refait la requête, jusqu'à 8 s sur le thread de la fenêtre (A18-02). C'est le cas de toute release sans paquet NVIDIA. (2) « Linux x86-64 […] Les moteurs sont téléchargés à part » : la 0.10.3 Linux est partie sans paquet TensorRT (A28-01). | CachyOS, RTX, 0.10.3 : pas de TensorRT à télécharger, et une requête bloquante à chaque lancement. | Garder la date d'un 404 comme une copie fraîche (un fichier vide daté). Ajouter aux notes une limite connue tant que le paquet manque. | Lu. |
| A38-11 | Mineur | `src/app/play/options.cpp:470`, `:497`, `src/app/play/settings.h:114-117`, `src/app/play/options.h:44` | `--help`, en plus d'A18-08, toujours présent (touche `b` fantôme, `z`, `x` et Ctrl+`-`/`+` absents, défauts d'avant les préréglages). « `--rife=N` ×N interpolation before SPAN » : avec l'ordre Auto, le warp passe après SPAN dès ×3, le défaut (`kDefaultRifeFactor = 3`), ou au-dessus de 1080p. « `--no-upscale` source picture as is » : la source est réduite à `work_height` (TRT-02). | `ajp --rife=3 ep.mkv` : l'aide dit « avant SPAN », la chaîne fait SPAN puis le warp ; la comparaison A/B d'un essai est fausse. | « ×N interpolation (order: Warp and SPAN, Auto by default) » ; corriger TRT-02 ou écrire « reduced to the working resolution ». | Lu. |
| A38-12 | Mineur | `src/app/play/settings_chain_tabs.cpp:260-262`, `:384-386`, `src/core/auto_rate.h:33-34`, `src/app/play/settings.h:108-117` | Deux infobulles du panneau sont fausses. « Stable follows […] in steps (×1 to ×6) » : le palier le plus haut est `kAutoMaxStep = 8`. « Warp and SPAN : Auto picks After above 1080p » oublie « ou dès ×3 », que dit le libellé (`settings.h:110`) : au défaut ×3, Auto prend toujours « Après », le choix rapide (A22-03). | Un utilisateur qui veut l'image la plus propre laisse Auto, sur la foi de « au-dessus de 1080p » : à 1440p de sortie et ×3, il a « Après » sans le savoir. | Recopier les bornes depuis les constantes (`kAutoMaxStep`, `kWarpAfterSpanFactor`) dans le texte. | Lu. |
| A38-13 | Mineur | `src/app/play/auto_config.h:19-22` | Le commentaire tient DistilDRBA hors de l'Auto « tant que sa porte de qualité n'est pas passée (docs/adr/0003) ». L'ADR 0003 ne parle pas de DistilDRBA, et aucun document ni test ne définit cette porte (`docs/` : aucune occurrence). La condition d'entrée ne peut donc jamais être remplie (voir A29-05). | Personne ne peut dire quelle mesure ferait passer `kDistilDrbaInAuto` à vrai : DistilDRBA reste réservé au choix manuel, même s'il est plus rapide à qualité proche. | Écrire la porte (par exemple PSNR ou VMAF de l'image du milieu face à RIFE 4.25 sur le clip de calibration, plus le temps par paire à ×2, ×3 et ×4) dans un ADR, et la faire mesurer par la calibration (A25-08). | Lu. |
| A38-14 | Mineur | `.github/ISSUE_TEMPLATE/bug.yml:56-60` | La liste « Niveau » n'offre ni Économe ni Personnalisé. Or tout réglage fait à la main passe en Personnalisé. | Un rapport d'un utilisateur qui a touché un réglage arrive avec « Je ne sais pas ». | Ajouter Economy / Économe et Custom / Personnalisé. | Lu. |
| A38-15 | Mineur | `src/render/shaders/sharpen.hook:1-5`, `:13`, `:33-35`, `README.md:39`, `tools/release/notes/v0.10.3.md:7` | La netteté est promise « sans halo » (README) et « sans liseré » (notes 0.10.3). Le hook laisse dépasser de 25 % de l'écart local (`pad = 0.25 * (hi - lo)`) : c'est un liseré borné, pas une absence. L'en-tête dit aussi « après SPAN et avant le filtre final », alors que le hook est sur `SCALED`, après le filtre final. | Trait noir sur un aplat clair, netteté à 1 : un liseré clair d'un quart du contraste est possible. | À confirmer à l'œil (instantané à 400 % sur un trait net, netteté 0 contre 1). Puis écrire « halo limité » ou ramener `pad` à 0. Corriger l'en-tête. | Lu ; effet à confirmer. |

## Inventaire des promesses

« Tenue » : vérifiée dans le code. « Fausse » : contredite, avec sa cause. « Invérifiable » : demande une mesure ou
un fichier hors du dépôt.

### README

| Promesse | Statut | Où ou cause |
|---|---|---|
| SPAN AnimeJaNai HD V3.1 | Tenue | `model_choice.h:34-37` |
| 720p ou 1080p deviennent 1440p ou 4K | Fausse | A38-04 (A49-02) |
| RIFE 4.25 en Qualité et Ultra | Fausse par défaut | A38-02 (A25-08) |
| DistilDRBA au choix | Tenue, à la main seulement | A29-05, A38-13 |
| Auto vise la fréquence de l'écran et redescend seul | Tenue | `chain_setup.cpp:139-142`, `core/auto_rate.h` |
| Coupes et dessins tenus non interpolés | Tenue | `rife_plan.cpp:10`, `upscale_drba.cpp:104`, `upscale_smooth.cpp:34` |
| HDR10 guidé par plan ; en SDR, surfaces jusqu'à 90 %, blanc à 95 %, éclats sur 5 % | Tenue | `anime-hdr-black.hook:214-223` |
| Netteté sans halo | À confirmer | A38-15 |
| Export sans grade, en SDR | Non vérifiée | Hors cible |
| Épisode suivant automatique | Tenue | `settings.h:254` |
| Fichiers de release (zip, tar.gz, macOS zip) | Tenue mais incomplète | A38-08 |
| Releases sur `Ta2b/ajp-releases` | Tenue dans le code (`src/app/CMakeLists.txt:69`) ; trois dépôts par défaut | A28-05 ; la page n'a pas pu être lue d'ici (403 du proxy) |
| Assistant : ordre, « Terminer », « quelques minutes » | Fausse | A38-05 (A25-03) |
| Tableau des niveaux | Fausse sur trois points | A38-03 (A22-10, A07-08) |
| Résolution de travail : 720p pour un écran 1440p | Tenue | `auto_config.cpp:139-141` |
| Menu en trois catégories, pages | Tenue | `settings_panel.cpp:309-321`, `:85-127` |
| HDR suivi sans bouton, seuil 350 nits, bascule en direct | Tenue | `host_facts.h:42`, `player_run.cpp:552` ; contrat d'écran figé (A11-01) |
| Pic lu dans le système (étalonnage Windows, EDID) | Tenue | `host_facts.cpp:244-256` ; A11-07 |
| Raccourcis du tableau | Tenus | `keys.cpp:180-191`, `input.cpp:94-116` |
| Raccourcis changeables dans `user.toml` | Fausse en partie | A38-07 (A24-06) |
| TensorRT : construction « une seule fois » | Fausse | A38-06 |
| TensorRT téléchargé au premier lancement | Fausse pour Linux 0.10.3 et 0.10.1 | A28-01 |
| Mises à jour : vous prévient ; manifestes signés Ed25519 | Fausse en pratique | A28-02 (aucun manifeste signé publié depuis 0.5.1), A26-02 (`engines-nvidia.lock` non signé) |
| Vos réglages sont gardés | Tenue | `settings_toml.cpp` (fusion) ; A22-02 en cas d'écriture refusée |
| Prérequis et scripts de construction | Tenus | tous les fichiers cités existent ; préréglages `linux-prefix`, `linux-release`, `debug` présents |
| Licences, notices complètes | Fausse | A38-09 |
| Version 0.10.0 | Fausse | A38-08 |

### Aide de la ligne de commande

| Promesse | Statut | Où ou cause |
|---|---|---|
| Touches de l'aide | Fausse | A18-08 (toujours présent) |
| Défauts de `--work-height`, `--rife`, `--rife-model` | Fausse | A18-08 |
| `--rife` « before SPAN » | Fausse au défaut | A38-11 |
| `--no-upscale` « source as is » | Fausse | TRT-02 |
| `--window` 1280x720, `--seconds` 5 à 120 (12), `--snapshot-at` 3 | Tenues | `options.h:77`, `options.cpp:378`, `:432` |
| `--ring-rgb10a2` « half the memory, default » | Tenue, mais c'est une baisse de qualité sans porte | A49-04 |
| Choisir DistilDRBA en ligne de commande | Absent | A18-09 |

### Assistant et panneau

| Promesse | Statut | Où ou cause |
|---|---|---|
| « Optimise now » garde la meilleure combinaison que la machine tient | Fausse | A07-01 (chaîne mesurée ≠ chaîne appliquée), A25-08 (ni 4.25 ni DistilDRBA essayés) |
| « Estimated: about N minutes » | Fausse | A25-03 (UX-05) |
| « Set it up myself » : l'Auto estimé | Tenue | `auto_config.cpp:131-187` |
| Mises à jour : « No checks » pour Jamais | Tenue | `player.cpp:134` (pas de relecture en ligne) |
| « Downloaded, installed at next launch » | Tenue dans le code ; jamais vu en pratique | A28-02 |
| « DistilDRBA : the lightest » | Fausse au-delà de ×4 | A23-08 (A03-13) |
| « Fused RIFE: about three times faster » | Tenue (1,3 ms contre 4,5 ms à 960×576) | `docs/fp8-interpolation.md:120` ; concordance faible (A03-02) |
| « Fused DistilDRBA: more than twice as fast » | Tenue (mesure) ; coupé en silence hors Équilibre | `docs/fp8-interpolation.md:24`, `:102` ; A22-12 |
| « Balanced is FP16 » (précision SPAN) | Fausse si un reçu INT8 existe | A49-07 |
| « Spline36 is AnimeJaNai's sharp choice; Hermite is sharp without halos » | Ambiguë : Hermite par défaut, plus doux | A22-14 |
| « Restore line art : cleaner lines for a quarter of the work » | Qualité jamais mesurée | A22-14, A49-02 |
| « Stable … ×1 to ×6 » ; « Auto picks After above 1080p » | Fausses | A38-12 |
| « 4K output: RIFE estimates motion on a lighter 720p copy » | Tenue (la baisse est dite) | A49-10 |
| « Scene cuts and held drawings are not interpolated » | Tenue | voir README |
| « Report a problem … without your user name or the folders of your videos » | Fausse | A38-01 (UX-13), A24-03, A23-14 |
| « Engine build cancelled: playing without upscaling or smoothing » | Fausse : source réduite | TRT-02, A31-05 |

### Notes de version 0.10.0 à 0.10.3

| Promesse | Statut | Où ou cause |
|---|---|---|
| 0.10.0 : assistant réordonné, choix écrits à chaque écran | Tenue | `wizard.cpp:10-24` |
| 0.10.0 : TensorRT aux seules RTX, pilote 580 au moins | Tenue | `first_run.h:49` |
| 0.10.0 : un seul lecteur, la vidéo passée à la fenêtre ouverte | Fausse pendant une construction | UX-03 (A29), A19-02 |
| 0.10.0 : sur batterie, l'Auto ne dépasse pas 60 i/s ; fenêtre cachée, interpolation suspendue | Tenue | `host_facts.h:49-53`, `player_run.cpp:414`, `chain_hdr.cpp:309` |
| 0.10.0 : épisode suivant actif par défaut | Tenue | `settings.h:254` |
| 0.10.0 : journal nettoyé (dossier personnel en ~) | Tenue à la lettre, insuffisante | A38-01 |
| 0.10.0 : `play.toml` lisible d'une version à l'autre | Tenue | voir audit 36 |
| 0.10.3 : netteté « sans liseré » | À confirmer | A38-15 |
| 0.10.3 : index relu « une fois par jour au plus » | Fausse après un 404 | A38-10, A18-02 |
| 0.10.3 : moteurs téléchargés à part (Linux) | Fausse pour Linux | A28-01 |
| `CHANGELOG.md` à jour | Fausse | A38-08 |

## Promesses qui couvrent une baisse de qualité

Consigne de Tatoub : sacrifier l'image pour la vitesse est un défaut, sauf mesure. Voici les textes qui annoncent
ou cachent une telle baisse, avec le gain mesuré quand il existe :

| Où | Baisse | Gain mesuré | Défaut |
|---|---|---|---|
| Auto calibré, RIFE 4.22 lite imposé | 4.25 jamais joué après calibration | aucun | A38-02, A25-08 |
| Auto, `kWorkCap` 720p sauf Ultra | source 1080p réduite avant SPAN | aucun pour l'image ; « un quart du travail » (`settings_chain_tabs.cpp:85-88`) | A38-04, A49-02, A22-14 |
| Réduction Hermite par défaut | plus doux que Spline36 | aucun | A22-14 |
| Ordre « Warp and SPAN » Auto | warp après SPAN dès ×3, au défaut | 8 ms contre 0,4 ms par passe à 1080p → 4K, sur une RTX 4070 Ti (`settings.h:103-104`) ; qualité jamais mesurée | A22-03, A38-12 |
| Anneau RGB10A2 par défaut | 10 bits au lieu de 16 dans l'anneau | mémoire divisée par deux (aide) ; banding jamais mesuré | A49-04 |
| Cran Équilibre annoncé FP16 | INT8 dès qu'un reçu existe | aucun | A49-07 |
| Netteté « sans halo » | dépassement de 25 % permis | sans objet | A38-15 |

## DistilDRBA : ce qui, dans les textes, l'empêche d'être pris ou cru plus rapide

- Le panneau le dit « le plus léger » à tout facteur : c'est faux au-delà de ×4, où il fait une passe par image
  créée (A23-08, A03-13). Le README, lui, n'annonce aucun plafond vrai en mode Rapide (A38-03), qui est justement
  le régime où il perd.
- Il ne peut pas entrer dans l'Auto : sa « porte de qualité » est citée sans exister (A38-13). La calibration ne
  l'essaie pas (A25-08), et aucune option de ligne de commande ne le choisit pour un essai `ajp compare` (A18-09).
- Le gain du DistilDRBA fusionné est mesuré, mais il se perd en silence sur un cran de précision autre
  qu'Équilibre (A22-12), et le panneau ne montre pas la variante qui tourne (A23-07).

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| TRT-02 | Toujours présent | `src/app/play/chain_setup.cpp:238` : `settings.upscale && !span` est faux quand `upscale` est coupé, donc la hauteur de travail s'applique. `player.cpp:526-533` (« Annuler et regarder ») et `--no-upscale` jouent la source réduite. Le message (`player.cpp:532`) et l'aide (`options.cpp:470`) disent le contraire. |
| UX-05 | Toujours présent | `src/app/play/engine_screen.cpp:223-227` (A25-03). La même promesse fausse est aussi dans le README (A38-05). |
| UX-13 | Toujours présent, plus large | A38-01 : chemins complets de chaque fichier ouvert (`chain.cpp:346-353`), nom d'utilisateur sous `/run/media/` |
| Renvoi 24 → 38 (`settings_system.cpp:507-509`) | Confirmé | A38-01. Dans le code actuel, l'infobulle est en `:509-511`. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 43 | `src/app/play/i18n_fr_*.cpp` | Environ 180 entrées de traduction dont le texte anglais n'apparaît dans aucun fichier du code (relevé par script, faux positifs possibles). Exemples : « Finish » (que le README promet), « Once the engines are installed, "Optimize for this machine" … in about a minute » (chemin et durée faux), l'HDR adaptatif, les styles de son, « Startup profile ». Code mort, et promesses fausses si un jour réaffichées. |
| 39 | `src/core/log_redact.cpp:52-73` | Le nettoyage du journal ne connaît que le dossier personnel, à la casse près (A38-01) ; à étendre aux racines de montage et aux noms courts 8.3. |
| 23 | `src/app/play/auto_config.cpp:167-169` | Le commentaire « les préréglages ne posent plus le FP8 » est contredit par la branche calibrée (`:202`) ; déjà au renvoi 49 → 23. |
| 28 | `tools/release/notes/v0.10.3.md` | Les notes publiées ne disent pas que la release Linux n'a pas de paquet TensorRT (A28-01). |
| 07 | `src/app/play/auto_config.cpp:203-204` | L'Auto calibré écrase le modèle RIFE de la tranche (A38-02) : à traiter avec A25-08 et A07-01. |

## Points positifs

- Le grade SDR fait exactement ce que le README chiffre (90 %, 95 %, 5 %), et le shader le dit en commentaire.
- Les gains annoncés pour RIFE fusionné et DistilDRBA fusionné reposent sur des mesures écrites dans `docs/`.
- Le HDR qui suit le système, son seuil de 350 nits et la bascule en direct sont tenus.
- Les raccourcis du README, les options de plage de l'aide et les fichiers cités pour la construction existent tous.
- Le nettoyage du journal a un test (`tests/unit/core/log_test.cpp:88-95`). Il ne couvre que le dossier
  personnel : c'est le test à étendre pour A38-01.

## Questions ouvertes

- A38-15 : la netteté à 1 laisse-t-elle un liseré visible ? Il faut un instantané à 400 % sur un trait net, sur la
  machine de Tatoub.
- A38-01 : sous Windows, `%TEMP%` est-il en nom court 8.3 sur un compte avec espace ou accent, et paraît-il dans le
  journal ? Il faut lancer une calibration sur un tel compte, puis chercher `~1` dans `ajp.log`.
