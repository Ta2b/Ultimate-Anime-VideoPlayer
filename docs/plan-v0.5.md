# Plan de la v0.5

Suivi de la prochaine version : **tout ce qui était prévu pour la 0.4, plus le son spatial. Rien n'est reporté.**
Tatoub n'est plus disponible pour valider : l'avancement est autonome, chaque décision est notée ci-dessous.
Ce fichier est mis à jour à chaque étape (cases cochées, journal daté). Les détails de conception sont dans
`docs/ux-parametres.md` (menu, configuration automatique, assistant, mise à jour) et
`docs/audio-spatialisation.md` (audio).

Légende : `[x]` fait et sur `main` ou en PR ouverte, `[~]` en cours, `[ ]` à faire.
Pull requests : une par étape ; quand une étape dépend d'une autre, la base est dite.

## Décisions prises (ne plus les rediscuter)

| Sujet | Décision | De qui |
|---|---|---|
| Version | 0.5 regroupe la 0.4 et le son spatial | Tatoub |
| Mises à jour par défaut | « Me prévenir » | Tatoub |
| Benchmark | proposé au premier lancement, « Plus tard » possible, refaisable dans Général > Configuration | Tatoub |
| Profils | l'écran est retiré (les profils restent dans `play.toml`) | Tatoub |
| HDR / Couleurs | deux catégories séparées | Tatoub |
| Langues | dans Général | Tatoub |
| Réglages avancés | une case globale en pied de l'arbre | Tatoub |
| Pastilles | Recommandé, Le plus rapide, Le plus fin, Le plus lourd, Expérimental | Tatoub |
| Temps mesurés | hors de l'interface (dans la documentation) | Tatoub |
| Poses DRBA | désactivées par défaut jusqu'à une mesure A/B | Tatoub |
| Son spatial | pas de détection casque ou enceintes ; « Comment écoutez-vous ? » au premier lancement | Tatoub |
| Son spatial : avfilter ou code maison, nombre de profils HRTF | **code maison** (`audio/mixer.h`), **un seul profil** : aucune HRTF mesurée livrée (licences, poids), modèle paramétrique de la tête (retard de Woodworth + ombre en passe-bas) ; crossfeed à 700 Hz, downmix à centre +2 dB ; mesure d'oreille à faire par Tatoub | Claude |
| Clé de signature des mises à jour | paire Ed25519 générée sur le PC de Tatoub (privée dans `~/.config/ajp-release/`, jamais dans le dépôt) ; **les releases se signent localement sur ce PC** tant qu'aucun secret CI n'existe : `docs/signature-des-releases.md` | Claude |

## Chantiers, dans l'ordre

### A. Configuration automatique
- [x] (a) Classement des cartes en tranches, `data/gpu_tiers.toml` : PR #22
- [x] (b) Préréglages, priorité, application de la chaîne : PR #23 (base : #22)
- [x] (c1) Décision du benchmark et `bench.toml` : PR #24 (base : #23)
- [x] (c2) `ajp bench` : lancement des essais (processus enfants isolés, son coupé), rapport `--report`, pilote dans la clé de mesure : PR de la branche `claude/bench-run`
- [x] Bouton « Évaluer ma machine » (Général > Configuration)
- [x] Comparateur A/B : outil `ajp compare` (PR `claude/ab-compare`). La mesure poses DRBA / DistilDRBA contre RIFE se lance avec la version CUDA en Release (`ajp compare VIDÉO --a-settings=… --b-settings=…`) ; en attendant, les poses restent désactivées par défaut
- [x] Bascule automatique en Personnalisé quand un réglage de chaîne est modifié à la main (`preset_chain_differs`)

### B. Textes, pastilles, infobulles
- [x] 720p au lieu de « 720 lignes », vocabulaire courant (PR `claude/textes-pastilles`)
- [x] `badge()` et pastilles (IA : résolution, réduction, moteurs, Smooth Motion) ; infobulles : Image, Fluidité (descriptions des moteurs sur leurs boutons) et Audio. **Restent en paragraphes** : les notes d'état de HDR, Couleurs, Export et Moteurs (grisés, avertissements, valeurs mesurées), conservées volontairement
- [x] Hermite en réduction (fait en F)

### C. Nouveau cadre des réglages et assistant
- [x] Arbre à gauche, page à droite, encadrés titrés (PR `claude/cadre-arbre`) ; lignes multiples et barre de défilement fine restent à faire
- [x] Catégories, case « Réglages avancés », Profils retiré ; dernière page mémorisée le temps de la session seulement (pas encore dans `play.toml`)
- [x] Assistant de premier démarrage (8 écrans, `wizard.h`), relançable par `--first-run` ; bouton « Relancer l'assistant » dans Général > Configuration (PR `claude/restes`)
- [x] « Ouvrir avec » au premier lancement, sans confirmation ; bouton « Faire de ce lecteur le lecteur par défaut »
  avec « Ne plus me le demander » (réapparaît en bas d'Avancé), détection xdg-mime sous Linux
- [ ] Recherche dans les réglages (par page d'abord) : **non faite**, prévue après la 0.5 (avec le repli de l'arbre en liste sur petit écran, les lignes multiples et la mémoire de la dernière page entre sessions)

### E. Audio et son spatial
- [x] Liste et choix de l'appareil de sortie (`audio/devices.h`) ; le défaut du système est suivi par SDL (flux sur l'appareil par défaut) : PR `claude/audio`
- [x] Format de la source par piste et de la sortie par appareil, conversion automatique (le décodeur rend les canaux voulus par le mixeur)
- [x] Profils Casque / Enceintes / Home cinéma, crossfeed, downmix soigné, profil par appareil, Fidèle (case + commande `SetListening`) ; touche **B** : bascule A/B traité / fidèle (PR `claude/restes`)
- [x] Binaural
- [x] Écran audio de l'assistant et page Audio des réglages (Audio > Sortie : Appareil, Écoute, Synchronisation)

### D. Mise à jour automatique signée
- [x] Vérification en arrière-plan et réglages (Me prévenir / Automatique / Jamais, canal, fréquence) : Général > Mises à jour (PR `claude/maj`)
- [x] Manifeste signé (Ed25519, `tools/release/sign-manifest.sh`), téléchargement vérifié (taille + SHA-256), application au prochain lancement, retour arrière (testés sur archive locale)
- [ ] Essais Linux puis Windows de bout en bout : la clé existe (`docs/signature-des-releases.md`) mais aucune release signée n'a encore été installée par le lecteur ; Windows renomme les fichiers en service en `.old`, non essayé

### F. Moteur
- [x] Hermite en réduction (`DownscaleFilter::Hermite`, cubique B = C = 0, clé `hermite`) : PR `claude/moteur`. Pas de pastille *Anime* (la liste des cinq pastilles est décidée) : l'infobulle le dit. Le filtre n'est pas défaut : la comparaison PSNR/regard contre Spline36 et Mitchell se fait avec `ajp compare` sur la machine
- [x] GMFSS Union : mesuré (`docs/reports/gmfss-distil-mesures-2026-10-28.md`) : aucun calcul indépendant du temps à déplacer, options TensorRT sans gain ; pas de changement
- [x] DistilDRBA : mesuré : 0,86 ms par passe à 720p (warps ≈ 30 % de 11,7 ms seulement à 2160p) ; pas de changement, décision motivée dans le rapport
- [x] Mode Auto : suivi par le fil d'audit (PR #38, fusionnée dans main avant l'étiquette) ; mesures sur la 4070 Ti : `docs/reports/mesures-auto-4070ti-2026-10-28.md`

### G. TensorRT sous Windows
- [x] `trtexec.exe` dans l'archive du moteur, fusion de `engines-nvidia.lock` dans `release-windows.yml`, `tools/release/attach-nvidia-windows.ps1`, `docs/nvidia-windows.md` (PR `claude/trt-windows`)
- [ ] Essai sur un poste Windows NVIDIA (liste dans `docs/nvidia-windows.md`) : aucune machine Windows ici. **La 0.5.0 Windows est livrée sans archive TensorRT : à attacher depuis un poste Windows NVIDIA** (`attach-nvidia-windows.ps1`), puis relancer Release Windows sur l'étiquette

### Release
- [x] Tests locaux complets (app 266, audio 69, core 176 verts), version CMake et vcpkg 0.5.0, notes `tools/release/notes/v0.5.0.md`, liste à tester `docs/tests-v0.5-windows-macos.md`.
- [x] **Étiquette `v0.5.0` posée le 28 octobre 2026 à 08:48 sur main (52a7907), sur l'instruction écrite de Tatoub dans le fil du projet (« pose l'étiquette v0.5.0 », 08:47)**. Les trois workflows de release (Linux, Windows, macOS) tournent sur l'étiquette.
- [ ] Après les workflows : signer le manifeste (`docs/signature-des-releases.md`, sur le PC de Tatoub), le joindre à la release, vérifier les assets (aucun fichier NVIDIA, pré-version, notes de la version).
- [ ] Essais Windows et macOS de `docs/tests-v0.5-windows-macos.md`, écoute du binaural, essais de mise à jour de bout en bout (Linux puis Windows) : à faire par Tatoub, aucune machine Windows ni Mac ici.

## Journal

- **2026-10-28** : plan créé. A(a), A(b), A(c1) en PR (#22, #23, #24). Plan des réglages unifié (PR #21),
  assistant de premier démarrage, audio et son spatial intégrés au plan de conception.
- **2026-10-28** : A(c2) en PR. `ajp bench <vidéo>` joue chaque niveau dans un processus enfant (`ajp play --mute
  --no-resume --data-dir=<jetable> --bench-level=N --bench-out=<mesures>`), le premier essai au niveau estimé puis
  `next_level` jusqu'au plus haut qui tient. `--data-dir` isole réglages, profils, `bench.toml` et journal ;
  **les moteurs construits et les modèles restent partagés** (sinon chaque essai reconstruirait ses moteurs :
  des minutes). La mesure de `bench.toml` porte maintenant la carte **et le pilote** ; un pilote qui change la
  rend caduque. Le bouton « Évaluer ma machine » arrive avec le nouveau cadre (étape C) et appellera `run_bench`.
- **2026-10-28** : A/B en PR. `ajp compare <vidéo> --a="OPTIONS" --b="OPTIONS" [--a-settings=play.toml]` joue A, B, B, A
  (le pire des deux passages de chacun, pour qu'un PC qui chauffe ne favorise personne) et met côte à côte pertes,
  retard p95, temps par image et composition. Aucun chiffre n'a été pris : le binaire construit ici est Debug sans
  CUDA. **Décision en attendant : poses désactivées par défaut** (déjà le cas).
- **2026-10-28** : B en PR. `panel::badge` / `panel::tip`, `Badge` et `badge_of` testés, « 720p », « Résolution de travail »,
  libellés de moteurs raccourcis (pastille à la place des parenthèses). Les paragraphes de Fluidité, HDR, Grade… passent en
  infobulles dans l'étape suivante, quand chaque ligne est refaite pour l'arbre.
- **2026-10-28** : C(cadre) en PR. Arbre : Général (Configuration, Lecteur par défaut), Image IA, HDR, Couleurs, Audio,
  Langues et sous-titres, Export, Moteurs, Avancé (Affichage, Réinitialiser, derrière la case). **Décision** : les pages
  de l'image restent modifiables avec un préréglage, et le premier réglage fait à la main bascule en Personnalisé
  (plutôt que des pages grisées). Choisir un préréglage écrit sa chaîne tout de suite (à appliquer par « Appliquer »).
  Pas encore : Mises à jour (étape 7), recherche, repli de l'arbre en liste sur petit écran, lignes multiples.
- **2026-10-28** : assistant en PR. `wizard.h/.cpp` (écrans selon la machine, passer, reprise par `first_run_step`, résumé,
  réglages écrits à « Terminer »). `settings.toml` gagne first_run_step, open_with_registered, update_mode/channel (défaut
  Me prévenir), listening ; `play.toml` gagne [default_player] dismissed. Écran Audio : le choix d'écoute est gardé, la liste
  des appareils vient avec l'étape audio ; l'écran HDR n'apparaît que sur écran HDR ; « Évaluer ma machine » se lance
  depuis Général > Configuration (il faut une vidéo). Pas fait : RTX HDR pas à pas (4b), bouton « Relancer l'assistant ».
- **2026-10-28** : audio en PR. Décisions : mixeur maison entre le décodeur et la file de sortie (pas d'avfilter : les blocs sont
  déjà en float entrelacé et la latence reste nulle) ; appareil stéréo reçoit de la stéréo, appareil multicanal garde les canaux de
  la piste ; changer d'appareil rouvre la vidéo, changer d'écoute se fait en direct. Réglages `[audio]` de play.toml :
  device, listening, faithful, devices.<nom>. Pas fait : la touche A/B, des événements d'ajout/retrait d'appareil (le défaut
  est suivi par SDL), et l'écoute à l'oreille du binaural (aucun casque ici).
- **2026-10-28** : mise à jour signée en PR. **La clé de signature n'existe pas** : `-DAJP_UPDATE_PUBKEY=<hex>` est vide, donc
  le lecteur dit « pas de clé de signature » et ne croit aucun manifeste (le reste est prêt). Pour l'activer, Tatoub (ou le
  coordinateur) crée la clé avec les deux commandes openssl du début de `tools/release/sign-manifest.sh`, compile avec la
  clé publique et signe chaque release avec `AJP_UPDATE_KEY`. Dépôt par défaut : `Ta2b/ajp-releases` (`AJP_RELEASE_REPO`).
  Vérification Ed25519 écrite à la main (`ed25519.cpp`, constantes recalculées, SHA-512 de libavutil) et recoupée avec une
  signature OpenSSL et le vecteur 1 de la RFC 8032. macOS : manuel (le .app ne s'écrase pas) ; Flatpak ou dossier en lecture
  seule : réglage grisé « mis à jour par votre gestionnaire de paquets ». Le défaut reste « Me prévenir ».
- **2026-10-28** : clé de signature créée (le coordinateur a demandé de contourner l'absence de Tatoub). La clé publique est la valeur par
  défaut de `AJP_UPDATE_PUBKEY` ; la privée reste sur le PC de Tatoub. Procédure : `docs/signature-des-releases.md`.
- **2026-10-28** : G (TensorRT Windows) en PR : parité avec Linux (archive = module + trtexec.exe, bibliothèques NVIDIA en lien direct,
  index `engines-nvidia.lock` joint avant l'étiquette). Les scripts PowerShell n'ont pas pu être exécutés ici (pas de pwsh) : l'essai est
  à faire sur le poste Windows.
- **2026-10-28** : restes en PR `claude/restes` : bouton « Relancer l'assistant », touche B (A/B audio), RTX HDR pas à pas (écran 4b :
  moteur TensorRT, SDK Maxine, carte ; « Vérifier de nouveau », « Renoncer : HDR adaptatif », HDR adaptatif en attendant),
  infobulles de Fluidité et d'Audio (descriptions des moteurs sur leurs boutons), pastilles centrées. **Vérification visuelle** faite
  avec `--snapshot` sur la 4070 Ti : assistant (écran 1) et menu (Configuration, Fluidité, Audio) ; corrigé : double barre de
  défilement (la fenêtre n'en a plus, les deux colonnes défilent seules). Pas encore vus : les écrans 2 à 8 de l'assistant, HDR et
  Couleurs, Mises à jour.
- **2026-10-28** : préparation de la release : version 0.5.0 (CMake, vcpkg), notes v0.5.0, liste Windows/macOS. L'étiquette n'est PAS posée :
  règle de Tatoub, rien n'est publié ni étiqueté sans permission explicite.
- **2026-10-28** : mesures du mode Auto et des moteurs sur la 4070 Ti : `docs/reports/mesures-auto-4070ti-2026-10-28.md`.
- **2026-10-28 (soir)** : #37 (version, notes, liste Windows/macOS), #38 (mode Auto), #40 (GMFSS/DistilDRBA : décision), #42 (clang-tidy), #43 (mesures) fusionnées. Étiquette `v0.5.0` posée à 08:48 sur
  l'instruction de Tatoub. Mesures GMFSS Union (lot, recouvrement au trtexec) demandées par l'audit : en cours, hors v0.5.
- **2026-10-28 (matin)** : v0.5.0 étiquetée, Linux et macOS publiés ; **le paquet TensorRT Linux avait été oublié** (l'index de la release n'avait pas `tensorrt`) : construit dans le conteneur de préflight au commit de l'étiquette, joint avec `engines-nvidia.lock`, Release Linux relancée (`--ref v0.5.0`). L'image de préflight datait d'avant `--disable-bzlib` : `tools/release/preflight/nvidia-package.sh` (PR #45) la reconstruit.
