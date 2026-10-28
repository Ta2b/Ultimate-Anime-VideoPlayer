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
| Clé de signature des mises à jour | si elle manque, ce qui n'en dépend pas avance et le reste est noté | Claude |

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
- [~] `badge()` et pastilles faits (IA : résolution, réduction, moteurs, Smooth Motion) ; infobulles à la place des paragraphes : onglet Image fait, Fluidité et le reste suivent avec le cadre en arbre
- [ ] Hermite en réduction (voir F)

### C. Nouveau cadre des réglages et assistant
- [x] Arbre à gauche, page à droite, encadrés titrés (PR `claude/cadre-arbre`) ; lignes multiples et barre de défilement fine restent à faire
- [x] Catégories, case « Réglages avancés », Profils retiré ; dernière page mémorisée le temps de la session seulement (pas encore dans `play.toml`)
- [x] Assistant de premier démarrage (8 écrans, `wizard.h`), relançable par `--first-run` ; le bouton « Relancer l'assistant » de Général reste à faire (PR `claude/assistant`)
- [x] « Ouvrir avec » au premier lancement, sans confirmation ; bouton « Faire de ce lecteur le lecteur par défaut »
  avec « Ne plus me le demander » (réapparaît en bas d'Avancé), détection xdg-mime sous Linux
- [ ] Recherche dans les réglages (par page d'abord)

### E. Audio et son spatial
- [x] Liste et choix de l'appareil de sortie (`audio/devices.h`) ; le défaut du système est suivi par SDL (flux sur l'appareil par défaut) : PR `claude/audio`
- [x] Format de la source par piste et de la sortie par appareil, conversion automatique (le décodeur rend les canaux voulus par le mixeur)
- [x] Profils Casque / Enceintes / Home cinéma, crossfeed, downmix soigné, profil par appareil, Fidèle (case + commande `SetListening`) ; la touche A/B reste à faire (nouvelle action de touche)
- [x] Binaural
- [x] Écran audio de l'assistant et page Audio des réglages (Audio > Sortie : Appareil, Écoute, Synchronisation)

### D. Mise à jour automatique signée
- [x] Vérification en arrière-plan et réglages (Me prévenir / Automatique / Jamais, canal, fréquence) : Général > Mises à jour (PR `claude/maj`)
- [x] Manifeste signé (Ed25519, `tools/release/sign-manifest.sh`), téléchargement vérifié (taille + SHA-256), application au prochain lancement, retour arrière (testés sur archive locale)
- [ ] Essais Linux puis Windows de bout en bout : demandent une release signée (clé absente, voir journal) ; Windows renomme les fichiers en service en `.old`, non essayé

### F. Moteur
- [ ] Hermite en réduction (`DownscaleFilter::Hermite`)
- [ ] GMFSS Union : optimisations
- [ ] DistilDRBA : warps hors du graphe
- [ ] Mode Auto : suivi par le fil d'audit (ne pas toucher `auto_rate`, `upscale_stage`, `live_settings` sans coordination)

### G. TensorRT sous Windows
- [ ] `trtexec.exe` dans l'index, condition du workflow `release-windows.yml`, essai sur un poste Windows

### Release
- [ ] Tests locaux complets, notes de version, étiquette v0.5.0 **seulement sur autorisation explicite** (règle de
  Tatoub : rien n'est publié ni étiqueté sans permission ; si elle n'est pas donnée, la release reste prête)

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
