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
| Son spatial : avfilter ou code maison, nombre de profils HRTF | à trancher en chantier E, en le notant ici | Claude |
| Clé de signature des mises à jour | si elle manque, ce qui n'en dépend pas avance et le reste est noté | Claude |

## Chantiers, dans l'ordre

### A. Configuration automatique
- [x] (a) Classement des cartes en tranches, `data/gpu_tiers.toml` : PR #22
- [x] (b) Préréglages, priorité, application de la chaîne : PR #23 (base : #22)
- [x] (c1) Décision du benchmark et `bench.toml` : PR #24 (base : #23)
- [x] (c2) `ajp bench` : lancement des essais (processus enfants isolés, son coupé), rapport `--report`, pilote dans la clé de mesure : PR de la branche `claude/bench-run`
- [ ] Bouton « Évaluer ma machine » (arrive avec C)
- [ ] Comparateur A/B (poses DRBA, DistilDRBA contre RIFE) : mesure puis décision
- [ ] Bascule automatique en Personnalisé quand un réglage de chaîne est modifié à la main (avec C)

### B. Textes, pastilles, infobulles
- [ ] 720p au lieu de « 720 lignes », vocabulaire courant
- [ ] `badge()` et pastilles, infobulles à la place des paragraphes
- [ ] Hermite en réduction (voir F)

### C. Nouveau cadre des réglages et assistant
- [ ] Arbre à gauche, page à droite, encadrés titrés, lignes multiples, barre de défilement fine
- [ ] 9 catégories, case « Réglages avancés », Profils retiré, dernière page mémorisée
- [ ] Assistant de premier démarrage (8 écrans), relançable depuis Général et par `--first-run`
- [ ] « Ouvrir avec » au premier lancement, sans confirmation ; bouton « Faire de ce lecteur le lecteur par défaut »
  avec « Ne plus me le demander » (réapparaît en bas d'Avancé)
- [ ] Recherche dans les réglages (par page d'abord)

### E. Audio et son spatial
- [ ] Liste et choix de l'appareil de sortie, événements de changement du défaut
- [ ] Détecteur de formats (source par piste, sortie par appareil) et conversion automatique
- [ ] Profils Casque / Enceintes / Home cinéma, crossfeed, downmix soigné, profil par appareil, Fidèle, touche A/B
- [ ] Binaural
- [ ] Écran audio de l'assistant et page Audio des réglages

### D. Mise à jour automatique signée
- [ ] Vérification en arrière-plan et réglages (Me prévenir / Automatique / Jamais, canal, fréquence)
- [ ] Manifeste signé (Ed25519), téléchargement, application au prochain lancement, retour arrière
- [ ] Essais Linux puis Windows

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
