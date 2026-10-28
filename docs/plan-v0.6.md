# Plan de la v0.6

Les retours de Tatoub du 28 octobre (après la v0.5.0, publiée en pré-version) forment la **v0.6**, pas une v0.5.1. La version CMake passe à 0.6.0 **à la
préparation de la release v0.6**, pas avant. Même règle qu'en v0.5 : une PR par étape, base main, clang-format avant chaque PR ; l'étiquette n'est posée
que sur un accord clair de Tatoub dans le fil. Détail des décisions et journal : ce fichier ; l'historique de la 0.5 : `plan-v0.5.md`.

## Étapes, dans l'ordre (état au 28/10, 15:20)
- [x] Catalogue RIFE réduit à 4.25 et 4.22 lite (#48) ; menu SPAN réduit au catalogue AnimeJaNai HD V3.1 (#55) : plus de profil « legacy » dans les listes
- [x] Fenêtre fantôme à la fermeture (Fedora, Wayland, GNOME) : correctif défendable par la lecture du code (#49) ; **à confirmer sur le Fedora GNOME**
- [x] Interpolation Auto par défaut pour tous les profils (#53)
- [ ] « Garder les poses DRBA » par défaut : **performance mesurée** (poses actives : tient, retard 0,0 ms, composition 1,2 ms ; inactives : ne tient pas, 15,1 ms, 6,0 ms ; 4070 Ti, Re:Zero 18, ×3) ; qualité : comparaison d'images en cours, puis décision
- [x] HDR : écran SDR sans conversion HDR (options grisées, note), mode SDR gradé pour imiter le rendu HDR, RTX HDR non installé : repli sur l'adaptatif (#68, #69) ; **à confirmer chez Tatoub**
- [x] Interface : style Showtime (commandes rondes au centre, titre en gras), fenêtre « lecteur par défaut » au démarrage, réglages avancés dans les onglets existants sans catégorie Avancé, Statistiques et Réinitialiser dans Général (#71, #72) ; reste le menu : molette, barre de défilement, libellés, disposition horizontale
- [ ] Benchmark autonome et silencieux : séquence de test écrite par le lecteur, fenêtre minuscule, étapes annoncées (#75) ; **reste** l'exploration des moteurs, multiplicateurs et résolutions, Auto en dernier
- [x] Moteurs : espace disque vérifié avant l'installation, dossier d'installation au choix (#54) ; installation TensorRT corrigée : architecture du GPU, progression unique, un seul essai (#60) ; paquet TensorRT découplé des étiquettes (#67)
- [x] Audio : profils de virtualisation (#64), distance, pièce, égaliseur, nuit, profil par appareil (#80), **à écouter** ; [ ] convolution HRIR à la HeSuVi (#82, en revue)
- [x] Empaquetage : AppImage (#74, essai local #77), installeur .exe Inno Setup + zip portable (#74, **à valider par la prochaine release Windows**), .dmg (#79)
- [x] Police : repli CJK vérifié (#59), chargeur FreeType pour le CFF2 de Fedora (#70) ; **à confirmer chez Tatoub**
- [x] CI allégée (#73) ; dépôt des releases séparé **gardé** (décision de Tatoub, 28/10)
- [ ] Release v0.6 : version 0.6.0, notes `tools/release/notes/v0.6.0.md` (brouillon #81), liste des essais Windows/macOS ; étiquette seulement sur le message direct de Tatoub

## Journal
- **2026-10-28** : v0.6 ouverte. Mesures du contrôleur Auto sur la tête 00f00ee : RIFE ×4, DistilDRBA ×5, GMFSS ×2 tenus en fin de run (`docs/reports/mesures-auto-4070ti-2026-10-28.md`).
- [x] Paquet TensorRT découplé des étiquettes : empreinte de ses entrées (`trt-fingerprint.sh`), release fixe `trt-<empreinte>` (`trt-package.sh`), Release Linux la lit sans relance ; plus de reconstruction ni de seconde passe par release (demande de Tatoub, 0.5.1)

## Liste vivante du détail demandé par Tatoub (28/10) : état

1. Audio : FAIT (#64, #80) : profils, distance, pièce, égaliseur, nuit, profil par appareil ; à écouter.
2. Réglages avancés : FAIT (#61, #72) : dans les onglets existants, OLED automatique, Réinitialiser et Statistiques hors des avancés.
3. Pop-up « lecteur par défaut » et réglage discret : FAIT (#71) ; composition finale et soins OLED hors de Couleurs : FAIT (#72, dans Agrandissement et HDR avec les avancés).
4. HDR : FAIT (#68, #69) ; à confirmer chez Tatoub.
5. Benchmark autonome et silencieux : EN PARTIE (#75) ; reste l'exploration des moteurs, multiplicateurs et résolutions.
6. Poses DRBA par défaut : EN COURS : performance mesurée (favorable), qualité à comparer.
7. Design Showtime FAIT (#71) ; AppImage FAIT (#74, #77) ; installeur .exe + zip portable FAIT (#74), à valider par la release Windows ; .dmg FAIT (#79).
8. Fedora GNOME : fenêtre fantôme (#49) et japonais (#59, #70) : À CONFIRMER chez Tatoub.
9. Convolution HRIR à la HeSuVi : EN REVUE (#82) ; préréglage « Voix d'anime » FAIT (#80).
10. Auto en VRR, demi-paliers ×0,5 : À FAIRE (trancher, mesurer, répondre).
11. Menu : molette, barre de défilement, moins de catégories, disposition horizontale, libellés français : À FAIRE (pastille « Le plus lent » FAITE).
12. Arbre : EN PARTIE (#72) ; à revoir : place de Langues, Export, Moteurs, Couleurs sans réglages sans rapport.
13. « Ouvrir avec » enregistré au premier lancement : FAIT (assistant).
14. Config auto (tranche du GPU et résolution de l'écran, réduire puis agrandir, SPAN Sharp1 Performance, modèle par tranche) : FAIT (#53, tests `[auto_config]`).
15. Assistant : choix guidé du moteur : À FAIRE.
16. Poses DRBA : contradiction du tableau de décisions (« désactivées ») : à corriger après la décision (point 6).
17. .dmg FAIT (#79) ; archive TensorRT Windows manquante : À FAIRE (même schéma que Linux, #67) ; build macOS corrigé (#66).
18. Police japonaise en 0.5.1 : CORRIGÉE (#70), à confirmer chez Tatoub.
