# Plan de la v0.6

Les retours de Tatoub du 28 octobre (après la v0.5.0, publiée en pré-version) forment la **v0.6**, pas une v0.5.1. La version CMake passe à 0.6.0 **à la
préparation de la release v0.6**, pas avant. Même règle qu'en v0.5 : une PR par étape, base main, clang-format avant chaque PR ; l'étiquette n'est posée
que sur un accord clair de Tatoub dans le fil. Détail des décisions et journal : ce fichier ; l'historique de la 0.5 : `plan-v0.5.md`.

## Étapes, dans l'ordre
- [x] Catalogue RIFE réduit à 4.25 et 4.22 lite (PR #48)
- [x] Fenêtre fantôme à la fermeture (Fedora, Wayland, GNOME) : correctif défendable par la lecture du code (PR #49) ; **à confirmer sur le Fedora GNOME**
- [ ] Interpolation Auto par défaut pour tous les profils, y compris après le benchmark
- [ ] « Garder les poses DRBA » activé par défaut, **si** la mesure A/B montre un gain de performance ET de qualité (sinon, chiffres au coordinateur)
- [ ] Profils « legacy » retirés de toutes les listes de sélection (réglages, assistant, benchmark), sans casser l'import d'un ancien play.toml
- [x] HDR : écran SDR sans route HDR (options grisées avec note), mode SDR GRADÉ pour imiter le rendu HDR (onglet Couleurs actif ; sur sortie HDR10, l'image SDR est gradée comme la route native sans l'aspect adaptatif), RTX HDR non installé : repli sur le HDR adaptatif ; à confirmer chez Tatoub
- [ ] Interface : lecteur par défaut en fenêtre au démarrage (« ne plus demander ») et réglage discret (Général > Mises à jour), composition finale et soin OLED hors de Couleurs, **audit complet « Réglages avancés »** (tableau dans `ux-parametres.md`, les réglages rares restent dans leur onglet), OLED auto si écran OLED détecté, Réinitialiser et debug hors des réglages avancés
- [ ] Benchmark autonome et silencieux (clip sévère embarqué 5 à 15 s, barre de progression, étapes « Chargement de tous les moteurs nécessaires au bench » puis « Benchmark en cours », recherche des modèles et résolutions, multiplicateur en Auto, plus d'écran de démarrage à chaque essai)
- [ ] Moteurs : espace disque vérifié avant l'installation (plus de boucle de téléchargement), dossier d'installation des moteurs au choix avec un défaut
- [ ] Audio : plusieurs profils de virtualisation nommés (Naturel, Binaural, Scène large, Cinéma), réglages par profil derrière « Réglages avancés » (largeur de scène, distance et angle, crossfeed, réverbération légère, centre, graves, égaliseur 3 bandes, compression « nuit », normalisation), profils personnalisés enregistrables, profil par appareil
- [ ] Empaquetage : AppImage (Linux), installateur .exe avec version portable zip (Windows), .dmg noté pour plus tard (macOS)
- [ ] Release v0.6 : version 0.6.0, notes `tools/release/notes/v0.6.0.md`, liste des essais Windows/macOS

## Journal
- **2026-10-28** : v0.6 ouverte. Mesures du contrôleur Auto sur la tête 00f00ee : RIFE ×4, DistilDRBA ×5, GMFSS ×2 tenus en fin de run (`docs/reports/mesures-auto-4070ti-2026-10-28.md`).
- [x] Paquet TensorRT découplé des étiquettes : empreinte de ses entrées (`trt-fingerprint.sh`), release fixe `trt-<empreinte>` (`trt-package.sh`), Release Linux la lit sans relance ; plus de reconstruction ni de seconde passe par release (demande de Tatoub, 0.5.1)

## Liste vivante du détail demandé par Tatoub (28/10, à n'oublier aucun avant la 0.6)

1. Audio : profils faits (#64) ; reste distance des enceintes virtuelles, pièce/réverbération légère, égaliseur, compression de dynamique « nuit », profils nommés enregistrables par appareil.
2. Réglages avancés : audit fait (#61) ; reste la page avancée « Rendu », l'OLED coché automatiquement à la détection, Réinitialiser hors des avancés, debug hors d'Avancé, les avancés prolongent les onglets existants.
3. Pop-up de démarrage « lecteur par défaut » avec « ne plus demander » + réglage discret ; composition finale et soin OLED hors de Couleurs.
4. HDR : fait (#68) — RTX HDR non installé : repli ; options HDR grisées sur écran SDR ; le mode SDR APPLIQUE le grade pour imiter le rendu HDR (voulu par Tatoub) ; à confirmer chez lui.
5. Benchmark autonome et silencieux : clip de 5 à 15 s généré ou choisi sans ouvrir de vidéo, aucune image, barre de progression seule, étapes « Chargement de tous les moteurs nécessaires au bench » puis « Benchmark en cours », explore moteurs, multiplicateurs et résolutions, multiplicateur Auto à la fin.
6. Poses DRBA par défaut après A/B performance ET qualité.
7. Design Showtime, AppImage, installeur .exe + zip portable, .dmg.
8. Confirmer sur Fedora GNOME : fenêtre fantôme à la fermeture, japonais.
9. Audio : convolution par réponse impulsionnelle à la HeSuVi (import WAV HRIR 14 canaux / stéréo / 7.1, FFT par blocs, rééchantillonnage, gain de sécurité, par appareil) ; aucune réponse tierce sans licence vérifiée ; par défaut le binaural paramétrique ; préréglage égaliseur/centre voix « anime ».
10. Auto en VRR : question ouverte des demi-paliers ×0,5 (cadence demi-entière) : trancher, mesurer, répondre à Tatoub.
11. Menu : la molette défile les onglets et catégories ; placement de la barre de défilement ; moins de catégories ; réglages en disposition horizontale quand ça tient ; tous les libellés français corrigés (1080p, etc.) ; pastille « Le plus lent ».
12. Arbre : place explicite de Langues, Export, Moteurs ; Général simple ; IA ; Couleurs sans Lanczos ni réglages sans rapport.
13. « Ouvrir avec » enregistré au premier lancement sans confirmation.
14. Config auto : tranche du GPU + résolution de l'écran, réduire puis agrandir (540p→1080p, 720p→1440p), SPAN Sharp1 Performance par défaut, modèle d'interpolation par tranche.
15. Assistant : choix guidé du moteur.
16. Poses DRBA : contradiction entre le tableau de décisions (« désactivées ») et la v0.6 (« activées par défaut ») : trancher après A/B performance ET qualité, corriger le document.
17. .dmg macOS à faire ; archive TensorRT Windows manquante (même schéma que Linux, #67) ; build macOS corrigé (#66).
18. Police japonaise toujours absente en 0.5.1 : enquête en cours (police CJK .ttc en CFF sous stb_truetype ? atlas dynamique ?).
