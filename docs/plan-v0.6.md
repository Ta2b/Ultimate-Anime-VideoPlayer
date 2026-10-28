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
- [ ] HDR : RTX HDR actif par défaut sans être installé ; options HDR sélectionnables sur un écran SDR ; mode SDR qui n'applique pas le grade ; HDR adaptatif sur écran SDR
- [ ] Interface : lecteur par défaut en fenêtre au démarrage (« ne plus demander ») et réglage discret (Général > Mises à jour), composition finale et soin OLED hors de Couleurs, **audit complet « Réglages avancés »** (tableau dans `ux-parametres.md`, les réglages rares restent dans leur onglet), OLED auto si écran OLED détecté, Réinitialiser et debug hors des réglages avancés
- [ ] Benchmark autonome et silencieux (clip sévère embarqué 5 à 15 s, barre de progression, étapes « Chargement de tous les moteurs nécessaires au bench » puis « Benchmark en cours », recherche des modèles et résolutions, multiplicateur en Auto, plus d'écran de démarrage à chaque essai)
- [ ] Moteurs : espace disque vérifié avant l'installation (plus de boucle de téléchargement), dossier d'installation des moteurs au choix avec un défaut
- [ ] Audio : plusieurs profils de virtualisation nommés (Naturel, Binaural, Scène large, Cinéma), réglages par profil derrière « Réglages avancés » (largeur de scène, distance et angle, crossfeed, réverbération légère, centre, graves, égaliseur 3 bandes, compression « nuit », normalisation), profils personnalisés enregistrables, profil par appareil
- [ ] Empaquetage : AppImage (Linux), installateur .exe avec version portable zip (Windows), .dmg noté pour plus tard (macOS)
- [ ] Release v0.6 : version 0.6.0, notes `tools/release/notes/v0.6.0.md`, liste des essais Windows/macOS

## Journal
- **2026-10-28** : v0.6 ouverte. Mesures du contrôleur Auto sur la tête 00f00ee : RIFE ×4, DistilDRBA ×5, GMFSS ×2 tenus en fin de run (`docs/reports/mesures-auto-4070ti-2026-10-28.md`).
