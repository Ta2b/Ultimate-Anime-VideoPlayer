# 0012 — Un lecteur simple : un seul HDR, qui suit le système, et un menu en trois catégories

## Contexte
Avant la release, Tatoub a demandé un grand nettoyage : un code qu'une autre personne (ou une IA) puisse
reprendre, et un menu très léger. Le lecteur portait plusieurs HDR (RTX HDR, émulation TrueHDR, HDR adaptatif
avec sa calibration, grade HDR anime), neuf profils et un profil de démarrage, un bouton « Méthode HDR » et un
arbre de douze pages en cinq catégories. Chaque chemin devait être tenu, testé et expliqué.

## Décision
- **Un seul grade HDR** : le grade HDR anime guidé par plan (`src/render/shaders/anime-hdr-anime/`). RTX HDR
  (0001), l'émulation TrueHDR et le HDR adaptatif sont retirés (PR #530).
- **Pas de profils** (PR #531) : `play.toml` garde la dernière chaîne jouée ; `user.toml` ne fournit plus que
  les raccourcis `[keys]` (`src/app/play/user_keys.h`). Les anciennes clés `profile_N` sont ignorées. Ctrl+0
  (sans traitement) reste.
- **Le HDR suit le système, sans bouton** (PR #532) : `system_hdr` (`src/app/play/host_facts.h`) met la sortie
  en HDR10 quand le système envoie du HDR et que la crête de l'écran atteint 350 nits ; sinon SDR gradée. L'état
  est relu toutes les 2 s pendant la lecture. `--sdr` force le SDR. Le pic visé reste un réglage avancé.
- **Trois catégories** (PR #533) : Vidéo (Post-traitement, Couleurs, Affichage), Audio et lecture (Son,
  Lecture), Général (Application, Moteurs, Export, Statistiques, À propos). Le niveau de traitement (Auto,
  Économe à Ultra) applique `apply_preset` ; le moteur est un réglage général.
- **Moteurs** : tout ce qui n'est pas NVIDIA passe par Vulkan. MIGraphX, OpenVINO et GMFSS restent dans les
  sources mais sont débranchés de la lecture (`docs/moteurs.md`, `docs/interpolation.md`).

## Conséquences
- Un ancien `play.toml` ou `user.toml` se lit toujours : les clés retirées sont ignorées.
- Un écran « HDR » sous 350 nits reste en SDR même si le système envoie du HDR ; seul `kMinHdrPeakNits` le
  change.
- Les documents d'avant ce tri sont marqués historiques (`docs/README.md`).
