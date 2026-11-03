# 0007 — SPAN basse précision par défaut

## Contexte
SPAN coûte la plus grosse part d'une image en lecture. Le paquet FP8 des noyaux CUTLASS (`span-fp8-cutlass.md`)
le calcule 34 à 37 % plus vite que TensorRT FP16 sur une RTX 4070 Ti, avec une puissance moitié moindre à 24 i/s,
mais il n'était pris qu'au cran Performance choisi à la main, pour le seul b8f64, à quelques tailles, avec une
porte jugée sur la moyenne d'images réelles et un reçu qui ne disait ni la carte ni la version des noyaux.

## Décision
Les préréglages posent SPAN au cran Performance, au sens de « la plus basse précision qui passe la porte » :
1. le paquet CUTLASS FP8, si un reçu v2 de cette architecture, de cette version des noyaux et de cette taille de
   travail passe la porte durcie (99,9e centile de l'erreur, pixels sombres, neuf cas durs synthétiques, images
   ramenées à la taille de travail par le filtre de réduction de la chaîne) ;
2. sinon l'INT8 qui passe sa porte ;
3. sinon le SPAN tel que livré en FP16.

Le FP8 par Q/DQ (seuils lâches) et un reçu v1 ne servent pas l'Auto. La préparation du paquet se fait en
silence, sans rouvrir la chaîne, sur un flux CUDA à la priorité la plus basse, refusée si la VRAM libre est sous
1,5 × le besoin ; le paquet sert à l'ouverture suivante. Une taille nouvelle ne demande qu'une validation seule.

## Conséquences
- Aucun nouveau réglage ; le préréglage Personnalisé garde le choix et l'ordre d'avant.
- Le b5f48 et toute taille paire sont pris (étapes 3.1, 3.2) ; le corps du réseau part en graphe CUDA (3.3) ;
  DepthToSpace et l'écriture de l'anneau sont fondus à `to_pixel` (3.4).
- Un paquet d'une carte ne sert pas sur une autre sans revalidation ; changer les noyaux demande d'augmenter
  `SpanFp8Net::kKernelsVersion`.
- `kBackendAbi` est monté à chaque étape (21 à 25) : les backends installés se réinstallent.
- À mesurer sur la RTX avant de tenir la décision pour acquise : seuils de la porte ramenés à 1,25 × ceux du
  b8f64 accepté, b5f48 FP8 ≤ 0,72 × FP16 en 720p et ≤ 0,70 en 1080p, 1 graphe et ≤ 2 lancements par SPAN,
  memcheck à 0 de 480p à 1080p, SPAN −30 % et J par image −25 % à 24 i/s. Si l'un manque, l'Auto repasse à
  Équilibre par `auto_config.cpp` et cette page est remplacée.
