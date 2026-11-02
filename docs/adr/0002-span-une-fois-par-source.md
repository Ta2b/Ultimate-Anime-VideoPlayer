# 0002 — SPAN une fois par image source, le warp après SPAN

## Contexte
SPAN (AnimeJaNai) est le réseau le plus cher de la chaîne : 1080p → 4K, environ 8 ms par passe sur une RTX
4070 Ti, contre 0,4 ms pour un warp. Agrandir chaque image créée par l'interpolation multiplie ce coût par le
facteur (×3 : trois passes par image source).

## Décision
Les moteurs de mouvement rendent un flot et un masque (RIFE mouvement, DistilDRBA mouvement, GMFSS) ; SPAN
agrandit chaque image source une fois, et le warp déforme les images agrandies à la taille de sortie, écrites
directement dans l'anneau. En Auto, le warp passe après SPAN dès que la sortie dépasse 1080 lignes ou que le
facteur peut atteindre ×3 (`warp_after_span`, `top_factor`). Le mouvement s'estime à 720p quand le travail est
plus grand (RIFE et DistilDRBA).

## Conséquences
- En 4K ×2,5, environ la moitié du temps GPU des moteurs qui agrandissaient chaque image créée.
- SPAN ne nettoie plus les défauts du warp : « Avant SPAN » reste un réglage pour qui préfère ce rendu.
