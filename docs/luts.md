# LUT 3D (.cube) du grade HDR anime

Le grade HDR anime (`docs/hdr-anime.md`) passe l'image par une pile de LUT 3D d'HDR (`anime_look_layers`,
`src/app/play/look_catalog.cpp`), appliquée par `anime-hdr-look.hook` après le grade et avant le vrai noir
OLED. Le shader parcourt la pile LUT par LUT, chacune lue telle quelle et mélangée à son intensité (11 LUT par
défaut, 19 aux curseurs à 100, quatre textures : une par fichier). Jusqu'en novembre 2026 la pile était réduite
à une table de 65 points : entre deux points, une table interpole la pile entière au lieu de la parcourir, et
l'image s'en écartait jusqu'à 12,7 codes PQ sur 1 023 (7,5 avec 129 points ; mesure cachée
`mesure : écart de la table composée à la pile exacte` de `tests/unit/render`, audit A41-01). Un curseur du
look ne change plus que des intensités, sans rien recalculer ; les LUT ne sont relues que si leur nombre
change. `--bake-looks` écrit toujours cette table de 65 points, pour un autre lecteur.

Une correction entoure la pile depuis novembre 2026 (`docs/hdr-anime.md`, « Niveaux à l'écran ») : sous RVB 48,
un pied adouci remplace le bas de la LUT de base, qui rendait noir tout ce qui était sous RVB 16. La dernière
passe ramène le blanc que la pile porte vers 6 300 nits sous la courbe de KWin, vers 70 % du pic de l'écran :
les curseurs de contraste et d'éclats le déplacent.

## La pile
Quatre fichiers de `data/luts/`, livrés dans un dossier `luts/` à côté du lecteur (Contents/Resources dans
AJP.app) :

| Ordre | LUT | Intensité au réglage 0 |
|---|---|---|
| 1 | `HDR grading base ver.9.0 33 point.cube` | 1 |
| 2 | `HDR-contrast-Enhancement-L.cube` | 3/7 du contraste (×7 en tout) |
| 3 | `Highlight-Enhancement-L.cube` | ×1,5 |
| 4 | `skin fix.cube` | 1 |
| 5 | `HDR-contrast-Enhancement-L.cube` | les 4/7 restants du contraste |

Deux curseurs avancés (Vidéo > Couleurs : `look_contrast`, `look_highlight` de `play.toml`, de −100 à 100)
règlent l'intensité du contraste et des éclats : −100 retire la LUT, 100 double son intensité. Une intensité
de plus de 1 devient des LUT entières, puis une partielle (`anime_look_mixes`).

Un fichier du même nom dans le dossier `luts` des données de l'utilisateur (Linux : `~/.local/share/ajp/ajp/luts`)
prend le pas sur celui du lecteur.

## Lecture des fichiers
- LUT 3D, `LUT_3D_SIZE` de 17 à 65, `DOMAIN_MIN`/`DOMAIN_MAX` et `LUT_3D_INPUT_RANGE` lus ; une LUT 1D ou un
  fichier illisible est refusé avec une ligne de journal « look : … » (`src/render/cube_lut.cpp`).
- Ce sont des LUT d'HDR, étalonnées en codes PQ BT.2020 : les codes PQ de l'image entrent directement dans la
  LUT. Sur une sortie SDR, la pile est coupée.
- Certaines LUT d'HDR rendent du noir pur hors de leur gamut d'entrée (un tiers du cube pour la LUT de base).
  Au chargement d'une LUT livrée, ces entrées sont recopiées du voisin valide le plus proche (le journal dit combien), pour que
  l'interpolation n'assombrisse pas les pixels du bord.

## Retiré
- Le choix et l'empilement de looks au menu, les looks « maison », les clés `look`, `look_mix`, `looks` et
  `looks_mix` de `play.toml` : un ancien fichier qui les porte se lit, elles sont ignorées.
- Les 51 autres LUT du pack « HDR LUT ver.9.1 » et la pile cuite en une LUT (`HDR pile unique 65 point.cube`),
  que rien ne chargeait (0.10.0).
- Le grade adaptatif : les clés `adaptive*`, `vibrance*`, `guard`, `expand`, `gamut_p3` de `[grade]` et
  `contrast`, `saturation`, `middle_gray`, `display_grades` de `[hdr]` sont ignorées.

## Origine et licence
Voir `data/luts/ATTRIBUTION.md` : auteur et licence non indiqués dans l'archive d'origine, redistribution sous la
responsabilité du propriétaire du dépôt.
