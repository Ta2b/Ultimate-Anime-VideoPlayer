# Séance de sonde : protocole (chantier 8.1, repris en 8.12)

Ce qu'on mesure : ce que l'écran affiche vraiment pour un contenu donné, avant de toucher au contrat d'écran
(8.2) et aux métadonnées (8.4). Attendu aujourd'hui : 203 nits de contenu → environ 140 nits à la sonde.
Résultats dans `docs/reports/sonde-<date>.md`.

## Matériel et préparation

- i1Display Pro et ArgyllCMS ; un `.ccss` QD-OLED s'il existe (installé par `oeminst`), sinon noter « sans
  correction ».
- Écran en mode HDR « Peak 1000 », 30 min de chauffe, pièce noire, luminosité à 100 %.
- Noter la luminosité SDR de KDE (265 nits, `docs/reports/display.md`) et la crête forcée (1000).

## Mires

```
python3 -I tools/grade/patterns.py ~/mires --size 2560x1440     # adapter à la définition de l'écran
```

- `patterns-hdr10.mkv` (HEVC 10 bits, PQ BT.2020) : fenêtres de 2, 10, 25, 50 et 100 % de l'aire à 203, 400,
  766 et 1 000 nits de contenu ; puis gris 18 % (26 nits, BT.2408) au centre dans un cadre à 1 000 nits qui
  couvre 0 à 60 % de l'aire. Une mire par chapitre, 6 s chacune.
- `patterns-sdr.mkv` (BT.709) : blanc, gris 25/50/75 %, peau sRGB 255/224/196.

## Lecture et mesure

```
ajp play ~/mires/patterns-hdr10.mkv --fullscreen --mute --no-resume --data-dir=/tmp/ajp-probe
```

Pause à chaque chapitre, sonde au centre, trois lectures par mire, la médiane retenue :

```
spotread -x -O -y e [-X qdoled.ccss] mesures.txt     # drapeaux à vérifier avec spotread -?
```

Ce que le lecteur annonce au compositeur :

```
WAYLAND_DEBUG=client ajp play … 2>&1 | grep -E "set_mastering|set_max_cll|set_max_fall|set_luminances|set_primaries"
```

## Tableau des résultats

| Mire | Contenu (nits) | Mesuré (nits, médiane) | Écart |
| --- | --- | --- | --- |
| fenêtre 10 % | 203 | | |
| … | | | |

Pour le cadre qui grandit : le gris central mesuré à chaque taille de cadre (critère 8.12 : stable à 3 %).
