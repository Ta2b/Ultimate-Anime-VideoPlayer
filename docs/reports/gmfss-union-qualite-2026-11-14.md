# GMFSS Fortuna Union face à RIFE 4.26 : qualité sur Re:Zero (14 novembre 2026)

> **Relevé daté** : une mesure ou une séance à sa date, gardée comme trace. Le code a pu changer depuis ; voir [`docs/README.md`](../README.md).

Question de Tatoub : réintroduire GMFSS Fortuna Union (retiré en 0.10.0) comme niveau d'interpolation au-dessus
d'Équilibré. Le guide demande une mesure face à RIFE avant tout retour ; la voici, pour la qualité. Les temps
(`tools/gmfss/banc_temps.sh`) restent à prendre sur une carte NVIDIA.

## Montage

- `tools/bench/qualite/gmfss_instants.py` : GMFSS en PyTorch (processeur), réseau et poids de la roue
  `vsgmfss_fortuna` 1.0.0, coupé comme l'export du lecteur (`tools/gmfss/gmfss_graph.py`) ; le montage rend GMFSS
  d'origine à moins de 1e-6. RIFE 4.26 par le graphe livré (flot et masque, warp de l'étage). Tout à 720p de
  travail, extrait `rezero-s4e18-1710.mkv` (1080p, 23,976 i/s).
- Vérité : des suites d'images toutes différentes d'un même plan (animé en uns) ; k et k+N forment la paire, les
  images du milieu sont les vérités à i/N. PSNR, pire bloc 64 × 64 et LPIPS (AlexNet ; plus bas, plus proche à l'œil).
- `tools/bench/qualite/gmfss_tenus.py` : vidéo côte à côte des dessins tenus lissés (`smooth_hold_u`), sans vérité.

## Résultats

Paires proches (k, k+3 ; 16 paires, 32 instants, mouvement moyen 7 px à 720p ; 4 instants d'un éclair compris) :

| Méthode | PSNR (dB) | LPIPS |
|---|---:|---:|
| RIFE 4.26 | 30,40 | 0,0734 |
| GMFSS union, flot entier | 29,92 | 0,0745 |
| GMFSS union, demi-flot (lecteur d'avant 0.10.0) | 29,47 | 0,0774 |
| « new union model » de 98mxr (affiné AnimeRun) | 30,18 | 0,0788 |
| Une fusion union au milieu, RIFE 4.26 sur chaque demi-paire | 30,12 | 0,0739 |
| Une fusion au milieu, autres instants déduits par le flot de GMFSS | 30,62 | 0,0903 |
| Splats seuls (sans GridNet) | 29,62 | 0,1092 |

Paires espacées (k, k+6 ; 10 paires, 50 instants, jusqu'à 76 px) : RIFE 4.26 27,22 dB / 0,139 ; union 26,52 /
0,145 ; nouveau union 26,68 / 0,154 ; milieu union + RIFE 27,03 / 0,143. RIFE 4.26 a le meilleur LPIPS sur
chacune des 10 paires.

## Lecture

- RIFE 4.26 égale ou bat union partout, en PSNR comme en LPIPS, petits et grands mouvements.
- Le demi-flot de l'ancien lecteur coûtait 0,5 dB en moyenne et jusqu'à 3,6 dB sur une paire (détails fins).
- Le « new union model » fait moins bien que l'ancien (que contenait la roue, identique au lien « union model »).
- Déduire les instants du milieu par le flot gagne en PSNR mais perd nettement en LPIPS : le PSNR préfère le flou
  (comme le montre l'article EISAI, ECCV 2022), d'où LPIPS et les vidéos.
- À l'œil, sur les dessins tenus lissés, les deux sont très proches.

## Licences

Code et poids de GMFSS Fortuna : MIT (98mxr ; roue de HolyWu, MIT). GMFlow : Apache 2.0. Le softmax splatting
d'origine (sniklaus) est réservé à l'usage académique : le lecteur ne livrait que ses propres noyaux. Le « new union
model » est affiné sur AnimeRun, dont la licence des données n'a pas été vérifiée.
