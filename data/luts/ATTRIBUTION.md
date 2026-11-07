# LUT 3D d'HDR (« HDR LUT ver.9.1 », dossier « HDR grading ver.9.0 »)

Quatre fichiers `.cube` (LUT 3D, 33 points, générés par DaVinci Resolve : en-tête `TITLE "Generate by Resolve"`),
tels quels, octet pour octet, ceux que charge le grade HDR anime : `HDR grading base ver.9.0 33 point.cube`,
`HDR-contrast-Enhancement-L.cube`, `Highlight-Enhancement-L.cube` et `skin fix.cube`. Les 51 autres fichiers du
pack, que le lecteur ne chargeait pas, sont retirés depuis la 0.10.0.

## Provenance
- Ajoutés au dépôt le 2026-10-31, sur instruction directe de son propriétaire (Ta2b), depuis l'archive
  « HDR LUT ver.9.1.zip » qu'il a fournie.
- **Auteur d'origine et licence : non indiqués dans l'archive** (elle ne contenait que 55 fichiers `.cube`,
  sans readme ni licence). Le propriétaire du dépôt répond de leur redistribution. À compléter ici, avec le nom
  de l'auteur et sa licence, dès qu'ils sont connus ; si l'auteur s'oppose à cette redistribution, retirer ce dossier.

## Usage dans le lecteur
Ce sont des LUT d'HDR (étalonnées pour des codes PQ BT.2020) : le lecteur les applique en sortie HDR et coupe la pile
en sortie SDR (docs/luts.md). Les fichiers se déposent aussi à la main dans le dossier `luts` des données du lecteur.
