# Looks du contrat d'écran v2 (chantier 8)

`ver9-couleur.cube` : la couleur de la pile par défaut du lecteur (`HDR grading base ver.9.0 33 point`, puis
`Black-Enhancement-L`, `Shadow-Enhancement-L`, `HDR-contrast-Enhancement-L` × 2, `Midtone-gamma-Enhancement-C`,
`boost black maintain chroma`, `HDR-contrast-Enhancement-L`), sans sa courbe neutre, dans le domaine SDR du
lecteur (BT.709, gamma 2,4, blanc 203 nits ; `# ajp-domain: sdr`). Dérivée des LUT de `data/luts/` (même
provenance, voir `data/luts/ATTRIBUTION.md`), par :

    cd data/luts && python3 -I ../../tools/look/split_hdr_lut.py "HDR grading base ver.9.0 33 point.cube" \
      Black-Enhancement-L.cube Shadow-Enhancement-L.cube HDR-contrast-Enhancement-L.cube \
      HDR-contrast-Enhancement-L.cube Midtone-gamma-Enhancement-C.cube "boost black maintain chroma.cube" \
      HDR-contrast-Enhancement-L.cube --out ../luts-v2 --compose ver9-couleur.cube

Rapport de la pile : courbe neutre 1 → 0,0, 5 → 7,7, 20 → 52,6, 50 → 146,4, 100 → 428,9, 203 → 1 816,6 nits ;
décalage de teinte de la couleur, moyenne 3,68°, p95 10,28° ; les gris restent gris (écart ≤ 2·10⁻⁶). Livrée dans `luts/` à côté des autres ; sous
`--hdr-contract-v2` (8.4, essai), elle remplace la pile par défaut, la courbe par plan (render/curve) prenant la
place de la courbe neutre (8.6).
