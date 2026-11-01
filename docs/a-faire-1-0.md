# À faire avant la 1.0 (suivi du 01/11)

Priorités de Tatoub, dans l'ordre : calibration HDR du grade adaptatif, correctif RTX HDR sous Linux (ABI 16, paquet TensorRT à reconstruire une fois, après les derniers changements du module).

## Reporté, à ne pas oublier
- **GMFSS Fortuna Union : artefacts** (fond vert saturé, formes en miroir, traînées), en FP16 comme en FP32, alors qu'Union est plus rapide que Fortuna Base. Seule différence avec Base : le RIFE intégré du graphe fusion (`tools/gmfss/gmfss_graph.py`, `Fusion.forward` : `ifnet(half0, half1, time)`, assemblage 9 canaux). À mesurer : sortie d'IFNet et valeur de `time` sur deux images, ou comparaison FP32 de l'export avec PyTorch. Pistes : parité des splats contre `t`, grille de warp figée à l'export (`export_gmfss.py`), `time` hors plage. Mesure GPU à lancer sur ordre de Tatoub. GMFSS est de toute façon trop lourd pour le PC portable.
- Calibration HDR : plan dans `/mnt/project-files/calibration/plan-calibration-hdr.md` (curseur de nits, deux images de référence, profil par écran).
- TensorRT Windows : à compiler une fois l'application terminée.
