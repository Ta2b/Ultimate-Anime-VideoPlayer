# GMFSS Union : lot et recouvrement au trtexec (RTX 4070 Ti, 28 octobre 2026)

Pour décider si l'audit code un lot (plusieurs instants dans un appel de la fusion) ou un second flux (paire n+1 pendant les fusions de n).
Graphes ONNX FP16 du lecteur (`gmfss_fortuna_union_s05_*`), TensorRT 11.3, `--noDataTransfers`, sous le verrou GPU ; bureau actif (~35 % du GPU
pris par Brave) : médianes. Taille de travail réelle : 540 lignes, `frames` 6×768×1280 (plans à 128), fusion à `half` 3×384×640.

## 1. Lot sur la fusion
La fusion a été ré-exportée avec un axe de lot dynamique `B` sur toutes les entrées et la sortie (`tools/gmfss/gmfss_graph.py` : `Fusion`, `time.reshape(-1,1,1,1)`
pour que l'IFNet embarqué reçoive un instant par élément ; export FP16 de `export_gmfss.py`, 1115 nœuds comme sans lot). Un moteur par lot, `--shapes` fixes.

| Lot | Médiane (min) pour le lot | **ms par instant** | Écart au lot 1 |
|---|---|---|---|
| 1 | 12,36 ms (11,51) | **12,36** | – |
| 2 | 24,54 ms (23,21) | **12,27** | −0,7 % |
| 3 | 36,80 ms (35,66) | **12,27** | −0,7 % |
| 5 | 63,21 ms (58,15) | **12,64** | +2,3 % |

**Cible (≥ 15 % de baisse de ms/instant au lot 2) : non atteinte.** Le temps croît linéairement avec le lot : la fusion à 720p de travail
(80 % de convolutions, voir `gmfss-distil-mesures-2026-10-28.md`) sature déjà le GPU avec un seul instant. Un lot ne gagne rien et, à 5, coûte 2 % de plus.
Ne pas coder le lot.

## 2. Recouvrement
`trtexec --streams=N` lance N exécutions concurrentes du même moteur :

| Graphe | Flux | Débit | Médiane d'une exécution |
|---|---|---|---|
| fusion (720p) | 1 | 81,6 qps | 12,2 ms |
| fusion (720p) | 2 | **75,0 qps** (−8 %) | 26,8 ms |
| paire (frames 6×768×1280) | 1 | 106,6 qps | 9,4 ms |

Deux fusions concurrentes vont **moins vite** qu'une seule (le débit baisse de 8 %, chaque exécution met 2,2 fois plus) : les convolutions occupent tout le
GPU, il n'y a pas de place libre pour recouvrir. La paire (9,4 ms au trtexec, 6,8–7,5 ms en lecture) est de même nature (convolutions de GMFlow et du FeatureNet).
Somme séquentielle paire + fusion : ≈ 21,6 ms ; **aucun gain de 20 % n'est attendu d'un second flux**.
Limite de la mesure : `trtexec` ne lance pas deux moteurs différents dans un même processus ; le recouvrement paire/fusion exact n'a pas été mesuré, mais la saturation
mesurée sur chacun des deux graphes (et le profil de la lecture : fusion 10–12 ms, paire 7 ms, sans gonflement par rapport au trtexec) va dans le même sens.

## Conclusion
Ni lot ni second flux ne diminuent le coût de GMFSS Union à 720p de travail sur la 4070 Ti. Les seules pistes restantes changent le réseau (le `upsample` final,
≈ 16 % de la fusion) ou la résolution (le flot à demi-échelle est déjà en place).
