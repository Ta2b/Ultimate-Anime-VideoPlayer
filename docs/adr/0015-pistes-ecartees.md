# 0015 — Pistes d'optimisation écartées

## Contexte
Le plan d'audit de la 0.9.1 (phase 6, P6.24) liste des pistes à ne pas suivre et demande que la raison de
chacune soit écrite, pour qu'elles ne reviennent pas sans élément nouveau.

## Décision
| Piste | Raison |
|---|---|
| Parcimonie 2:4 des poids (cœurs Tensor « sparse ») | Demande de réentraîner SPAN et RIFE avec l'élagage ; les modèles AnimeJaNai et RIFE sont repris tels quels, sans données ni budget d'entraînement. Le gain annoncé (×2 sur les GEMM) ne touche qu'une partie des couches. |
| NvFRUC (interpolation d'image de NVIDIA) tel quel | Sous licence du SDK Optical Flow, non redistribuable dans nos archives (aucun fichier de NVIDIA livré) ; rendu pensé pour le jeu, qui mélange les traits fins des dessins. Le flux matériel seul reste une piste (P6.10). |
| Assembleur écrit à la main (SASS, PTX, SPIR-V) | Ni maintenable ni portable entre générations de cartes ; TensorRT, les noyaux CUTLASS et les shaders GLSL sont relus par leurs compilateurs à chaque version. |
| Retouche du code machine GPU après compilation | Même raison, plus un binaire que personne ne peut reconstruire depuis les sources. |
| `-march=x86-64-v3` imposé au lecteur livré | Le CPU ne porte pas le calcul lourd (GPU) ; imposer AVX2 exclurait des machines sans gain mesurable. Le préréglage `linux-prefix-cuda-machine` le garde pour un poste connu. Le SHA-256 choisit SHA-NI à l'exécution (P4.10). |
| PGO (optimisation guidée par profil) | Un profil par plateforme à produire et à garder à jour, pour un gain sur le CPU qui n'est pas le goulot ; la LTO des préréglages livrés (P5.20) prend l'essentiel sans profil. |

## Conséquences
- Une piste de ce tableau ne revient qu'avec une mesure qui contredit sa raison, dans une nouvelle ADR.
- Les pistes de la phase 6 qui demandent une carte (P6.6 à P6.22) se mesurent sur le poste de référence ; leur
  résultat, gardé ou abandonné, va dans `docs/` avec la mesure.
