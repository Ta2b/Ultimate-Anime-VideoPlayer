# NVIDIA CUTLASS, copie partielle

- Source : https://github.com/NVIDIA/cutlass, étiquette `v4.8.1`, commit
  `2c85f497547eb441f0fe80bfb0fb2e6f9add1719`.
- Contenu : `LICENSE.txt` et les 259 en-têtes de `include/` que les noyaux du SPAN FP8 incluent
  (`docs/span-fp8.md`). Aucun fichier n'est modifié : chacun est identique à celui de l'amont à ce commit.
- `FILES.sha256` : la somme SHA-256 de chaque fichier copié. La liste des fichiers est celle de ce fichier.
- `tools/deps/cutlass-copy.sh --check` vérifie la copie contre `FILES.sha256` (sans réseau, lancé par la CI des
  règles) ; `tools/deps/cutlass-copy.sh --upstream` la compare à l'amont au commit ci-dessus ;
  `tools/deps/cutlass-copy.sh <commit>` refait la copie depuis un autre commit et réécrit les sommes.
