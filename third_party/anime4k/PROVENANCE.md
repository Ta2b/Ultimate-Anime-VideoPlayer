# Anime4K, copie partielle

- Source : https://github.com/bloc97/Anime4K, branche `master` (v4.0.1 et ses corrections de shaders), commit
  `7684e9586f8dcc738af08a1cdceb024cc184f426`.
- Licence : MIT (`LICENSE`, et l'en-tête de chaque shader). Le texte est repris dans `THIRD_PARTY_NOTICES.md`,
  livré avec le lecteur.
- Contenu : `LICENSE` et les neuf shaders GLSL de mpv que le lecteur embarque (`src/render/anime4k.h`) : le
  plafonnement des hautes lumières, Restore et Restore Soft, Upscale ×2 et Upscale Denoise ×2, chacun en M
  (Rapide) et en VL (Qualité). Aucun fichier n'est modifié : chacun est identique à celui de l'amont à ce commit,
  dans le même dossier.
- `FILES.sha256` : la somme SHA-256 de chaque fichier copié. La liste des fichiers est celle de ce fichier.
- `tools/deps/anime4k-copy.sh --check` vérifie la copie contre `FILES.sha256` (sans réseau, lancé par la CI des
  règles) ; `tools/deps/anime4k-copy.sh --upstream` la compare à l'amont au commit ci-dessus ;
  `tools/deps/anime4k-copy.sh <commit>` refait la copie depuis un autre commit et réécrit les sommes.
- Ces fichiers sont hors de `src/` : la règle des 800 lignes ne vaut pas pour une copie de l'amont (les shaders VL
  en font environ 870).
