# Journal des versions

Le détail de chaque version est dans ses notes de release, `tools/release/notes/v<version>.md`, que la release
reprend telles quelles. La version vient de `project()` dans `CMakeLists.txt` ; `vcpkg.json` la recopie
(`tools/ci/check-version.sh`).

| Version | Notes |
|---|---|
| 0.13.0 | [v0.13.0-rc.1](tools/release/notes/v0.13.0-rc.1.md) : carte de la scène par le flux optique de NVIDIA, calques et plaque de décor, tuiles de SPAN, RIFE natif par défaut, lecture à 165 Hz |
| 0.11.0 | [v0.11.0](tools/release/notes/v0.11.0.md) : la chaîne suit la source, deux audits, Anime4K, RIFE fusionné et SPAN FP8 plus rapides, sous-titres après un saut ; [v0.11.0-rc.8](tools/release/notes/v0.11.0-rc.8.md) : second audit, Anime4K et ArtCNN C4F16, RIFE fusionné plus rapide, niveaux d'interpolation, options RIFE et peau des visages à l'essai ; [v0.11.0-rc.7](tools/release/notes/v0.11.0-rc.7.md) : corrections de l'audit (qualité, interpolation, écran, TensorRT), préversion sans macOS |
| 0.10.3 | [v0.10.3](tools/release/notes/v0.10.3.md) : corrections Windows et Vulkan (ArtCNN, netteté) |
| 0.10.2 | [v0.10.2](tools/release/notes/v0.10.2.md) : la 0.10.1 reconstruite (modèle ArtCNN livré, macOS) |
| 0.10.1 | [v0.10.1](tools/release/notes/v0.10.1.md) : commandes rejouées en boucle, trtexec de Linux ; ne s'est pas construite |
| 0.10.0 | [v0.10.0](tools/release/notes/v0.10.0.md) : reprise de l'audit de la 0.9.1, phases 1 à 5 |
| 0.9.1 | [v0.9.1](tools/release/notes/v0.9.1.md) : correctifs de la 0.9.0, surtout sous Windows |
| 0.9.0 | [v0.9.0](tools/release/notes/v0.9.0.md) |
| 0.8.0 | [v0.8.0](tools/release/notes/v0.8.0.md) |
