# Journal des versions

Le détail de chaque version est dans ses notes de release, `tools/release/notes/v<version>.md`, que la release
reprend telles quelles. La version vient de `project()` dans `CMakeLists.txt` ; `vcpkg.json` la recopie
(`tools/ci/check-version.sh`).

| Version | Notes |
|---|---|
| 0.11.0 | [v0.11.0-rc.4](tools/release/notes/v0.11.0-rc.4.md) : corrections de l'audit (qualité, résilience, Windows, TensorRT), préversion |
| 0.10.3 | [v0.10.3](tools/release/notes/v0.10.3.md) : corrections Windows et Vulkan (ArtCNN, netteté) |
| 0.10.2 | [v0.10.2](tools/release/notes/v0.10.2.md) : la 0.10.1 reconstruite (modèle ArtCNN livré, macOS) |
| 0.10.1 | [v0.10.1](tools/release/notes/v0.10.1.md) : commandes rejouées en boucle, trtexec de Linux ; ne s'est pas construite |
| 0.10.0 | [v0.10.0](tools/release/notes/v0.10.0.md) : reprise de l'audit de la 0.9.1, phases 1 à 5 |
| 0.9.1 | [v0.9.1](tools/release/notes/v0.9.1.md) : correctifs de la 0.9.0, surtout sous Windows |
| 0.9.0 | [v0.9.0](tools/release/notes/v0.9.0.md) |
| 0.8.0 | [v0.8.0](tools/release/notes/v0.8.0.md) |
