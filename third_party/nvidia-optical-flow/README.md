# En-têtes de NVIDIA Optical Flow (API 2.0)

`nvOpticalFlowCommon.h` et `nvOpticalFlowCuda.h`, copiés tels quels de
<https://github.com/NVIDIA/NVIDIAOpticalFlowSDK>, commit `edb50da3cf849840d680249aa6dbef248ebce2ca`
(la version qu'OpenCV télécharge pour `cudaoptflow`). Licence BSD à trois clauses de NVIDIA, en tête
de chaque fichier.

Seuls les types et les signatures servent : la bibliothèque (`libnvidia-opticalflow.so.1` sous Linux,
`nvofapi64.dll` sous Windows) vient avec le pilote NVIDIA et se charge à l'exécution
(`src/gpu/cuda/backend/nvof_session.cpp`). Hors de `src/` : ni clang-format ni la limite de taille ne
s'appliquent à ces copies. Le pilote accepte cette version 2.0 de l'API sur les RTX récentes.
