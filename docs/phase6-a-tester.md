# Phase 6 : ce qui est codé et reste à tester sur une vraie carte

Les pistes de la phase 6 du plan d'audit sont codées sans carte (vérifiées sur Vulkan logiciel quand c'est
possible), mais leur intérêt se juge sur le poste de référence. Chacune est éteinte par défaut, ou réservée au
choix manuel, tant que son critère n'est pas mesuré. Une piste qui échoue à son critère est retirée, et sa
mesure gardée dans `docs/reports/`.

| Piste | Ce qui est codé | Comment l'allumer | Ce qu'il faut mesurer | Critère |
|---|---|---|---|---|
| P6.7 ArtCNN | ArtCNN C4F32 (MIT) enveloppé en RGB (`tools/upscale/wrap_luma_rgb.py`), livré dans `models/onnx`, proposé dans la liste des modèles SPAN ; dans les tranches Économe et Léger derrière `kLightSpanInAuto` | Choisir « ArtCNN C4F32 » dans Vidéo > Modèle ; pour l'Auto, `kLightSpanInAuto = true` (`auto_config.h`) | Temps par image et image, sur un iGPU (Intel Xe, Radeon 680M) : Économe actuel (sans agrandissement) contre ArtCNN ×2 ; puis Rapide : SPAN Performance contre ArtCNN | Meilleure image qu'Économe au même coût ; sur RTX, vérifier aussi TensorRT et la préparation FP8, qui n'a jamais vu ce graphe |
