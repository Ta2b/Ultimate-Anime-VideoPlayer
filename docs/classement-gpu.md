# Classement des cartes graphiques pour ajp

Le préréglage Auto part de la tranche de la carte (`data/gpu_tiers.toml`, `src/app/play/gpu_tier.cpp`). Les
cartes y sont classées **par puissance pour ajp**, pas par date ni par résultats en jeu : ajp passe son temps
dans des réseaux (SPAN, RIFE, GMFSS), dont le débit suit surtout le calcul FP16 matriciel. Une RTX 2060 y dépasse
une GTX 1080 Ti, plus rapide en jeu mais sans cœurs Tensor ; une RTX 2080 Ti y dépasse une RTX 3060 Ti.

## Ce que règle chaque tranche

| Tranche | Auto (`auto_config.cpp`) | Moteur conseillé |
|---|---|---|
| Économe | grade HDR seul : ni agrandissement ni interpolation | Vulkan |
| Rapide | agrandissement depuis 540p, interpolation jusqu'à ×2 | TensorRT sur RTX, sinon Vulkan |
| Normal | agrandissement depuis 720p, interpolation jusqu'à ×3 | TensorRT sur RTX, sinon Vulkan |
| Qualité | la source jusqu'à 1080p, modèles fins, interpolation jusqu'à ×4 | TensorRT sur RTX, sinon Vulkan |
| Ultra | comme Qualité, avec de la marge | TensorRT sur RTX, sinon Vulkan |

## Méthode

Repère : la RTX 4070 Ti, seule carte mesurée (14,25 ms de SPAN b8f64 par image 1080p sous TensorRT, pour
41,7 ms par image source). Chaque carte est placée par son débit FP16 matriciel face à elle :

| Débit face à la 4070 Ti | Tranche |
|---|---|
| 1,2 fois et plus | Ultra |
| 0,7 à 1,2 | Qualité |
| 0,4 à 0,7 | Normal |
| 0,2 à 0,4 | Rapide |
| moins de 0,2 | Économe |

Les cartes sans unités matricielles (GTX, Radeon avant RDNA 3, Intel sans XMX) passent par Vulkan en FP32 ou en
FP16 simple, environ quatre fois moins efficace : elles vont en Économe, sauf les plus fortes (GTX 1080 Ti, TITAN
Xp, TITAN X Pascal, TITAN V), en Rapide. Un portable va d'ordinaire une tranche sous la carte de bureau du même
nom (moins de watts).

## NVIDIA, bureau (de la plus puissante à la moins puissante)

- **Ultra** : RTX 5090, RTX PRO 6000, RTX 4090, RTX 6000 Ada, RTX 5080, RTX PRO 5000, RTX 4080 (Super), RTX 4070
  Ti Super, RTX 5070 Ti.
- **Qualité** : RTX 4070 Ti, 3090 Ti, RTX 5000 Ada, RTX PRO 4500, 3090, 4070 Super, RTX A6000, 3080 Ti, 5070,
  RTX 4500 Ada, RTX A5500, RTX PRO 4000, 4070.
- **Normal** : RTX 3080, TITAN RTX, Quadro RTX 8000 et 6000, 2080 Ti, RTX A5000, 3070 Ti, RTX A4500, 4060 Ti,
  5060 Ti, 2080 (Super), 3070, RTX 4000 Ada, RTX A4000, Quadro RTX 5000, 5060, RTX PRO 2000, 2070 Super, 3060 Ti.
- **Rapide** : RTX 4060, 2070, RTX 2000 Ada, Quadro RTX 4000, 2060 Super, 3060, 2060, 5050, RTX A2000, Quadro
  RTX 3000, 3050, RTX A1000, RTX PRO 1000 ; sous Vulkan, TITAN V, TITAN Xp, TITAN X Pascal, GTX 1080 Ti.
- **Économe** : RTX A500, RTX A400, les autres GTX (1080, 1070, 16xx, 10xx, 9xx), les GT.

## NVIDIA, portables

- **Qualité** : RTX 5090, 4090, 5080, 4080, RTX PRO 5000.
- **Normal** : RTX 5070 Ti, 3080 Ti, 5070, 4070, 3080, RTX PRO 4000, 2080 (Max-Q).
- **Rapide** : les autres RTX portables (3070 Ti, 3070, 5060, 4060, 3060, 4050, 2070, 2060, 3050, RTX A… et PRO…).
- **Économe** : RTX 2050, les GTX et MX portables.

## AMD et Intel (Vulkan)

- **Qualité** : RX 9070 (XT), RX 7900.
- **Normal** : Radeon 8060S (iGPU), RX 9060, RX 7800, RX 7700, RX 6900 et 6800 ; Intel Arc B5xx, A7xx.
- **Rapide** : RX 7600, RX 6700, RX 6600, RX 5700, Radeon VII, Vega 64 et 56, Radeon 8xxM (iGPU) ; Intel Arc
  A5xx, Arc 140V (iGPU).
- **Économe** : RX 6500 et 6400, RX 5500 et 5600, RX 400 et 500, Radeon 7xxM et les autres iGPU ; Intel Arc A3xx,
  Arc et Iris Xe intégrés, UHD.

## À mesurer

Les places sont des estimations, à corriger par une mesure dès qu'une carte est disponible : la GTX 1080 Ti en
Rapide sous Vulkan, une RTX 2060 en Rapide, une RX 7900 XTX en Qualité, une RTX 3060 Ti en Normal. La mesure : la
vidéo du banc 60 s au préréglage de la tranche, binaire `linux-release` ou `windows-release`, sans image sautée
(`--bench-level`).
