# Complément de l'audit 45. Moteur Vulkan : calcul, noyaux et shaders

Branche : `claude/project-thread-lxz2e1`. Commit audité : `9ead2be`. Date : `2026-11-09`.

## Verdict

Des deux renvois, un seul apporte un défaut nouveau : rien ne fixe ni ne vérifie l'arrondi FP16 et les
sous-normaux des shaders (A45-15). La porte à 2 % est un doublon d'A45-02, mais sa portée est plus large que
le rapport ne le disait. Sur AMD et Intel, Vulkan est le seul moteur : la porte y règle donc le rendu normal,
pas seulement un repli.

## Cadrage retenu

Aucune question posée à Tatoub, sur consigne du coordinateur. Seuls les deux renvois de
`complements/45.md` sont vérifiés. La règle reste : une baisse de qualité faite pour la vitesse est un défaut,
sauf si une mesure la justifie. Le fil n'a pas de GPU. Une mesure a été faite sur CPU (onnxruntime, DistilDRBA
livré, quatre triplets de `calibration.mkv`, script hors dépôt). Le comportement du matériel reste « à confirmer ».

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A45-15 | Mineur | `src/gpu/vulkan/shaders/net_tensor.glsl:75-77`, `src/gpu/vulkan/shaders/net_conv_coopmat.glsl:294-295`, `src/kernels/glsl/rife_fused.glsl:9-11`, `src/kernels/ref/half.h:1-4`, `src/gpu/vulkan/compute/context_create.cpp:301-306`, `tests/unit/vk_compute/transfer_test.cpp:296-299` | **L'arrondi f32 → f16 et le traitement des sous-normaux F16 sont laissés au pilote, et rien ne les vérifie.** Toutes les conversions des shaders utilisent `float16_t(v)` (`AJP_STORE`), la conversion d'une tuile coopmat float en float16 (épilogue `DIRECT_EPI`) ou `packHalf2x16` (`rounded()` de RIFE et DistilDRBA). Aucun shader ne déclare de mode d'exécution `SPV_KHR_float_controls` (`RoundingModeRTE`, `DenormPreserve`), et `grep -rn "Denorm\|RoundingMode\|float_controls" src` ne trouve rien. Les propriétés `shaderRoundingModeRTEFloat16` et `shaderDenormPreserveFloat16` ne sont ni lues ni écrites au journal. Sans ces déclarations, la spécification Vulkan (« Precision and Operation of SPIR-V Instructions ») permet d'arrondir vers zéro ou au plus proche, au choix de l'implémentation, et de mettre à zéro tout sous-normal. Pourtant, l'hôte (poids, biais, entrées de la porte : `net_weights.cpp:20`, `pass_fuse.cpp:104-107`) et la référence `kernels/ref` arrondissent au plus proche, égalités au pair, en gardant les sous-normaux. `half.h:3` le note (« au plus proche en pratique, sans décoration RTE ») sans le vérifier. Les tests GPU tolèrent 1/1024 en relatif (`transfer_test.cpp:297`), soit un ULP : un pilote qui arrondit vers zéro passe. | Pilote qui arrondit vers zéro (RTZ) ou qui met les sous-normaux F16 à zéro, ce qui est permis sur Intel, lavapipe ou un futur pilote. Chaque rangement FP16 d'activation (une vingtaine de couches) aurait alors une erreur biaisée vers zéro, d'un ULP au plus. Ce biais s'additionne linéairement, et non en racine comme l'arrondi au plus proche : les aplats sombres s'assombrissent légèrement. La porte (A45-02) ne le verrait pas, parce que la référence passe par le même pilote. Pour les sous-normaux, la mesure ci-dessous montre que les poids d'A44-06 ne posent pas de problème. | Déclarer `RoundingModeRTE` et `DenormPreserve` pour la largeur 16 quand l'appareil les annonce, par `GL_EXT_spirv_intrinsics` (`spirv_execution_mode`) ou par une passe sur le SPIR-V. Écrire les propriétés float controls au journal de démarrage du backend. Ajouter un test GPU exact (sans tolérance) qui range des égalités, des cas proches de la moitié d'un ULP et des sous-normaux, et compare bit à bit avec `ref::float_to_half`. | Lu (grep). Sous-normaux mesurés : voir la note. À confirmer sur chaque famille : `vulkaninfo \| grep -i "denorm\|roundingmode"` sur la 4070 Ti, RDNA et Intel, puis le test exact proposé. |

Note de mesure (sous-normaux, A44-06). Le chemin Vulkan fond le facteur `beta` des ResConv dans les poids
F16 (`pass_fuse.cpp:103-107`). Le DistilDRBA livré (`distil_drba_v2_lite.onnx`) a ainsi **316 191 poids et
biais sous-normaux** après fusion, sur 24 convolutions. Dans plusieurs canaux, ce sont eux qui portent tout le
poids du canal (block0 convblock.7, block1 convblock.6). Mesure : le graphe fondu tel quel, contre le même
graphe avec ces valeurs mises à zéro, sur quatre triplets 1280×704 de `calibration.mkv` (t = 5, 30, 60 et
90 s), au milieu (u = 0,25), sous onnxruntime en FP16. Résultat : PSNR de 88,8 à 97,4 dB, et seulement 3 à 34
pixels sur 901 120 qui s'écartent de plus de 1/255 (écart maximal 5,7/255). Le graphe fondu non mis à zéro est
identique bit à bit au graphe d'origine. **Si un pilote met à zéro les sous-normaux des poids, A44-06 reste
invisible.** La mesure ne couvre pas la mise à zéro des activations sous-normales : à confirmer avec le test
exact d'A45-15.

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| Renvoi 44 → 45 : arrondi f32 → f16 (RTE ou RTZ) | **Confirmé** : A45-15 | Arrondi laissé au pilote, sans déclaration ni vérification. Au plus proche en pratique sur NVIDIA et AMD (à confirmer avec `vulkaninfo`) ; rien ne le garantit ailleurs. |
| Renvoi 44 → 45 : sous-normaux F16 | **Confirmé** dans A45-15, **réfuté pour son effet sur A44-06** | La mise à zéro est permise, faute de `DenormPreserve`. Mesurée sur les poids du DistilDRBA livré : 88,8 dB au pire, invisible (note ci-dessus). |
| Renvoi 44 → 45 : accumulation des coopmat | **Doublon d'A45-01** | `net_conv_coopmat.glsl:29-33`, `:201` : l'accumulateur est en float32 par défaut (`coopmat_f16` : produits FP16, sommes FP32) et en float16 dans `coopmat_acc16` seulement (`ACC_T float16_t`). Les sommes de `coopmat_acc16` sont reconverties en float pour l'épilogue (`:304-306`), donc après la perte. Le choix est mesuré par couche (`kPrecisionTie`, `net_autotune.cpp:370-382`). Une précision à ajouter à A45-01 : l'arrondi interne des Tensor Cores n'est pas réglable, et les travaux publiés sur les générations Volta à Ampere (Fasi, Higham, Mikaitis, Pranesh, 2021) rapportent une troncature pendant l'accumulation. La simulation d'A45-01, qui arrondit au plus proche à chaque pas de 16, peut donc **sous-estimer** l'erreur de `coopmat_acc16` : à confirmer sur Ada. |
| Renvoi 46 → 45 : `net_gate.h:24-25`, porte FP16 à 2 % sur du bruit | **Doublon d'A45-02** (et d'A46-11 pour les tables), **avec une portée plus large** | Mêmes constats qu'A45-02 : `kNetworkDriftLow = 0.02` (`net_gate.h:26`), entrée de bruit ramenée à [0, 1] (`net_gate.cpp:28`, `:33`), référence aux noyaux `direct` FP16 et non FP32, tolérance par couche de 3 % du pic plus 2 % (`net_autotune.cpp:101-115`). Ce qui change, c'est la portée. La règle est la même sur toutes les cartes, et les variantes à sommes FP16 sont proposées partout où l'appareil le permet : `direct_f16p` et `winograd_f16p` dès `shaderFloat16`, `coopmat_acc16` dès qu'une tuile FP16 → FP16 est annoncée (`device_profile.cpp:61`, `net_lower_conv.cpp:590-596`). Sur AMD et Intel, Vulkan est le seul moteur (ADR 0012), donc cette porte règle le **rendu normal**. Sur une carte à FP16 empaqueté sans matrices coopératives (RDNA 2, Intel Xe), `direct_f16p` gagne sans doute plus de 5 % et sera retenu partout. Gain : mesuré par couche, plus de 5 %, mais écrit nulle part. Perte : jamais mesurée sur une image. Je propose de passer A45-02 en **Majeur** pour AMD et Intel. La correction reste celle d'A45-02 : juger contre une sortie FP32 sur une vraie image, au seuil de TensorRT, et écrire le gain dans le journal ou la table. |

## Renvois

Aucun.

## Points positifs

- L'hôte et la référence arrondissent de la même façon : au plus proche, égalités au pair, sous-normaux gardés
  (`kernels/ref/half.h`). Les poids, les biais et les entrées de la porte sont donc identiques sur toutes les
  cartes.
- L'accumulation par défaut des matrices coopératives est en float32, et l'épilogue `DIRECT_EPI` n'existe qu'en
  sommes float (`net_conv_coopmat.glsl:205-210`, `:276`).
- Entre une variante à sommes FP16 et une variante à sommes float, la porte garde celle en float dès que l'écart
  de vitesse tombe sous 5 % ou dans le bruit. C'est le bon sens de départage, même si le seuil reste à juger sur
  image.

## Questions ouvertes

Aucune posée à Tatoub. Il reste à mesurer sur la carte : `vulkaninfo` (propriétés float controls) sur NVIDIA,
AMD et Intel, puis le test exact d'A45-15.
