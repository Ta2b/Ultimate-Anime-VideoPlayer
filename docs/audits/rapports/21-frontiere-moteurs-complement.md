# Complément de l'audit 21. Frontière des moteurs (interface et ABI)

Branche : `claude/project-thread-8vbpnc`. Commit audité : `0cfcdee`. Date : `2026-11-09`.

## Verdict

Sur les trois renvois, un seul est nouveau : `IngestSpec::rife_conversion` est une option morte du contrat
(A21-11). Les deux autres sont des doublons de défauts déjà au tableau (A32-02 et A45-14 pour le profil GPU,
A03-01 et A40-07 pour le rééchantillonnage de l'ingestion).

## Cadrage retenu

Seuls les renvois de `complements/21.md` sont vérifiés, dans le code. Consigne suivie : toute baisse de qualité
faite pour la vitesse est un défaut sauf mesure qui la justifie. Aucune construction (dépendances absentes du
conteneur) : les preuves sont lues.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A21-11 | Mineur | `src/gpu/kernel_ops.h:49-53` ; `src/gpu/vulkan/backend/kernel_frames.cpp:406-409` ; `src/gpu/cuda/backend/tensorrt_kernels.cpp:97-102` ; `tests/unit/vk_backend/kernel_ops_test.cpp:77-85` ; `tests/unit/gpu/backend_test.cpp:148` | Option morte du contrat. `IngestSpec::rife_conversion` n'est posé à `true` par aucun appelant de `src/` : les cinq ingestions du lecteur passent `IngestSpec{}` ou un `spec` dont seuls `reduce` et `deband` changent (`upscale_stage.cpp:286-299`, `upscale_ahead.cpp:44,85`, `upscale_drba.cpp:139`). La conversion de l'entrée de RIFE (BT.709, siting gauche, chroma bilinéaire) se fait ailleurs : dans le moteur RIFE fusionné à partir des tenseurs de la chaîne, et dans `rife_pad` (`kernel_pair.cpp:51-65`), lui-même mort (A21-10). Pourtant chaque moteur doit encore traiter le drapeau : Vulkan le convertit (`kernel_frames.cpp:406-409`) et le teste à chaque passage du test d'ingestion ; TensorRT le refuse, avec un message qui parle de formats (« NV12, P010 ou P016 … ») au lieu de dire que la conversion RIFE est absente. Le commentaire du contrat (`kernel_ops.h:49-51`) décrit un chemin que personne ne prend. | Aucun effet sur l'image aujourd'hui. Un développeur qui suit le commentaire et pose `rife_conversion` pour nourrir RIFE obtient une conversion juste sous Vulkan et un refus trompeur sous TensorRT, c'est-à-dire un repli silencieux d'une RTX vers un autre chemin, sans savoir pourquoi. Le test Vulkan double sa durée d'ingestion pour une branche que le lecteur n'emprunte pas. | Retirer `rife_conversion`, sa branche dans `kernel_frames.cpp:406-409`, sa condition dans `tensorrt_kernels.cpp:97`, la boucle `rife` de `kernel_ops_test.cpp:77-85` et la vérification de `backend_test.cpp:148` ; réécrire le commentaire `kernel_ops.h:49-51`. À faire avec A21-10, dans le même changement : `kernel_ops.h` est haché, donc nouvelle version d'ABI et nouveau paquet TensorRT (empreinte). | Lu : `grep -rn rife_conversion` ne trouve que les deux moteurs, le contrat et deux tests. |

## Renvois traités

| De l'audit | Où | Conclusion | Preuve |
|---|---|---|---|
| 32 | `src/gpu/gpu_profile.h` — `gpu_profile` sans équivalent Vulkan pour l'écran | **Doublon** de A32-02 (côté écran) et de A45-14 (côté backend Vulkan). `KernelOps::gpu_profile` a un défaut vide (`kernel_ops.h:268-270`), que seul `TensorRtBackend` remplace (`tensorrt_backend.cpp:486`) ; `VulkanKernelOps` ne le remplace pas, donc sous le backend Vulkan `keep_gpu_profile` sort sans rien écrire (`gpu_profile_out.cpp:20`, `if (stages.empty()) return;`) : c'est A45-14. Les passes libplacebo de l'écran ne passent par aucun backend et ne sont mesurées sous aucun moteur ; le contrat ne dit pas non plus que les durées des étapes se chevauchent entre couloirs : c'est A32-02 (« tout le côté Vulkan », « les sommes ne sont pas additives »). Rien de plus à la frontière : le correctif de A32-02 (temps par passe via `pl_render_params.info_callback`) vit côté écran, hors `KernelOps`. Seul ajout utile, à faire avec A32-02 : une ligne au commentaire `kernel_ops.h:268-269` disant que les étapes peuvent se recouvrir entre voies. | Lu. |
| 45 | `src/gpu/kernel_ops.h:49-53` — `IngestSpec::rife_conversion` posé par personne | **Confirmé** : A21-11 ci-dessus. | Lu. |
| 45 | `src/gpu/kernel_ops.h:42-56` — rééchantillonnage différent sous TensorRT et Vulkan | **Doublon** de A03-01 (TensorRT ignore `IngestSpec::chroma` et prend du bilinéaire, `slice_kernels.cu:23-53`) et de A40-07 (le test TensorRT fige ce bilinéaire). Le reste de la conversion concorde : matrice et plage par les mêmes `kernels::coefficients` et `yuv_scale` (`tensorrt_kernels.cpp:115-117`), siting identique pour les trois valeurs de `ChromaSiting` (`tensorrt_kernels.cpp:142-143` contre `detail::color_spec`, `kernel_frames.cpp:46`), mêmes quatre filtres de réduction (`tensorrt_kernels.cpp:129-132`). Le contrat dit bien « jamais une conversion silencieuse » (`kernel_ops.h:6-8`) : TensorRT le viole, ce que A03-01 couvre déjà. **Qualité contre vitesse** : aucune mesure de gain de vitesse ne justifie le bilinéaire ; l'en-tête le dit « provisoire » (`slice_kernels.h:7-10`) et ne cite qu'un écart « sous un niveau 8 bits sur les aplats », qui ne dit rien des contours colorés où Spline36 et bilinéaire diffèrent. La baisse reste donc un défaut, au niveau Majeur déjà donné par A03-01. Un écart moindre, sans défaut nouveau : le débandage à l'ingestion n'existe que sous TensorRT (A45-03). | Lu. |

## Points positifs

- Les conversions de couleur des deux moteurs partent des mêmes fonctions de `kernels/spec` : seul le filtre de
  chroma diverge, ce qui borne la correction de A03-01 à un noyau.
- `rife_conversion` est refusé, pas ignoré, sous TensorRT : l'option morte ne produit pas d'image fausse.
