# TrueHDR réel (Maxine VFX 1.3) — chantier vendor-hdr-real

Date : 19 octobre 2026. Branche `pc/vendor-hdr-real` (base `origin/w1/integration` f8cbcc0),
commit a8455e5, non poussée (le relecteur pousse).
Machine : RTX 4070 Ti, pilote 615.78.08, SDK `/usr/local/VideoFX` = VFXSDK_linux 1.3.0.0 et
fonction nvvfxtruehdr 1.3.0.0 (`features/nvvfxtruehdr/include/nvVFXTrueHDR.h:13`).
Chemins de code relatifs au dépôt.

## Résumé

- **Le chargeur de la vague 1 ne pouvait pas marcher sur le vrai SDK.** Le sélecteur du pic
  s'appelle `"Luminance"`, pas `"MaxLuminance"`. Le contrôle d'ABI refusait donc tout vrai SDK
  dès l'import. Le marqueur de version supposé (`version.h`) n'existe pas non plus. C'est corrigé
  dans `src/vendor_hdr/maxine_abi.h:68` et `src/vendor_hdr/sdk_version.cpp:87`.
- Les autres constantes, signatures et symboles sont justes. Ils citent maintenant leurs lignes
  d'en-tête.
- TrueHDR tourne pour de vrai sur la RTX. Il est ouvert depuis la copie du magasin, sur des
  tampons CUDA, sans LD_LIBRARY_PATH, avec le vrai encadrement d'événements entre un flux non
  bloquant et le flux 0.
- **Temps par image** : mesuré sur la RTX. Les temps mesurés avec TrueHDR ne se publient pas (licence de NVIDIA) : ils restent sur le PC.
- **Déterministe** : mêmes octets d'un passage à l'autre, d'une instance à l'autre et après
  recréation. Aucun état ne passe d'une image à la suivante.
- **Rampe de 32 marches** : faisable sur une seule instance. Elle est strictement croissante et
  ne dépend pas de l'ordre de passage.
- **Plusieurs instances** Maxine peuvent vivre en même temps.
- **Magasin** : l'import réel copie environ 135 Mo au lieu de 4,3 Go. Il n'emporte que les
  bibliothèques que TrueHDR charge vraiment.

## 1. Vérification contre les vrais en-têtes et bibliothèques

Sources :
- `/usr/local/VideoFX/include/{nvCVStatus.h,nvCVImage.h,nvVideoEffects.h}` ;
- `features/nvvfxtruehdr/include/nvVFXTrueHDR.h` ;
- `nm -D` et `readelf -d` sur les bibliothèques ;
- `/proc/self/maps` d'un processus qui a créé, chargé et exécuté TrueHDR (sonde dans le
  scratchpad, compilée contre le SDK).

| Élément | Vague 1 | Réel | Verdict |
|---|---|---|---|
| NVCV_SUCCESS / NVCV_ERR_UNIMPLEMENTED | 0 / -2 | nvCVStatus.h:47,49 | juste |
| NVCV_RGBA / NVCV_RGB10A2 | 6 / 13 | nvCVImage.h:49,61 | juste |
| NVCV_U8 / NVCV_P32 | 1 / 11 | nvCVImage.h:68,78 | juste |
| NVCV_INTERLEAVED / NVCV_GPU | 0 / 1 | nvCVImage.h:92,140 | juste |
| Struct NvCVImage (64 octets, champs) | — | nvCVImage.h:151-166 | juste |
| "TrueHDR", "SrcImage0", "DstImage0", "ModelDir", "CudaStream" | — | nvVideoEffects.h:303-314, nvVFXTrueHDR.h:15 | juste |
| Contrast / Saturation / MiddleGray | — | nvVFXTrueHDR.h:17-19 | juste |
| **NVVFXTRUEHDR_LUMINANCE** | `"MaxLuminance"` | **`"Luminance"`** (nvVFXTrueHDR.h:20) | **faux, corrigé** |
| Signatures NvVFX_* (CreateEffect, DestroyEffect, SetU32, SetString, SetImage, SetCudaStream, Load, Run) | — | nvVideoEffects.h:75-191 | justes |
| NvCVImage_Init, NvCV_GetErrorStringFromCode | — | nvCVImage.h:261, nvCVStatus.h:130 | justes (enums passés en int, même taille en x86-64) |
| Symboles | NvVFX_* dans libVideoFX, NvCVImage_* dans libNVCVImage | `nm -D` | justes |
| **Marqueur de version** | `version.h` ou `include/version.h`, `<P>_MAJOR_VERSION` | **`share/videofx_version.h`**, `NVIDIA_VIDEOEFFECTS_SDK_VERSION_{MAJOR,MINOR,RELEASE,BUILD}` | **faux, corrigé** |
| Nom réel de libVideoFX | `libVideoFX.so.X.Y` | `lib/libVideoFX.so -> .so.1 -> .so.1.3 -> .so.1.3.0` | juste (repli) |
| Version de la fonction | non lue | `#define NVVFXTRUEHDR_VERSION "1.3.0.0"` | maintenant exigée, même majeur et mineur |
| NvVFX_GetVersion | non utilisé | `(maj<<24)\|(min<<16)\|(build<<8)` (nvVideoEffects.h:71-72), 0x01030000 | vérifié à l'ouverture (`maxine_vfx.cpp:159`) |
| **Bibliothèques chargées** | 13 exigées, dont `libnvinfer.so.10`, plus tout `external/` en option | libcudart.so.12, libnppc, libnppial, libnppicc, libnppidei, libnppig (.so.12), **libcudnn.so.9** (NEEDED par libVideoFXLocal), libNVCVImage, libVideoFX, libVideoFXLocal, libnvngxruntime, libnvVFXTrueHDR. Le modèle libnvidia-ngx-truehdr.so.1.2.3 est chargé par NGX lui-même (memfd), avec le libnvidia-ngx.so du pilote. | **TensorRT, cuBLAS et cuFFT ne sont jamais chargés** ; jeu explicite (`sdk_layout.cpp:27`) |
| Codes d'erreur du faux SDK | -3, -4, -5, -6 | SELECTOR -5, PARAMETER -7, PIXELFORMAT -9, INITIALIZATION -12 | faux, corrigé |

Comparaison avec le chemin Linux du lecteur actuel (`aji-rtx-linux/aji_rtx_vfx.cpp`, lecture
seule) :
- même ordre d'appels ;
- RGBA8 en entrée, RGB10A2 en sortie ;
- `ModelDir` à `lib/models`, qui répond UNIMPLEMENTED (vérifié : -2, toléré seul) ;
- flux nullptr.

Il préchargeait `features/*/lib/*.so*` en RTLD_GLOBAL et comptait sur LD_LIBRARY_PATH pour le
reste. Le nouveau chargeur ouvre le jeu exact par chemin absolu
(`src/vendor_hdr/maxine_runtime.cpp:19`) : RTLD_LOCAL pour CUDA, NPP et le cœur, RTLD_GLOBAL
pour la fonction.

**Comportement des réglages** (sur la RTX) :
- Plages identiques au plan : Contrast 0-200, Saturation 0-200, MiddleGray 10-100, Luminance
  400-2000.
- Hors plage, le SDK renvoie NVCV_ERR_PARAMETER (-7) et garde l'ancienne valeur.
- Défauts du SDK : 100, 100, 50 et **650 nits**. Le lecteur passe toujours 1000 (défaut du plan).
- Un réglage changé s'applique sans recharger le modèle.
- Le nouveau réglage `DebandingOff` (0/1) n'est pas exposé.

**Non vérifiable ici** : le RTX Video SDK 1.1 (Windows). Ni l'archive, ni les DLL, ni une
chaîne Windows ne sont sur ce PC, et `rtx_video_abi.h` reste « à confirmer ».

## 2. TrueHDR sur la RTX, par le module

Tests `[nvidia][cuda]` : `tests/unit/vendor_hdr/real_true_hdr_test.cpp`, aides
`cuda_driver.{h,cpp}` (libcuda.so.1 par dlopen) et `real_sdk.{h,cpp}`.

- **Du magasin au GPU** (`real_true_hdr_test.cpp:92`) :
  - import réel dans un dossier de données temporaire ;
  - ouverture par `open_true_hdr` depuis la copie (le runtime NGX y trouve son modèle) ;
  - 1920x1080 ; entrée posée et sortie relue **sur le flux non bloquant seulement**, donc
    l'ordre ne tient que par `hand_to_legacy` / `take_from_legacy` (événements réels) ;
  - alpha 3 partout ; rampe de gris monotone (noir < 64, blanc > 600 en PQ 10 bits) ;
  - deuxième passage aux mêmes octets ;
  - contraste 200 change la sortie, et le retour à 100 redonne les mêmes octets ;
  - `vendor_time()` compte les appels.
- **Contrat d'entrée** (`:170`) :
  - source Y'CbCr 10 bits BT.709 limitée, convertie par `kernels/spec/color.h`
    (`normalize` puis `to_rgb`) ;
  - tramage triangulaire de ±1 code 8 bits avant quantification ;
  - sur la rampe quasi noire (4 codes 10 bits par niveau 8 bits), le plus grand cran entre
    colonnes voisines vaut **17 codes PQ sans tramage, 4,8 avec** ;
  - le vrai SDK n'est jamais appelé pour une entrée non tramée, une matrice BT.601 ou une plage
    pleine (refus InputMismatch).
- **Pas de ligne à marge** (`:200`) : un pas de 4 × largeur + 256 donne les mêmes octets.
- **Flux** (sonde) :
  - `NvVFX_Run(effet, 0)` sur le flux 0 rend la main quand l'image est écrite : le flux 0 est
    au repos au retour dans 500 cas sur 500. Le temps de `process()` est donc du temps hôte
    bloqué (VendorTime) ;
  - sur un flux non bloquant à soi, il faut recharger après `SetCudaStream`, sinon -12 ;
  - **la première image après ce rechargement est fausse** (tous les pixels diffèrent, jusqu'à
    720 codes) ; les suivantes sont justes. Cela confirme le risque « TrueHDR hors flux 0 » du
    plan : garder le flux 0 ;
  - `async=1` sur un flux à soi rend la main avant la fin.

## 3. Mesures et questions du plan

**Temps par image.** Commande :
`ajp-gpu build/linux-prefix/tests/unit/vendor_hdr/ajp_vendor_hdr_tests "[bench]"`.
Protocole : 20 images d'échauffement, puis 5 passes de 100 images, entrée tramée. Les temps mesurés avec TrueHDR ne se publient pas (licence de NVIDIA) : ils restent sur le PC.

**Déterminisme** : oui.
- Deux passages donnent les mêmes octets.
- La suite A, B, A rend A identique.
- Deux instances vivantes rendent les mêmes octets.
- Une instance détruite puis recréée (bibliothèques refermées puis rouvertes) aussi.

La route TrueHDR peut donc rester dans le test framemd5 de l'export.

**Rampe de 32 marches quasi noires.** Elle est faisable sur une seule instance :
- 32 images unies, niveaux 0 à 31 sur 255, réglages par défaut, pic 1000 : sortie R
  0, 8, 25, 39, 51, 62, 72, 82, 90, 99, … 211, 215, 219 (PQ 10 bits) ;
- strictement croissante, identique en montée et en descente ;
- après `set_params`, la même instance rend une autre rampe, elle aussi croissante ;
- coût : 32 images, ou une seule si on accepte des bandes.

**Recréer l'instance.** Pour Maxine, la règle d'instance unique ne s'applique pas : plusieurs
instances vivent ensemble et la recréation est sûre (testé, ASan compris). Elle reste une règle
du RTX Video SDK, non vérifiable ici.

## 4. Magasin des SDK

`SdkStore::import("/usr/local/VideoFX")` vers un dossier temporaire
(`real_sdk_test.cpp:54`) :
- état Ready, version 1.3.0.0 ;
- relu Ready par un nouveau magasin.

**Copie** : `lib/`, `include/`, `share/`, `features/nvvfxtruehdr/`, et d'`external/` les seules
7 bibliothèques chargées (lien et cible : `src/vendor_hdr/sdk_copy.cpp:65`).
- Taille : **133 Mo** au lieu de 4,3 Go. Le Manager actuel copie tout `external/`, soit
  8,1 Go dans `optional-sdks/truehdr`.
- `external/tensorrt` et cuBLAS ne sont pas copiés.

**Version** :
- lue dans `share/videofx_version.h` ;
- recoupée avec le nom réel de `libVideoFX.so` (« SDK mélangé » s'ils divergent) ;
- la fonction doit avoir le même majeur et mineur (`sdk_version.cpp:70-120`).

## Tests et commandes

```
cmake --preset linux-prefix
ajp-build cmake --build --preset linux-prefix                       # 0 avertissement
ajp-run ctest --preset linux-prefix -R '^vendor_hdr: ' -E '^vendor_hdr: Maxine sur la RTX'   # 89/89
ajp-gpu ctest --preset linux-prefix -R '^vendor_hdr: Maxine sur la RTX'                       # 5/5
```

**Sans SDK** (`-DAJP_TEST_MAXINE_SDK=/nonexistent/VideoFX`) : les 7 tests réels sont
« Skipped » (vérifié). Sur la CI, sans libcuda, ils sont aussi SKIP.

**ASan et UBSan** (preset linux-prefix-asan, cible `ajp_vendor_hdr_tests` seule, dossier
supprimé après) :
- hors GPU : 89/89 avec `LSAN_OPTIONS=suppressions=~/ajp-ci/lsan.supp` ;
- sur la RTX : 5/5 avec `ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0`. Le pilote NVIDIA
  exige `protect_shadow_gap=0`, sinon cuInit échoue sous ASan. À ajouter aux règles.

**Format et analyse** :
- clang-format 20.1.8 : `--dry-run --Werror` propre sur tout le dépôt ;
- clang-tidy 20.1.0 et 23.1.3 propres sur `src/vendor_hdr/*.cpp`, sauf
  `sdk_store.cpp:204` : `readability-redundant-lambda-parameter-list`, un contrôle de
  clang-tidy 23 seulement, sur du code non touché.
- Provenance : les deux outils 20 viennent de PyPI (`clang-format==20.1.8`,
  `clang-tidy==20.1.0`) et sont installés dans un venv du scratchpad, hors dépôt.

**Incident.** La vérification du SKIP a lancé une fois les tests GPU sous `ajp-run` au lieu
d'`ajp-gpu`. Ils ont créé un contexte CUDA moins d'une seconde avant de SKIP. Depuis, les tests
vérifient le SDK avant de toucher au GPU.

## Questions ouvertes

1. **`DebandingOff`** (nvVFXTrueHDR.h:21, nouveau en 1.3) : il allège TrueHDR. Sur la mire
   synthétique, il ne change aucun pixel. Faut-il l'exposer comme réglage, ou l'activer sur
   sources propres ? Il demande un rechargement du modèle.
2. **Rampe de calibration** en une seule image à 32 bandes plutôt que 32 images : à mesurer.
   TrueHDR peut dépendre du voisinage.
3. **Fusion avec `origin/w2/fondations`** (205f7d1). `git merge-tree` donne deux conflits
   triviaux :
   - `src/vendor_hdr/maxine_runtime.cpp` : garder cette branche ; la ligne `log()` que w2
     renomme n'existe plus ;
   - `tests/unit/vendor_hdr/CMakeLists.txt` : réunir les deux en
     `DEPS ajp::core ajp::kernels ${CMAKE_DL_LIBS}`.
   - `maxine_vfx.cpp` fusionne seul.
4. **RTX Video SDK 1.1** (DLL, symboles, signatures) : toujours à vérifier sur Windows avec
   l'archive.
5. **Défaut du pic** : le plan dit 1000 nits, le SDK 650 nits. Le lecteur passe toujours sa
   valeur, rien à changer, mais à savoir pour l'émulation native.
