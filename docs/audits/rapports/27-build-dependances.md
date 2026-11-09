# Rapport de l'audit 27. Build, dépendances et ports

Branche : `claude/project-thread-bdgcal`. Commit audité : `d1622ba`. Date : `2026-11-09`.

## Verdict

La construction est saine sur l'essentiel (versions épinglées par commit, aucune option qui dégrade l'image,
archives reproductibles), mais libplacebo a sept mois et 247 commits de retard sur l'amont, dont des correctifs
de rendu (noir relevé par le tramage, YUV plein, mise à l'échelle), et notre correctif de présentation ne
s'applique plus sur la version 7.387.0 ; le `RUNPATH` posé par `patchelf` laisse bien `LD_LIBRARY_PATH` passer
devant le paquet (démontré). Quatre des cinq défauts connus sont toujours présents, SEC-19 l'est en partie.

## Cadrage retenu

Pas de questions posées : Tatoub a demandé d'auditer au maximum sans cadrage. Retenu :

- cible NVIDIA sous TensorRT, Windows 11 et CachyOS ; macOS (`tools/deps/macos-from-source.sh`) hors cible, lu
  seulement pour ce qu'il partage avec Linux ; le VRR est hors sujet ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant. Question suivie : **ce qui, dans la construction,
  empêcherait DistilDRBA d'être bien plus rapide que RIFE** (réponse : rien, voir « Qualité et vitesse ») ;
- consigne de Tatoub : toute baisse de qualité d'image faite pour la vitesse est un défaut, sauf mesure qui la
  justifie ;
- vérification demandée par le coordinateur : `patchelf --set-rpath` et `LD_LIBRARY_PATH`
  (`tools/release/package-linux.sh:124`).

Dans ce conteneur : pas de GPU ni de kit CUDA. Les dépendances du cœur (`tools/deps/linux-from-source.sh --core`)
ont été construites, ce qui a permis de construire `linux-prefix-core` avec et sans `AJP_PCH`. Le reste est lu, ou
vérifié contre l'amont (`git ls-remote`, clone de libplacebo, ports de vcpkg à la baseline de `vcpkg.json`).

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A27-01 | Majeur | `ports/libplacebo/portfile.cmake:8`, `tools/deps/linux-from-source.sh:37` | libplacebo est épinglé à v7.360.1 (14/03/2026). L'amont a publié v7.387.0 (19/10/2026) : 247 commits, dont des correctifs de rendu que le lecteur n'a pas. `ef14f120` « dithering: don't let the dither pattern lift true black off zero » ; `e00b597a` « use real absolute black (0) instead of PL_COLOR_HDR_BLACK » ; `d9922613` « fix full-range YUV decoding » (point neutre de la chroma faux en YUV plein) ; `5ed89596` « preserve energy in ortho scaling » ; `431c3170` « dither onto the sample grid of the target » ; `d4624cbf` « vulkan/command: fix write-after-write hazard on layout transition » ; plus le tone mapping (`ab96f735`, `0321aab1`, `d73ef897`) et `4d82c689` (HDR10+). Et `0001-ajp-present-timing.patch` ne s'applique plus sur 7.387.0 (`src/vulkan/swapchain.c:747`, probablement à cause de `de357dcd`, qui touche la même fonction de présentation) : la mise à jour est bloquée par notre correctif. | Une scène sombre d'anime sur une sortie SDR 8 bits : libplacebo tramant vers 8 bits, des valeurs presque nulles issues des arrondis sont amplifiées en bruit sur le noir pur (visible sur un QD-OLED, l'écran de Tatoub). Une source en YUV plein (capture d'écran, certains encodages web) prend une dérive de couleur. Les deux sont corrigés en amont, pas chez nous. | Rebaser `0001-ajp-present-timing.patch` sur v7.387.0 (ou sur le commit de `de357dcd`), monter les deux épinglages ensemble, puis comparer des captures A/B (`--snapshot`) sur une scène sombre et une source YUV plein. Faire échouer `deps-watch` (renvoi 28) tant que libplacebo a une version de retard. | Lu : `git log v7.360.1..v7.387.0` (247 commits), `git apply --check` échoue sur 7.387.0. Effet visible à confirmer : capture d'un fondu au noir avant et après la montée, comparée en 8 bits. |
| A27-02 | Mineur | `tools/release/package-linux.sh:124`, `:135` ; `CMakeLists.txt:72-74` | `patchelf --set-rpath` pose un `DT_RUNPATH`, même sur un binaire construit avec `DT_RPATH` (`--disable-new-dtags`) : `LD_LIBRARY_PATH` passe alors devant `$ORIGIN/lib`. Le paquet charge donc une autre bibliothèque de même nom si l'utilisateur en exporte une. Le build, lui, est en `DT_RPATH` (préréglages `prefix`) : les tests ne voient jamais la règle du paquet. | `LD_LIBRARY_PATH` pointe un dossier qui contient un `libass.so.9` officiel ou un `libplacebo.so.360` sans notre correctif (mpv construit à la main, environnement conda, lanceur Steam) : `ajp` ne démarre pas (« undefined symbol: ass_set_render_thread_count » ou `pl_vulkan_swapchain_submit`). Un `libnvinfer.so.11` d'une autre 11.x devant celui de NVIDIA livré : le module et `trtexec` construisent avec une version que le paquet n'a pas validée. | `patchelf --force-rpath --set-rpath` (un `DT_RPATH` passe avant `LD_LIBRARY_PATH`, et vaut aussi pour les dépendances indirectes), ici et pour la copie de `trtexec` (renvoi 05). Ajouter à `check_portable` un `readelf -d` qui refuse tout `RUNPATH` dans le paquet. | Démontré (patchelf 0.19.3) : un exécutable construit en `RPATH`, passé par `patchelf --set-rpath '$ORIGIN/lib'`, montre `RUNPATH [$ORIGIN/lib]` et charge la bibliothèque de `LD_LIBRARY_PATH` ; avec `--force-rpath`, il garde `RPATH` et charge celle du paquet. Script ci-dessous. |
| A27-03 | Mineur | `cmake/Dependencies.cmake:64`, `:68` | Le configure accepte un libplacebo sans notre correctif et un libass officiel : il ne demande que `libplacebo>=7.360` et `libass` sans version. L'erreur arrive à l'édition de liens, loin de sa cause. | Sous CachyOS, un `PKG_CONFIG_PATH` où le système passe avant le préfixe (ou un essai sans préfixe) : `cmake` réussit, la construction échoue à la fin sur `undefined reference to pl_vulkan_swapchain_submit` ou `ass_set_render_thread_count`. | Après `pkg_check_modules`, un `check_symbol_exists` (ou `check_c_source_compiles`) de `pl_vulkan_swapchain_submit` et de `ass_set_render_thread_count`, avec un message qui nomme le script du préfixe. | Lu (`src/app/play/placebo_screen.cpp:458`, `src/subs/subtitle_renderer.cpp:118` appellent ces symboles sans repli). |
| A27-04 | Mineur | `cmake/Dependencies.cmake:4-5`, `tools/deps/linux-from-source.sh:5-7`, `tools/release/versions.env:12-16` | « Mêmes versions » sous Linux et Windows n'est vrai que pour ce que `DEPS` construit. dav1d, lcms2, shaderc, FriBidi (livrés dans `lib/` du paquet Linux) et FreeType, HarfBuzz (au moment du build) viennent d'apt sans version : environ dav1d 1.4.1, lcms2 2.14, shaderc 2023.8, FriBidi 1.0.13 sous Ubuntu 24.04, contre dav1d 1.5.4, lcms2 2.19.1, shaderc 2026.2, FriBidi 1.0.18 dans la baseline de vcpkg. Rien ne vérifie l'accord, et `deps-watch.yml` monte la baseline de vcpkg seule (Windows change, Linux non). Catch2 diffère déjà (3.16.0 contre 3.16.1). | Le banc de parité (`docs/parite-windows-linux.md`) compare deux binaires dont libplacebo compile ses shaders avec deux shaderc distants de trois ans, et dont l'AV1 logiciel passe par deux dav1d différents : un écart de temps peut venir des dépendances, pas du lecteur. | Construire dav1d, lcms2 et shaderc dans `DEPS` (ils sont dans `lib/` du paquet de toute façon), et ajouter à la CI un contrôle qui compare chaque référence de `DEPS` à la version du port de la baseline. | Lu ; versions de vcpkg lues dans les `vcpkg.json` des ports au commit `65886d79`. Versions Ubuntu à confirmer : `apt-cache policy libdav1d-dev liblcms2-dev libshaderc-dev libfribidi-dev` sur le runner. |
| A27-05 | Mineur | `tools/deps/linux-from-source.sh:135-138`, `ports/libplacebo/portfile.cmake:38-52` | Les options de libplacebo ne sont fixées que sous Windows (`xxhash`, `unwind`, `libdovi`, `glslang`, `vk-proc-addr` coupés). Sous Linux elles restent en `auto` : elles suivent ce qui est installé sur la machine qui construit. `glslang-dev` est dans `APT_PLAYER`, donc le libplacebo de la CI se construit avec glslang ; sur CachyOS, `xxhash` et `libunwind` sont presque toujours là. | Le libplacebo du poste de Tatoub (CachyOS) n'a pas les mêmes options que celui du paquet Linux, ni que celui de Windows : un écart de comportement ou de vitesse mesuré chez lui peut ne pas exister ailleurs. | Passer les mêmes `-D…=disabled` (ou `enabled`) dans `PLACEBO_OPTS` que dans le port, et les mettre dans la signature du marqueur. | Lu. |
| A27-06 | Mineur | `tools/deps/linux-from-source.sh:58-66`, `:129`, `:142`, `:165` | Le marqueur d'une dépendance construite ignore les bibliothèques du système qu'elle lie (dav1d, shaderc, lcms2, FreeType…) et le compilateur. Les workflows restaurent un préfixe par `restore-keys: deps-player-gcc-13-`, donc parfois construit sur une image plus ancienne du runner. | L'image `ubuntu-24.04` monte `libdav1d` ou `libshaderc` : le FFmpeg et le libplacebo du cache restent liés aux anciennes. Au mieux le paquet est refusé par `check_portable` ; au pire il livre une copie que plus rien ne reconstruit. Sous CachyOS, un passage de GCC 15 à 16 ne refait rien. | Mettre dans chaque signature les versions `pkg-config --modversion` de ce que la dépendance lie, et `"$CC" -dumpfullversion`. | Lu. |
| A27-07 | Mineur | `CMakeLists.txt:28`, `CMakePresets.json:136` | Le module livré se compile pour `75-real;86-real;89-real;120-real;120-virtual`, mais la tâche CUDA de chaque pull request ne compile que `89-real`. Les noyaux pour Turing, Ampere et Blackwell ne passent par nvcc et ptxas (`--warn-on-spills` en erreur) qu'au moment du paquet TensorRT. | Une modification d'un noyau qui déborde en mémoire locale sur sm_75, ou qui emploie un intrinsèque absent de sm_75, passe la CI et casse `trt-package.yml` le jour de la release. | Faire compiler la liste complète à la tâche `cuda` (au moins sur `main`), ou ajouter `AJP_CUDA_ARCHITECTURES` complet à une tâche du soir. | Lu (`.github/workflows/ci.yml:250`). |
| A27-08 | Mineur | `ports/libplacebo/0001-ajp-present-timing.patch:153-170`, `:228-232` | Le chemin `QUEUE_FULL` du correctif suppose que la présentation refusée exécute quand même l'attente de son sémaphore, ce que le commentaire affirme sans référence à la spécification ; et chaque barrière orpheline coûte une attente de 50 ms dans `vk_sw_start_frame` avant d'être remplacée. | Si la file de timings se remplit (pilote 615.78.08, vu par le correctif), l'image suivante attend 50 ms : un à-coup visible à 24 i/s interpolé vers 165 Hz. Si le sémaphore n'est pas consommé, le signal suivant sur ce sémaphore est invalide (couche de validation). | Citer le passage de VK_EXT_present_timing qui garantit l'attente du sémaphore, sinon remplacer le sémaphore avec la barrière. Remplacer la barrière orpheline sans attendre si son image n'a pas été présentée. Compter ces remplacements dans les journaux. | À confirmer : lecture avec `--present` au niveau Timing, couche de validation active, jusqu'à un `QUEUE_FULL` ; relever les messages et le temps de `pl_swapchain_start_frame`. |
| A27-09 | Mineur | `cmake/Dependencies.cmake:115-117`, `vcpkg.json:58-64` | Sous Windows, ImGui vient du port vcpkg sans sa fonction `freetype` ; `AJP_WITH_FREETYPE` n'est défini que pour la copie du préfixe Linux. L'interface Windows rastérise donc ses polices avec stb_truetype, sans les polices CFF2 et variables, alors que Linux passe par FreeType. | Sous Windows, une police de repli CFF2 ou variable est refusée (« police de repli … refusée par le chargeur de polices », `src/app/play/ui.cpp:187`) et ses caractères manquent ; le texte est rendu autrement que sous Linux. | Ajouter la fonction `freetype` au port `imgui` de `vcpkg.json`, et définir `AJP_WITH_FREETYPE` quand la cible vcpkg l'a (`IMGUI_ENABLE_FREETYPE` dans ses définitions). | Lu (port `imgui` de la baseline : la fonction `freetype` existe). |
| A27-10 | Mineur | `CMakePresets.json:18`, `:31` ; `CMakeLists.txt:16-18` | `AJP_WARNINGS_AS_ERRORS` vaut OFF par défaut « pour qu'un compilateur plus récent construise encore », mais tous les préréglages le mettent à ON, y compris `linux-prefix-cuda-machine` que `docs/nvidia-linux.md:26` donne pour CachyOS. Aucune tâche de CI ne construit avec le compilateur d'Arch (GCC 15 ou 16, Clang 21) : seulement GCC 13 et Clang 20. | Une mise à jour de GCC sous CachyOS ajoute un avertissement : le lecteur de Tatoub ne se construit plus, sans que la CI l'ait vu. | Un préréglage local sans `-Werror` (ou `AJP_WARNINGS_AS_ERRORS` hérité de l'environnement), et une tâche de CI dans un conteneur `archlinux` qui construit le cœur. | À confirmer : `cmake --preset linux-prefix-core` avec le GCC courant de CachyOS. |
| A27-11 | Mineur | `cmake/AjpSpirv.cmake:35`, `:60-63` | `spirv-val` est facultatif et son absence passe sans un mot. Sous Windows (vcpkg) il n'est pas trouvé : les shaders de calcul du repli Vulkan n'y sont jamais validés. | Un shader valide pour glslc mais refusé par `spirv-val` (même cible `vulkan1.x`) passe la construction Windows et n'échoue qu'au chargement chez l'utilisateur. | `message(STATUS)` quand il manque, et `REQUIRED` en CI ; sous Windows, le prendre du port `spirv-tools` (outil de l'hôte). | Lu (la règle de CI `check-shaders.sh` ne tourne que sous Linux). |
| A27-12 | Mineur | `third_party/cutlass/` | CUTLASS est recopié en partie (259 en-têtes) sans commit ni somme : seule `THIRD_PARTY_NOTICES.md:25` dit « 4.8.1 », et `cutlass/version.h` n'est pas copié. Aucun script ne refait la copie. | Une mise à jour de CUTLASS ou une vérification de la copie (un en-tête modifié à la main) ne peut pas être refaite ni prouvée ; les noyaux FP8 du SPAN en dépendent. | Noter le commit de la 4.8.1 et une somme de l'arbre copié dans un fichier du dossier, avec le script qui sélectionne les 259 fichiers. | Lu. |
| A27-13 | Mineur | `cmake/AjpModule.cmake:61-77`, `:38-46` | Les en-têtes précompilés ne s'appliquent qu'aux modules, pas aux exécutables de test (47 des 87 unités de compilation de `linux-prefix-core`). Sur les modules du cœur, aucun gain mesuré (QUA-09). | Un développeur active `-DAJP_PCH=ON` comme le conseille `CLAUDE.md` : la reconstruction n'est pas plus courte. | Mesurer sur `ajp_app_play` (le module le plus lourd, qui journalise) ; si le gain n'y est pas, retirer l'option et la phrase de `CLAUDE.md`. Sinon l'étendre aux tests. | Mesuré : `ninja ajp_core ajp_core_log ajp_kernels`, GCC 13, Debug, 4 fils, deux passes chacun : 14,1 et 14,1 s sans, 14,6 et 14,7 s avec. |

### Démonstration du `RUNPATH` (A27-02)

```sh
gcc -shared -fPIC -Wl,-soname,libq.so.1 a.c -o build/libq.so.1   # « paquet »
gcc -shared -fPIC -Wl,-soname,libq.so.1 b.c -o autre/libq.so.1   # « LD_LIBRARY_PATH »
gcc m.c -Lbuild -lq -Wl,--disable-new-dtags -Wl,-rpath,$PWD/build -o build/ajp   # RPATH, comme le préfixe
cp build/ajp pkg/ && cp build/libq.so.1 pkg/lib/ && patchelf --set-rpath '$ORIGIN/lib' pkg/ajp
readelf -d pkg/ajp | grep PATH            # (RUNPATH) Library runpath: [$ORIGIN/lib]
LD_LIBRARY_PATH=$PWD/autre pkg/ajp        # LD_LIBRARY_PATH
# avec patchelf --force-rpath --set-rpath : (RPATH) [$ORIGIN/lib], et « paquet » malgré LD_LIBRARY_PATH
```

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| SEC-01 | Toujours présent | `src/app/CMakeLists.txt:68` : `AJP_UPDATE_PUBKEY_BACKUP` vide, et aucun préréglage ni workflow ne le pose (`grep PUBKEY .github tools CMakePresets.json`). Le code de révocation existe (`src/app/play/updater.cpp:251`, `:429`), mais sans clé de secours la révocation de la clé active ne laisse aucune clé. Le fond (garde de la clé, P1.14) relève de l'audit 26. |
| QUA-03 | Toujours présent | Revérifié par les audits 05 (A05-08) et 20 (A20-10) ; `src/core/drawings.cpp`, `src/core/profile.cpp`, `src/prepare/receipt.cpp` toujours compilés par leurs `CMakeLists.txt`. |
| QUA-09 | Toujours présent | Aucun préréglage n'active `AJP_PCH` (`CMakePresets.json`). Le gain annoncé par `CLAUDE.md` (« environ un quart ») n'est pas retrouvé sur le cœur : A27-13. |
| SEC-19 | Corrigé en partie | CUDA : toujours la dernière 13.4.x du jour (`tools/deps/linux-tensorrt.sh:32-33`, `tools/deps/windows-tensorrt.ps1:35-39`), et l'empreinte du paquet TensorRT ne le voit pas (renvoi 28). `trtexec` : l'archive est lue en flux et `tar` l'extrait dans un dossier de travail avant la comparaison de son SHA, mais rien n'est gardé si elle échoue (`linux-tensorrt.sh:115-126`) : corrigé en substance. Marqueurs : toujours la référence et non le commit (`linux-from-source.sh:58-66`), plus A27-06. `pip` : Meson est épinglé (`meson==$MESON_VERSION`, `versions.env:6`) mais sans `--require-hashes`. |
| SEC-20 | Toujours présent | `src/gpu/cuda/backend/CMakeLists.txt:95` : `AJP_TRTEXEC_NO_FALLBACK` en `Release` seulement ; `linux-prefix-cuda-release` et `linux-prefix-cuda-machine` (`CMakePresets.json:144`, `:152`), le lecteur de Tatoub, sont en `RelWithDebInfo`. Conséquence : A05-05. Correction : `$<NOT:$<CONFIG:Debug>>`. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 28 | `tools/release/trt-fingerprint.sh:17-19` | L'empreinte du paquet TensorRT ignore le `CMakeLists.txt` racine (dont `AJP_CUDA_ARCHITECTURES`, ligne 28), `CMakePresets.json` (Release, LTO de `linux-release`), `third_party/cutlass` (noyaux FP8) et `tools/deps/linux-tensorrt.sh` (version de CUDA non épinglée) : ajouter une architecture ou changer les options réutilise l'ancien paquet. |
| 28 | `.github/workflows/deps-watch.yml` | La baseline de vcpkg se monte seule, sans `DEPS` de Linux (A27-04) ; et libplacebo en retard d'une version n'y bloque rien (A27-01). |
| 28 | `.github/workflows/release-linux.yml:67-70` | `restore-keys: deps-player-gcc-13-` restaure un préfixe d'une autre image du runner ; avec A27-06, des copies périmées peuvent entrer dans le paquet. |
| 05 | `src/gpu/cuda/backend/CMakeLists.txt:83` | La copie de `trtexec` est rangée par `patchelf --set-rpath "$ORIGIN"` : même `RUNPATH` que A27-02, `LD_LIBRARY_PATH` peut lui donner un autre `libnvinfer` que celui du module. `--force-rpath` change son SHA, recalculé au configure. |
| 49 | `src/gpu/cuda/backend/CMakeLists.txt:27-31` | Baisse de précision pour la vitesse, activée par défaut à la construction sous Linux : les noyaux FP8 du SPAN (cran Performance, pris par défaut dans les préréglages sur RTX 40 et 50, `docs/nvidia-linux.md:45`). Mesure qui la justifie : 9,1 à 9,3 ms contre 14,25 ms en FP16, pour 50 à 52 dB (`docs/plan-v0.6.md:122`). Sous Windows le défaut est OFF : deux images différentes selon le système (déjà renvoyé 02 → 34). À trancher par Tatoub. |

## Qualité et vitesse

- **Aucune option de construction ne dégrade l'image.** Ni `-ffast-math`, ni `--use_fast_math` pour nvcc, ni
  `-Ofast` ; glslc en `-O` seul (pas de précision relâchée) ; MSVC en `/fp:precise` ; `ajp_kernels` et les tests
  en `-ffp-contract=off`. Les approximations relevées ailleurs (`__expf` de RIFE, A03-11) sont dans le code, pas
  dans les options.
- **DistilDRBA.** Rien dans la construction ne le freine face à RIFE : mêmes options pour `drba_fused_kernels.cu`
  et `rife_fused_kernels.cu`, `-lineinfo` sans coût à l'exécution, code réel pour chaque architecture RTX (pas
  de compilation JIT au premier lancement, sauf au-delà de sm_120). Les freins sont dans le code et le modèle
  (A03-13, A03-14, A03-18).
- **La seule baisse de précision voulue que la construction active** est le SPAN FP8 sous Linux, justifiée par
  une mesure (renvoi 49).
- **libplacebo en retard** (A27-01) est aujourd'hui la plus grosse perte de qualité de cette zone : des
  correctifs de rendu publiés en amont ne sont pas chez nous.

## Points positifs

- Chaque dépendance construite est épinglée par son commit complet, vérifié après le clonage
  (`linux-from-source.sh:46-52`), et le correctif de libplacebo entre dans la signature du marqueur.
- Les archives de TensorRT et la roue PyPI sont vérifiées par une empreinte épinglée ; sous Windows, avant
  l'extraction.
- Archives Linux reproductibles (`tar --sort=name`, `--mtime` du commit, `gzip -n`) et `-ffile-prefix-map` qui
  sort les chemins de la machine des binaires.
- Le paquet refuse toute bibliothèque système dont le nom diffère entre distributions (`check_portable`), et
  libstdc++, FreeType ou HarfBuzz sont laissés au système, ce qui évite les conflits avec Mesa.
- Les correctifs de toml++ sont appliqués au configure et arrêtent la construction dès que l'amont change le
  code visé.
- `-fvisibility=hidden`, `--exclude-libs,ALL` et `--no-undefined` sur le module TensorRT : pas de collision de
  symboles avec le spdlog du lecteur.
- Versions de vcpkg et de `DEPS` identiques pour tout ce que `DEPS` construit (FFmpeg 9.0.2, SDL 3.4.20, ImGui
  1.93.0, spdlog 1.17.0, Vulkan-Headers 1.4.363, toml++ 3.4.0) ; libass du fork à jour (0.17.5, dernière
  version de libass).

## Questions ouvertes

- Monter libplacebo à 7.387.0 demande de rebaser le correctif de présentation : à faire avant la 1.0 ? Les
  correctifs de tramage et de noir absolu concernent chaque image SDR.
- Le SPAN FP8 par défaut sous Linux (renvoi 49) : Tatoub garde-t-il ce choix, sachant que Windows ne le fait
  pas ?
