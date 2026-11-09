# Complément de l'audit 15. Renvois reçus des autres audits

Branche : `claude/project-thread-0xktxo`. Commit audité : `9be1f5a`. Date : `2026-11-09`.

## Verdict

Aucun des deux renvois n'ouvre de défaut nouveau : `libcuda.so.1` chargé par son nom est le doublon de A34-08
(avec une précision sur le vrai scénario), et l'AV1 sur RTX 20 passe bien en logiciel puis par la copie hôte du
thread de soumission, ce qui est le doublon de A01-03 ; le décodage logiciel lui-même est une limite du matériel,
pas un défaut du choix des essais.

## Cadrage retenu

Celui du rapport 15 (NVIDIA sous TensorRT, Windows 11 et CachyOS, aucune question à Tatoub). Pas de carte NVIDIA ni
de dépendance dans le conteneur : chaque conclusion vient de la lecture du code au commit `9be1f5a` et du comportement
connu de FFmpeg et de glibc, cité là où il sert.

## Renvois traités

| Renvoi | De | Conclusion | Preuve |
|---|---|---|---|
| `src/media/cuda_driver.cpp:105-109`, `libcuda.so.1` chargé par son nom | 34 | **Doublon de A34-08**, gravité Mineur maintenue, avec une précision sur le scénario. Sous Linux, `open_library` fait `dlopen("libcuda.so.1", RTLD_NOW \| RTLD_LOCAL)` (`cuda_driver.cpp:105-108`), alors que Windows se limite à System32 (`:95-97`) et que la télémétrie cherche NVML dans une liste de dossiers absolus (`src/app/play/gpu_telemetry.cpp:269-272`). Un nom seul passe par `LD_LIBRARY_PATH`, puis le `RUNPATH` de l'objet qui appelle (`$ORIGIN/lib`, posé sur `ajp` par `tools/release/package-linux.sh:135`), avant le cache de `ldconfig`. Le scénario de A34-08 (un `libcuda.so.1` déposé dans `lib/`) ne donne rien de plus que d'écrire dans le dossier du lecteur, qui permet déjà de remplacer `ajp`. Le cas réaliste est autre : un `LD_LIBRARY_PATH` qui pointe vers un `libcuda.so.1` d'une autre version que le module noyau, par exemple le paquet de compatibilité ascendante `cuda-compat` (`/usr/local/cuda/compat`) laissé dans le profil du shell. Le pilote chargé refuse alors le module noyau (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, 803), NVDEC est écarté, et comme `libcudart` du module TensorRT reprend le même `soname` déjà chargé, TensorRT tombe avec lui. | Lu. La correction de A34-08 (dossiers absolus d'abord, nom seul en dernier recours, journalisé avec le chemin retenu par `dlinfo`) règle aussi ce cas : une fois le vrai pilote chargé, le `dlopen("libcuda.so.1")` de `libcudart` rend la même bibliothèque, par son `soname`. À ajouter au journal : la version rendue par `cuDriverGetVersion`, pour reconnaître le cas dans un rapport d'utilisateur. |
| `src/app/play/chain_setup.cpp:43`, RTX 20 : AV1 en logiciel puis copie hôte sur le thread de soumission | 37 | **Doublon de A01-03** pour la copie ; **réfuté comme défaut** pour le choix des essais. Le chemin est bien celui que décrit le renvoi. Sous TensorRT, les essais sont `Cuda` puis `Software` (`chain_setup.cpp:43`). L'AV1 n'est pas dans `hardware_hopeless` (`video_decoder.cpp:320-323`, H.264 seulement), donc `Cuda` s'ouvre avec le décodeur `av1` natif de FFmpeg, qui porte l'accélération NVDEC (`hw_device.cpp:57-66`, `video_decoder.cpp:265`). Au premier en-tête de séquence, l'initialisation NVDEC de FFmpeg interroge `cuvidGetDecoderCaps`, que Turing refuse pour l'AV1 ; `ff_get_format` retire alors `AV_PIX_FMT_CUDA` et rappelle `get_format`, qui bascule (`fall_back`, `video_decoder.cpp:101-103`) et rouvre en logiciel avec les paquets rejoués (`reopen_software`, `:121-158`). `avcodec_find_decoder(AV_CODEC_ID_AV1)` rend alors `libdav1d`, présent dans toutes les constructions (`vcpkg.json`, fonction `dav1d` ; `tools/deps/linux-from-source.sh:127`), sur 8 threads au plus (`kAutoDecoderThreads`, `src/media/hw_api.h:78`). Comme `download = false` sous TensorRT (`chain_setup.cpp:69`), l'image `yuv420p10` reste en mémoire hôte telle quelle (`map_pixel_format` l'accepte, aucune conversion par `to_chain_yuv`, `frame_data.cpp:283-286`) et part par `HostUploads::upload` sur le thread de la chaîne (`tensorrt_kernels.cpp:191`, `host_upload.cpp:82-129`) : environ 25 Mo copiés par image en 4K 10 bits, exactement le cas chiffré par A01-03. Il n'existe pas de meilleur essai à ajouter : Turing n'a aucun décodeur AV1 matériel, ni par NVDEC, ni par D3D11VA, ni par Vulkan Video. L'essai `Cuda` inutile ne coûte qu'une ouverture de contexte et le rejeu d'un paquet. | Lu. Aucune baisse de qualité : `libdav1d` décode à l'identique et l'image garde ses 10 bits et sa chroma. Ce qui reste à mesurer, et qui appartient à A01-03 : le temps de `upload` dans le profil GPU (`--stats`) sur une AV1 4K 10 bits à 24 i/s sur une RTX 20, et la charge CPU de `libdav1d` sur 8 threads à côté. Si la copie dépasse quelques millisecondes, la correction de A01-03 (copie sur le thread du décodeur, ou `cudaHostRegister` sur le tampon de FFmpeg) suffit ; rien n'est à changer dans `decoder_attempts`. Petit gain possible, sans enjeu : sauter `Cuda` d'avance pour l'AV1 quand l'architecture est sm 7.5, comme `hardware_hopeless` le fait pour H.264, pour un journal plus clair (« aucun décodeur matériel pour l'AV1 sur cette carte » au lieu d'un essai puis d'un repli). |

## Défauts

Aucun nouveau. Le dernier identifiant du rapport 15 reste A15-14.

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A34-08 | Toujours présent | `src/media/cuda_driver.cpp:105-108`. Scénario à remplacer par celui du `LD_LIBRARY_PATH` ci-dessus. |
| A01-03 | Toujours présent | `src/gpu/cuda/backend/host_upload.cpp:110-129`. Couvre aussi l'AV1 sur RTX 20. |

## Renvois

Aucun.

## Points positifs

- Le repli de NVDEC vers le logiciel avant la première image rejoue les paquets déjà envoyés et rouvre sur tous
  les threads demandés : une AV1 sur RTX 20 démarre sans image perdue.
- Sous Windows, le pilote CUDA ne se charge que depuis System32 (`cuda_driver.cpp:95-97`).

## Questions ouvertes

Aucune.
