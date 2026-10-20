Préversion de test pour Linux x86-64 : GPU AMD, Intel et NVIDIA par le moteur Vulkan, et des moteurs à télécharger depuis l'écran de bienvenue (TensorRT et RTX HDR, MIGraphX, OpenVINO). Rien de cette version n'a encore tourné sur un vrai GPU : des choses peuvent casser.

## Nouveautés

### Moteurs à télécharger depuis l'application
- L'écran de bienvenue (ou `./ajp --first-run`) propose les moteurs qui conviennent au GPU ; ceux que vous cochez se téléchargent depuis cette release, vérifiés par leur SHA-256.
- **TensorRT** (NVIDIA, 1,2 Go) : la licence de NVIDIA s'affiche et s'accepte à l'écran avant le téléchargement.
- **RTX HDR** (NVIDIA) : la conversion SDR vers HDR de NVIDIA, sous Linux aussi. Il faut un écran HDR et le pilote NVIDIA 550 ou plus.
- **MIGraphX** (AMD) : construit pour ROCm 7.2 ; il faut ROCm 7 installé (`sudo apt install migraphx` depuis le dépôt ROCm 7 d'AMD). Avec ROCm 6 seulement, l'écran le dit.
- **OpenVINO** (Intel) : les GPU Arc et Xe, intégrés ou non. Il faut le pilote compute d'Intel (`intel-opencl-icd libze-intel-gpu1`).

### Modèles inclus
- L'archive contient les modèles SPAN AnimeJaNai HD V3.1 (Sharp1 et normal, Performance et Balanced) et RIFE 4.25 et 4.22 lite : la lecture avec agrandissement et interpolation marche dès l'extraction.

### Décodage matériel sous Linux
- Avec le moteur Vulkan, la vidéo était décodée par le processeur. Elle passe maintenant par VA-API, puis repli sur le processeur.
- Avec un iGPU Intel activé (UHD 770 par exemple), c'est lui qui décode (AV1, HEVC 10 bits, VP9), et la carte graphique reste libre pour SPAN et RIFE. S'il ne gère pas le codec, la carte graphique ou le processeur prend le relais.

### Moteur Vulkan sur AMD RDNA (RX 5000, RX 6000)
- L'autotuner essaie les deux tailles de vague (32 et 64) et des convolutions qui calculent 2 ou 4 pixels par thread. Chaque modèle est remesuré une fois au premier lancement.
- La mesure de la paire suivante (coupes de scène, dessins tenus) se fait en avance, pendant RIFE et SPAN.
- À ×3 et plus, l'entrée de RIFE n'est écrite qu'une fois par paire.

- RX 7000 et RX 9000 : convolutions par matrices coopératives (WMMA), écartées d'office si leur sortie diffère du noyau direct. Le profil 1 est proposé dès 11 Go.

### NVIDIA sous Linux (RTX 4050 portable)
- Le moteur Vulkan sert aussi les cartes NVIDIA, avec les cœurs Tensor (matrices coopératives) quand le pilote les annonce. TensorRT se télécharge depuis l'écran de bienvenue.
- La mémoire réservée suit celle de la carte (6 Go sur une RTX 4050 portable) au lieu de 2 Go fixes. Le lecteur démarre sur le profil 7 « Léger 1080p ».
- Sur un portable à deux GPU, lancez le lecteur sur la carte NVIDIA : `prime-run ./ajp play "épisode.mkv"`, ou `__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only ./ajp play "épisode.mkv"`.

### Écran de bienvenue
- MIGraphX n'est plus proposé aux GPU AMD que ROCm ne gère pas (RX 6500, RX 6600, RX 6700, GPU intégrés).

## Installation

Téléchargez `ajp-linux-x64.tar.gz` (vérifiable avec `SHA256SUMS`), puis :

```
tar xf ajp-linux-x64.tar.gz
cd ajp-linux-x64
./ajp play "épisode.mkv" --fullscreen
```

Il faut une distribution récente (glibc 2.38 ou plus : Ubuntu 24.04, Debian 13, Fedora 39, Arch). Paquets du système nécessaires (Ubuntu/Debian) : `libvulkan1 mesa-vulkan-drivers libva2 libva-drm2 libva-x11-2 intel-media-va-driver`, plus le pilote NVIDIA propriétaire (550 ou plus) sur une carte NVIDIA. Le fichier `LISEZMOI.txt` de l'archive donne les équivalents Fedora et Arch, les licences des modèles et les réglages conseillés pour les GPU de 4 Go.

Sur un GPU de 4 Go (RX 6500, RX 6500M) : commencez par le profil 7 « Léger 1080p » et passez en 720p si `--stats` laisse de la marge. Branchez l'écran sur la carte graphique.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01HskzHcmMh1oNuwF54Lto5e
