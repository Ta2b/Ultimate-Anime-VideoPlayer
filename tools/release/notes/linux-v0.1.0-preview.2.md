Préversion de test pour Linux x86-64, centrée sur les GPU AMD et Intel par le moteur Vulkan. Rien de cette version n'a encore tourné sur un vrai GPU : des choses peuvent casser.

## Nouveautés

### Modèles inclus
- L'archive contient les modèles SPAN AnimeJaNai HD V3.1 (Sharp1 et normal, Performance et Balanced) et RIFE 4.25 et 4.22 lite : la lecture avec agrandissement et interpolation marche dès l'extraction.

### Décodage matériel sous Linux
- Avec le moteur Vulkan, la vidéo était décodée par le processeur. Elle passe maintenant par VA-API, puis repli sur le processeur.
- Avec un iGPU Intel activé (UHD 770 par exemple), c'est lui qui décode (AV1, HEVC 10 bits, VP9), et la carte graphique reste libre pour SPAN et RIFE. S'il ne gère pas le codec, la carte graphique ou le processeur prend le relais.

### Moteur Vulkan sur AMD RDNA (RX 5000, RX 6000)
- L'autotuner essaie les deux tailles de vague (32 et 64) et des convolutions qui calculent 2 ou 4 pixels par thread. Chaque modèle est remesuré une fois au premier lancement.
- La mesure de la paire suivante (coupes de scène, dessins tenus) se fait en avance, pendant RIFE et SPAN.
- À ×3 et plus, l'entrée de RIFE n'est écrite qu'une fois par paire.

### Écran de bienvenue
- MIGraphX n'est plus proposé aux GPU AMD que ROCm ne gère pas (RX 6500, RX 6600, RX 6700, GPU intégrés).

## Installation

Téléchargez `ajp-linux-x64.tar.gz` (vérifiable avec `SHA256SUMS`), puis :

```
tar xf ajp-linux-x64.tar.gz
cd ajp-linux-x64
./ajp play "épisode.mkv" --fullscreen
```

Il faut une distribution récente (glibc 2.39 ou plus : Ubuntu 24.04, Fedora 40, Arch). Paquets du système nécessaires (Ubuntu/Debian) : `libvulkan1 mesa-vulkan-drivers libva2 libva-drm2 libva-x11-2 intel-media-va-driver`. Le fichier `LISEZMOI.txt` de l'archive donne les équivalents Fedora et Arch, les licences des modèles et les réglages conseillés pour les GPU de 4 Go.

Sur un GPU de 4 Go (RX 6500, RX 6500M) : commencez par le profil 7 « Léger 1080p » et passez en 720p si `--stats` laisse de la marge. Branchez l'écran sur la carte graphique.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01HskzHcmMh1oNuwF54Lto5e
