Préversion de test pour Linux x86-64 : GPU AMD, Intel et NVIDIA par le moteur Vulkan, et des moteurs à télécharger depuis l'écran de bienvenue (TensorRT et RTX HDR, MIGraphX, OpenVINO). Elle remplace la preview.2 et reprend les corrections de la preview.3, qui n'a pas été publiée. Elle n'a tourné que dans les tests automatiques et sur une seule carte NVIDIA : des choses peuvent casser, surtout sur AMD et Intel.

## Corrections
- **Gel du PC à la fermeture** (moteur Vulkan) : fermer le lecteur, ou changer un réglage qui reconstruit la chaîne, pouvait bloquer la carte graphique une seconde puis laisser le lecteur figé pour de bon. Il rendait sa mémoire GPU pendant que la carte travaillait encore dessus. Il attend maintenant la fin de ce travail, et aucune attente ne peut plus le bloquer plus de 3 s.
- **TensorRT et RTX HDR « pas encore publiés »** : l'écran de bienvenue refusait tout l'index des moteurs de la preview.2 à cause de l'entrée RTX HDR. Ils se téléchargent maintenant, et une entrée inconnue d'une version future est écartée seule.
- **Mémoire pleine sans ReBAR** (RTX 4050 portable, Optimus, cartes plus anciennes) : le moteur Vulkan refusait la lecture en 4K quand le petit tas de 256 Mo partagé avec le processeur était plein. Il passe maintenant à la mémoire système pour les envois et les relectures ; les réseaux restent dans la mémoire de la carte.
- **Fichiers de réglages abîmés** : un fichier de réglages ou de profils mal formé (`play.toml`, `settings.toml`, `user.toml`) pouvait faire planter le lecteur. Il est maintenant refusé proprement.
- **Coupure de courant** : les réglages, les points de reprise et les profils sont écrits sur le disque avant de remplacer l'ancien fichier, donc une coupure juste après un enregistrement ne laisse plus un fichier vide (btrfs, NTFS).
- **Couleurs** : les vidéos en BT.2020 à luminance constante, SMPTE 240M ou FCC se lisent avec la matrice la plus proche au lieu d'être refusées.
- **TensorRT** : les vidéos que NVDEC ne décode pas, comme le H.264 10 bits des fansubs Hi10P, sont décodées par le processeur au lieu d'être refusées.
- **Écran sans fréquence connue** : quand le système ne donne pas la fréquence de l'écran, les images ne partent plus par deux, l'une collée à l'autre.
- **Interface** : le panneau de réglages s'affiche aussitôt quand on l'ouvre au clavier (P), le nom d'un profil peut être effacé puis retapé, et les messages à l'écran gardent leurs lignes vides et ne cassent plus sur une accolade dans un titre.

## Nouveautés

### Lecture
- **Écrans à fréquence fixe** : la cadence se cale sur le rafraîchissement quand il tombe juste (24 im/s sur 120 Hz par exemple), et le son suit en se rééchantillonnant. Le lecteur vise l'instant où l'image apparaît vraiment, et le réglage « Latence de l'écran » corrige son estimation.
- **Warp et SPAN** : nouveau réglage, sur « Auto » par défaut. Au-dessus d'une sortie 1080p, l'interpolation se fait après SPAN, donc SPAN ne tourne qu'une fois par image source (en ×3 vers la 4K, trois fois moins). « Avant SPAN » garde la qualité maximale.
- **RIFE à ×3 et plus** : tant que le mouvement reste fort, la passe du milieu qui servait seulement à le mesurer est sautée.
- **Avances et retours rapides** dans les MKV : le lecteur lit l'image clé dans l'index du fichier au lieu de chercher jusqu'à trois fois.
- En pause, le lecteur ne se réveille plus chaque milliseconde, et l'ouverture d'une chaîne ne relit plus chaque modèle en entier.

### Fluidité mesurée
- Les mesures (Tab) ont une ligne « fluidité » : les images neuves par seconde face à la cadence attendue, les images manquées et le retard p95 des images sur l'horloge. Les paires laissées sans interpolation exprès (coupes de scène, dessins tenus) sont comptées à part, entre parenthèses.
- `--stats` ajoute les colonnes `missing`, `lag_us` et `uninterpolated`. `ajp play --one-lane` fait tout passer par une seule file GPU, pour lire le coût propre de chaque réseau.

### DistilDRBA v2 Lite
- L'interpolation DistilDRBA est livrée dans `models/rife/`, à côté de RIFE : elle se choisit dans les réglages d'interpolation.

### Export
- L'extension du fichier choisit le conteneur : `.mkv` (par défaut, garde toutes les pistes), `.mp4` ou `.mov`, où chaque image tombe à son temps exact, même à 120 im/s. Le panneau d'export propose MP4 ou MOV à partir de 100 im/s. MP4 et MOV ne gardent que les pistes qu'ils acceptent (pas de sous-titres ASS ni de TrueHD).
- Quand le pilote le permet, les images rendues vont droit à l'encodeur sur la carte graphique, sans repasser par le processeur.

### Premier lancement plus court
- L'autotuner Vulkan essaie chaque variante une fois et ne mesure en entier que les plus rapides. Une mise à jour du pilote ne relance plus toutes les mesures : chaque choix est seulement revérifié.
- TensorRT garde un cache de mesures par GPU, partagé entre les modèles et les tailles d'image.
- Les GPU AMD RDNA3 et RDNA4 pas encore listés, et les nouvelles cartes NVIDIA, se voient proposer les matrices coopératives, vérifiées par l'autotuner.
- Les polices de l'interface se chargent pendant l'ouverture de l'écran.

### Optimisations
- **Intel Core Ultra (NPU)** : au premier lancement, OpenVINO mesure SPAN sur le NPU et sur le GPU, et garde le NPU s'il est presque aussi rapide et donne la même image. Le GPU reste alors libre pour RIFE et le rendu. Il faut le pilote NPU d'Intel (`linux-npu-driver`).
- **GPU Intel intégrés** : avec OpenVINO, le réseau lit et écrit directement les images du lecteur, sans copie.
- **GPU sans matrices coopératives** (Arc intégré des Core Ultra, Iris Xe, Radeon RX 6000) : convolutions 3 × 3 par Winograd, avec 2,25 fois moins de multiplications. L'autotuner ne les garde que si elles sont plus rapides et donnent la même sortie.
- **NVIDIA (moteur Vulkan)** : matrices coopératives avec sommes en fp16 quand le pilote les annonce, retenues seulement si leur sortie concorde.
- **MIGraphX** : les réseaux s'enchaînent avec la chaîne Vulkan sur le GPU, sans attente du processeur.
- **Moteur Vulkan** : une soumission au GPU par image au lieu d'une par noyau. Chaque modèle est remesuré une fois au premier lancement.

### Moteurs à télécharger depuis l'application
- L'écran de bienvenue (ou `./ajp --first-run`) propose les moteurs qui conviennent au GPU ; ceux que vous cochez se téléchargent depuis cette release, vérifiés par leur SHA-256.
- **TensorRT** (NVIDIA, 1,2 Go) : la licence de NVIDIA s'affiche et s'accepte à l'écran avant le téléchargement.
- **RTX HDR** (NVIDIA) : la conversion SDR vers HDR de NVIDIA, sous Linux aussi. Il faut un écran HDR et le pilote NVIDIA 550 ou plus.
- **MIGraphX** (AMD) : construit pour ROCm 7.2 ; il faut ROCm 7 installé (`sudo apt install migraphx` depuis le dépôt ROCm 7 d'AMD). Avec ROCm 6 seulement, l'écran le dit.
- **OpenVINO** (Intel) : les GPU Arc et Xe, intégrés ou non. Il faut le pilote compute d'Intel (`intel-opencl-icd libze-intel-gpu1`).

### Modèles inclus
- L'archive contient les modèles SPAN AnimeJaNai HD V3.1 (Sharp1 et normal, Performance et Balanced), RIFE 4.25 et 4.22 lite, et DistilDRBA v2 Lite : la lecture avec agrandissement et interpolation marche dès l'extraction.

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
