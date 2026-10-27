# AMD, Windows et RX 9070 XT

Suite de `docs/nvidia-linux.md`, d'après l'audit AMD/Windows (constats AW-01 à AW-16). Verdict de
l'audit : pour la RX 9070 XT (Navi 48, RDNA4), le chemin spécialisé est un **profil RDNA4 dans le moteur
Vulkan**, pas une voie séparée. MIGraphX (ROCm ou Windows ML) ne reçoit pas de travail de plus avant
mesure : le code est soigné, mais rien ne montre qu'il bat Vulkan sur cette carte (AW-07, AW-08).

## Ce qui est fait

| Constat | Changement |
|---|---|
| AW-11 | Sous Windows, D3D11VA est essayé après Vulkan Video et avant le décodage logiciel (copie en mémoire hôte). HEVC et AV1 10 bits ne passent plus par le CPU quand Vulkan Video ne les prend pas. |
| AW-12 | L'offre DirectML est retirée de l'écran d'accueil : aucun moteur DirectML n'existe. Les anciens AMD restent sur Vulkan. |
| AW-14 | Le manifeste Windows active les chemins longs (noms de fichiers d'anime à l'export). |
| AW-03 | Une table d'autotune peut être réservée à un pilote (`PresetTable::driver`). Nom de fichier : `AmdRdna4-radv-rx9070xt-1.tsv` (`radv`, `amd` pour le pilote Adrenalin, `nvidia`). Le journal de construction du moteur donne la famille et la version du pilote. |
| AW-05 | Krackan (0x1114) et Strix Halo (0x1586) sont reconnus comme RDNA3 par identifiant PCI, sans dépendre du nom RADV. |

## Ce qui reste, et pourquoi

- **AW-01 tables RDNA4** : il faut trois lectures sur la 9070 XT, puis
  `cp autotune.tsv tools/vk/presets/AmdRdna4-radv-rx9070xt-1.tsv` (ou `-amd-` sous Adrenalin) et
  `python3 tools/vk/presets_from_cache.py`. Sans mesure, aucune table n'est inventée.
- **AW-02 INT8/FP8 matriciel** : demande un modèle quantifié qui n'existe pas encore ; vérifier d'abord
  `vulkaninfo` (propriétés de `VK_KHR_cooperative_matrix`, présence de float8).
- **AW-04** : l'écran d'accueil détecte par nom de produit, le moteur par identifiant PCI. Les deux
  concordent pour la 9070 XT ; l'unification passe par un `GpuFacts::family`, non fait ici.
- **AW-06, AW-09** : priorité de file et Resizable BAR (Smart Access Memory). L'état du BAR s'affiche
  déjà sur la ligne GPU de l'écran d'accueil et au journal d'ouverture (comme pour NVIDIA).
- **AW-13** Windows ML (tenseurs par l'hôte, pas de LUID), **AW-15** HDR Windows (niveau de blanc SDR
  du système non lu), **AW-16** minuteur Windows (risque faible) : à traiter après mesure.
- **AW-14** : l'entrée `tensorrt` du workflow Windows reste une case à cocher à la main ; rien de
  NVIDIA n'entre dans une release publique sans elle.

## Mesures à faire sur la 9800X3D + 9070 XT

M1 Vulkan contre MIGraphX (Re:Zero ép. 18, 16:00 à 18:00, `--stats`) · M2 RADV contre Adrenalin ·
M3 trois lectures par profil utile pour les tables · M4 `vulkaninfo` · M5 BAR (`lspci -vv`) ·
M6 décodage Windows : HEVC 10 bits, AV1, Hi10P, le journal dit-il `d3d11va` ou `logiciel` ? ·
M7 HDR10 sur le QD-OLED.

Point de départ conseillé : profil 1 (1440p, RIFE ×3), qui est celui que le lecteur propose déjà à
une RDNA3/RDNA4 de 12 Go et plus.
