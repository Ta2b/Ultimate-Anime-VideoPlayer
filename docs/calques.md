# Interpolation par calques et plaque de décor

Phase 4 du plan du flux NVIDIA (« Usages du flux optique matériel NVIDIA »). Aucun modèle nouveau : RIFE et
SPAN tels quels, plus un warp par le mouvement de caméra que donne la carte de la scène (`kernels/spec/scene_flow.h`,
`PairFacts::scene`). Les deux fonctions sont éteintes par défaut, sans option de ligne de commande : elles
s'allument dans `play.toml`, et ne tournent qu'avec la carte, donc avec l'option d'essai `--scene-map` (sans
elle, une ligne au journal le dit). En lecture seulement : l'export rejoue la chaîne au bit près, et la carte
arrive quand elle peut (`layer_setup`, `chain_layers.cpp`).

## Ce que ça fait

- **Calques** (`[interpolation] layers = true`). RIFE tourne sur toute l'image, comme aujourd'hui.
  Là où la carte voit du décor qui suit la caméra, l'image intermédiaire reprend ce décor de a et b déplacées par
  le seul mouvement de caméra à l'instant t, au lieu du warp de RIFE : plus de traits qui ondulent ni d'aplats
  qui bavent sur un panoramique, un zoom ou un travelling. Les dessins, les effets et tout ce dont la carte doute
  gardent RIFE. Avec « Garder les poses », le décor suit la caméra pendant que les dessins gardent leur rythme
  (le rythme par calque).
- **Plaque de décor** (`[interpolation] background_plate = true`). Une image clé par plan
  passe par un second SPAN, le modèle tel que livré en FP16 ; chaque source suivante du plan reprend son décor
  de cette clé, recadrée au mouvement de caméra cumulé. Le décor d'une lecture en SPAN FP8 ou INT8 prend la
  précision du FP16, et il ne scintille plus d'une image à l'autre (mêmes pixels, déplacés). Une nouvelle clé
  quand la plaque couvre moins des trois quarts de l'image, que le zoom a bougé de plus de 3 % ou après
  240 paires ; plus de plaque dès qu'une paire perd la caméra, jusqu'à la paire sûre suivante.

Sans carte (AMD, Intel, Turing, NVOF refusé, pas encore relue), ou dès qu'elle doute de la paire (coupe, fondu,
caméra non ajustée ou non affinée par Lucas-Kanade, résidu au-delà de 0,5 pixel de travail, moins de 5 % de
décor), l'image est exactement celle
d'aujourd'hui. Un refus du backend (Vulkan n'a pas le noyau) coupe les deux pour la lecture, une ligne au
journal.

## Comment

- La caméra : celle que Lucas-Kanade affine sur les lumas de la paire (`SceneMotion::camera_refined`,
  `kernels/spec/scene_camera.h`, porte 2.1 : 97 % des paires sous 0,5 pixel). L'affine de NVOF seule (affinage
  fini trop tard, pas lancé) invente un cisaillement d'environ 2 pixels sur les panoramiques lents : la paire
  n'a alors ni calques ni plaque (« caméra non affinée » au journal). La carte compte en coordonnées continues
  (centre du pixel i en i + ½), les calques en indices : `index_camera`.
- Décision par paire, sur l'hôte (`src/app/play/layer_plan.h`) : tuiles décor sûres (Background à confiance
  0,5 ou plus, ou Uncertain qui ne change pas), puis tout le chemin d'un dessin, d'un effet ou d'une tuile
  incertaine qui change (de sa tuile dans a à sa tuile dans b, élargi d'une tuile) est retiré.
- Composition, sur le GPU (`kernels/spec/layers.h`, `ref::compose_layers`, `gpu/cuda/backend/layer_kernels.cu`,
  `KernelOps::compose_layers`, ABI 51) : un bit par tuile passé par valeur, masque bilinéaire `2 g − 1` qui reste
  dans le décor, déplacements affines exacts (la caméra portée à la taille de sortie, `M_t = I + t (M − I)`
  inversé), lecture bilinéaire ou Catmull-Rom (celle du warp de RIFE), et une garde par pixel : là où a et b
  déplacées diffèrent de plus de 4 à 12 codes (plaque : 6 à 18 contre SPAN de b), l'image de RIFE (ou SPAN) reste.
- Dans l'étage (`layer_stage.h`) : la paire arrive (`outputs`), les images intermédiaires sont composées après le
  warp (`blend`, `blend_poses`), la plaque après SPAN de b et avant le grade (`span_current`). Avec les calques,
  les images intermédiaires passent par un tenseur au lieu d'être écrites directement dans le slot
  (`layers_ring_direct`) : une copie de plus par image.
- Hors du champ : DistilDRBA, GMFSS, RIFE image et le Smooth (les instants n'y sont plus ceux de la caméra)
  gardent leur chemin ; seuls les warps de RIFE mouvement sur la paire de la carte composent (`blend`,
  `layered`). Une clé dont le SPAN de la plaque n'a pas tourné (SPAN de b repris d'un dessin tenu, ou fait en
  avance par le Smooth) est refaite à la paire suivante ; les tuiles qui bougent sont coupées avec la plaque. RIFE tourne toujours sur toute l'image : le gain est la qualité, pas le temps.

Journal de fin : `calques : N paires avec calques (X % de décor en moyenne), M images composées ; sans : … ;
plaque : K clés, P sources reprises`. Profil GPU : étapes « calques » et « plaque de décor ».

## Rejeu sur processeur

Les décisions se rejouent sans le PC, sur les flux NVOF exportés une fois (lumas, vecteurs et coûts à la taille
du flux du lecteur, le format de `carte_rejeu`) : `carte_rejeu --calques` rejoue la carte puis `plan_layers` et
la plaque (compilés tels quels) et mesure ce que la garde refuserait sur les lumas ;
`tools/bench/qualite/calques_rejeu.py` en tire, par extrait, la part des paires avec calques et les raisons des
autres, la garde des calques et de la plaque, et des vues (décor repris en bleu, rendu à RIFE en rouge). La garde
y est approchée : le lecteur compare des images agrandies, en RVB, et sa plaque passe par SPAN. Le PC ne garde que
l'essai ci-dessous : images, temps et retards.

## Essai (avec le reste, à la fin)

Pas de banc à part : un panoramique de 20 s, calques et plaque éteints, puis calques, puis calques et plaque,
deux tours (ABAB), sur le binaire `linux-release` d'un nouveau paquet Linux (`src/gpu`, `src/kernels` changent
l'empreinte TensorRT). L'export n'a jamais de calques : l'essai se fait en lecture, sur une RTX 30 ou après.

Tests d'abord : `kirari_backend_tensorrt_tests "[layers]"` (le noyau contre la référence),
`kirari_kernels_tests "[layers]"`, `kirari_app_tests "[layers]"`.

```sh
# Trois dossiers de réglages, copies des vrais (--data-dir : moteurs et modèles restent partagés).
D=~/.local/share/kirari; E=/tmp/calques; PANO=panoramique.mkv; S=0  # l'extrait et sa seconde de début
for v in eteint calques plaque; do
  mkdir -p $E/$v
  for f in settings.toml user.toml play.toml; do [ -f "$D/$f" ] && cp "$D/$f" $E/$v/; done
  sed -i '/^layers = /d;/^background_plate = /d' $E/$v/play.toml
done
sed -i '/^\[interpolation\]/a layers = true' $E/calques/play.toml
sed -i '/^\[interpolation\]/a layers = true\nbackground_plate = true' $E/plaque/play.toml
# RIFE mouvement ×2 (les calques ne composent que ses warps), 24 présentations de suite à 10 s.
for t in 1 2; do for v in eteint calques plaque; do
  kirari --debug --scene-map --motion=rife --rife=2 --mute --no-resume --fullscreen --start=$S --seconds=20 \
    --data-dir=$E/$v --stats=$E/$v/t$t.stats --snapshot=$E/$v/t$t.ppm --snapshot-at=10 --snapshot-frames=24 $PANO
  cp $E/$v/kirari.log $E/$v/t$t.log
done; done
grep -h "calques\|plaque\|carte de la scène :\|profil GPU · .*SPAN" $E/*/t*.log
```

À relever :

- `calques : N paires avec calques (X % de décor en moyenne), M images composées ; sans : …` : sur un
  panoramique, au moins la moitié des paires avec calques ; les raisons de « sans » (« caméra non affinée »,
  « décor hors du modèle de caméra » : le résidu, « trop peu de décor »). Avec la plaque : `K clés, P sources
  reprises` (0 clé : le warp n'est pas après SPAN, la plaque ne tourne pas ; le dire).
- `carte de la scène : … affinée sur …, (… affinages trop tard …)` : les paires où Lucas-Kanade arrive trop
  tard n'ont pas de calques.
- Profil GPU : les étapes « calques » et « plaque de décor » sous 0,3 ms par appel en 4K ; le SPAN de la
  plaque, une fois par clé.
- `--stats` : pas plus d'images en retard que la variante éteinte.
- Les 24 images de chaque variante côte à côte : les traits du décor n'ondulent plus et les aplats ne bavent
  plus au bord des personnages (calques) ; le décor ne scintille plus d'une image à l'autre (plaque) ; aucun
  personnage effacé ni doublé, aucune couture de tuile.

Si le décor ondule encore ou qu'un personnage s'efface : les seuils de `layer_plan.h` et les gardes de
`layers.h` se règlent ; si le décor sort du modèle de caméra (parallaxe), les calques restent éteints.
