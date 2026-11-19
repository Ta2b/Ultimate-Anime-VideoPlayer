# Interpolation par calques et plaque de décor

Phase 4 du plan du flux NVIDIA (« Usages du flux optique matériel NVIDIA »). Aucun modèle nouveau : RIFE et
SPAN tels quels, plus un warp par le mouvement de caméra que donne la carte de la scène (`kernels/spec/scene_flow.h`,
`PairFacts::scene`). Sans option de ligne de commande, les deux se règlent dans `play.toml` : les calques sont
allumés par défaut depuis le second essai du 19 novembre (un `play.toml` qui les a éteints les garde éteints),
la plaque reste éteinte. Les deux ne tournent qu'avec la carte, donc avec l'option d'essai `--scene-map` (sans
elle, une ligne au journal le dit). En lecture seulement : l'export rejoue la chaîne au bit près, et la carte
arrive quand elle peut (`layer_setup`, `chain_layers.cpp`).

## Ce que ça fait

- **Calques** (`[interpolation] layers = true`). RIFE tourne sur toute l'image, comme aujourd'hui.
  Là où la carte voit du décor qui suit la caméra, l'image intermédiaire reprend ce décor de a et b déplacées par
  le seul mouvement de caméra à l'instant t, au lieu du warp de RIFE : plus de traits qui ondulent ni d'aplats
  qui bavent sur un panoramique, un zoom ou un travelling. Les dessins, les effets et tout ce dont la carte doute
  gardent RIFE. Avec « Garder les poses », le décor suit la caméra pendant que les dessins gardent leur rythme
  (le rythme par calque).
- **Plaque de décor** (`[interpolation] background_plate = true`). Une image clé par plan ; chaque source
  suivante du plan reprend son décor de cette clé, recadrée au mouvement de caméra cumulé : il ne scintille plus
  d'une image à l'autre (mêmes pixels, déplacés). La clé est la sortie de SPAN de l'image, quand SPAN tourne en
  FP16 ou mieux ; en FP8 ou en INT8, elle passe par un second SPAN, le modèle tel que livré en FP16, et le décor
  prend la précision du FP16 (une passe de SPAN de plus par clé). Une nouvelle clé
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
  fini trop tard, pas lancé) invente un cisaillement d'environ 2 pixels sur les panoramiques lents ; elle ne sert
  jamais. Une paire dont l'affinage est en retard reprend la caméra lk d'une des deux paires d'avant du plan,
  si les deux s'accordent à un demi-pixel de travail près aux coins et au centre (`with_neighbour`) ; sinon,
  ni calques ni plaque (« caméra non affinée » au journal). Au rejeu des dix extraits de flux-carte, la caméra lk
  de la paire d'avant est à 0,03 à 0,16 pixel médian de la vraie sur sept extraits, celle de NVOF à 0,2 à 1,8 ;
  la reprise garde 53 % des paires en retard, dont 1 % à plus d'un demi-pixel. La carte compte en coordonnées
  continues (centre du pixel i en i + ½), les calques en indices : `index_camera`.
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

Journal de fin : `calques : N paires avec calques (dont V avec la caméra lk d'une voisine, X % de décor en
moyenne), M images composées ; sans : … ; plaque : K clés (reprises de SPAN | SPAN FP16 de plus), P sources
reprises`. Profil GPU : étapes « calques » et « plaque de décor ».

## Rejeu sur processeur

Les décisions se rejouent sans le PC, sur les flux NVOF exportés une fois (lumas, vecteurs et coûts à la taille
du flux du lecteur, le format de `carte_rejeu`) : `carte_rejeu --calques` rejoue la carte puis `plan_layers` et
la plaque (compilés tels quels) et mesure ce que la garde refuserait sur les lumas ;
`tools/bench/qualite/calques_rejeu.py` en tire, par extrait, la part des paires avec calques et les raisons des
autres, la garde des calques et de la plaque, et des vues (décor repris en bleu, rendu à RIFE en rouge). La garde
y est approchée : le lecteur compare des images agrandies, en RVB, et sa plaque passe par SPAN. Le PC ne garde que
l'essai ci-dessous : images, temps et retards. `--sans-affinage` rejoue le pire du lecteur, l'affinage toujours
en retard : la carte part avec la caméra de NVOF et seule la caméra lk d'une voisine sert.

Rejeu du 19 novembre, dix extraits de 160 images (huit de carte_lecteur, lueur, aura) : avec l'affinage à
l'heure, 82 à 100 % des paires ont des calques ; affinage toujours en retard, 47 à 97 % sur neuf extraits (17 %
sur le panoramique rapide de Re:Zero 12:10 : 33 pixels par paire avec un balancement vertical, puis des secousses
de 20 à 28 pixels). Deux essais écartés :

- garder la clé de la plaque à travers une paire sans calques à caméra sûre : 12 % de clés en moins, mais une
  plaque qui dérive (garde de la plaque au 90e centile de 13 à 44 % sur lueur) ;
- extrapoler la caméra voisine (vitesse et accélération, 2 c₋₁ − c₋₂) : sur ce panoramique, elle reste à 0,7
  pixel médian de la caméra lk (la voisine seule aussi), et NVOF, à 2,3 pixels, ne peut pas la valider ; prise
  quand elle est la plus proche de NVOF, elle fait passer les dix extraits de 65 à 70 % des paires en retard
  reprises, mais 0,6 % au lieu de 0,4 % à plus d'un demi-pixel. Seul l'affinage à l'heure rattrape ce plan.

## Essai du 19 novembre (#946, RTX 4070 Ti)

Rien de cassé (aucun personnage effacé ni doublé, aucune couture), mais pas la porte : 34 % des paires avec
calques (177 sans caméra lk : affinage trop tard sur 45 % des paires), calques 0,82 ms par appel en 4K, plaque
0,30 ms mais un retard p95 de 4,7 à 5 ms. Le retard venait du second SPAN de la plaque sur chaque clé (37) : quand
SPAN tourne déjà en FP16 ou mieux, la plaque reprend désormais sa sortie (`plate_copy`). Les calques n'ajoutent
pas de retard (p95 0 ms, comme éteints). L'affinage à l'heure revient à la carte (`scene_map.cpp`).

## Second essai du 19 novembre (tête de #939 : #957 et l'affinage parti derrière le flux)

Porte des calques passée : 74 % des paires avec calques sur le panoramique (359 sur 485, « caméra non affinée »
16 au lieu de 177) et 83 % sur Re:Zero 60 s (1 194 sur 1 444). Retard p95 de 2,8 ms contre 3,0 éteint sur le
panoramique, 2,0 contre 1,8 sur Re:Zero, et 36 images perdues contre 54. Calques 0,83 à 0,89 ms par appel. La
carte affine 424 paires sur 424, dont 22 trop tard mais toutes relues. Les calques s'allument donc par défaut,
avec la carte.

La plaque ne passe pas. En FP8, son second SPAN FP16 porte le p95 à 6,8 ms contre 3,0 (panoramique) et à 6,4
contre 1,8 (Re:Zero). En Équilibre, la copie de la sortie de SPAN donne 3,2 contre 1,5 ms, avec moins
d'images manquées (26 contre 102, un seul tour). Elle reste éteinte.

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
sed -i '/^\[interpolation\]/a layers = false' $E/eteint/play.toml  # allumés par défaut
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

- `calques : N paires avec calques (dont V avec la caméra lk d'une voisine, …), M images composées ; sans : …` : sur un
  panoramique, au moins la moitié des paires avec calques ; les raisons de « sans » (« caméra non affinée »,
  « décor hors du modèle de caméra » : le résidu, « trop peu de décor »). Avec la plaque : `K clés, P sources
  reprises` (0 clé : le warp n'est pas après SPAN, la plaque ne tourne pas ; le dire).
- `carte de la scène : … affinée sur …, (… affinages trop tard …)` : les paires où Lucas-Kanade arrive trop
  tard n'ont pas de calques.
- Profil GPU : « calques » vers 0,85 ms par appel en 4K (deux lectures Catmull-Rom du décor, aux essais du
  19 novembre ; les 0,3 ms visés d'abord n'étaient qu'une estimation), « plaque de décor » vers 0,3 ms ; le
  SPAN de la plaque, une fois par clé, seulement quand SPAN tourne sous le FP16. Ce qui tranche : la ligne
  suivante.
- `--stats` : pas plus d'images en retard que la variante éteinte.
- Les 24 images de chaque variante côte à côte : les traits du décor n'ondulent plus et les aplats ne bavent
  plus au bord des personnages (calques) ; le décor ne scintille plus d'une image à l'autre (plaque) ; aucun
  personnage effacé ni doublé, aucune couture de tuile.

Si le décor ondule encore ou qu'un personnage s'efface : les seuils de `layer_plan.h` et les gardes de
`layers.h` se règlent ; si le décor sort du modèle de caméra (parallaxe), les calques restent éteints.
