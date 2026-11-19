# Carte de la scène (#939) : mesures sur le PC

> Depuis bbdc188c (préparation de la première release) : la carte est allumée par défaut (`--scene-map` reste
> accepté, `--no-scene-map` l'éteint) et les éclats sur les effets de la carte sont éteints par défaut. Pour
> les captures du point 2, ajouter `--effect-highlights` aux lignes `fx-` et `vue-`. Les lueurs vues dans
> l'image (#974) restent allumées ; `--no-effect-highlights` coupe tout.

Branche `claude/project-thread-3e8oz4`, commit **9baf5c9c** (main efb30a22 + #939 + l'affinage à l'heure).
Le module TensorRT change (src/kernels, src/gpu) : une seule construction couvre les trois mesures.
La CI tourne en parallèle sur ce commit ; la mesure ne l'attend pas.

```sh
cd ~/kirari-player
git fetch origin claude/project-thread-3e8oz4 && git checkout --detach 9baf5c9c
cmake --preset linux-prefix-cuda-release && cmake --build --preset linux-prefix-cuda-release
K=$PWD/build/linux-prefix-cuda-release/src/app/kirari
E=<chemin de rezero-s4e18.mkv>          # l'épisode de référence (celui des captures 16:20 / 16:43)
O=$HOME/kirari-seance/carte3 && mkdir -p $O
```

## 1. Coût de la carte et affinage à l'heure (PC seul)

```sh
flock ~/kirari-reports/gpu-bench.lock tools/bench/banc.sh --kirari=$K --a="--debug" --b="--debug --scene-map" \
  --extraits=extraits.tsv --sortie=$O/banc --tours=3 --secondes=60
grep -h "carte de la scène :" $O/banc/extrait*-B-*.log
```

À relever, par extrait : « caméra ajustée sur N, affinée sur M », « K affinages trop tard, dont R relus ensuite,
P pas lancés ; L lancés derrière le flux depuis la caméra du plan, X écartés », et l'écart A/B du rapport (i/s,
images perdues). Attendu d'après le rejeu processeur : environ 91 % des paires affinées à l'heure (1 346 sur
1 479), et les affinages perdus passent de 46 à 3 sur l'ensemble. « relus ensuite » doit être proche de « trop
tard » : un affinage en retard n'est plus perdu, la caméra des calques le reprend.

## 2. Éclats HDR sur les effets (16:20 et 16:43)

Peut partager le PC avec une autre mesure de qualité (pas une mesure de temps).

```sh
cd $O
for s in 970 993; do
  $K --debug --scene-map --mute --no-resume --start=$s --snapshot=fx-$s.pq.ppm --snapshot-at=10 --seconds=11 "$E" > fx-$s.log 2>&1
  $K --debug --scene-map --no-effect-highlights --mute --no-resume --start=$s --snapshot=nofx-$s.pq.ppm --snapshot-at=10 --seconds=11 "$E" > nofx-$s.log 2>&1
  $K --debug --scene-map --effect-mask-view --mute --no-resume --start=$s --snapshot=vue-$s.pq.ppm --snapshot-at=10 --seconds=11 "$E" > vue-$s.log 2>&1
done
grep -h "instantané\|effet sûr" fx-*.log nofx-*.log
```

À relever : la ligne « instantané … : masque des effets » (part d'effet sûr, marge retenue), la part de pixels
qui gagnent plus de 10 % (2,07 % avant) et le pic (632 nits avant). Avant #939, ces deux moments (une lueur,
une aura sur Ram) ne classaient aucune tuile en effet : la part d'effet sûr doit maintenant être au-dessus de
zéro, et `vue-*.pq.ppm` doit montrer du vert sur la lueur et l'aura (rouge : lumières, bleu : marge retenue).
Joindre les six images.

## 3. Carte sur les extraits du banc (porte 2.3)

Peut partager le PC avec une autre mesure de qualité.

```sh
cmake -S tools/bench/qualite/carte_rejeu -B build/carte_rejeu -DCMAKE_BUILD_TYPE=Release && cmake --build build/carte_rejeu
python3 -I tools/bench/qualite/carte_lecteur.py --outil build/nvof_dump/nvof_dump --rejeu build/carte_rejeu/carte_rejeu \
  <les 8 --extrait TYPE=CHEMIN@DÉBUT de la carte précédente, ~/kirari-seance/carte2> \
  --extrait lueur="$E"@16:17 --extrait aura="$E"@16:40 --sortie $O/carte
```

(`build/nvof_dump` est celui de la carte précédente ; le reconstruire seulement s'il manque.)

À relever dans `$O/carte/carte.md` : la part d'effets sur lueur et aura (zéro avant), la section « Caméra
affinée » et la ligne « tot » (affinages lancés derrière le flux, gardés et écartés), et aucune régression sur
les 8 extraits par rapport à `~/kirari-seance/carte2/carte/carte.md`.

Renvoyer : la ligne du journal de chaque extrait B, les lignes « instantané », les six captures et `carte.md`.
