# Séance PC : RIFE natif après la rc.1, chronométrage seulement

La séance du 19 novembre (e05a3a5) est dépouillée dans [`rife-bilan.md`](rife-bilan.md), partie 0 : le natif y a
tenu, il est le défaut depuis #961. Celle-ci vient après l'étiquette de la rc.1, sur la branche de #987 (la garde
des morceaux de la porte à 48 dB), sans attendre ci.yml (seule la fusion l'attend). Le bilan (partie 4) fait foi
s'ils divergent.

Avec le binaire de lecture CUDA, pas `linux-prefix` en Debug. Rien d'autre d'ouvert sur le GPU. Extrait :
Re:Zero S4E18, `--start=960`. Le verdict gardé se refait seul à la première ouverture (l'empreinte des sources
du RIFE natif a changé) : c'est elle qu'on lit à l'étape 0.

```sh
git fetch origin claude/project-thread-i22zy2 && git checkout --detach FETCH_HEAD
cmake --preset linux-prefix-cuda-release && cmake --build --preset linux-prefix-cuda-release
L="--rife-model=rife_v4.22_lite.onnx"

# 0. Verdict de l'essai au chargement, 4.25 puis lite.
for m in "" "$L"; do
  kirari --debug play $m --rife=2 --no-upscale --start=960 extrait.mkv 2>&1 | grep -i "rife natif\|mélange"
done

# 1. Où passe le temps des petits noyaux (blocs grossiers), 4.25, natif puis TensorRT. --delay saute le
#    chargement : la capture ne contient que la lecture.
for v in "" "--no-rife-native"; do
  o=rife${v:+-trt}
  nsys profile -o $o --delay=15 --duration=20 --cuda-graph-trace=node \
    kirari --debug play $v --rife=2 --no-upscale --start=960 extrait.mkv
  nsys export --type sqlite -o $o.sqlite $o.nsys-rep
  python3 tools/rife/nsys_passe.py $o.sqlite
done
```

À relever :

- **Étape 0**, pour chaque modèle : la ligne « Référence du mélange » (native attendue sur les deux modèles
  maintenant, avec la barre retenue), le mélange et le temps d'une passe contre TensorRT. Attendu sur le lite :
  l'encodeur natif entre, vers 2,65 ms contre 2,95.
- **Étape 1** : la sortie de `nsys_passe.py` des deux captures (passe, temps dans les noyaux, petites grilles et
  l'écart juste avant elles, les quinze noyaux les plus lourds), et l'occupation de `nsys stats --report
  cuda_gpu_kern_sum` si on la veut.

Décision ensuite :

- #987 se fusionne si le verdict tient la porte et que la passe ne ralentit pas.
- Les blocs grossiers : si l'écart avant les petits noyaux pèse au moins le tiers de leur temps, un lancement
  par bloc grossier (noyau persistant, barrière de grille entre les couches, les sommes partielles finies dans
  la même grille) ; sinon le temps est dans les noyaux, et c'est leur forme qu'il faut reprendre (partie 4, c).
