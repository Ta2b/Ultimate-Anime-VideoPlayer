# Séance PC : RIFE natif (4.25 et 4.22 lite), chronométrage seulement

Sur la branche telle quelle, sans attendre ci.yml (seule la fusion l'attend) : commit e05a3a5 de
`claude/project-thread-i22zy2` (#961, entrée à la volée, plus la porte contre le calcul exact et la fin fondue,
main d4866bb fusionnée). Le bilan ([`rife-bilan.md`](rife-bilan.md), partie 4) fait foi s'ils divergent.

Avec le binaire de lecture CUDA, pas `linux-prefix` en Debug. Rien d'autre d'ouvert sur le GPU. Extrait :
Re:Zero S4E18, `--start=960`.

```sh
git fetch origin claude/project-thread-i22zy2 && git checkout --detach e05a3a5
cmake --preset linux-prefix-cuda-release && cmake --build --preset linux-prefix-cuda-release
L="--rife-model=rife_v4.22_lite.onnx"

# 0. Verdict de l'essai au chargement, 4.25 puis lite (la première ouverture construit les moteurs).
for m in "" "$L"; do
  kirari --debug play --rife-native $m --rife=2 --no-upscale --start=960 extrait.mkv 2>&1 | grep -i "rife natif"
done

# 1. ABAB dans le lecteur, 60 s, trois tours chacun (A natif, B TensorRT), 4.25 puis lite.
for m in "" "$L"; do
  kirari compare extrait.mkv --a="--rife-native $m --rife=2 --no-upscale --one-lane --start=960" \
    --b="$m --rife=2 --no-upscale --one-lane --start=960" --seconds=60 --report="rife${m:+-lite}-x2.txt"
done

# 2. Occupation du GPU (cible : moins de 23 %), Nsight Systems sur 30 s, natif puis TensorRT, 4.25.
nsys profile -o rife-natif --capture-range=none --duration=30 \
  kirari --debug play --rife-native --rife=2 --no-upscale --start=960 extrait.mkv
nsys profile -o rife-trt --capture-range=none --duration=30 \
  kirari --debug play --rife=2 --no-upscale --start=960 extrait.mkv
nsys stats --report cuda_gpu_kern_sum rife-natif.nsys-rep rife-trt.nsys-rep

# 3. Avec SPAN (sans --no-upscale) et ×3 : les étapes 0 et 1 seulement.
```

À relever :

- **Étape 0**, pour chaque modèle :
  - la ligne « référence du mélange » : native (la chaîne f32) ou TensorRT, et pourquoi ; pour le lite, native
    attendue ;
  - les lignes par morceau (temps et dB rôle par rôle de chaque réseau) ;
  - le verdict : retenu ou écarté, le mélange, ses dB contre la référence, contre TensorRT et contre le natif
    f32, le temps d'une passe contre TensorRT. Attendu : 4.25 vers 2,4 ms (3,39 pour TensorRT), lite vers
    2,6 à 2,7 ms (3,31) ;
  - l'entrée du dernier bloc à la volée gardée ou non, avec les deux temps.
- **Étape 1** : les images perdues, la médiane et le p95 du moteur RIFE fusionné, A contre B.
- **Étape 2** : l'occupation, les dix noyaux les plus lourds ; `finish_conv_kernel` doit remplacer la
  transposée du dernier bloc et `fused_finish`.

Décision ensuite : natif par défaut s'il tient la porte et passe sous 23 % d'occupation (PR courte avec
`--no-rife-native` et le cache du verdict) ; sinon le profil dit quel noyau perd.
