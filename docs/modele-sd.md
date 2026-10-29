# SPAN pour les vieux animes en SD : AniSD AC

Demande de Tatoub : ajouter aux modèles SPAN « le AnimeJaNai SD, celui des vieux animes ».

## Ce qui existe, et ce qui ne convient pas

- **AnimeJaNai** (the-database, mpv-upscale-2x_animejanai) ne publie en SD que des **Compact** (V1/V2 Standard et
  Strong, « 2x_AnimeJaNai_SD_V1beta34_Compact »). Le lecteur ne lance que des SPAN : un Compact (ESRGAN léger,
  `SRVGGNetCompact`) n'est pas pris. Il n'existe pas d'« AnimeJaNai SD SPAN ». Les AnimeJaNai HD V3.1 du catalogue
  sont déjà des SPAN, mais pour du HD.
- **AniSD** (Sirosky, dépôt Upscale-Hub, release « AniSD », 20 avril) : une suite pour restaurer les DVD et WEB
  d'anime en SD, avec des variantes SPAN (notre architecture), Compact, SwinIR, RealPLKSR, DAT2, CRAFT.
  - AniSD : sources SD récentes (WEB), peu compressées.
  - **AniSD AC** (« Augmented Clean ») : compression moyenne à forte, dotcrawl, bruit, halos, chroma : les vieux DVD.
  - AniSD DC (« Deep Clean ») : le plus agressif, pour les sources très abîmées (années 90), plus lent, moins fidèle.

## Choix : `2x_AniSD_AC_G6i2b_SPAN_190K`

C'est le modèle que l'auteur recommande en premier pour la plupart des DVD (« Quickstart » de la release et du
wiki), et il est en SPAN, que l'auteur donne pour mieux généraliser que Compact. Le DC SPAN (92 500 itérations) nettoie
plus mais lisse davantage et n'est pas le défaut de l'auteur ; le baseline (G6i1, 215 000) vise des sources propres. Un
seul modèle est donc ajouté. Le DC reste utilisable par `--model` après la même conversion.

| | |
|---|---|
| Source | https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD/2x_AniSD_AC_G6i2b_SPAN_190K_fp32.onnx |
| Licence | **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0), `LICENSE.md` du dépôt Upscale-Hub (le dépôt n'affiche pas de licence à GitHub ; le texte est CC BY-NC 4.0). Pas de partage à l'identique, contrairement à AnimeJaNai (BY-NC-SA). |
| Taille source | 1 656 315 octets |
| SHA-256 source (fp32, notre téléchargement ; Sirosky ne publie pas d'empreinte) | `c0d52ae736a41803af7389911333f6a70f4b8f4bee795c01713a3b8bd81b284d` |
| Fichier du lecteur | `2x_AniSD_AC_G6i2b_SPAN_190K_fp16.onnx`, 834 868 octets |
| SHA-256 du fichier du lecteur | `6cdca47792cf40516443e40dfddfea26c5141461861587fba5373a3a3fd82bcb` |

Autres SPAN de la même release (non retenus) : `2x_AniSD_G6i1_SPAN_215K_fp32.onnx` `ec446acf…8fedf`,
`2x_AniSD_DC_SPAN_92500_fp32.onnx` `2a6aafad…933c3`.

## Compatibilité avec le lecteur

Le graphe est un SPAN à **48 canaux, 6 blocs**, entrée 3 canaux (sans « unshuffle » : pas de `PixelUnshuffle` en
tête), attention sans paramètre (Sigmoid − 0,5), `Concat` de 4 tenseurs, `Conv` 48→12, `DepthToSpace` ×2. Entrée
`1×3×H×W`, sortie `1×3×2H×2W` : les mêmes que le modèle HD. Le fichier publié est en **FP32** à axes `width`/`height`
intervertis ; TensorRT 11, fortement typé, suivrait le FP32 : `tools/span/convert_anisd_fp16.py` passe poids, constantes,
entrée et sortie en FP16 (aucun PyTorch : `onnx` et `numpy`), écart maximal 1e-3 sur une entrée aléatoire (ONNX Runtime
CPU).

- **TensorRT FP16** (Équilibre) : le chemin normal du lecteur.
- **Cran Performance (FP8 CUTLASS)** : **non proposé**. Les noyaux ne prennent que le SPAN b8f64 d'AnimeJaNai (64
  canaux, 8 blocs, `conv_1.1.eval_conv`, attention 1×1 appris) ; `package_from_onnx` refuse ce graphe, et l'interface
  grise le bouton « Préparer le mode Performance » avec une note (`span_fp8_supported`). Le cran n'existe donc pas pour
  lui, sans mention d'un FP8 dans le menu au-delà du grisé habituel.
- **Cran Qualité (FP32)** : aucune variante installée, grisé comme pour les modèles HD.

## Intégration

- Catalogue (`model_choice.h`) : `kSpanSdModel`, affiché « AniSD AC · SD, vieux animes ».
- Config automatique (`auto_config.cpp`) : source de **576 lignes ou moins** (hauteur connue) → ce modèle ; sinon le
  modèle du lecteur (HD). Absent du dossier, `choose_model` retombe sur le modèle HD et le signale au journal.
  « Personnalisé » n'est pas touché.
- `tools/release/fetch-models.sh` : télécharge le fichier depuis la release `models-span-sd-1` d'ajp-releases avec son
  `SHA256SUMS` (facultatif tant qu'elle n'existe pas).
- Publication : `tools/release/publier-modele-sd.sh DOSSIER [--publier]` (essai à blanc par défaut), à lancer par
  Tatoub.
