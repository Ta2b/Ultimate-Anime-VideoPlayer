# Audit 46. Outils de recherche, modèles et données livrées

**Audit par zone. Priorité : réduit : données livrées.** 63 fichiers, 6 731 lignes ; tests rattachés : 0 fichiers, 0 lignes.

## Mission

Les scripts hors produit (export de modèles, FP8, bancs, recherche FlowJaNai) et les données livrées avec le lecteur (calibration, LUT).

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- **Périmètre réduit.** N'auditer que ce qui est livré avec le lecteur (`data/`, modèles téléchargés et leur index). Les scripts de recherche de `tools/` sont hors cible.
- Scripts qui supposent des chemins personnels, seulement s'ils servent à produire ce qui est livré.
- Modèles livrés : empreintes vérifiées, version du modèle connue du lecteur.
- Ce qui peut sortir du dépôt (recherche terminée, données inutilisées).

## Frontières

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| TRT-07 | Majeur | Le menu propose ArtCNN en FP8, toujours refusé. Le fichier « fp16 » calcule en FP32, sous TensorRT comme sous Vulkan (rapport G). |

## Fichiers

| Fichier | Lignes |
|---|---|
| `data/calibration/calibration.mkv` | 0 |
| `data/luts/HDR grading base ver.9.0 33 point.cube` | 0 |
| `data/luts/HDR-contrast-Enhancement-L.cube` | 0 |
| `data/luts/Highlight-Enhancement-L.cube` | 0 |
| `data/luts/skin fix.cube` | 0 |
| `tools/bench/banc.sh` | 66 |
| `tools/bench/banc_resume.py` | 140 |
| `tools/bench/endurance.py` | 174 |
| `tools/bench/extraits.exemple.tsv` | 0 |
| `tools/bench/fp8/drba_fusion/abab.sh` | 19 |
| `tools/bench/fp8/drba_fusion/commun.py` | 84 |
| `tools/bench/fp8/drba_fusion/construire.sh` | 5 |
| `tools/bench/fp8/drba_fusion/decoupe.py` | 85 |
| `tools/bench/fp8/drba_fusion/drba_fusion.cuh` | 341 |
| `tools/bench/fp8/drba_fusion/drba_fusion_bench.cu` | 496 |
| `tools/bench/fp8/drba_fusion/essai.sh` | 14 |
| `tools/bench/fp8/drba_fusion/mkin.py` | 14 |
| `tools/bench/fp8/drba_fusion/moteurs.sh` | 22 |
| `tools/bench/fp8/drba_fusion/plie.py` | 46 |
| `tools/bench/fp8/drba_fusion/qualite.py` | 32 |
| `tools/bench/fp8/drba_fusion/ref_bornes.py` | 48 |
| `tools/bench/fp8/drba_fusion/resultats.txt` | 48 |
| `tools/bench/parite.py` | 101 |
| `tools/bench/rife_triplets.py` | 144 |
| `tools/bench/span_flicker.py` | 95 |
| `tools/drba/export_distil_drba.py` | 206 |
| `tools/fp8/abab_pieces.sh` | 13 |
| `tools/fp8/abab_trt.sh` | 36 |
| `tools/fp8/build_interp_fp8.sh` | 52 |
| `tools/fp8/conv_microbench.py` | 76 |
| `tools/fp8/fp8_probe.py` | 136 |
| `tools/fp8/interp_fp8.py` | 409 |
| `tools/grade/check_hdr.py` | 269 |
| `tools/grade/compare_shots.sh` | 79 |
| `tools/grade/de_itp.py` | 127 |
| `tools/grade/emitters/dataset.py` | 104 |
| `tools/grade/emitters/labels.py` | 129 |
| `tools/grade/emitters/train.py` | 138 |
| `tools/grade/episodes.exemple.tsv` | 0 |
| `tools/grade/make_stills.sh` | 23 |
| `tools/grade/patterns.py` | 194 |
| `tools/grade/shots30.tsv` | 0 |
| `tools/grade/sim/curve.py` | 213 |
| `tools/grade/sim/simulate.py` | 127 |
| `tools/look/bake_stack.py` | 169 |
| `tools/nvof_interp/dataset.py` | 86 |
| `tools/nvof_interp/export_onnx.py` | 96 |
| `tools/nvof_interp/model.py` | 122 |
| `tools/nvof_interp/nvof_dump/CMakeLists.txt` | 14 |
| `tools/nvof_interp/nvof_dump/nvof_dump.cpp` | 183 |
| `tools/nvof_interp/precompute_nvof.py` | 102 |
| `tools/nvof_interp/train.py` | 137 |
| `tools/rife/convert_rife_fp16.py` | 226 |
| `tools/ring/banding.py` | 95 |
| `tools/span/derive_span_fp8.py` | 333 |
| `tools/span/quantize_span.py` | 417 |
| `tools/trt/layer_report.py` | 161 |
| `tools/upscale/models/2x_ArtCNN_C4F32_rgb_fp16.onnx` | 0 |
| `tools/upscale/models/ArtCNN-LICENSE.txt` | 21 |
| `tools/upscale/wrap_luma_rgb.py` | 84 |
| `tools/vk/presets/NvidiaAda-rtx4070ti-1.tsv` | 0 |
| `tools/vk/presets/NvidiaAda-rtx4070ti-2.tsv` | 0 |
| `tools/vk/presets_from_cache.py` | 180 |

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
