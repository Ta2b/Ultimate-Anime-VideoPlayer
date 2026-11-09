# Audit 45. Moteur Vulkan : calcul, noyaux et shaders

**Audit par zone. Priorité : réduit : repli sur une RTX.** 217 fichiers, 26 562 lignes ; tests rattachés : 57 fichiers, 11 874 lignes.

## Mission

Le calcul sous Vulkan : appareil et files, abaissement des réseaux en dispatchs, noyaux coopmat, FP16 et FP8, autotune, colle de RIFE fusionné, noyaux du socle (ingestion, sortie YUV, Spline36) et leurs références.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- **Périmètre réduit.** N'auditer que ce qui sert sur une RTX : le repli quand TensorRT manque, pendant une construction de moteurs et pour « Annuler et regarder », et les noyaux du socle (ingestion, sortie YUV, Spline36) s'ils servent aussi sous TensorRT. Ne pas auditer la performance ni ce qui ne sert qu'à AMD et Intel.
- Le repli démarre-t-il bien sur une RTX sous Windows et sous CachyOS, et rend-il une image juste ?
- Passage de Vulkan à TensorRT quand la construction finit : pas de fuite, pas d'image figée.
- Erreurs Vulkan : `DEVICE_LOST`, `OUT_OF_DEVICE_MEMORY`, timelines.
- Autotune sur une RTX : concordance, durée du premier lancement, cache.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (42 inclusions)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) (39 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (16 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (4 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (2 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (2 inclusions)

Utilisé par :

- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (16 inclusions)
- [48. Moteur Core ML (macOS)](48-coreml-macos.md) (12 inclusions)
- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) (7 inclusions)
- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (7 inclusions)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) (5 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (4 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (3 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (2 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (1 inclusion)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) (1 inclusion)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| MED-02 | Majeur | La conversion P0.11 ne traite que les images hôte. Une source 4:4:4 ou 4:2:2 décodée par NVDEC ou Vulkan Video s'arrête dès la première image. |
| QUA-05 | Majeur | Six écritures restent faites à la main, sans fsync, contre P5.11. Une coupure peut laisser un moteur Vulkan ou un cache vide. |
| GPU-17 | Mineur | Un GPU bloqué sans être perdu n'est pas détecté. La fermeture peut rester bloquée dans `pl_gpu_finish`. |
| MED-04 | Mineur | Position de chroma fixée en dur et profondeur incohérente selon le 4:4:4. Gris, RGB planaire, YCgCo et ICtCp restent refusés, et seulement en cours de lecture. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/gpu/vulkan/CMakeLists.txt` | 6 |
| `src/gpu/vulkan/backend/CMakeLists.txt` | 119 |
| `src/gpu/vulkan/backend/deferred_steps.cpp` | 66 |
| `src/gpu/vulkan/backend/deferred_steps.h` | 72 |
| `src/gpu/vulkan/backend/device_profile.cpp` | 126 |
| `src/gpu/vulkan/backend/device_profile.h` | 34 |
| `src/gpu/vulkan/backend/drba_fused_engine.cpp` | 359 |
| `src/gpu/vulkan/backend/drba_fused_engine.h` | 102 |
| `src/gpu/vulkan/backend/entry.cpp` | 13 |
| `src/gpu/vulkan/backend/entry_points.h` | 68 |
| `src/gpu/vulkan/backend/family.cpp` | 184 |
| `src/gpu/vulkan/backend/family.h` | 61 |
| `src/gpu/vulkan/backend/host_tensor.cpp` | 211 |
| `src/gpu/vulkan/backend/kernel_frames.cpp` | 466 |
| `src/gpu/vulkan/backend/kernel_image.cpp` | 181 |
| `src/gpu/vulkan/backend/kernel_ops.cpp` | 241 |
| `src/gpu/vulkan/backend/kernel_ops.h` | 366 |
| `src/gpu/vulkan/backend/kernel_ops_detail.h` | 185 |
| `src/gpu/vulkan/backend/kernel_pair.cpp` | 200 |
| `src/gpu/vulkan/backend/kernel_ring.cpp` | 119 |
| `src/gpu/vulkan/backend/kernel_slice.cpp` | 270 |
| `src/gpu/vulkan/backend/kernel_stats.cpp` | 62 |
| `src/gpu/vulkan/backend/net_alias.cpp` | 310 |
| `src/gpu/vulkan/backend/net_alias.h` | 41 |
| `src/gpu/vulkan/backend/net_artifact.cpp` | 391 |
| `src/gpu/vulkan/backend/net_artifact.h` | 55 |
| `src/gpu/vulkan/backend/net_autotune.cpp` | 649 |
| `src/gpu/vulkan/backend/net_autotune.h` | 159 |
| `src/gpu/vulkan/backend/net_build.cpp` | 232 |
| `src/gpu/vulkan/backend/net_build.h` | 65 |
| `src/gpu/vulkan/backend/net_engine.cpp` | 350 |
| `src/gpu/vulkan/backend/net_engine.h` | 154 |
| `src/gpu/vulkan/backend/net_gate.cpp` | 171 |
| `src/gpu/vulkan/backend/net_gate.h` | 51 |
| `src/gpu/vulkan/backend/net_hoist.cpp` | 338 |
| `src/gpu/vulkan/backend/net_hoist.h` | 23 |
| `src/gpu/vulkan/backend/net_lower.cpp` | 452 |
| `src/gpu/vulkan/backend/net_lower.h` | 104 |
| `src/gpu/vulkan/backend/net_lower_conv.cpp` | 750 |
| `src/gpu/vulkan/backend/net_lower_copy.cpp` | 189 |
| `src/gpu/vulkan/backend/net_lower_detail.h` | 53 |
| `src/gpu/vulkan/backend/net_measure.cpp` | 100 |
| `src/gpu/vulkan/backend/net_measure.h` | 58 |
| `src/gpu/vulkan/backend/net_params.cpp` | 329 |
| `src/gpu/vulkan/backend/net_params.h` | 76 |
| `src/gpu/vulkan/backend/net_presets.cpp` | 40 |
| `src/gpu/vulkan/backend/net_profile.cpp` | 91 |
| `src/gpu/vulkan/backend/net_profile.h` | 62 |
| `src/gpu/vulkan/backend/net_program.h` | 89 |
| `src/gpu/vulkan/backend/net_query.cpp` | 57 |
| `src/gpu/vulkan/backend/net_query.h` | 63 |
| `src/gpu/vulkan/backend/net_runtime.cpp` | 189 |
| `src/gpu/vulkan/backend/net_runtime.h` | 234 |
| `src/gpu/vulkan/backend/net_weights.cpp` | 188 |
| `src/gpu/vulkan/backend/net_weights.h` | 51 |
| `src/gpu/vulkan/backend/queue.cpp` | 187 |
| `src/gpu/vulkan/backend/queue.h` | 148 |
| `src/gpu/vulkan/backend/reduce_weights.cpp` | 99 |
| `src/gpu/vulkan/backend/reduce_weights.h` | 21 |
| `src/gpu/vulkan/backend/rife_fused_engine.cpp` | 365 |
| `src/gpu/vulkan/backend/rife_fused_engine.h` | 130 |
| `src/gpu/vulkan/backend/shared_tensor.cpp` | 372 |
| `src/gpu/vulkan/backend/shared_tensor.h` | 62 |
| `src/gpu/vulkan/backend/vulkan_backend.cpp` | 493 |
| `src/gpu/vulkan/backend/vulkan_backend.h` | 180 |
| `src/gpu/vulkan/backend/vulkan_backend_detail.h` | 41 |
| `src/gpu/vulkan/backend/vulkan_backend_runtime.cpp` | 408 |
| `src/gpu/vulkan/backend/vulkan_drba_fused.cpp` | 82 |
| `src/gpu/vulkan/backend/vulkan_rife_fused.cpp` | 86 |
| `src/gpu/vulkan/compute/CMakeLists.txt` | 78 |
| `src/gpu/vulkan/compute/buffer.cpp` | 144 |
| `src/gpu/vulkan/compute/buffer.h` | 71 |
| `src/gpu/vulkan/compute/caps.cpp` | 242 |
| `src/gpu/vulkan/compute/caps.h` | 127 |
| `src/gpu/vulkan/compute/context.cpp` | 156 |
| `src/gpu/vulkan/compute/context.h` | 287 |
| `src/gpu/vulkan/compute/context_create.cpp` | 351 |
| `src/gpu/vulkan/compute/descriptors.cpp` | 157 |
| `src/gpu/vulkan/compute/descriptors.h` | 70 |
| `src/gpu/vulkan/compute/destroyed_log.h` | 62 |
| `src/gpu/vulkan/compute/functions.cpp` | 55 |
| `src/gpu/vulkan/compute/functions.h` | 116 |
| `src/gpu/vulkan/compute/image.cpp` | 216 |
| `src/gpu/vulkan/compute/image.h` | 58 |
| `src/gpu/vulkan/compute/kernel_dispatch.cpp` | 88 |
| `src/gpu/vulkan/compute/kernel_dispatch.h` | 202 |
| `src/gpu/vulkan/compute/kernel_runner.cpp` | 168 |
| `src/gpu/vulkan/compute/kernel_runner.h` | 125 |
| `src/gpu/vulkan/compute/kernel_types.cpp` | 159 |
| `src/gpu/vulkan/compute/kernel_types.h` | 144 |
| `src/gpu/vulkan/compute/kernels_color.cpp` | 199 |
| `src/gpu/vulkan/compute/kernels_planes.cpp` | 164 |
| `src/gpu/vulkan/compute/kernels_slice.cpp` | 133 |
| `src/gpu/vulkan/compute/kernels_window.cpp` | 197 |
| `src/gpu/vulkan/compute/library.cpp` | 101 |
| `src/gpu/vulkan/compute/library.h` | 41 |
| `src/gpu/vulkan/compute/log.cpp` | 11 |
| `src/gpu/vulkan/compute/log.h` | 14 |
| `src/gpu/vulkan/compute/memory.cpp` | 340 |
| `src/gpu/vulkan/compute/memory.h` | 149 |
| `src/gpu/vulkan/compute/pipeline.cpp` | 271 |
| `src/gpu/vulkan/compute/pipeline.h` | 126 |
| `src/gpu/vulkan/compute/queue.cpp` | 410 |
| `src/gpu/vulkan/compute/recording.cpp` | 292 |
| `src/gpu/vulkan/compute/recording.h` | 198 |
| `src/gpu/vulkan/compute/refuse.cpp` | 62 |
| `src/gpu/vulkan/compute/refuse.h` | 25 |
| `src/gpu/vulkan/compute/spirv_access.cpp` | 77 |
| `src/gpu/vulkan/compute/spirv_access.h` | 15 |
| `src/gpu/vulkan/compute/sync.cpp` | 78 |
| `src/gpu/vulkan/compute/sync.h` | 54 |
| `src/gpu/vulkan/compute/vulkan.h` | 26 |
| `src/gpu/vulkan/compute/weights.cpp` | 118 |
| `src/gpu/vulkan/compute/weights.h` | 45 |
| `src/gpu/vulkan/shaders/net_activation.glsl` | 51 |
| `src/gpu/vulkan/shaders/net_avg_pool.comp` | 44 |
| `src/gpu/vulkan/shaders/net_binary.comp` | 50 |
| `src/gpu/vulkan/shaders/net_conv.comp` | 4 |
| `src/gpu/vulkan/shaders/net_conv.glsl` | 198 |
| `src/gpu/vulkan/shaders/net_conv_consts.glsl` | 41 |
| `src/gpu/vulkan/shaders/net_conv_coopmat.glsl` | 329 |
| `src/gpu/vulkan/shaders/net_conv_coopmat_acc16.comp` | 6 |
| `src/gpu/vulkan/shaders/net_conv_coopmat_f16.comp` | 5 |
| `src/gpu/vulkan/shaders/net_conv_coopmat_fp8.glsl` | 226 |
| `src/gpu/vulkan/shaders/net_conv_coopmat_fp8_f16.comp` | 5 |
| `src/gpu/vulkan/shaders/net_conv_f16p.comp` | 6 |
| `src/gpu/vulkan/shaders/net_conv_reduce.comp` | 41 |
| `src/gpu/vulkan/shaders/net_conv_transpose.comp` | 94 |
| `src/gpu/vulkan/shaders/net_conv_wino.glsl` | 207 |
| `src/gpu/vulkan/shaders/net_conv_wino_f16.comp` | 5 |
| `src/gpu/vulkan/shaders/net_conv_wino_f16p.comp` | 6 |
| `src/gpu/vulkan/shaders/net_copy.comp` | 40 |
| `src/gpu/vulkan/shaders/net_depth_space.comp` | 39 |
| `src/gpu/vulkan/shaders/net_grid_sample.comp` | 67 |
| `src/gpu/vulkan/shaders/net_pad.comp` | 45 |
| `src/gpu/vulkan/shaders/net_reduce_mean.comp` | 40 |
| `src/gpu/vulkan/shaders/net_resize.comp` | 77 |
| `src/gpu/vulkan/shaders/net_tensor.glsl` | 87 |
| `src/gpu/vulkan/shaders/net_unary.comp` | 40 |
| `src/gpu/vulkan/shaders/ring_store.comp` | 36 |
| `src/kernels/CMakeLists.txt` | 43 |
| `src/kernels/error.h` | 56 |
| `src/kernels/glsl.cpp` | 336 |
| `src/kernels/glsl.h` | 64 |
| `src/kernels/glsl/block_diff.comp` | 75 |
| `src/kernels/glsl/block_diff_planes.comp` | 69 |
| `src/kernels/glsl/common.glsl` | 34 |
| `src/kernels/glsl/drba_fused.glsl` | 52 |
| `src/kernels/glsl/drba_fused_finish.comp` | 83 |
| `src/kernels/glsl/drba_fused_glue.comp` | 70 |
| `src/kernels/glsl/drba_fused_post.comp` | 47 |
| `src/kernels/glsl/drba_fused_pre.comp` | 78 |
| `src/kernels/glsl/flow_max.comp` | 46 |
| `src/kernels/glsl/frame_stats.comp` | 229 |
| `src/kernels/glsl/ingest_chroma_h.comp` | 55 |
| `src/kernels/glsl/ingest_chroma_h_planes.comp` | 46 |
| `src/kernels/glsl/ingest_combine.comp` | 56 |
| `src/kernels/glsl/ingest_combine_planes.comp` | 51 |
| `src/kernels/glsl/output_chroma_v.comp` | 70 |
| `src/kernels/glsl/output_matrix.comp` | 63 |
| `src/kernels/glsl/planes.glsl` | 33 |
| `src/kernels/glsl/resample_h.comp` | 41 |
| `src/kernels/glsl/resample_v.comp` | 40 |
| `src/kernels/glsl/rgb_pair_diff.comp` | 63 |
| `src/kernels/glsl/rgb_window.comp` | 37 |
| `src/kernels/glsl/rgba16_store.comp` | 30 |
| `src/kernels/glsl/rife_constants.comp` | 45 |
| `src/kernels/glsl/rife_frames.comp` | 38 |
| `src/kernels/glsl/rife_fused.glsl` | 45 |
| `src/kernels/glsl/rife_fused_entry.comp` | 41 |
| `src/kernels/glsl/rife_fused_finish.comp` | 72 |
| `src/kernels/glsl/rife_fused_glue.comp` | 150 |
| `src/kernels/glsl/scene_diff.comp` | 61 |
| `src/kernels/glsl/scene_diff_planes.comp` | 57 |
| `src/kernels/glsl/tensor.glsl` | 15 |
| `src/kernels/glsl/transfer.comp` | 150 |
| `src/kernels/glsl/warp_blend.comp` | 190 |
| `src/kernels/glsl/yuv.glsl` | 119 |
| `src/kernels/glsl/yuv_window.comp` | 75 |
| `src/kernels/glsl/yuv_window_planes.comp` | 74 |
| `src/kernels/glsl_main.cpp` | 38 |
| `src/kernels/image.h` | 87 |
| `src/kernels/ref/check.h` | 78 |
| `src/kernels/ref/color.cpp` | 147 |
| `src/kernels/ref/color.h` | 39 |
| `src/kernels/ref/detect.cpp` | 76 |
| `src/kernels/ref/detect.h` | 24 |
| `src/kernels/ref/frame_stats.cpp` | 194 |
| `src/kernels/ref/frame_stats.h` | 21 |
| `src/kernels/ref/half.cpp` | 70 |
| `src/kernels/ref/half.h` | 26 |
| `src/kernels/ref/resample.cpp` | 167 |
| `src/kernels/ref/resample.h` | 37 |
| `src/kernels/ref/rife.cpp` | 156 |
| `src/kernels/ref/rife.h` | 37 |
| `src/kernels/ref/rife_fused.cpp` | 174 |
| `src/kernels/ref/rife_fused.h` | 38 |
| `src/kernels/ref/transfer.cpp` | 100 |
| `src/kernels/ref/transfer.h` | 49 |
| `src/kernels/ref/warp.cpp` | 216 |
| `src/kernels/ref/warp.h` | 46 |
| `src/kernels/spec.h` | 24 |
| `src/kernels/spec/color.h` | 293 |
| `src/kernels/spec/detect.h` | 55 |
| `src/kernels/spec/distil_drba.h` | 42 |
| `src/kernels/spec/drba_fused.h` | 115 |
| `src/kernels/spec/fp_off_begin.h` | 15 |
| `src/kernels/spec/fp_off_end.h` | 10 |
| `src/kernels/spec/frame_stats.h` | 103 |
| `src/kernels/spec/poses.h` | 48 |
| `src/kernels/spec/resample.h` | 114 |
| `src/kernels/spec/rife.h` | 106 |
| `src/kernels/spec/rife_fused.h` | 121 |
| `src/kernels/spec/ring_pack.h` | 53 |
| `src/kernels/spec/tolerance.h` | 56 |
| `src/kernels/spec/transfer.h` | 139 |
| `src/kernels/spec/warp.h` | 89 |

## Tests rattachés

- `tests/unit/audio/sync_test.cpp`
- `tests/unit/backend_tensorrt/drba_fused_test.cpp`
- `tests/unit/backend_tensorrt/rife_fused_test.cpp`
- `tests/unit/gpu_ir/memory_test.cpp`
- `tests/unit/kernels/color_test.cpp`
- `tests/unit/kernels/detect_test.cpp`
- `tests/unit/kernels/distil_drba_test.cpp`
- `tests/unit/kernels/dither_test.cpp`
- `tests/unit/kernels/drba_fused_test.cpp`
- `tests/unit/kernels/engine_oracle.h`
- `tests/unit/kernels/glsl_test.cpp`
- `tests/unit/kernels/half_test.cpp`
- `tests/unit/kernels/parity_test.cpp`
- `tests/unit/kernels/resample_test.cpp`
- `tests/unit/kernels/rife_fused_test.cpp`
- `tests/unit/kernels/rife_test.cpp`
- `tests/unit/kernels/ring_pack_test.cpp`
- `tests/unit/kernels/tolerance_test.cpp`
- `tests/unit/kernels/transfer_test.cpp`
- `tests/unit/kernels/warp_test.cpp`
- `tests/unit/media/color_test.cpp`
- `tests/unit/vk_backend/deferred_steps_test.cpp`
- `tests/unit/vk_backend/drba_fused_chain_test.cpp`
- `tests/unit/vk_backend/fake_backend.cpp`
- `tests/unit/vk_backend/family_test.cpp`
- `tests/unit/vk_backend/host_import_test.cpp`
- `tests/unit/vk_backend/kernel_planes_test.cpp`
- `tests/unit/vk_backend/net_alias_test.cpp`
- `tests/unit/vk_backend/net_conv_test.cpp`
- `tests/unit/vk_backend/net_gate_test.cpp`
- `tests/unit/vk_backend/net_hoist_test.cpp`
- `tests/unit/vk_backend/net_measure_test.cpp`
- `tests/unit/vk_backend/net_ops_test.cpp`
- `tests/unit/vk_backend/net_profile_test.cpp`
- `tests/unit/vk_backend/net_rig.cpp`
- `tests/unit/vk_backend/net_rig.h`
- `tests/unit/vk_backend/net_runtime_test.cpp`
- `tests/unit/vk_backend/rife_fused_chain_test.cpp`
- `tests/unit/vk_backend/rife_fused_test.cpp`
- `tests/unit/vk_backend/runtime_queue_test.cpp`
- `tests/unit/vk_backend/shared_tensor_test.cpp`
- `tests/unit/vk_backend/slice_ops_test.cpp`
- `tests/unit/vk_backend/test_models.cpp`
- `tests/unit/vk_backend/test_models.h`
- `tests/unit/vk_backend/vendor_runtime_test.cpp`
- `tests/unit/vk_compute/color_test.cpp`
- `tests/unit/vk_compute/context_test.cpp`
- `tests/unit/vk_compute/detect_test.cpp`
- `tests/unit/vk_compute/execution_test.cpp`
- `tests/unit/vk_compute/fake_loader.cpp`
- `tests/unit/vk_compute/memory_test.cpp`
- `tests/unit/vk_compute/resample_test.cpp`
- `tests/unit/vk_compute/rife_test.cpp`
- `tests/unit/vk_compute/rig.cpp`
- `tests/unit/vk_compute/rig.h`
- `tests/unit/vk_compute/transfer_test.cpp`
- `tests/unit/vk_compute/warp_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
