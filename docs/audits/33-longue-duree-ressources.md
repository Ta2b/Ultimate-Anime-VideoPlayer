# Audit 33. Longue durée, mémoire et ressources

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Ce qui ne casse qu'après des heures : fuites de mémoire, de mémoire vidéo, de handles ou de threads, journaux et caches qui grossissent.

## Méthode

1. Suivre dans le code chaque ressource allouée à l'ouverture d'un fichier ou d'une chaîne, et sa libération.
2. Décrire un essai de 8 heures, un essai de 100 réouvertures de chaîne et un essai de 50 fichiers enchaînés, avec ce qu'il faut relever.
3. Relever les caches et fichiers qui grossissent sans borne.

## À couvrir

- Pool CUDA, moteurs, anneau, poignées d'import partagées.
- Threads créés par fichier ou par chaîne.
- Journal : rotation seulement au lancement.
- Caches des plans, des pipelines et des moteurs : purge.
- Dossiers temporaires.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Ressources à suivre

| Point | Où | Zone |
|---|---|---|
| création et destruction des voies | `src/gpu/cuda/backend/tensorrt_lanes.cpp:55` | [01](01-tensorrt-execution.md) |
| réserve des envois | `src/gpu/cuda/backend/host_upload.cpp:53` | [01](01-tensorrt-execution.md) |
| anneau partagé | `src/app/play/placebo_screen.cpp:68` | [10](10-presentation-libplacebo.md) |
| journal | `src/core/log.cpp:64` | [20](20-socle-core.md) |
| cache des plans | `src/media/shot_cache.cpp:81` | [42](42-plans-coupes.md) |
| cache des moteurs | `src/prepare/cache.cpp:61` | [05](05-construction-moteurs.md) |

### Allocations GPU et partagées à suivre

Pour chaque ligne : où la ressource est rendue, et si elle l'est aussi quand une étape échoue. (45 lignes, zones de la cible seulement.)

| Où | Zone | Code |
|---|---|---|
| `src/gpu/cuda/backend/host_upload.cpp:77` | [01](01-tensorrt-execution.md) | `if (auto ok = cuda_check(cudaMalloc(&slot.device, bytes), "cudaMalloc"); !ok) return ok;` |
| `src/gpu/cuda/backend/ring_slot.cpp:66` | [01](01-tensorrt-execution.md) | `if (const cudaError_t e = cudaImportExternalMemory(&imported->memory_, &memory); e != cuda` |
| `src/gpu/cuda/backend/ring_slot.cpp:76` | [01](01-tensorrt-execution.md) | `if (const cudaError_t e = cudaImportExternalMemory(&imported->memory_, &memory); e != cuda` |
| `src/gpu/cuda/backend/ring_slot.cpp:126` | [01](01-tensorrt-execution.md) | `if (const cudaError_t error = cudaImportExternalSemaphore(&imported->semaphore_, &semaphor` |
| `src/gpu/cuda/backend/ring_slot.cpp:135` | [01](01-tensorrt-execution.md) | `if (const cudaError_t error = cudaImportExternalSemaphore(&imported->semaphore_, &semaphor` |
| `src/gpu/cuda/backend/tensorrt_backend.cpp:366` | [01](01-tensorrt-execution.md) | `if (auto ok = cuda_check(cudaMallocAsync(&memory, bytes, stream_), "cudaMallocAsync"); !ok` |
| `src/gpu/cuda/backend/tensorrt_backend.cpp:402` | [01](01-tensorrt-execution.md) | `if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep) != cudaSuccess \|` |
| `src/gpu/cuda/backend/tensorrt_backend.cpp:403` | [01](01-tensorrt-execution.md) | `cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies, &internal) != cud` |
| `src/gpu/cuda/backend/tensorrt_export.cpp:58` | [01](01-tensorrt-execution.md) | `cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&start), starts, stream), "cudaMallocA` |
| `src/gpu/cuda/backend/tensorrt_export.cpp:62` | [01](01-tensorrt-execution.md) | `cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&weights), taps, stream), "cudaMallocA` |
| `src/gpu/cuda/backend/tensorrt_export.cpp:113` | [01](01-tensorrt-execution.md) | `cuda_check(cudaMallocAsync(&yuv_.rows, yuv_rows_bytes(width, height), stream_), "cudaMallo` |
| `src/gpu/cuda/backend/tensorrt_kernels.cpp:163` | [01](01-tensorrt-execution.md) | `if (auto ok = cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&ingest_rows_), rows, st` |
| `src/gpu/cuda/backend/tensorrt_kernels.cpp:224` | [01](01-tensorrt-execution.md) | `if (auto ok = cuda_check(cudaMallocAsync(&deband_out_, bytes, stream_), "cudaMallocAsync")` |
| `src/gpu/cuda/backend/tensorrt_lanes.cpp:83` | [01](01-tensorrt-execution.md) | `if (cudaMallocAsync(&memory, bytes, stream_) != cudaSuccess \|\|` |
| `src/gpu/cuda/backend/tensorrt_stats.cpp:33` | [01](01-tensorrt-execution.md) | `ready = cuda_check(cudaMalloc(&slot.device, kCountsBytes), "cudaMalloc");` |
| `src/gpu/cuda/backend/tensorrt_stats.cpp:34` | [01](01-tensorrt-execution.md) | `if (ready) ready = cuda_check(cudaMallocHost(&slot.host, kCountsBytes), "cudaMallocHost");` |
| `src/gpu/cuda/backend/trt_engine.cpp:71` | [01](01-tensorrt-execution.md) | `if (cudaMalloc(&memory, bytes) != cudaSuccess) {` |
| `src/gpu/cuda/backend/trt_engine.cpp:329` | [01](01-tensorrt-execution.md) | `if (error == cudaSuccess && queued && captured != nullptr) error = cudaGraphInstantiate(&e` |
| `src/gpu/cuda/backend/span_fp8_engine.cpp:43` | [02](02-span-fp8.md) | `if (cudaError_t e = cudaMalloc(&scratch.input, in_bytes); e != cudaSuccess) return failed(` |
| `src/gpu/cuda/backend/span_fp8_engine.cpp:44` | [02](02-span-fp8.md) | `if (cudaError_t e = cudaMalloc(&scratch.output, out_bytes); e != cudaSuccess)` |
| `src/gpu/cuda/backend/span_fp8_engine.cpp:242` | [02](02-span-fp8.md) | `error = cudaGraphInstantiate(&exec, captured, 0);` |
| `src/gpu/cuda/backend/span_fp8_layout.cuh:35` | [02](02-span-fp8.md) | `const cudaError_t error = cudaMalloc(&ptr_, bytes);` |
| `src/gpu/cuda/backend/span_fp8_run.cpp:126` | [02](02-span-fp8.md) | `CU(cudaMalloc(&workspace, net ? static_cast<std::size_t>(net->workspace_bytes()) : 1U));` |
| `src/gpu/cuda/backend/span_fp8_run.cpp:127` | [02](02-span-fp8.md) | `CU(cudaMalloc(&input, pixels * 2));` |
| `src/gpu/cuda/backend/span_fp8_run.cpp:128` | [02](02-span-fp8.md) | `CU(cudaMalloc(&output, pixels * 8));` |
| `src/gpu/cuda/backend/span_fp8_run.cpp:159` | [02](02-span-fp8.md) | `CU(cudaGraphInstantiate(&exec, graph, 0));` |
| `src/gpu/cuda/backend/tensorrt_span_derive.cpp:138` | [02](02-span-fp8.md) | `if (auto e = cudaMalloc(&scratch.input, in_bytes); e != cudaSuccess) return cuda_refusal(e` |
| `src/gpu/cuda/backend/tensorrt_span_derive.cpp:139` | [02](02-span-fp8.md) | `if (auto e = cudaMalloc(&scratch.output_ref, in_bytes * 4); e != cudaSuccess)` |
| `src/gpu/cuda/backend/tensorrt_span_derive.cpp:141` | [02](02-span-fp8.md) | `if (auto e = cudaMalloc(&scratch.output_fp8, in_bytes * 4); e != cudaSuccess)` |
| `src/gpu/cuda/backend/tensorrt_span_derive.cpp:203` | [02](02-span-fp8.md) | `if (auto e = cudaMalloc(&scratch.workspace, static_cast<std::size_t>(net->workspace_bytes(` |
| `src/gpu/cuda/backend/drba_fused_engine.cpp:161` | [03](03-interpolation-tensorrt.md) | `if (cudaError_t e = cudaMalloc(buffer, in_bytes); e != cudaSuccess) return failed(e, "cuda` |
| `src/gpu/cuda/backend/drba_fused_engine.cpp:164` | [03](03-interpolation-tensorrt.md) | `if (cudaError_t e = cudaMalloc(&scratch.output, out_bytes); e != cudaSuccess)` |
| `src/gpu/cuda/backend/drba_fused_engine.cpp:385` | [03](03-interpolation-tensorrt.md) | `error = cudaGraphInstantiate(&graph.exec, captured, 0);` |
| `src/gpu/cuda/backend/rife_fused_engine.cpp:168` | [03](03-interpolation-tensorrt.md) | `if (cudaError_t e = cudaMalloc(&scratch.input, in_bytes); e != cudaSuccess) return failed(` |
| `src/gpu/cuda/backend/rife_fused_engine.cpp:169` | [03](03-interpolation-tensorrt.md) | `if (cudaError_t e = cudaMalloc(&scratch.output, out_bytes); e != cudaSuccess)` |
| `src/gpu/cuda/backend/rife_fused_engine.cpp:361` | [03](03-interpolation-tensorrt.md) | `error = cudaGraphInstantiate(&graph.exec, captured, 0);` |
| `src/gpu/cuda/backend/tensorrt_rife.cpp:173` | [03](03-interpolation-tensorrt.md) | `ready = cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device), sizeof(PairSums)), "` |
| `src/gpu/cuda/backend/tensorrt_rife.cpp:175` | [03](03-interpolation-tensorrt.md) | `ready = cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host), sizeof(PairSums)),` |
| `src/gpu/cuda/backend/tensorrt_warp.cpp:159` | [03](03-interpolation-tensorrt.md) | `ready = cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device), kHistogramBytes), "c` |
| `src/gpu/cuda/backend/tensorrt_warp.cpp:162` | [03](03-interpolation-tensorrt.md) | `cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host), kHistogramBytes), "cudaMal` |
| `src/app/play/placebo_ring.cpp:147` | [10](10-presentation-libplacebo.md) | `slot.tex = pl_tex_create(gpu, &params);` |
| `src/app/play/placebo_snapshot.cpp:35` | [10](10-presentation-libplacebo.md) | `pl_tex tex = pl_tex_create(gpu, &params);` |
| `src/app/play/placebo_imgui.cpp:100` | [24](24-interface-rendu.md) | `tex = pl_tex_create(gpu_, &params);` |
| `src/app/play/placebo_grade.cpp:305` | [41](41-hdr-grade.md) | `pl_tex in = pl_tex_create(gpu, &tex);` |
| `src/app/play/placebo_grade.cpp:311` | [41](41-hdr-grade.md) | `pl_tex out = pl_tex_create(gpu, &tex);` |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| TRT-06 | Mineur | Le pool CUDA par défaut ne rend jamais la mémoire vidéo. Sur une carte de 4 à 6 Go, Vulkan et libplacebo peuvent en manquer. |
| TRT-13 | Mineur | Si `create` échoue à mi-chemin, files et événements CUDA fuient. |
| TRT-18 | Mineur | Dossiers temporaires à nom fixe, partagés entre utilisateurs puis effacés par `remove_all`. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md)
- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md)
- [20. Socle commun (core)](20-socle-core.md)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md)
- [41. HDR, grade et looks](41-hdr-grade.md)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md)
