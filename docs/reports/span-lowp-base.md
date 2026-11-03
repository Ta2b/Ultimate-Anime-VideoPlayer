# SPAN basse précision : banc de départ (étape 3.0)

À remplir lors de la passe de mesures groupée sur la RTX 4070 Ti. Rien n'a été mesuré dans le cloud (pas de GPU).

## Protocole
ABAB × 3 sous le verrou GPU, médianes à ±2 % entre tours :

```
ajp_span_fp8_run b8f64.span8 1080 1920 in.f32 out.f32 --bench 300
ajp_span_fp8_run b8f64.span8 720 1280 in.f32 out.f32 --bench 300
trtexec --loadEngine=<moteur du cache> --useCudaGraph --noDataTransfers --warmUp=500 --iterations=300
```

`trtexec` pour le b8f64 et le b5f48, en 720p et 1080p ; trace `nsys` d'une lecture au cran Performance
(`nsys stats -r cuda_api_sum` : 1 `cudaGraphLaunch` et au plus 2 `cudaLaunchKernel` par SPAN).

## Résultats

| Mesure | Tour 1 | Tour 2 | Tour 3 |
|---|---|---|---|
| b8f64 FP8, 1080p | | | |
| b8f64 FP8, 720p | | | |
| b8f64 TensorRT FP16, 1080p | | | |
| b8f64 TensorRT FP16, 720p | | | |
| b5f48 FP8, 1080p | | | |
| b5f48 FP8, 720p | | | |
| b5f48 TensorRT FP16, 1080p | | | |
| b5f48 TensorRT FP16, 720p | | | |
