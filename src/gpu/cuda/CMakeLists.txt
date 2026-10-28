# Backends NVIDIA, seulement avec AJP_WITH_CUDA : le backend TensorRT chargé à l'exécution
# (backend/). Le socle de noyaux CUDA (compute/) et l'enveloppe TensorRT (trt/) s'y ajoutent.
add_subdirectory(backend)
