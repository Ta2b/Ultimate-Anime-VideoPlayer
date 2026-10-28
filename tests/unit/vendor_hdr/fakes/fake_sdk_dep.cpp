// Bibliothèque factice qui tient lieu des dépendances d'un SDK (CUDA, NPP, TensorRT, NGX) :
// elle se charge, et rien d'autre.
extern "C" int fake_sdk_dep_marker() {
  return 1;
}
