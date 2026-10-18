// Bibliothèque partagée factice des tests de vk_compute : elle s'ouvre comme un chargeur
// Vulkan, mais n'exporte pas vkGetInstanceProcAddr. Le contexte doit la refuser
// (LibraryUnavailable) au lieu d'appeler un symbole absent.
extern "C" __attribute__((visibility("default"))) int ajp_fake_vulkan_loader() {
  return 0;
}
