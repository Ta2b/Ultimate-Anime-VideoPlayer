// Bibliothèque partagée factice des tests de vk_compute : elle s'ouvre comme un chargeur
// Vulkan, mais n'exporte pas vkGetInstanceProcAddr. Le contexte doit la refuser
// (LibraryUnavailable) au lieu d'appeler un symbole absent.
#if defined(_WIN32)
#define KIRARI_FAKE_EXPORT __declspec(dllexport)
#else
#define KIRARI_FAKE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" KIRARI_FAKE_EXPORT int kirari_fake_vulkan_loader() {
  return 0;
}
