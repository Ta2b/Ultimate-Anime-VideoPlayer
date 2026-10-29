# core_types : vocabulaire partagé et contrats entre modules (types.h, inbox.h, frame.h,
# thread.h, gpu_device.h, ring.h…), sans dépendance hors bibliothèque standard.
ajp_add_module(core_types)

# core_log : le canal de journal de chaque module et l'installation des puits par app.
# Seule dépendance : spdlog. Les modules la lient en privé.
ajp_add_module(core_log
  SOURCES log.cpp
  DEPS ajp::dep::spdlog)

# core : profil TOML v5, resolve(), Plan, choix des moteurs.
# Seul propriétaire de la politique ; les autres modules exécutent le Plan.
ajp_add_module(core
  SOURCES cadence.cpp capabilities.cpp debug_flags.cpp drawings.cpp engines.cpp ini_import.cpp
          ini_import_steps.cpp keys.cpp model_index.cpp pacing.cpp plan.cpp profile.cpp resolve.cpp
          auto_rate.cpp free_rate.cpp free_grid.cpp auto_memory.cpp file_sync.cpp inflate.cpp settings.cpp update_manifest.cpp user_overlay.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::toml)
