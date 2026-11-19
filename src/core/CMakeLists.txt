# core_types : vocabulaire partagé et contrats entre modules (types.h, inbox.h, frame.h,
# thread.h, gpu_device.h, ring.h…), sans dépendance hors bibliothèque standard.
kirari_add_module(core_types)

# core_log : le canal de journal de chaque module et l'installation des puits par app.
# Seule dépendance : spdlog. Les modules la lient en privé.
kirari_add_module(core_log
  SOURCES log.cpp log_file.cpp log_redact.cpp
  DEPS kirari::dep::spdlog)

# core_os : la place des threads face à la lecture (thread_role.h), MMCSS sous Windows ; bibliothèques
# ouvertes à l'exécution (dynamic_library.h).
kirari_add_module(core_os
  SOURCES dynamic_library.cpp thread_role.cpp
  DEPS kirari::core_types
  PRIVATE_DEPS $<$<PLATFORM_ID:Windows>:avrt> ${CMAKE_DL_LIBS})

# core : réglages, cadence, choix des moteurs, raccourcis, écritures atomiques.
kirari_add_module(core
  SOURCES cadence.cpp engines.cpp moving_area.cpp
          keys.cpp model_index.cpp sha256_blocks.cpp pacing.cpp
          auto_rate.cpp cost_model.cpp steady_rate.cpp steady_grid.cpp free_rate.cpp free_grid.cpp vsync_phase.cpp auto_memory.cpp vblank_clock.cpp present_ledger.cpp file_sync.cpp inflate.cpp settings.cpp update_manifest.cpp text_file.cpp
  DEPS kirari::core_types
  PRIVATE_DEPS kirari::dep::toml kirari::dep::zlib)
