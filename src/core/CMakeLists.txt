# core_types : vocabulaire partagé (types.h, inbox.h), sans dépendance.
ajp_add_module(core_types)

# core : profil TOML v5, resolve(), Plan, choix des moteurs.
# Seul propriétaire de la politique ; les autres modules exécutent le Plan.
ajp_add_module(core
  SOURCES cadence.cpp drawings.cpp engines.cpp pacing.cpp plan.cpp profile.cpp resolve.cpp
  DEPS ajp::core_types
  PRIVATE_DEPS ajp::dep::toml)
