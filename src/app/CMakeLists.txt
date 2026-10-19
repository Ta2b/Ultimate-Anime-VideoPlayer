# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse. core_log : app seul installe
# les puits du journal (install_log_sinks), depuis main.
ajp_add_module(app
  DEPS ajp::core ajp::core_log ajp::media ajp::gpu ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::present ajp::export)
