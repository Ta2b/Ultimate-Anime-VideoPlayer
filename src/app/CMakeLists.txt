# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse.
ajp_add_module(app
  DEPS ajp::core ajp::media ajp::gpu ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::present ajp::export)
