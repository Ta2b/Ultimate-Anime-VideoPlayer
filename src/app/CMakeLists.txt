# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse. core_log : app seul installe
# les puits du journal (install_log_sinks), depuis main.
#
# play/ : la tranche verticale « ajp play » (session du thread main, câblage des
# threads source, gpu, audio et subs, horloge audio maîtresse).
ajp_add_module(app
  SOURCES
    play/bridges.cpp play/frame_queue.cpp play/frame_stats.cpp play/options.cpp play/pipeline.cpp
    play/ring_stage.cpp play/session.cpp play/upscale_stage.cpp play/wiring.cpp
  DEPS ajp::core ajp::core_log ajp::media ajp::gpu ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::present ajp::export
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::dep::sdl3)
