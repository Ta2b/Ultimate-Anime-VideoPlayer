# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse.
#
# La bibliothèque porte tout ce qui se teste (ligne de commande, détection du
# système, téléchargements, écran « Moteurs ») ; l'exécutable ajp n'ajoute que main().
ajp_add_module(app
  SOURCES
    cli.cpp cli_help.cpp commands.cpp commands_engines.cpp data_paths.cpp download.cpp
    engine_chooser.cpp engine_text.cpp install.cpp library_loader.cpp runtime_probe.cpp
    system_report.cpp vulkan_probe.cpp
  DEPS ajp::core ajp::media ajp::gpu ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::present ajp::export ajp::vendor_hdr
  PRIVATE_DEPS ajp::dep::curl ajp::dep::sdl3 ajp::dep::vulkan ${CMAKE_DL_LIBS})

add_executable(ajp main.cpp)
target_link_libraries(ajp PRIVATE ajp::app)
ajp_set_warnings(ajp)
target_compile_definitions(ajp_app PRIVATE AJP_VERSION="${PROJECT_VERSION}")
