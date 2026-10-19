ajp_add_unit_tests(app
  SOURCES
    cli_test.cpp
    commands_test.cpp
    data_paths_test.cpp
    download_test.cpp
    engine_chooser_test.cpp
    http_server.cpp
    install_test.cpp
    probe_test.cpp
    support.cpp
    system_probe_test.cpp
  DEPS ajp::dep::vulkan)

if(WIN32)
  target_link_libraries(ajp_app_tests PRIVATE ws2_32)
endif()
