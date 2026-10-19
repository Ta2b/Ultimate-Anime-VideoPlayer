# present : ordonnanceur, horloge de l'écran, détection de l'écran, sans écran ni GPU.
ajp_add_unit_tests(present
  SOURCES
    display_info_test.cpp scheduler_events_test.cpp scheduler_sim.cpp scheduler_test.cpp
    vsync_estimator_test.cpp)

# La fenêtre réelle à part : SDL3, libplacebo et Vulkan liés directement, et un délai borné
# (une swapchain FIFO dont la fenêtre serait cachée pourrait attendre sans fin).
add_executable(ajp_present_window_tests window_test.cpp)
target_link_libraries(ajp_present_window_tests PRIVATE
  ajp::present ajp::dep::sdl3 ajp::dep::placebo ajp::dep::vulkan Catch2::Catch2WithMain)
ajp_set_warnings(ajp_present_window_tests)
catch_discover_tests(ajp_present_window_tests TEST_PREFIX "present: " DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT 60)
