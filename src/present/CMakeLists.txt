# present : fenêtre, swapchain, cadence, détection de l'écran. Thread main.
# Public : core (DisplayInfo, CadenceChoice, Ratio) et ses contrats. SDL3, libplacebo, Vulkan
# et spdlog restent privés : les en-têtes publics ne font que déclarer leurs types.
set(present_sources
  display_clock.cpp display_info.cpp display_sdl.cpp frame_vsyncs.cpp output.cpp refresh.cpp scheduler.cpp
  scheduler_queue.cpp surface.cpp swapchain.cpp vsync_estimator.cpp window.cpp)
if(WIN32)
  list(APPEND present_sources display_win32.cpp)
endif()

ajp_add_module(present
  SOURCES ${present_sources}
  DEPS ajp::core
  PRIVATE_DEPS ajp::dep::sdl3 ajp::dep::placebo ajp::dep::vulkan ajp::core_log)

if(WIN32)
  target_link_libraries(ajp_present PRIVATE dxgi)
endif()
