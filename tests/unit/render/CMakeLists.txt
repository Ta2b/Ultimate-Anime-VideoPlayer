# Les tests [gpu] tournent sur le GPU de la machine (RTX sur le PC de référence, lavapipe en CI)
# et se sautent avec une raison sans appareil Vulkan ; [.bench] est caché.
ajp_add_unit_tests(render
  SOURCES
    bench_test.cpp compose_test.cpp device_test.cpp grade_params_test.cpp osd_test.cpp support.cpp
    vk_import.cpp
  DEPS ajp::dep::placebo ajp::dep::vulkan ajp::dep::imgui)
target_compile_definitions(ajp_render_tests PRIVATE VK_NO_PROTOTYPES)
