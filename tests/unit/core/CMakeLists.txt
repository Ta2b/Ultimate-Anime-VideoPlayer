ajp_add_unit_tests(core
  SOURCES cadence_test.cpp contracts_test.cpp moving_area_test.cpp
          auto_rate_test.cpp auto_rate_vrr_test.cpp cadence_bounds_test.cpp free_test.cpp free_heavy_test.cpp free_starved_test.cpp steady_rate_test.cpp steady_grid_test.cpp auto_memory_test.cpp engines_test.cpp headers_test.cpp inbox_test.cpp inflate_test.cpp keys_test.cpp log_test.cpp
          model_index_test.cpp sha256_blocks_test.cpp pacing_test.cpp

          file_sync_test.cpp file_space_test.cpp settings_test.cpp thread_test.cpp update_manifest_test.cpp text_file_test.cpp vrr_floor_test.cpp present_test.cpp present_ledger_test.cpp cost_model_test.cpp sample_reservoir_test.cpp
  DEPS ajp::core_log ajp::core_os)

# Le répartiteur d'un backend (install_module_log_sinks) vaut pour tout son binaire : à part.
ajp_add_unit_tests(core_log SOURCES log_module_test.cpp)

# close_log ferme spdlog pour tout le binaire : à part (le module reste core_log).
add_executable(ajp_core_log_close_tests log_close_test.cpp)
target_link_libraries(ajp_core_log_close_tests PRIVATE ajp::core_log Catch2::Catch2WithMain)
ajp_set_warnings(ajp_core_log_close_tests)
catch_discover_tests(ajp_core_log_close_tests TEST_PREFIX "core_log_close: " DISCOVERY_MODE PRE_TEST)

# Le journal asynchrone ne se pose qu'une fois par processus : à part (le module reste core_log).
add_executable(ajp_core_log_async_tests log_async_test.cpp)
target_link_libraries(ajp_core_log_async_tests PRIVATE ajp::core_log Catch2::Catch2WithMain)
ajp_set_warnings(ajp_core_log_async_tests)
catch_discover_tests(ajp_core_log_async_tests TEST_PREFIX "core_log_async: " DISCOVERY_MODE PRE_TEST)

# C27-02 : le configure refuse un toml++ dont les fichiers à corriger manquent.
add_test(NAME "core: TomlFixes refuse un toml++ sans impl/parser.inl"
         COMMAND ${CMAKE_COMMAND} -DSOURCE=${PROJECT_SOURCE_DIR} -DWORK=${CMAKE_CURRENT_BINARY_DIR}/toml_fixes
                 -DTOML_INCLUDE=${ajp_toml_include} -P ${CMAKE_CURRENT_SOURCE_DIR}/toml_fixes_test.cmake)
