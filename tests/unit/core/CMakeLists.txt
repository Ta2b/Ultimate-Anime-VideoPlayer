ajp_add_unit_tests(core
  SOURCES cadence_test.cpp contracts_test.cpp moving_area_test.cpp
          auto_rate_test.cpp free_test.cpp free_starved_test.cpp steady_rate_test.cpp steady_grid_test.cpp auto_memory_test.cpp engines_test.cpp headers_test.cpp inbox_test.cpp inflate_test.cpp keys_test.cpp log_test.cpp
          model_index_test.cpp pacing_test.cpp
          file_sync_test.cpp settings_test.cpp thread_test.cpp update_manifest_test.cpp text_file_test.cpp vrr_floor_test.cpp present_test.cpp present_ledger_test.cpp cost_model_test.cpp sample_reservoir_test.cpp
  DEPS ajp::core_log ajp::core_os)

# Le répartiteur d'un backend (install_module_log_sinks) vaut pour tout son binaire : à part.
ajp_add_unit_tests(core_log SOURCES log_module_test.cpp)

# close_log ferme spdlog pour tout le binaire : à part (le module reste core_log).
add_executable(ajp_core_log_close_tests log_close_test.cpp)
target_link_libraries(ajp_core_log_close_tests PRIVATE ajp::core_log Catch2::Catch2WithMain)
ajp_set_warnings(ajp_core_log_close_tests)
catch_discover_tests(ajp_core_log_close_tests TEST_PREFIX "core_log_close: " DISCOVERY_MODE PRE_TEST)
