# Fuzz des entrées venues de l'extérieur : settings.toml et play.toml, l'EDID de l'écran, les
# journaux de construction du cache, la mémoire de reprise, le texte de l'OSD échappé en ASS.
# Chaque cible (fuzz_<cible>.cpp, fuzz_targets.h) lit des octets quelconques et vérifie aussi ses
# propriétés (fuzz_check.h : relecture à l'identique, plages, rien ne sort d'une ligne ASS).
#
# Par défaut, un pilote de rejeu (replay_main.cpp) les nourrit de leur corpus (corpus/<cible>/) et
# de mutations déterministes de ce corpus : un test ctest « fuzz: <cible> » chacune, fuzz court
# sous les préréglages ASan/UBSan. AJP_FUZZ_LIBFUZZER (Clang) construit en plus ajp_fuzz_<cible>,
# liée à libFuzzer avec ASan et UBSan, pour une campagne longue qui enrichit le corpus ; les
# bibliothèques du lecteur ne sont instrumentées (couverture suivie par libFuzzer) qu'avec
# -DCMAKE_CXX_FLAGS=-fsanitize=fuzzer-no-link,address,undefined au configure :
#   build/<préréglage>/tests/fuzz/ajp_fuzz_settings tests/fuzz/corpus/settings -max_total_time=600
# POSIX seulement (le pilote nomme l'entrée fautive depuis un gestionnaire de signal).
option(AJP_FUZZ_LIBFUZZER "Cibles libFuzzer en plus du rejeu (Clang : -fsanitize=fuzzer)" OFF)

set(ajp_fuzz_targets settings play_settings edid build_log resume osd_script)
set(ajp_fuzz_sources)
foreach(name IN LISTS ajp_fuzz_targets)
  list(APPEND ajp_fuzz_sources fuzz_${name}.cpp)
endforeach()

add_executable(ajp_fuzz_replay replay_main.cpp ${ajp_fuzz_sources})
target_link_libraries(ajp_fuzz_replay PRIVATE ajp::app)
ajp_set_warnings(ajp_fuzz_replay)
# La reprise écrit et relit un fichier par entrée : moins de mutations, pour quelques secondes.
set(ajp_fuzz_mutations_resume 300)
foreach(name IN LISTS ajp_fuzz_targets)
  add_test(NAME "fuzz: ${name}" COMMAND ajp_fuzz_replay ${name} "${CMAKE_CURRENT_SOURCE_DIR}/corpus/${name}"
                                        ${ajp_fuzz_mutations_${name}})
  set_tests_properties("fuzz: ${name}" PROPERTIES LABELS fuzz)
endforeach()

if(AJP_FUZZ_LIBFUZZER)
  foreach(name IN LISTS ajp_fuzz_targets)
    add_executable(ajp_fuzz_${name} libfuzzer_entry.cpp fuzz_${name}.cpp)
    target_link_libraries(ajp_fuzz_${name} PRIVATE ajp::app)
    target_compile_definitions(ajp_fuzz_${name} PRIVATE AJP_FUZZ_TARGET=${name})
    target_compile_options(ajp_fuzz_${name} PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(ajp_fuzz_${name} PRIVATE -fsanitize=fuzzer,address,undefined)
    ajp_set_warnings(ajp_fuzz_${name})
  endforeach()
endif()
