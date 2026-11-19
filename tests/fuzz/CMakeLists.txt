# Fuzz des entrées venues de l'extérieur : settings.toml et play.toml, l'EDID de l'écran, les
# journaux de construction du cache, la mémoire de reprise, le texte de l'OSD échappé en ASS, les scripts ASS rendus
# par libass, [keys] de user.toml, le paquet .span8, les chapitres d'un fichier Matroska.
# Chaque cible (fuzz_<cible>.cpp, fuzz_targets.h) lit des octets quelconques et vérifie aussi ses
# propriétés (fuzz_check.h : relecture à l'identique, plages, rien ne sort d'une ligne ASS).
#
# Par défaut, un pilote de rejeu (replay_main.cpp) les nourrit de leur corpus (corpus/<cible>/) et
# de mutations déterministes de ce corpus : un test ctest « fuzz: <cible> » chacune, fuzz court
# sous les préréglages ASan/UBSan. KIRARI_FUZZ_LIBFUZZER (Clang) construit en plus kirari_fuzz_<cible>,
# liée à libFuzzer avec ASan et UBSan, pour une campagne longue qui enrichit le corpus ; les
# bibliothèques du lecteur ne sont instrumentées (couverture suivie par libFuzzer) qu'avec
# -DCMAKE_CXX_FLAGS=-fsanitize=fuzzer-no-link,address,undefined au configure :
#   build/<préréglage>/tests/fuzz/kirari_fuzz_settings tests/fuzz/corpus/settings -max_total_time=600
# POSIX seulement (le pilote nomme l'entrée fautive depuis un gestionnaire de signal).
option(KIRARI_FUZZ_LIBFUZZER "Cibles libFuzzer en plus du rejeu (Clang : -fsanitize=fuzzer)" OFF)

set(kirari_fuzz_targets settings play_settings edid build_log resume osd_script ass_render mkv_chapters)
# Les lecteurs de fichiers et de flux (P1.29) : une source pour toutes, fuzz_parsers.cpp (keys et span_package :
# fuzz_formats.cpp).
set(kirari_fuzz_parser_targets onnx srt cube inflate manifest options engine_index update_state prepare_onnx keys
    span_package)
set(kirari_fuzz_sources)
foreach(name IN LISTS kirari_fuzz_targets)
  list(APPEND kirari_fuzz_sources fuzz_${name}.cpp)
endforeach()

add_executable(kirari_fuzz_replay replay_main.cpp ${kirari_fuzz_sources} fuzz_parsers.cpp fuzz_formats.cpp)
target_link_libraries(kirari_fuzz_replay PRIVATE kirari::app kirari::core kirari::gpu_ir kirari::gpu_span_fp8 kirari::media kirari::prepare kirari::render
                                              kirari::subs)
kirari_set_warnings(kirari_fuzz_replay)
# La reprise écrit et relit un fichier par entrée : moins de mutations, pour quelques secondes.
set(kirari_fuzz_mutations_resume 300)
# Le rendu ASS lance quatre fils de libass par instant ; le paquet .span8 de synthèse pèse 600 Ko par entrée.
set(kirari_fuzz_mutations_ass_render 150)
set(kirari_fuzz_mutations_span_package 300)
foreach(name IN LISTS kirari_fuzz_targets kirari_fuzz_parser_targets)
  add_test(NAME "fuzz: ${name}" COMMAND kirari_fuzz_replay ${name} "${CMAKE_CURRENT_SOURCE_DIR}/corpus/${name}"
                                        ${kirari_fuzz_mutations_${name}})
  set_tests_properties("fuzz: ${name}" PROPERTIES LABELS fuzz)
endforeach()

if(KIRARI_FUZZ_LIBFUZZER)
  foreach(name IN LISTS kirari_fuzz_targets kirari_fuzz_parser_targets)
    if(name IN_LIST kirari_fuzz_parser_targets)
      add_executable(kirari_fuzz_${name} libfuzzer_entry.cpp fuzz_parsers.cpp fuzz_formats.cpp)
    else()
      add_executable(kirari_fuzz_${name} libfuzzer_entry.cpp fuzz_${name}.cpp)
    endif()
    target_link_libraries(kirari_fuzz_${name} PRIVATE kirari::app kirari::core kirari::gpu_ir kirari::gpu_span_fp8 kirari::media kirari::prepare
                                                   kirari::render kirari::subs)
    target_compile_definitions(kirari_fuzz_${name} PRIVATE KIRARI_FUZZ_TARGET=${name})
    target_compile_options(kirari_fuzz_${name} PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all)
    target_link_options(kirari_fuzz_${name} PRIVATE -fsanitize=fuzzer,address,undefined)
    kirari_set_warnings(kirari_fuzz_${name})
  endforeach()
endif()
