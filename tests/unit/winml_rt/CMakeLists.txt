# Tests du runtime MIGraphX de Windows ML qui se passent de Windows : le manifeste de l'artefact,
# le choix de l'appareil, et les refus de make_runtime hors d'un GPU AMD et hors de Windows.
ajp_add_unit_tests(winml_rt SOURCES winml_test.cpp)
