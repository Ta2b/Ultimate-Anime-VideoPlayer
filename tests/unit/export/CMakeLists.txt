# Le choix de l'encodeur, logique pure : l'encodage lui-même demande un GPU et n'est pas testé ici.
ajp_add_unit_tests(export
  SOURCES encoder_choice_test.cpp)
