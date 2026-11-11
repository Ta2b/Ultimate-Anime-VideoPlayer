// Ouvre une région où a × b + c ne se contracte pas en FMA : les formules inline de spec/ s'évaluent
// dans le fichier qui les inclut, avec ses options ; la région rend leurs arrondis ceux écrits ici sans
// imposer -ffp-contract=off au reste du lecteur (P5.22). Se referme par fp_off_end.h. Pas de garde :
// chaque en-tête de spec/ l'inclut une fois. MSVC ne contracte pas en /fp:precise. Rien sous GCC (B50-06) :
// `#pragma GCC optimize` donne aux fonctions un attribut d'optimisation que celui de leurs appelants hors de
// kernels n'a pas, et GCC refuse alors de les inliner ; kernels et les tests (ajp_add_unit_tests) se
// compilent avec -ffp-contract=off, et le reste du lecteur n'a pas besoin des arrondis du contrat au bit
// près. NOLINTBEGIN(llvm-header-guard)
#if defined(__CUDACC__)
// nvcc : les ports CUDA suivent leurs propres réglages.
#elif defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#endif
// NOLINTEND(llvm-header-guard)
