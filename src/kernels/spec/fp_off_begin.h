// Ouvre une région où a × b + c ne se contracte pas en FMA : les formules inline de spec/ s'évaluent
// dans le fichier qui les inclut, avec ses options ; la région rend leurs arrondis ceux écrits ici sans
// imposer -ffp-contract=off au reste du lecteur (P5.22). Se referme par fp_off_end.h. Pas de garde :
// chaque en-tête de spec/ l'inclut une fois. MSVC ne contracte pas en /fp:precise.
// NOLINTBEGIN(llvm-header-guard)
#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
// NOLINTEND(llvm-header-guard)
