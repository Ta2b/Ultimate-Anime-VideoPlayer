// Constantes de spécialisation communes à la Conv directe (net_conv.glsl) et à la somme de ses tranches de
// K (net_conv_reduce.comp, V1) : mêmes identifiants, même liste envoyée par l'abaissement.
layout(constant_id = 0) const uint TX = 0;
layout(constant_id = 1) const uint TY = 0;
layout(constant_id = 2) const uint TR = 0;
layout(constant_id = 3) const uint LX = 0;
layout(constant_id = 4) const uint LY = 0;
layout(constant_id = 5) const uint LR = 0;
layout(constant_id = 6) const uint CXP = 16;
layout(constant_id = 7) const uint CYP = 16;
layout(constant_id = 8) const uint CRP = 16;
layout(constant_id = 9) const uint N = 1;
layout(constant_id = 10) const uint CIN = 1;
layout(constant_id = 11) const uint HIN = 1;
layout(constant_id = 12) const uint WIN = 1;
layout(constant_id = 13) const uint COUT = 1;
layout(constant_id = 14) const uint HOUT = 1;
layout(constant_id = 15) const uint WOUT = 1;
layout(constant_id = 16) const uint KH = 1;
layout(constant_id = 17) const uint KW = 1;
layout(constant_id = 18) const uint SH = 1;
layout(constant_id = 19) const uint SW = 1;
layout(constant_id = 20) const uint DH = 1;
layout(constant_id = 21) const uint DW = 1;
layout(constant_id = 22) const uint PT = 0;
layout(constant_id = 23) const uint PL = 0;
layout(constant_id = 24) const uint GROUPS = 1;
layout(constant_id = 25) const uint ACT = 0;
layout(constant_id = 26) const float ALPHA = 0.0;
layout(constant_id = 27) const float LO = 0.0;
layout(constant_id = 28) const float HI = 0.0;
layout(constant_id = 29) const uint RESIDUAL = 0;
layout(constant_id = 30) const uint QUADS = 1;
layout(constant_id = 32) const uint PIX = 1;  // 31 : la taille du groupe de travail
layout(constant_id = 33) const uint SPACE = 0;      // DepthToSpace fondu : blocs de SPACE × SPACE
layout(constant_id = 34) const uint SPACE_CRD = 0;  // 1 : ordre CRD, 0 : DCR
layout(constant_id = 35) const float IN_Q = 0.0;    // > 0 : x arrondi à la grille E4M3 de cette échelle
layout(constant_id = 36) const float OUT_Q = 0.0;   // > 0 : résultat arrondi de même, après le biais
// Découpe de K (V1) : SPLIT tranches des canaux d'entrée, chacune dans son groupe de travail ; les sommes
// partielles vont dans le tampon de travail (binding 6), net_conv_reduce les ajoute et applique l'épilogue.
layout(constant_id = 37) const uint SPLIT = 1;
