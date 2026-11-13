// Le SPAN d'un paquet « .span8 » calculé en f32 sur le processeur, sans quantification ni CUTLASS : un calcul
// écrit à part des noyaux, contre lequel le backend juge, au chargement, ce que les noyaux CUTLASS rendent
// sur la carte (SpanFp8Engine, essai au chargement). Il sert surtout aux architectures où les noyaux n'ont
// jamais tourné devant nous (RTX 50 : docs/span-fp8-cutlass.md, SPAN FP16 natif).
//
// Le graphe est celui du SPAN « unshuffle » d'AnimeJaNai (onnx_weights.h) : dégroupage 2×2 (canal
// c·4 + dy·2 + dx), conv_1 (3×3), blocs (c1, Relu, c2, Relu, c3 ; sortie (x + c3)·tanh(att(c3))),
// concaténation de la sortie de conv_1, de celle du dernier bloc, de celle du premier et du premier Relu du
// dernier bloc, conv_cat (1×1), to_pixel (3×3), DepthToSpace(4, CRD). Remplissage nul au bord de chaque
// convolution, comme l'ONNX. Les canaux réels du paquet (real_channels) seulement : les 16 de complément d'un
// b5f48 sont nuls.
//
// Lent (une boucle simple) : quelques dizaines de milliers de pixels d'entrée, pas une image entière.
#pragma once

#include <span>
#include <vector>

#include "gpu/span_fp8/package.h"

namespace ajp::gpu::span_fp8 {

// Les échelles et le drapeau FP8 du paquet sont ignorés : les poids FP16 tels quels, calcul en f32.
// rgb : 3 plans de height × width (hauteur et largeur paires), dans [0, 1] en pratique. Rend 3 plans de
// 2·height × 2·width ; vide si la taille ne va pas (impaire, nulle, ou rgb de la mauvaise longueur).
std::vector<float> forward(const Package& package, std::span<const float> rgb, int height, int width);

// Pixels de sortie, depuis le haut et la gauche, qu'une découpe height × width de l'image calcule comme
// l'image entière : au-delà, le remplissage nul du bord droit et du bas de la découpe s'y fait sentir. Une
// convolution 3×3 au demi-plan de travail mange une ligne : conv_1, trois par bloc, to_pixel.
int exact_side(int side, int blocks);

}  // namespace ajp::gpu::span_fp8
