// Colle de DistilDRBA fusionné (kernels/spec/drba_fused.h), commune aux noyaux drba_fused_pre, _post, _glue
// et _finish : portage des noyaux CUDA (gpu/cuda/backend/drba_fused_kernels.cu), formules et arrondis FP16
// aux mêmes endroits, la grille des warps exceptée (rife_fused.glsl, pixel_tap : en FP32, depuis la position
// en pixels). S'inclut après tensor.glsl et rife_fused.glsl (rounded, pixel_tap, mix4).
#ifndef AJP_DRBA_FUSED_GLSL
#define AJP_DRBA_FUSED_GLSL

// Le côté choisi du graphe : vrai si tau < 1 (image 0), faux sinon (image 2) ; tau = 2 × plan 6 de pair01.
bool drba_before(float half_tau) {
  float tau = 2.0 * half_tau;
  return clamp((1.0 - tau) * 4096.0, 0.0, 1.0) >= 0.5;
}

// Resize bilinéaire demi-pixel de facteur s (coordonnées source bornées) : les quatre voisins et fractions.
struct DrbaEnlarge {
  uint x0;
  uint x1;
  uint y0;
  uint y1;
  float wx;
  float wy;
};

DrbaEnlarge drba_enlarge(uint px, uint py, uint s, uint zh, uint zw) {
  float inv = 1.0 / float(s);
  precise float sy = (float(py) + 0.5) * inv - 0.5;
  precise float sx = (float(px) + 0.5) * inv - 0.5;
  sy = min(max(sy, 0.0), float(zh - 1u));
  sx = min(max(sx, 0.0), float(zw - 1u));
  DrbaEnlarge e;
  e.y0 = uint(sy);
  e.x0 = uint(sx);
  e.y1 = min(e.y0 + 1u, zh - 1u);
  e.x1 = min(e.x0 + 1u, zw - 1u);
  e.wy = sy - float(e.y0);
  e.wx = sx - float(e.x0);
  return e;
}

// DepthToSpace (CRD, 2) lu à la volée : Z[c][zy][zx] = Y[4c + 2 (zy & 1) + (zx & 1)][zy >> 1][zx >> 1].
// Le noyau qui inclut ce fichier définit DRBA_Y(index) : la lecture de Y.
#define DRBA_READ_Z(c, zy, zx, zh, zw) \
  float(DRBA_Y(((4u * (c) + 2u * ((zy) & 1u) + ((zx) & 1u)) * ((zh) / 2u) + ((zy) >> 1u)) * ((zw) / 2u) + ((zx) >> 1u)))

#endif  // AJP_DRBA_FUSED_GLSL
