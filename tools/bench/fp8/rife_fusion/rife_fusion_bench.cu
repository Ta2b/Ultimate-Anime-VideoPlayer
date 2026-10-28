// Banc du prototype RIFE v4.25 fusionné : six moteurs TensorRT + noyaux de colle FP16, dans un graphe CUDA.
//
//   rife_fusion_bench verifier <dossier_cas> <dossier_moteurs>   noyaux seuls contre la référence ORT
//   rife_fusion_bench executer <dossier_moteurs> in.bin out.bin  chaîne complète, sortie fp16 brute
//   rife_fusion_bench plein <moteur> in.bin out.bin              réseau entier TensorRT, sortie fp16 brute
//   rife_fusion_bench mesurer <dossier_moteurs> [in.bin]         médianes : chaîne, noyaux, moteurs
//   rife_fusion_bench chaine_mesure <dossier_moteurs> [in.bin]   médiane de la chaîne seule
//   rife_fusion_bench plein_mesure <moteur> [in.bin]             médiane du réseau entier (même méthode)
// Sans in.bin les tampons sont à zéro (flot nul, lectures trop faciles) : toujours passer une vraie entrée.
#include <NvInfer.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "rife_fusion.cuh"

#define VERIF(x)                                                                               \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "%s:%d %s : %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

namespace {

constexpr int H = 768;
constexpr int W = 1280;
// Dimensions des entrées Xk (k = 0..4) et des sorties Yk (52 canaux) des blocs.
constexpr int kD[5] = {8, 8, 4, 2, 1};
constexpr int kCanauxX[5] = {15, 28, 28, 28, 28};
constexpr int kS[4] = {8, 8, 4, 2};

class Journal : public nvinfer1::ILogger {
 public:
  void log(Severity gravite, const char* msg) noexcept override {
    if (gravite <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
  }
};

std::vector<char> lire_fichier(const std::string& chemin) {
  std::ifstream f(chemin, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "fichier illisible : %s\n", chemin.c_str());
    std::exit(1);
  }
  const auto n = static_cast<std::size_t>(f.tellg());
  std::vector<char> d(n);
  f.seekg(0);
  f.read(d.data(), static_cast<std::streamsize>(n));
  return d;
}

void ecrire_fichier(const std::string& chemin, const void* data, std::size_t n) {
  std::ofstream f(chemin, std::ios::binary);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
}

struct Moteur {
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  std::string nom_in, nom_out;
  std::size_t octets_in = 0, octets_out = 0;
};

Moteur charger(nvinfer1::IRuntime& rt, const std::string& chemin) {
  Moteur m;
  const auto donnees = lire_fichier(chemin);
  m.engine.reset(rt.deserializeCudaEngine(donnees.data(), donnees.size()));
  if (!m.engine) {
    std::fprintf(stderr, "désérialisation impossible : %s\n", chemin.c_str());
    std::exit(1);
  }
  m.ctx.reset(m.engine->createExecutionContext());
  for (int i = 0; i < m.engine->getNbIOTensors(); ++i) {
    const char* nom = m.engine->getIOTensorName(i);
    const auto dims = m.engine->getTensorShape(nom);
    std::size_t n = 2;  // fp16
    for (int d = 0; d < dims.nbDims; ++d) n *= static_cast<std::size_t>(dims.d[d]);
    if (m.engine->getTensorIOMode(nom) == nvinfer1::TensorIOMode::kINPUT) {
      m.nom_in = nom;
      m.octets_in = n;
    } else {
      m.nom_out = nom;
      m.octets_out = n;
    }
    if (m.engine->getTensorDataType(nom) != nvinfer1::DataType::kHALF ||
        m.engine->getTensorFormat(nom) != nvinfer1::TensorFormat::kLINEAR)
      std::fprintf(stderr, "avertissement : %s n'est pas fp16 linéaire\n", nom);
  }
  return m;
}

void lier(Moteur& m, void* in, void* out) {
  m.ctx->setTensorAddress(m.nom_in.c_str(), in);
  m.ctx->setTensorAddress(m.nom_out.c_str(), out);
}

template <typename T>
T* alloue(std::size_t octets) {
  void* p = nullptr;
  VERIF(cudaMalloc(&p, octets));
  VERIF(cudaMemset(p, 0, octets));
  return static_cast<T*>(p);
}

// Tampons partagés de la chaîne.
struct Tampons {
  __half* in;
  __half* enc;
  __half* x[5];
  __half* y[5];
  __half* out;
  int xh[5], xw[5];  // dimensions des entrées
};

Tampons tampons() {
  Tampons t{};
  t.in = alloue<__half>(11ull * H * W * 2);
  t.enc = alloue<__half>(8ull * H * W * 2);
  t.out = alloue<__half>(3ull * H * W * 2);
  for (int k = 0; k < 5; ++k) {
    t.xh[k] = H / kD[k];
    t.xw[k] = W / kD[k];
    t.x[k] = alloue<__half>(static_cast<std::size_t>(kCanauxX[k]) * t.xh[k] * t.xw[k] * 2);
    t.y[k] = alloue<__half>(52ull * (t.xh[k] / 2) * (t.xw[k] / 2) * 2);
  }
  return t;
}

dim3 grille(int w, int h, dim3 bloc) {
  return dim3((w + bloc.x - 1) / bloc.x, (h + bloc.y - 1) / bloc.y);
}

void lancer_entree(const Tampons& t, cudaStream_t s) {
  rf::ParamsEntree p{H, W, kD[0], t.xh[0], t.xw[0], t.in, t.enc, t.x[0]};
  const dim3 bloc(32, 8);
  rf::noyau_entree<<<grille(p.ow, p.oh, bloc), bloc, 0, s>>>(p);
}

// Colle du bloc b : produit l'entrée du bloc b + 1.
void lancer_colle(const Tampons& t, int b, cudaStream_t s) {
  rf::ParamsColle p{};
  p.h = H;
  p.w = W;
  p.d = kD[b + 1];
  p.oh = t.xh[b + 1];
  p.ow = t.xw[b + 1];
  p.nflot = b + 1;
  for (int k = 0; k <= b; ++k) p.f[k] = {t.y[k], t.xh[k] / 2, t.xw[k] / 2, kS[k]};
  p.echelle_flot = 1.0F / static_cast<float>(p.d);
  p.in = t.in;
  p.enc = t.enc;
  p.x = t.x[b + 1];
  const dim3 bloc(32, 4);
  rf::noyau_colle<<<grille(p.ow, p.oh, bloc), bloc, 0, s>>>(p);
}

void lancer_fin(const Tampons& t, cudaStream_t s) {
  rf::ParamsFin p{H, W, t.in, t.enc, t.x[4], t.y[4], t.xh[4] / 2, t.xw[4] / 2, t.out};
  const dim3 bloc(32, 8);
  rf::noyau_fin<<<grille(W, H, bloc), bloc, 0, s>>>(p);
}

struct Chaine {
  Moteur enc, bloc[5];
  Tampons t;
};

Chaine charger_chaine(nvinfer1::IRuntime& rt, const std::string& dossier) {
  Chaine c;
  c.enc = charger(rt, dossier + "/enc.engine");
  for (int b = 0; b < 5; ++b) c.bloc[b] = charger(rt, dossier + "/bloc" + std::to_string(b) + ".engine");
  c.t = tampons();
  lier(c.enc, c.t.in, c.t.enc);  // les 6 premiers plans de l'entrée = 2 × 3 × H × W (images A puis B)
  for (int b = 0; b < 5; ++b) lier(c.bloc[b], c.t.x[b], c.t.y[b]);
  return c;
}

void enfiler(Chaine& c, cudaStream_t s) {
  c.enc.ctx->enqueueV3(s);
  lancer_entree(c.t, s);
  for (int b = 0; b < 5; ++b) {
    c.bloc[b].ctx->enqueueV3(s);
    if (b < 4) lancer_colle(c.t, b, s);
  }
  lancer_fin(c.t, s);
}

// Capture d'une séquence dans un graphe CUDA.
template <typename F>
cudaGraphExec_t capturer(cudaStream_t s, F&& f) {
  cudaGraph_t g;
  VERIF(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  f();
  VERIF(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t e;
  VERIF(cudaGraphInstantiate(&e, g, 0));
  VERIF(cudaGraphDestroy(g));
  return e;
}

// Médiane (ms) de `n` lancements du graphe, chacun mesuré par événements après `chauffe` lancements.
double mediane(cudaGraphExec_t g, cudaStream_t s, int chauffe, int n, double* minimum = nullptr) {
  cudaEvent_t a, b;
  VERIF(cudaEventCreate(&a));
  VERIF(cudaEventCreate(&b));
  for (int i = 0; i < chauffe; ++i) VERIF(cudaGraphLaunch(g, s));
  VERIF(cudaStreamSynchronize(s));
  std::vector<double> v;
  for (int i = 0; i < n; ++i) {
    VERIF(cudaEventRecord(a, s));
    VERIF(cudaGraphLaunch(g, s));
    VERIF(cudaEventRecord(b, s));
    VERIF(cudaEventSynchronize(b));
    float ms;
    VERIF(cudaEventElapsedTime(&ms, a, b));
    v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  if (minimum) *minimum = v.front();
  return v[v.size() / 2];
}

struct Ecart {
  double psnr, maxabs, pic;
};

Ecart comparer(const __half* gpu, const std::vector<char>& ref, std::size_t n) {
  std::vector<__half> h(n);
  VERIF(cudaMemcpy(h.data(), gpu, n * 2, cudaMemcpyDeviceToHost));
  const auto* r = reinterpret_cast<const __half*>(ref.data());
  double se = 0, mx = 0, pic = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = __half2float(h[i]);
    const double b = __half2float(r[i]);
    const double d = a - b;
    se += d * d;
    mx = std::max(mx, std::fabs(d));
    pic = std::max(pic, std::fabs(b));
  }
  const double mse = se / static_cast<double>(n);
  return {mse == 0 ? 200.0 : 10.0 * std::log10(pic * pic / mse), mx, pic};
}

void televerser(__half* dst, const std::vector<char>& src) {
  VERIF(cudaMemcpy(dst, src.data(), src.size(), cudaMemcpyHostToDevice));
}

int verifier(const std::string& cas, const std::string& dossier_moteurs) {
  (void)dossier_moteurs;
  Tampons t = tampons();
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  televerser(t.in, lire_fichier(cas + "/in.bin"));
  televerser(t.enc, lire_fichier(cas + "/enc.bin"));
  for (int k = 0; k < 5; ++k) televerser(t.y[k], lire_fichier(cas + "/y" + std::to_string(k) + ".bin"));
  auto affiche = [&](const char* nom, const __half* gpu, const char* fichier, std::size_t n) {
    const auto ref = lire_fichier(cas + "/" + fichier);
    const Ecart e = comparer(gpu, ref, n);
    std::printf("%-24s PSNR %7.2f dB (pic %.3g)  max|diff| %.4g\n", nom, e.psnr, e.pic, e.maxabs);
  };
  lancer_entree(t, s);
  VERIF(cudaStreamSynchronize(s));
  affiche("noyau entree -> x0", t.x[0], "x0.bin", 15ull * t.xh[0] * t.xw[0]);
  for (int b = 0; b < 4; ++b) {
    lancer_colle(t, b, s);
    VERIF(cudaStreamSynchronize(s));
    char nom[64], f[16];
    std::snprintf(nom, sizeof nom, "noyau colle %d -> x%d", b, b + 1);
    std::snprintf(f, sizeof f, "x%d.bin", b + 1);
    affiche(nom, t.x[b + 1], f, 28ull * t.xh[b + 1] * t.xw[b + 1]);
  }
  // Le noyau final consomme l'entrée du bloc 4 de référence.
  televerser(t.x[4], lire_fichier(cas + "/x4.bin"));
  lancer_fin(t, s);
  VERIF(cudaStreamSynchronize(s));
  affiche("noyau fin -> sortie", t.out, "out.bin", 3ull * H * W);
  return 0;
}

int executer(const std::string& dossier, const std::string& in, const std::string& out) {
  Journal j;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(j));
  Chaine c = charger_chaine(*rt, dossier);
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  televerser(c.t.in, lire_fichier(in));
  enfiler(c, s);
  VERIF(cudaStreamSynchronize(s));
  std::vector<__half> h(3ull * H * W);
  VERIF(cudaMemcpy(h.data(), c.t.out, h.size() * 2, cudaMemcpyDeviceToHost));
  ecrire_fichier(out, h.data(), h.size() * 2);
  return 0;
}

int plein(const std::string& moteur, const std::string& in, const std::string& out, bool mesure) {
  Journal j;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(j));
  Moteur m = charger(*rt, moteur);
  __half* din = alloue<__half>(m.octets_in);
  __half* dout = alloue<__half>(m.octets_out);
  lier(m, din, dout);
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  if (!mesure) {
    televerser(din, lire_fichier(in));
    m.ctx->enqueueV3(s);
    VERIF(cudaStreamSynchronize(s));
    std::vector<char> h(m.octets_out);
    VERIF(cudaMemcpy(h.data(), dout, h.size(), cudaMemcpyDeviceToHost));
    ecrire_fichier(out, h.data(), h.size());
    return 0;
  }
  if (!in.empty())
    televerser(din, lire_fichier(in));  // vraies données : les flots réels décident des lectures du warp
  m.ctx->enqueueV3(s);
  VERIF(cudaStreamSynchronize(s));
  cudaGraphExec_t g = capturer(s, [&] { m.ctx->enqueueV3(s); });
  double mn;
  const double med = mediane(g, s, 200, 200, &mn);
  std::printf("PLEIN graphe CUDA : médiane %.4f ms, min %.4f ms\n", med, mn);
  return 0;
}

int mesurer(const std::string& dossier, const std::string& in, bool seulement_chaine = false) {
  Journal j;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(j));
  Chaine c = charger_chaine(*rt, dossier);
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  if (!in.empty()) televerser(c.t.in, lire_fichier(in));  // vraies données (sinon zéros : warp trivial)
  enfiler(c, s);
  VERIF(cudaStreamSynchronize(s));
  if (seulement_chaine) {
    cudaGraphExec_t g = capturer(s, [&] { enfiler(c, s); });
    double mn;
    const double med = mediane(g, s, 200, 200, &mn);
    std::printf("CHAINE graphe CUDA : médiane %.4f ms, min %.4f ms\n", med, mn);
    return 0;
  }
  constexpr int kRep = 20;  // répétitions dans un graphe pour les éléments courts
  auto temps = [&](const char* nom, auto&& f) {
    cudaGraphExec_t g = capturer(s, [&] {
      for (int i = 0; i < kRep; ++i) f();
    });
    double mn;
    const double med = mediane(g, s, 20, 100, &mn);
    std::printf("%-28s %8.4f ms\n", nom, med / kRep);
    return med / kRep;
  };
  double somme_moteurs = 0, somme_noyaux = 0;
  somme_moteurs += temps("moteur encode", [&] { c.enc.ctx->enqueueV3(s); });
  somme_noyaux += temps("noyau entree (x0)", [&] { lancer_entree(c.t, s); });
  for (int b = 0; b < 5; ++b) {
    char nom[64];
    std::snprintf(nom, sizeof nom, "moteur bloc %d", b);
    somme_moteurs += temps(nom, [&] { c.bloc[b].ctx->enqueueV3(s); });
    if (b < 4) {
      std::snprintf(nom, sizeof nom, "noyau colle %d (-> x%d)", b, b + 1);
      somme_noyaux += temps(nom, [&] { lancer_colle(c.t, b, s); });
    }
  }
  somme_noyaux += temps("noyau fin (sortie)", [&] { lancer_fin(c.t, s); });
  std::printf("somme moteurs %.4f ms ; somme noyaux %.4f ms ; total isolé %.4f ms\n", somme_moteurs,
              somme_noyaux, somme_moteurs + somme_noyaux);
  cudaGraphExec_t g = capturer(s, [&] { enfiler(c, s); });
  double mn;
  const double med = mediane(g, s, 200, 200, &mn);
  std::printf("CHAINE graphe CUDA : médiane %.4f ms, min %.4f ms\n", med, mn);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "verifier" && argc >= 3) return verifier(argv[2], argc > 3 ? argv[3] : "");
  if (mode == "executer" && argc >= 5) return executer(argv[2], argv[3], argv[4]);
  if (mode == "plein" && argc >= 5) return plein(argv[2], argv[3], argv[4], false);
  const std::string donnees = argc >= 4 ? argv[3] : "";
  if (mode == "plein_mesure" && argc >= 3) return plein(argv[2], donnees, "", true);
  if (mode == "mesurer" && argc >= 3) return mesurer(argv[2], donnees);
  if (mode == "chaine_mesure" && argc >= 3) return mesurer(argv[2], donnees, true);
  std::fprintf(stderr, "usage : voir l'en-tête du fichier\n");
  return 2;
}
