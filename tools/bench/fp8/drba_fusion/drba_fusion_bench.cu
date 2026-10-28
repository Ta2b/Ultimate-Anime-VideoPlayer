// Banc du prototype DistilDRBA v2 lite fusionné : quatre moteurs TensorRT + noyaux FP16, dans un graphe CUDA.
//
//   drba_fusion_bench verifier <dossier_cas>                       noyaux seuls contre la référence ORT
//   drba_fusion_bench executer <dossier_moteurs> in01 in12 out     chaîne complète, sortie fp16 brute
//   drba_fusion_bench plein <moteur> in01 in12 out                 réseau entier TensorRT, sortie fp16 brute
//   drba_fusion_bench mesurer <dossier_moteurs> in01 in12          médianes : chaîne, noyaux, moteurs
//   drba_fusion_bench chaine_mesure <dossier_moteurs> in01 in12    médiane de la chaîne seule
//   drba_fusion_bench plein_mesure <moteur> in01 in12              médiane du réseau entier (même méthode)
//   drba_fusion_bench info <moteur>                                tenseurs d'entrée-sortie du moteur
// Sans entrées réelles les warps sont triviaux : toujours passer de vraies entrées pour les mesures.
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

#include "drba_fusion.cuh"

#define VERIF(x)                                                                               \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "%s:%d %s : %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

namespace {

constexpr int H = 1088;
constexpr int W = 1920;
constexpr int H4 = H / 4, W4 = W / 4, H8 = H / 8, W8 = W / 8, H16 = H / 16, W16 = W / 16;
constexpr std::size_t kPlan = static_cast<std::size_t>(H) * W;
constexpr std::size_t kPlan4 = static_cast<std::size_t>(H4) * W4;
constexpr std::size_t kPlan8 = static_cast<std::size_t>(H8) * W8;
constexpr std::size_t kPlan16 = static_cast<std::size_t>(H16) * W16;

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
  std::vector<std::string> entrees, sorties;  // dans l'ordre du moteur
  std::vector<std::size_t> octets_entrees, octets_sorties;
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
      m.entrees.push_back(nom);
      m.octets_entrees.push_back(n);
    } else {
      m.sorties.push_back(nom);
      m.octets_sorties.push_back(n);
    }
    if (m.engine->getTensorDataType(nom) != nvinfer1::DataType::kHALF ||
        m.engine->getTensorFormat(nom) != nvinfer1::TensorFormat::kLINEAR)
      std::fprintf(stderr, "avertissement : %s n'est pas fp16 linéaire\n", nom);
  }
  return m;
}

void lier(Moteur& m, void* in, void* out) {
  m.ctx->setTensorAddress(m.entrees.at(0).c_str(), in);
  m.ctx->setTensorAddress(m.sorties.at(0).c_str(), out);
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
  __half* in01;
  __half* in12;
  __half* x0;  // 58 × 68 × 120
  __half* x1;  // 109 × 136 × 240
  __half* x2;  // 109 × 272 × 480 (entrée de l'encodeur = plans 0-8, sortie = plans 9-56)
  __half* y0;  // 52 × 34 × 60
  __half* y1;  // 52 × 68 × 120
  __half* y2;  // 52 × 136 × 240
  __half* out;
};

Tampons tampons() {
  Tampons t{};
  t.in01 = alloue<__half>(7 * kPlan * 2);
  t.in12 = alloue<__half>(7 * kPlan * 2);
  t.x0 = alloue<__half>(df::kCanauxX0 * kPlan16 * 2);
  t.x1 = alloue<__half>(df::kCanauxX * kPlan8 * 2);
  t.x2 = alloue<__half>(df::kCanauxX * kPlan4 * 2);
  t.y0 = alloue<__half>(52 * (kPlan16 / 4) * 2);
  t.y1 = alloue<__half>(52 * (kPlan8 / 4) * 2);
  t.y2 = alloue<__half>(52 * (kPlan4 / 4) * 2);
  t.out = alloue<__half>(3 * kPlan * 2);
  return t;
}

dim3 grille2d(int w, int h, dim3 bloc) {
  return dim3((w + bloc.x - 1) / bloc.x, (h + bloc.y - 1) / bloc.y);
}

df::PasGrille pas(int n) {
  const float c32 = static_cast<float>(2.0 / (n - 1));
  return {c32, __half2float(__double2half(2.0 / (n - 1)))};
}

void lancer_pre(const Tampons& t, cudaStream_t s) {
  df::ParamsPre p{H, W, t.in01, t.in12, t.x0, t.x1, t.x2};
  const std::size_t n = 9 * (kPlan4 + kPlan8 + kPlan16) + kPlan4 + kPlan8 + kPlan16;
  df::noyau_pre<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p);
}

void lancer_post(const Tampons& t, cudaStream_t s) {
  df::ParamsPost p{H4, W4, t.x2, t.x1, t.x0};
  const std::size_t n = 48 * (kPlan8 + kPlan16);
  df::noyau_post<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p);
}

// Colle k (1 ou 2) : complète Xk depuis la sortie du bloc k - 1.
void lancer_colle(const Tampons& t, int k, cudaStream_t s) {
  df::ParamsColle p{};
  p.h = k == 1 ? H8 : H4;
  p.w = k == 1 ? W8 : W4;
  p.z = {k == 1 ? t.y0 : t.y1, p.h, p.w};
  p.z.zh = p.h / 2;
  p.z.zw = p.w / 2;
  p.pair01 = t.in01;
  p.plan_pleine = kPlan;
  p.x = k == 1 ? t.x1 : t.x2;
  p.gx = pas(p.w);
  p.gy = pas(p.h);
  const dim3 bloc(32, 4);
  df::noyau_colle<<<grille2d(p.w, p.h, bloc), bloc, 0, s>>>(p);
}

void lancer_fin(const Tampons& t, cudaStream_t s) {
  df::ParamsFin p{};
  p.h = H;
  p.w = W;
  p.z = {t.y2, H4, W4};
  p.pair01 = t.in01;
  p.pair12 = t.in12;
  p.out = t.out;
  p.gx = pas(W);
  p.gy = pas(H);
  const dim3 bloc(32, 8);
  df::noyau_fin<<<grille2d(W, H, bloc), bloc, 0, s>>>(p);
}

struct Chaine {
  Moteur enc, bloc[3];
  Tampons t;
};

Chaine charger_chaine(nvinfer1::IRuntime& rt, const std::string& dossier) {
  Chaine c;
  c.enc = charger(rt, dossier + "/enc.engine");
  for (int b = 0; b < 3; ++b) c.bloc[b] = charger(rt, dossier + "/bloc" + std::to_string(b) + ".engine");
  c.t = tampons();
  lier(c.enc, c.t.x2, c.t.x2 + 9 * kPlan4);
  lier(c.bloc[0], c.t.x0, c.t.y0);
  lier(c.bloc[1], c.t.x1, c.t.y1);
  lier(c.bloc[2], c.t.x2, c.t.y2);
  return c;
}

void enfiler(Chaine& c, cudaStream_t s) {
  lancer_pre(c.t, s);
  c.enc.ctx->enqueueV3(s);
  lancer_post(c.t, s);
  c.bloc[0].ctx->enqueueV3(s);
  lancer_colle(c.t, 1, s);
  c.bloc[1].ctx->enqueueV3(s);
  lancer_colle(c.t, 2, s);
  c.bloc[2].ctx->enqueueV3(s);
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
  std::size_t differents;
};

// `gpu` (n éléments fp16 sur le GPU) contre la référence `ref` ; tranche [debut, debut + n) de la référence.
Ecart comparer(const __half* gpu, const std::vector<char>& ref, std::size_t debut, std::size_t n) {
  std::vector<__half> h(n);
  VERIF(cudaMemcpy(h.data(), gpu, n * 2, cudaMemcpyDeviceToHost));
  const auto* r = reinterpret_cast<const __half*>(ref.data()) + debut;
  double se = 0, mx = 0, pic = 0;
  std::size_t diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = __half2float(h[i]);
    const double b = __half2float(r[i]);
    const double d = a - b;
    se += d * d;
    mx = std::max(mx, std::fabs(d));
    pic = std::max(pic, std::fabs(b));
    diff += a != b;
  }
  const double mse = se / static_cast<double>(n);
  return {mse == 0 ? 200.0 : 10.0 * std::log10(pic * pic / mse), mx, pic, diff};
}

void televerser(__half* dst, const std::vector<char>& src, std::size_t octets_debut = 0) {
  VERIF(cudaMemcpy(dst, src.data() + octets_debut, src.size() - octets_debut, cudaMemcpyHostToDevice));
}

void televerser_tranche(__half* dst, const std::vector<char>& src, std::size_t debut, std::size_t n) {
  VERIF(cudaMemcpy(dst, src.data() + debut * 2, n * 2, cudaMemcpyHostToDevice));
}

int verifier(const std::string& cas) {
  Tampons t = tampons();
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  televerser(t.in01, lire_fichier(cas + "/in01.bin"));
  televerser(t.in12, lire_fichier(cas + "/in12.bin"));
  const auto enc_in = lire_fichier(cas + "/enc_in.bin");
  const auto enc_out = lire_fichier(cas + "/enc_out.bin");
  const auto x0 = lire_fichier(cas + "/x0.bin"), x1 = lire_fichier(cas + "/x1.bin"),
             x2 = lire_fichier(cas + "/x2.bin");
  const auto y0 = lire_fichier(cas + "/y0.bin"), y1 = lire_fichier(cas + "/y1.bin"),
             y2 = lire_fichier(cas + "/y2.bin");
  const auto complet = lire_fichier(cas + "/complet.bin");
  auto affiche = [&](const char* nom, const __half* gpu, const std::vector<char>& ref, std::size_t debut,
                     std::size_t n) {
    const Ecart e = comparer(gpu, ref, debut, n);
    std::printf("%-34s PSNR %7.2f dB (pic %.3g)  max|diff| %.4g  différents %zu / %zu\n", nom, e.psnr, e.pic,
                e.maxabs, e.differents, n);
  };
  // pre
  lancer_pre(t, s);
  VERIF(cudaStreamSynchronize(s));
  affiche("pre -> images au quart (X2[0-8])", t.x2, enc_in, 0, 9 * kPlan4);
  affiche("pre -> images au huitieme (X1[0-8])", t.x1, x1, 0, 9 * kPlan8);
  affiche("pre -> images au seizieme (X0[0-8])", t.x0, x0, 0, 9 * kPlan16);
  affiche("pre -> temps X0[57]", t.x0 + 57 * kPlan16, x0, 57 * kPlan16, kPlan16);
  affiche("pre -> temps X1[95]", t.x1 + 95 * kPlan8, x1, 95 * kPlan8, kPlan8);
  affiche("pre -> temps X2[95]", t.x2 + 95 * kPlan4, x2, 95 * kPlan4, kPlan4);
  // post : l'encodage de référence est mis à sa place dans X2.
  televerser_tranche(t.x2 + 9 * kPlan4, enc_out, 0, 48 * kPlan4);
  lancer_post(t, s);
  VERIF(cudaStreamSynchronize(s));
  affiche("post -> encodages au demi (X1[9-56])", t.x1 + 9 * kPlan8, x1, 9 * kPlan8, 48 * kPlan8);
  affiche("post -> encodages au quart (X0[9-56])", t.x0 + 9 * kPlan16, x0, 9 * kPlan16, 48 * kPlan16);
  // colle 1 : X1 de référence sauf les canaux produits par la colle.
  televerser(t.x1, x1);
  VERIF(cudaMemset(t.x1 + 57 * kPlan8, 0, 38 * kPlan8 * 2));
  VERIF(cudaMemset(t.x1 + 96 * kPlan8, 0, 13 * kPlan8 * 2));
  televerser(t.y0, y0);
  lancer_colle(t, 1, s);
  VERIF(cudaStreamSynchronize(s));
  affiche("colle 1 -> warps (X1[57-94])", t.x1 + 57 * kPlan8, x1, 57 * kPlan8, 38 * kPlan8);
  affiche("colle 1 -> masque/attributs/flot", t.x1 + 96 * kPlan8, x1, 96 * kPlan8, 13 * kPlan8);
  affiche("colle 1 -> X1 entier", t.x1, x1, 0, df::kCanauxX * kPlan8);
  // colle 2
  televerser(t.x2, x2);
  VERIF(cudaMemset(t.x2 + 57 * kPlan4, 0, 38 * kPlan4 * 2));
  VERIF(cudaMemset(t.x2 + 96 * kPlan4, 0, 13 * kPlan4 * 2));
  televerser(t.y1, y1);
  lancer_colle(t, 2, s);
  VERIF(cudaStreamSynchronize(s));
  affiche("colle 2 -> warps (X2[57-94])", t.x2 + 57 * kPlan4, x2, 57 * kPlan4, 38 * kPlan4);
  affiche("colle 2 -> masque/attributs/flot", t.x2 + 96 * kPlan4, x2, 96 * kPlan4, 13 * kPlan4);
  affiche("colle 2 -> X2 entier", t.x2, x2, 0, df::kCanauxX * kPlan4);
  // fin
  televerser(t.y2, y2);
  lancer_fin(t, s);
  VERIF(cudaStreamSynchronize(s));
  affiche("fin -> sortie", t.out, complet, 0, 3 * kPlan);
  return 0;
}

int executer(const std::string& dossier, const std::string& in01, const std::string& in12,
             const std::string& out) {
  Journal j;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(j));
  Chaine c = charger_chaine(*rt, dossier);
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  televerser(c.t.in01, lire_fichier(in01));
  televerser(c.t.in12, lire_fichier(in12));
  enfiler(c, s);
  VERIF(cudaStreamSynchronize(s));
  std::vector<__half> h(3 * kPlan);
  VERIF(cudaMemcpy(h.data(), c.t.out, h.size() * 2, cudaMemcpyDeviceToHost));
  ecrire_fichier(out, h.data(), h.size() * 2);
  return 0;
}

// Réseau entier : deux entrées (pair01, pair12) et une sortie.
struct Plein {
  Moteur m;
  __half *in01, *in12, *out;
};

Plein charger_plein(nvinfer1::IRuntime& rt, const std::string& moteur) {
  Plein p;
  p.m = charger(rt, moteur);
  if (p.m.entrees.size() != 2 || p.m.sorties.size() != 1) {
    std::fprintf(stderr, "moteur entier : 2 entrées et 1 sortie attendues\n");
    std::exit(1);
  }
  p.in01 = alloue<__half>(p.m.octets_entrees[0]);
  p.in12 = alloue<__half>(p.m.octets_entrees[1]);
  p.out = alloue<__half>(p.m.octets_sorties[0]);
  // pair01 d'abord : les noms du modèle d'origine.
  for (int i = 0; i < 2; ++i) {
    const std::string& nom = p.m.entrees[static_cast<std::size_t>(i)];
    p.m.ctx->setTensorAddress(nom.c_str(), nom == "pair01" ? p.in01 : p.in12);
  }
  p.m.ctx->setTensorAddress(p.m.sorties[0].c_str(), p.out);
  return p;
}

int plein(const std::string& moteur, const std::string& in01, const std::string& in12, const std::string& out,
          bool mesure) {
  Journal j;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(j));
  Plein p = charger_plein(*rt, moteur);
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  if (!in01.empty()) {
    televerser(p.in01, lire_fichier(in01));
    televerser(p.in12, lire_fichier(in12));
  }
  p.m.ctx->enqueueV3(s);
  VERIF(cudaStreamSynchronize(s));
  if (!mesure) {
    std::vector<char> h(p.m.octets_sorties[0]);
    VERIF(cudaMemcpy(h.data(), p.out, h.size(), cudaMemcpyDeviceToHost));
    ecrire_fichier(out, h.data(), h.size());
    return 0;
  }
  cudaGraphExec_t g = capturer(s, [&] { p.m.ctx->enqueueV3(s); });
  double mn;
  const double med = mediane(g, s, 200, 200, &mn);
  std::printf("PLEIN graphe CUDA : médiane %.4f ms, min %.4f ms\n", med, mn);
  return 0;
}

int mesurer(const std::string& dossier, const std::string& in01, const std::string& in12,
            bool seulement_chaine) {
  Journal j;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(j));
  Chaine c = charger_chaine(*rt, dossier);
  cudaStream_t s;
  VERIF(cudaStreamCreate(&s));
  if (!in01.empty()) {
    televerser(c.t.in01, lire_fichier(in01));  // vraies données (sinon zéros : warp trivial)
    televerser(c.t.in12, lire_fichier(in12));
  }
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
  somme_noyaux += temps("noyau pre", [&] { lancer_pre(c.t, s); });
  somme_moteurs += temps("moteur encodeur", [&] { c.enc.ctx->enqueueV3(s); });
  somme_noyaux += temps("noyau post", [&] { lancer_post(c.t, s); });
  somme_moteurs += temps("moteur bloc 0", [&] { c.bloc[0].ctx->enqueueV3(s); });
  somme_noyaux += temps("noyau colle 1", [&] { lancer_colle(c.t, 1, s); });
  somme_moteurs += temps("moteur bloc 1", [&] { c.bloc[1].ctx->enqueueV3(s); });
  somme_noyaux += temps("noyau colle 2", [&] { lancer_colle(c.t, 2, s); });
  somme_moteurs += temps("moteur bloc 2", [&] { c.bloc[2].ctx->enqueueV3(s); });
  somme_noyaux += temps("noyau fin", [&] { lancer_fin(c.t, s); });
  std::printf("somme moteurs %.4f ms ; somme noyaux %.4f ms ; total isolé %.4f ms\n", somme_moteurs,
              somme_noyaux, somme_moteurs + somme_noyaux);
  cudaGraphExec_t g = capturer(s, [&] { enfiler(c, s); });
  double mn;
  const double med = mediane(g, s, 200, 200, &mn);
  std::printf("CHAINE graphe CUDA : médiane %.4f ms, min %.4f ms\n", med, mn);
  return 0;
}

int info(const std::string& moteur) {
  Journal j;
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(j));
  Moteur m = charger(*rt, moteur);
  for (std::size_t i = 0; i < m.entrees.size(); ++i)
    std::printf("entrée %s : %zu octets\n", m.entrees[i].c_str(), m.octets_entrees[i]);
  for (std::size_t i = 0; i < m.sorties.size(); ++i)
    std::printf("sortie %s : %zu octets\n", m.sorties[i].c_str(), m.octets_sorties[i]);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  auto arg = [&](int i) { return std::string(i < argc ? argv[i] : ""); };
  if (mode == "verifier" && argc >= 3) return verifier(argv[2]);
  if (mode == "info" && argc >= 3) return info(argv[2]);
  if (mode == "executer" && argc >= 6) return executer(argv[2], argv[3], argv[4], argv[5]);
  if (mode == "plein" && argc >= 6) return plein(argv[2], argv[3], argv[4], argv[5], false);
  if (mode == "plein_mesure" && argc >= 3) return plein(argv[2], arg(3), arg(4), "", true);
  if (mode == "mesurer" && argc >= 3) return mesurer(argv[2], arg(3), arg(4), false);
  if (mode == "chaine_mesure" && argc >= 3) return mesurer(argv[2], arg(3), arg(4), true);
  std::fprintf(stderr, "usage : voir l'en-tête du fichier\n");
  return 2;
}
