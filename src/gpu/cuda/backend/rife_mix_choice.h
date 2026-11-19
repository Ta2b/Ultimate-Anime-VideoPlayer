// Le choix du mélange du RIFE natif (rife_native_mix.cpp, docs/rife-natif.md), en C++ pur : testé sans CUDA
// (tests/unit/gpu/rife_mix_choice_test.cpp). Un mélange donne à chaque morceau (l'encodeur, puis chaque bloc)
// TensorRT ou un réseau natif. Il se juge contre une référence : la chaîne native f32, la plus proche du
// calcul exact, quand la garde la reconnaît (reference_of) ; sinon la chaîne de TensorRT, à 45 dB comme
// avant. Pourquoi : à travers des blocs FP16, tout changement d'un morceau, même d'un ulp, redistribue leurs
// arrondis ; jugé contre TensorRT, un morceau plus juste que le sien se lit au niveau du bruit propre de
// TensorRT (43,6 dB sur la chaîne du 4.22 lite, où aucun encodeur différent du sien ne tenait 45 dB, même
// exact).
#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"

namespace kirari::gpu::cuda::rife_mix {

// L'option de chaque morceau : 0 TensorRT, 1 + n le réseau natif n.
using Options = std::vector<int>;
// Une valeur par morceau et par option : [morceau][option].
using Grid = std::vector<std::vector<double>>;

// « encodeur » pour le morceau 0, « bloc k » pour le morceau 1 + k.
std::string piece_name(int piece);

// Les groupes de plans jugés à part, en plans de taille plane : l'image (3 plans), ou le flot (4 plans) et le
// masque (1 plan).
std::vector<std::pair<std::size_t, std::size_t>> groups_of(bool motion);
// Le pic de chaque groupe de want : max(1, |want| le plus grand du groupe) ; l'image va de 0 à 1, le flot en
// pixels peut dépasser 1.
std::vector<double> group_peak(const std::vector<float>& want, std::size_t plane, bool motion);
// L'erreur quadratique moyenne de got contre want, groupe par groupe ; infinie sur une valeur non finie
// d'un côté ou de l'autre (une référence fautive ne laisse rien passer).
std::vector<double> group_mse(const std::vector<float>& want, const std::vector<float>& got,
                              std::size_t plane, bool motion);
// Le PSNR d'un groupe : 200 dB sans erreur, moins l'infini sur une erreur non finie.
double db_of(double peak, double mse);
// Le PSNR du groupe le plus bas.
double worst_db(const std::vector<double>& peak, const std::vector<double>& mse);
// Le PSNR de chaque groupe de got contre want.
std::vector<double> group_db(const std::vector<float>& want, const std::vector<float>& got, std::size_t plane,
                             bool motion);
double worst_psnr(const std::vector<float>& want, const std::vector<float>& got, std::size_t plane,
                  bool motion);

// Un morceau contre celui de TensorRT sur la même entrée : le PSNR du rôle le plus bas, chacun à son pic
// (max(1, |want|)). channels : 0 pour l'encodage (4 plans, un seul rôle) ; sinon Y d'un bloc, par pixel, dont
// les canaux 4c + parité sont le flot (c < 4) puis le masque (c = 4), le reste les traits. Un défaut du seul
// masque (4 canaux sur 52) se lit ainsi à sa taille, sans la dilution d'un PSNR sur tout le tenseur.
double piece_db(const std::vector<float>& want, const std::vector<float>& got, int channels);

// Les options qu'un morceau peut prendre (times : [morceau][option]) : TensorRT, et chaque réseau plus rapide
// que lui (un réseau plus lent ne gagne rien sur TensorRT).
std::vector<std::vector<int>> candidates(const Grid& times);
bool all_trt(const Options& options);
// L'option la plus rapide de chaque morceau.
Options fastest_options(const Grid& times);

// La garde de la référence native. Chaque morceau du réseau f32 sort, sur la même entrée, à kPieceFloorDb au
// moins de celui de TensorRT, rôle par rôle (piece_db) : une comparaison sans le bruit des blocs suivants,
// qui voit un défaut du natif plus gros que les arrondis de TensorRT. Ces arrondis bornent la garde : ses
// blocs tournent en FP16, et nos morceaux f32 sortent à 50,9 à 62,4 dB des siens sur le PC (bloc 1 de 4.25 le
// plus bas), nos morceaux f16 plus près de lui que nos f32. Un plancher à 60 dB refusait donc un natif juste.
// À 48, un défaut du natif qui pèse autant que le bruit propre de TensorRT fait encore tomber le morceau
// dessous. Et la chaîne de TensorRT est à kChainFloorDb au moins de la chaîne native f32 (43,6 dB sur le 4.22
// lite, 49,3 sur le 4.25) : la barre d'un mélange ne descend jamais sous elle. Sinon la référence reste
// TensorRT : un natif faux ne devient jamais sa propre référence, et un refus à tort ne fait que revenir à
// l'ancienne règle. La garde suppose le natif f32 proche du calcul exact, ce que seuls les morceaux
// attestent. Un mélange qui tient la barre contre lui peut s'écarter de la sortie de TensorRT un peu plus que
// la barre (les deux écarts s'ajoutent).
inline constexpr double kPieceFloorDb = 48.0;
inline constexpr double kChainFloorDb = 42.0;

struct Reference {
  bool native = false;         // vrai : la chaîne native f32 ; faux : celle de TensorRT
  std::vector<double> bar_db;  // ce que chaque groupe d'un mélange doit tenir contre elle
  std::string why;             // pour le journal
};

// pieces_db : chaque morceau du réseau f32 contre celui de TensorRT, sur la même entrée ; trt_db : la chaîne
// de TensorRT contre la chaîne native f32, par groupe ; min_db : la porte. Native, la barre de chaque groupe
// est min_db, ou ce que TensorRT y tient s'il n'y arrive pas lui-même : le mélange est au moins aussi proche
// que lui du calcul exact. Sinon, min_db contre TensorRT.
Reference reference_of(const std::vector<double>& pieces_db, const std::vector<double>& trt_db,
                       double min_db);

// L'effet d'une option sur la chaîne, seule à s'écarter de la référence, en dB (le groupe le plus bas).
struct Effect {
  int piece = 0;
  int option = 0;
  double db = 0.0;
};

// Le mélange retenu : l'option de chaque morceau (toutes à 0 : TensorRT garde la chaîne), sa sortie, ses dB
// contre la référence (le groupe le plus bas) et ceux que la somme des effets prévoyait, les effets mesurés.
struct Mix {
  Options options;
  std::vector<float> out;
  double db = 0.0;
  double predicted = 0.0;
  std::vector<Effect> effects;
};

// La sortie de la chaîne pour des options.
using Chain = std::function<std::expected<std::vector<float>, Refusal>(const Options&)>;

// Ce contre quoi un mélange se juge : want, la sortie de la chaîne dont chaque morceau prend l'option base
// (0 : TensorRT ; le réseau f32 pour la référence native), les groupes (plans de taille plane), la barre de
// chaque groupe.
struct Judge {
  const std::vector<float>* want = nullptr;
  std::size_t plane = 0;
  bool motion = false;
  std::vector<double> bar_db;
  int base = 0;
};

// Mesure l'effet de chaque option sur la chaîne (ce morceau seul hors de base), puis prend le mélange le plus
// rapide (la somme des temps seuls) dont l'erreur prévue tient la barre, vérifié sur la chaîne. Les erreurs
// de morceaux différents s'ajoutent à peu près : l'erreur prévue d'un groupe va de la somme des effets
// (erreurs indépendantes) au carré de la somme de leurs racines (erreurs de même signe aux mêmes pixels),
// selon une corrélation qui part de 0 et que chaque vérification manquée recale, avec une marge de plus pour
// les mélanges de plusieurs morceaux hors de base. Le dernier essai ne prend qu'un mélange d'au plus un tel
// morceau, mesuré tel quel ; sans mélange qui tienne, ou si le plus rapide est tout à TensorRT, tout à
// TensorRT. Refusé si les tailles ne s'accordent pas (temps, base, barre, sortie de want).
std::expected<Mix, Refusal> choose_mix(const Grid& times, const Judge& judge, const Chain& chain);

}  // namespace kirari::gpu::cuda::rife_mix
