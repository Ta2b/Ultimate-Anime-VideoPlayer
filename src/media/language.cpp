// Repris de the-database-mpv misc/language.c:30-250 (table ISO 639 langmap)
// et 252-299 (mp_match_lang), réécrit en C++ sans talloc. Comportement fixé
// par tests/unit/media/language_test.cpp.
#include "media/language.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

namespace ajp::media {
namespace {

using LangPair = std::pair<std::string_view, std::string_view>;

// Balise reconnue → forme canonique, triée par balise pour la recherche dichotomique.
// clang-format off
constexpr std::array<LangPair, 204> kLanguages{{
    {"aa", "aar"}, {"ab", "abk"}, {"ae", "ave"}, {"af", "afr"}, {"ak", "aka"}, {"am", "amh"},
    {"an", "arg"}, {"ar", "ara"}, {"as", "asm"}, {"av", "ava"}, {"ay", "aym"}, {"az", "aze"},
    {"ba", "bak"}, {"be", "bel"}, {"bg", "bul"}, {"bh", "bih"}, {"bi", "bis"}, {"bm", "bam"},
    {"bn", "ben"}, {"bo", "tib"}, {"bod", "tib"}, {"br", "bre"}, {"bs", "bos"}, {"ca", "cat"},
    {"ce", "che"}, {"ces", "cze"}, {"ch", "cha"}, {"co", "cos"}, {"cr", "cre"}, {"cs", "cze"},
    {"cu", "chu"}, {"cv", "chv"}, {"cy", "wel"}, {"cym", "wel"}, {"da", "dan"}, {"de", "ger"},
    {"deu", "ger"}, {"dv", "div"}, {"dz", "dzo"}, {"ee", "ewe"}, {"el", "gre"}, {"ell", "gre"},
    {"en", "eng"}, {"eo", "epo"}, {"es", "spa"}, {"et", "est"}, {"eu", "baq"}, {"eus", "baq"},
    {"fa", "per"}, {"fas", "per"}, {"ff", "ful"}, {"fi", "fin"}, {"fj", "fij"}, {"fo", "fao"},
    {"fr", "fre"}, {"fra", "fre"}, {"fy", "fry"}, {"ga", "gle"}, {"gd", "gla"}, {"gl", "glg"},
    {"gn", "grn"}, {"gu", "guj"}, {"gv", "glv"}, {"ha", "hau"}, {"he", "heb"}, {"hi", "hin"},
    {"ho", "hmo"}, {"hr", "hrv"}, {"ht", "hat"}, {"hu", "hun"}, {"hy", "arm"}, {"hye", "arm"},
    {"hz", "her"}, {"ia", "ina"}, {"id", "ind"}, {"ie", "ile"}, {"ig", "ibo"}, {"ii", "iii"},
    {"ik", "ipk"}, {"io", "ido"}, {"is", "ice"}, {"isl", "ice"}, {"it", "ita"}, {"iu", "iku"},
    {"ja", "jpn"}, {"jv", "jav"}, {"ka", "geo"}, {"kat", "geo"}, {"kg", "kon"}, {"ki", "kik"},
    {"kj", "kua"}, {"kk", "kaz"}, {"kl", "kal"}, {"km", "khm"}, {"kn", "kan"}, {"ko", "kor"},
    {"kr", "kau"}, {"ks", "kas"}, {"ku", "kur"}, {"kv", "kom"}, {"kw", "cor"}, {"ky", "kir"},
    {"la", "lat"}, {"lb", "ltz"}, {"lg", "lug"}, {"li", "lim"}, {"ln", "lin"}, {"lo", "lao"},
    {"lt", "lit"}, {"lu", "lub"}, {"lv", "lav"}, {"mg", "mlg"}, {"mh", "mah"}, {"mi", "mao"},
    {"mk", "mac"}, {"mkd", "mac"}, {"ml", "mal"}, {"mn", "mon"}, {"mr", "mar"}, {"mri", "mao"},
    {"ms", "may"}, {"msa", "may"}, {"mt", "mlt"}, {"my", "bur"}, {"mya", "bur"}, {"na", "nau"},
    {"nb", "nob"}, {"nd", "nde"}, {"ne", "nep"}, {"ng", "ndo"}, {"nl", "dut"}, {"nld", "dut"},
    {"nn", "nno"}, {"no", "nor"}, {"nr", "nbl"}, {"nv", "nav"}, {"ny", "nya"}, {"oc", "oci"},
    {"oj", "oji"}, {"om", "orm"}, {"or", "ori"}, {"os", "oss"}, {"pa", "pan"}, {"pi", "pli"},
    {"pl", "pol"}, {"ps", "pus"}, {"pt", "por"}, {"qu", "que"}, {"rm", "roh"}, {"rn", "run"},
    {"ro", "rum"}, {"ron", "rum"}, {"ru", "rus"}, {"rw", "kin"}, {"sa", "san"}, {"sc", "srd"},
    {"sd", "snd"}, {"se", "sme"}, {"sg", "sag"}, {"si", "sin"}, {"sk", "slo"}, {"sl", "slv"},
    {"slk", "slo"}, {"sm", "smo"}, {"sn", "sna"}, {"so", "som"}, {"sq", "alb"}, {"sqi", "alb"},
    {"sr", "srp"}, {"ss", "ssw"}, {"st", "sot"}, {"su", "sun"}, {"sv", "swe"}, {"sw", "swa"},
    {"ta", "tam"}, {"te", "tel"}, {"tg", "tgk"}, {"th", "tha"}, {"ti", "tir"}, {"tk", "tuk"},
    {"tl", "tgl"}, {"tn", "tsn"}, {"to", "ton"}, {"tr", "tur"}, {"ts", "tso"}, {"tt", "tat"},
    {"tw", "twi"}, {"ty", "tah"}, {"ug", "uig"}, {"uk", "ukr"}, {"ur", "urd"}, {"uz", "uzb"},
    {"ve", "ven"}, {"vi", "vie"}, {"vo", "vol"}, {"wa", "wln"}, {"wo", "wol"}, {"xh", "xho"},
    {"yi", "yid"}, {"yo", "yor"}, {"za", "zha"}, {"zh", "chi"}, {"zho", "chi"}, {"zu", "zul"},
}};
// clang-format on

bool iequal(std::string_view a, std::string_view b) {
  return std::ranges::equal(
      a, b, [](unsigned char x, unsigned char y) { return std::tolower(x) == std::tolower(y); });
}

bool iless(std::string_view a, std::string_view b) {
  return std::ranges::lexicographical_compare(
      a, b, [](unsigned char x, unsigned char y) { return std::tolower(x) < std::tolower(y); });
}

std::vector<std::string_view> split_subtags(std::string_view tag) {
  std::vector<std::string_view> parts;
  while (!tag.empty()) {
    const std::size_t dash = tag.find('-');
    parts.push_back(tag.substr(0, dash));
    tag = dash == std::string_view::npos ? std::string_view{} : tag.substr(dash + 1);
  }
  return parts;
}

}  // namespace

std::string_view canonical_language(std::string_view primary) {
  const auto* it = std::ranges::lower_bound(kLanguages, primary, iless, &LangPair::first);
  if (it != kLanguages.end() && iequal(it->first, primary)) return it->second;
  return primary;
}

int language_score(std::span<const std::string> priorities, std::string_view tag) {
  const std::vector<std::string_view> parts = split_subtags(tag);
  if (parts.empty()) return 0;
  const auto part_count = static_cast<int>(parts.size());

  int best = 0;
  for (std::size_t rank = 0; rank < priorities.size(); ++rank) {
    const std::vector<std::string_view> wanted = split_subtags(priorities[rank]);
    if (wanted.empty() || !iequal(canonical_language(parts[0]), canonical_language(wanted[0]))) continue;
    // Une langue plus haut dans la liste l'emporte ; chaque sous-balise de la
    // piste différente ou non demandée coûte 1000.
    int score = INT_MAX - static_cast<int>(rank);
    int matched = 1;
    for (std::size_t i = 1; i < wanted.size() && matched < part_count; ++i, ++matched) {
      if (!iequal(parts[static_cast<std::size_t>(matched)], wanted[i])) score -= 1000;
    }
    score -= (part_count - matched) * 1000;
    best = std::max(best, score);
  }
  return best;
}

}  // namespace ajp::media
