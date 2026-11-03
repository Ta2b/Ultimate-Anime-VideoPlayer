// Les refontes des shaders du grade (fonctions partagées, passes fondues) ne doivent pas changer un code de
// sortie : l'empreinte de l'image gradée de chaque cas est comparée à celle relevée avant la refonte, sur le
// Vulkan logiciel de la CI (lavapipe), le seul où elle est reproductible ; ailleurs le test se saute.
// AJP_GRADE_GOLDEN_PRINT=1 écrit les empreintes du code courant.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vulkan/vulkan.h>

#include "grade_gpu_support.h"
#include "render/cube_lut.h"
#include "render/grade.h"

namespace ajp::render {
using namespace gpu_test;

namespace {

std::uint64_t fnv(const std::vector<std::uint16_t>& codes) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const std::uint16_t c : codes) {
    h ^= c & 0xFFU;
    h *= 0x100000001b3ULL;
    h ^= c >> 8U;
    h *= 0x100000001b3ULL;
  }
  return h;
}

bool software_device(const Gpu& gpu) {
  const auto get = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      gpu.inst->get_proc_addr(gpu.inst->instance, "vkGetPhysicalDeviceProperties"));
  if (get == nullptr) return false;
  VkPhysicalDeviceProperties props{};
  get(gpu.vk->phys_device, &props);
  return std::strstr(props.deviceName, "llvmpipe") != nullptr;
}

// Pans sombre et clair, un éclat, une rampe orange, du rouge et du bleu saturés : chaque branche du grade.
std::vector<std::uint16_t> scene() {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (int y = 0; y < kSide; ++y)
    for (int x = 0; x < kSide; ++x) {
      double r = x < kSide / 2 ? 20.0 : 250.0;
      double g = r;
      double b = r;
      if (y >= kSide / 2) {
        r = 900.0 * (x + 1) / kSide;
        g = r * 0.6;
        b = r * 0.03;
      }
      if (y >= 3 * kSide / 4 && x < kSide / 4) {
        r = 120.0;
        g = 8.0;
        b = 4.0;
      }
      if (y >= 3 * kSide / 4 && x >= 3 * kSide / 4) {
        r = 10.0;
        g = 30.0;
        b = 140.0;
      }
      if (x >= 60 && x < 63 && y >= 60 && y < 63) r = g = b = 800.0;
      const auto at = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4;
      pixels[at] = static_cast<std::uint16_t>(std::lround(nits_to_pq(r) * 65535.0));
      pixels[at + 1] = static_cast<std::uint16_t>(std::lround(nits_to_pq(g) * 65535.0));
      pixels[at + 2] = static_cast<std::uint16_t>(std::lround(nits_to_pq(b) * 65535.0));
      pixels[at + 3] = 65535;
    }
  return pixels;
}

// La même scène en codes SDR (gamma 2,2, blanc à 1).
std::vector<std::uint16_t> sdr_image() {
  std::vector<std::uint16_t> pixels = scene();
  for (std::size_t i = 0; i < pixels.size(); i += 4)
    for (std::size_t c = 0; c < 3; ++c) {
      const double nits = pq_to_nits(pixels[i + c] / 65535.0);
      pixels[i + c] =
          static_cast<std::uint16_t>(std::lround(std::pow(std::min(nits / 203.0, 1.0), 1.0 / 2.2) * 65535.0));
    }
  return pixels;
}

struct Case {
  const char* name;
  GradeParams params;
  bool looks = false;  // deux LUT empilées (une teinte chaude, puis une courbe)
  bool kwin = false;   // la compression de KWin dans la dernière passe
};

std::vector<Case> cases() {
  std::vector<Case> out;
  out.push_back({"defaut", [] {
                   GradeParams p;
                   p.peak = kPeak;
                   return p;
                 }()});
  out.push_back({"eclats", strong(1.0)});
  out.push_back({"etendu", expanded(0.8)});
  out.push_back({"truehdr", native_look(0.0)});
  out.push_back({"vibrance-punch", [] {
                   GradeParams p = strong(0.5);
                   p.vibrance = 0.6;
                   p.punch = 0.4;
                   p.chroma = 1.2;
                   p.shadows = 0.5;
                   p.gamut_p3 = 1.0;
                   return p;
                 }()});
  out.push_back({"noir", [] {
                   GradeParams p = strong(0.5);
                   p.black_lo = 0.002;
                   p.black_hi = 0.02;
                   return p;
                 }()});
  out.push_back({"looks",
                 [] {
                   GradeParams p = strong(0.5);
                   p.look_mix = 0.8;
                   p.look_mix2 = 0.5;
                   return p;
                 }(),
                 true});
  out.push_back({"looks-noir-kwin",
                 [] {
                   GradeParams p = strong(0.5);
                   p.look_mix = 1.0;
                   p.black_lo = 0.002;
                   p.black_hi = 0.02;
                   return p;
                 }(),
                 true, true});
  out.push_back({"sdr", [] {
                   GradeParams p = sdr_neutral();
                   p.vibrance = 0.4;
                   p.shadows = 0.4;
                   return p;
                 }()});
  return out;
}

CubeLut warm() {
  CubeLut lut = identity_lut(17);
  for (std::size_t i = 0; i < lut.rgb.size(); i += 3) {
    lut.rgb[i] = std::min(1.0F, lut.rgb[i] * 1.08F);
    lut.rgb[i + 2] *= 0.9F;
  }
  return lut;
}

CubeLut curve() {
  CubeLut lut = identity_lut(17);
  for (float& v : lut.rgb) v = v * v * (3.0F - 2.0F * v);
  return lut;
}

}  // namespace

TEST_CASE("grade fusionné : empreintes des images gradées inchangées par les refontes",
          "[render][gpu][golden]") {
  // La chaîne d'avant 2.2 (look dans sa passe) garde les empreintes du départ ; le corps fusionné les garde
  // sans look, et les siennes avec (2.2 : le look lu dans la passe du grade, sans l'étape en demi-flottants).
  const GradeBody body = GENERATE(GradeBody::Fused, GradeBody::FusedSplit);
  const char* prefix = body == GradeBody::Fused ? "" : "split/";
  Gpu gpu(body);
  require_device(gpu);
  REQUIRE(gpu.grade);
  const bool print = std::getenv("AJP_GRADE_GOLDEN_PRINT") != nullptr;
  if (!print && !software_device(gpu)) SKIP("empreintes relevées sur lavapipe seulement");
  // Relevées sur lavapipe au commit d'avant les refontes du chantier 2a.
  struct Golden {
    const char* name;
    std::uint64_t hash;
  };
  constexpr Golden kGolden[] = {
      {"defaut", 0x982fd46364b7d7bfULL},
      {"eclats", 0xa0d9c6dfab01903ULL},
      {"etendu", 0x88b752715c2042aeULL},
      {"truehdr", 0x43c7f9075d6b0bd0ULL},
      {"vibrance-punch", 0xfaed7dfec8eed895ULL},
      {"noir", 0x28e0e7de6f239eafULL},
      {"looks", 0x351f506e24ab7ed1ULL},
      {"looks-noir-kwin", 0x814afb1ec82427cbULL},
      {"sdr", 0xd5f812967d41a93aULL},
      {"split/defaut", 0x982fd46364b7d7bfULL},
      {"split/eclats", 0xa0d9c6dfab01903ULL},
      {"split/etendu", 0x88b752715c2042aeULL},
      {"split/truehdr", 0x43c7f9075d6b0bd0ULL},
      {"split/vibrance-punch", 0xfaed7dfec8eed895ULL},
      {"split/noir", 0x28e0e7de6f239eafULL},
      {"split/looks", 0x7c51537821d3ad96ULL},
      {"split/looks-noir-kwin", 0x5da88f97717fcc6cULL},
      {"split/sdr", 0xd5f812967d41a93aULL},
  };
  const std::vector<std::uint16_t> source = scene();
  const CubeLut a = warm();
  const CubeLut b = curve();
  for (const Case& c : cases()) {
    std::array<const CubeLut*, kLookSlots> luts{};
    if (c.looks) {
      luts[0] = &a;
      luts[1] = &b;
    }
    REQUIRE(gpu.grade->set_looks(luts));
    gpu.grade->set_kwin_match(c.kwin ? std::optional(Grade::KwinMatch{.reference = 200.0, .peak = 800.0})
                                     : std::nullopt);
    const std::vector<std::uint16_t> image = c.params.sdr_out > 0.5 ? sdr_image() : source;
    const std::uint64_t h = fnv(graded_image(gpu, image, c.params));
    const std::string name = std::string(prefix) + c.name;
    if (print) {
      WARN("{\"" << name << "\", 0x" << std::hex << h << "ULL},");
      continue;
    }
    INFO(name);
    bool found = false;
    for (const Golden& g : kGolden) {
      if (std::string(g.name) != name) continue;
      found = true;
      CHECK(g.hash == h);
    }
    CHECK(found);
  }
}

TEST_CASE("look lu dans la passe du grade = passe du look", "[render][gpu][lut]") {
  Gpu fused(GradeBody::Fused);
  require_device(fused);
  REQUIRE(fused.grade);
  Gpu split(GradeBody::FusedSplit);
  REQUIRE(split.grade);
  const CubeLut a = warm();
  const CubeLut b = curve();
  for (const Case& c : cases()) {
    if (!c.looks) continue;
    INFO(c.name);
    for (Gpu* gpu : {&fused, &split}) {
      REQUIRE(gpu->grade->set_looks({&a, &b}));
      gpu->grade->set_kwin_match(c.kwin ? std::optional(Grade::KwinMatch{.reference = 200.0, .peak = 800.0})
                                        : std::nullopt);
    }
    const std::vector<std::uint16_t> x = graded_image(fused, scene(), c.params);
    const std::vector<std::uint16_t> y = graded_image(split, scene(), c.params);
    REQUIRE(x.size() == y.size());
    // La chaîne à deux passes range l'image entre le grade et le look en demi-flottants (11 bits de mantisse
    // : jusqu'à 32 codes 16 bits au-dessus du code 0,5) ; la passe unique ne l'arrondit plus. L'écart est
    // donc borné par ce pas, étiré au plus deux fois par la LUT et l'encodage du look, plus 2 codes (critère
    // du plan).
    std::size_t outside = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (i % 4 == 3) continue;
      const double v = std::max(int(y[i]), 1) / 65535.0;
      const double half_step = std::ldexp(1.0, static_cast<int>(std::floor(std::log2(v))) - 10) * 65535.0;
      if (std::abs(int(x[i]) - int(y[i])) > 2.0 + 2.0 * half_step) ++outside;
    }
    CHECK(outside == 0);
  }
}

}  // namespace ajp::render
