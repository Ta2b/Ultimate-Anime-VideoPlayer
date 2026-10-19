// Réflexion SPIR-V (spirv_reflect.h), sans appareil : l'interface de chaque shader embarqué, les
// modules malformés ou hors de la disposition fixe refusés (mutations d'un vrai module), et la
// comparaison aux descriptions et aux capacités de l'appareil.
#include "gpu/vulkan/compute/spirv_reflect.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "ajp_vk_compute_spirv.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;

namespace {

// Numéros de la spécification SPIR-V utilisés par les mutations.
constexpr std::uint32_t kOpEntryPoint = 15;
constexpr std::uint32_t kOpVariable = 59;
constexpr std::uint32_t kOpDecorate = 71;
constexpr std::uint32_t kOpMemberDecorate = 72;
constexpr std::uint32_t kDecorationSpecId = 1;
constexpr std::uint32_t kDecorationBinding = 33;
constexpr std::uint32_t kDecorationDescriptorSet = 34;
constexpr std::uint32_t kDecorationOffset = 35;
constexpr std::uint32_t kStorageUniform = 2;
constexpr std::uint32_t kStoragePushConstant = 9;
constexpr std::uint32_t kStorageStorageBuffer = 12;
constexpr std::uint32_t kCapStorageBuffer16BitAccess = 4433;
constexpr std::uint32_t kCapInt64 = 11;
constexpr std::uint32_t kCapImageQuery = 50;
constexpr std::uint32_t kCapGroupNonUniformArithmetic = 63;
constexpr std::size_t kHeaderWords = 5;

std::vector<std::uint32_t> module_of(std::string_view shader) {
  const std::span<const std::uint32_t> words = spirv::words(shader);
  REQUIRE_FALSE(words.empty());
  return {words.begin(), words.end()};
}

// Indice du premier mot de chaque instruction d'opcode op dont les opérandes vérifient match.
std::vector<std::size_t> find_all(const std::vector<std::uint32_t>& words, std::uint32_t op,
                                  const std::function<bool(std::span<const std::uint32_t>)>& match) {
  std::vector<std::size_t> found;
  for (std::size_t i = kHeaderWords; i < words.size();) {
    const std::uint32_t count = words[i] >> 16U;
    REQUIRE(count > 0);
    if ((words[i] & 0xFFFFU) == op && match(std::span(words).subspan(i + 1, count - 1U))) found.push_back(i);
    i += count;
  }
  return found;
}

std::size_t find_one(const std::vector<std::uint32_t>& words, std::uint32_t op,
                     const std::function<bool(std::span<const std::uint32_t>)>& match) {
  const std::vector<std::size_t> found = find_all(words, op, match);
  REQUIRE_FALSE(found.empty());
  return found.front();
}

bool is_decoration(std::span<const std::uint32_t> a, std::uint32_t decoration) {
  return a.size() >= 3 && a[1] == decoration;
}

PipelineDesc transfer_desc(std::span<const std::uint32_t> constants) {
  return {.shader = "transfer", .constants = constants, .buffers = 2, .images = 0, .push_bytes = 32};
}

}  // namespace

TEST_CASE("SPIR-V : interface de chaque shader embarqué, disposition fixe", "[vk_compute][spirv]") {
  const std::span<const spirv::SpirvShader> shaders = spirv::shaders();
  REQUIRE(shaders.size() >= 20);
  for (const spirv::SpirvShader& shader : shaders) {
    INFO(shader.name);
    const ShaderInterface reflected = require_value(reflect(shader.words, shader.name));
    CHECK(reflected.name == shader.name);
    // Les tampons d'abord, puis les images, sans trou : la disposition de PipelineCache.
    std::vector<std::uint32_t> all = reflected.storage_buffers;
    all.insert(all.end(), reflected.storage_images.begin(), reflected.storage_images.end());
    for (std::size_t i = 0; i < all.size(); ++i) CHECK(all[i] == i);
    CHECK(reflected.push_bytes % 4 == 0);
    for (std::size_t i = 0; i < reflected.spec_ids.size(); ++i) CHECK(reflected.spec_ids[i] == i);
    // Seules les variantes float16 demandent le stockage 16 bits.
    const bool f16 = shader.name.ends_with("_f16");
    const bool declares_16bit = std::ranges::find(reflected.capabilities, kCapStorageBuffer16BitAccess) !=
                                reflected.capabilities.end();
    CHECK(declares_16bit == f16);
  }
}

TEST_CASE("SPIR-V : interfaces connues et lecture sans appareil", "[vk_compute][spirv]") {
  const ShaderInterface transfer = require_value(PipelineCache::interface("transfer"));
  CHECK(transfer.storage_buffers == std::vector<std::uint32_t>{0, 1});
  CHECK(transfer.storage_images.empty());
  CHECK(transfer.push_bytes == 32);
  CHECK(transfer.spec_ids == std::vector<std::uint32_t>{0});

  const ShaderInterface store = require_value(PipelineCache::interface("rgba16_store"));
  CHECK(store.storage_buffers == std::vector<std::uint32_t>{0});
  CHECK(store.storage_images == std::vector<std::uint32_t>{1});
  CHECK(store.push_bytes == 8);

  require_refusal(PipelineCache::interface("shader_inexistant"), RefusalCode::FeatureUnavailable);
}

TEST_CASE("SPIR-V : modules malformés refusés", "[vk_compute][spirv]") {
  const std::vector<std::uint32_t> good = module_of("transfer");
  require_refusal(reflect({}, "vide"), RefusalCode::InputMismatch);

  std::vector<std::uint32_t> magic = good;
  magic[0] = 0x03022307U;  // mauvais boutisme
  require_refusal(reflect(magic, "boutisme"), RefusalCode::InputMismatch);

  // Une instruction qui annonce plus de mots qu'il n'en reste (le point d'entrée coupé après son
  // modèle d'exécution), puis une de zéro mot.
  const std::size_t entry = find_one(good, kOpEntryPoint, [](auto) { return true; });
  const std::vector<std::uint32_t> truncated(good.begin(),
                                             good.begin() + static_cast<std::ptrdiff_t>(entry + 2));
  require_refusal(reflect(truncated, "tronqué"), RefusalCode::InputMismatch);
  std::vector<std::uint32_t> zero = good;
  zero[kHeaderWords] &= 0xFFFFU;
  require_refusal(reflect(zero, "zéro mot"), RefusalCode::InputMismatch);

  // Point d'entrée Vertex au lieu de GLCompute.
  std::vector<std::uint32_t> vertex = good;
  vertex[entry + 1] = 0;
  require_refusal(reflect(vertex, "vertex"), RefusalCode::InputMismatch);
}

TEST_CASE("SPIR-V : interface hors de la disposition fixe refusée", "[vk_compute][spirv]") {
  const std::vector<std::uint32_t> good = module_of("transfer");
  REQUIRE(reflect(good, "transfer"));

  SECTION("ensemble de descripteurs autre que 0") {
    std::vector<std::uint32_t> words = good;
    const std::size_t at =
        find_one(words, kOpDecorate, [](auto a) { return is_decoration(a, kDecorationDescriptorSet); });
    words[at + 3] = 1;
    const auto refused = reflect(words, "ensemble 1");
    require_refusal(refused, RefusalCode::InputMismatch);
    CHECK(refused.error().detail.find("ensemble 1") != std::string::npos);
  }
  SECTION("binding en double") {
    std::vector<std::uint32_t> words = good;
    for (const std::size_t at :
         find_all(words, kOpDecorate, [](auto a) { return is_decoration(a, kDecorationBinding); })) {
      words[at + 3] = 0;
    }
    const auto refused = reflect(words, "binding double");
    require_refusal(refused, RefusalCode::InputMismatch);
    CHECK(refused.error().detail.find("deux fois") != std::string::npos);
  }
  SECTION("uniform buffer au lieu d'un tampon de stockage") {
    std::vector<std::uint32_t> words = good;
    const std::size_t at =
        find_one(words, kOpVariable, [](auto a) { return a.size() >= 3 && a[2] == kStorageStorageBuffer; });
    words[at + 3] = kStorageUniform;
    require_refusal(reflect(words, "uniform"), RefusalCode::InputMismatch);
  }
  SECTION("SpecId déclaré deux fois") {
    std::vector<std::uint32_t> words = good;
    const std::size_t at =
        find_one(words, kOpDecorate, [](auto a) { return is_decoration(a, kDecorationSpecId); });
    // Une seconde décoration SpecId 0, sur un autre identifiant : l'ordre ne compte pas ici.
    words.insert(words.end(),
                 {(4U << 16U) | kOpDecorate, words[at + 1] + 1, kDecorationSpecId, words[at + 3]});
    require_refusal(reflect(words, "SpecId double"), RefusalCode::InputMismatch);
  }
  SECTION("bloc push qu'on ne sait pas mesurer") {
    std::vector<std::uint32_t> words = good;
    const std::size_t variable =
        find_one(words, kOpVariable, [](auto a) { return a.size() >= 3 && a[2] == kStoragePushConstant; });
    const std::uint32_t pointer = words[variable + 1];
    // Le type pointé du pointeur PushConstant : OpTypePointer %pointer PushConstant %struct.
    std::uint32_t block = 0;
    for (std::size_t i = kHeaderWords; i < words.size(); i += words[i] >> 16U) {
      if ((words[i] & 0xFFFFU) == 32 && words[i + 1] == pointer) block = words[i + 3];
    }
    REQUIRE(block != 0);
    // Le premier membre perd sa décoration Offset (remplacée par RelaxedPrecision, numéro 0).
    const std::size_t at = find_one(words, kOpMemberDecorate, [block](auto a) {
      return a.size() >= 4 && a[0] == block && a[1] == 0 && a[2] == kDecorationOffset;
    });
    words[at + 3] = 0;
    require_refusal(reflect(words, "push"), RefusalCode::InputMismatch);
  }
}

TEST_CASE("SPIR-V : description comparée à l'interface du shader", "[vk_compute][spirv]") {
  const ShaderInterface transfer = require_value(PipelineCache::interface("transfer"));
  const std::array one{0U};
  const std::array two{0U, 1U};
  CHECK(check_interface(transfer, transfer_desc(one)));

  PipelineDesc desc = transfer_desc(one);
  desc.buffers = 3;
  require_refusal(check_interface(transfer, desc), RefusalCode::InputMismatch);
  desc = transfer_desc(one);
  desc.buffers = 1;
  desc.images = 1;
  require_refusal(check_interface(transfer, desc), RefusalCode::InputMismatch);
  desc = transfer_desc(one);
  desc.push_bytes = 28;
  require_refusal(check_interface(transfer, desc), RefusalCode::InputMismatch);
  // Une constante oubliée garderait sa valeur par défaut ; une de trop n'irait nulle part.
  require_refusal(check_interface(transfer, transfer_desc({})), RefusalCode::InputMismatch);
  require_refusal(check_interface(transfer, transfer_desc(two)), RefusalCode::InputMismatch);
}

TEST_CASE("SPIR-V : capacités comparées aux fonctions activées de l'appareil", "[vk_compute][spirv]") {
  const ShaderInterface half = require_value(PipelineCache::interface("transfer_f16"));
  DeviceCaps caps{.device_name = "appareil de test"};
  const auto refused = check_capabilities(half, caps);
  require_refusal(refused, RefusalCode::FeatureUnavailable);
  CHECK(refused.error().detail.find("4433") != std::string::npos);
  caps.enabled.storage_buffer_16bit = true;
  CHECK(check_capabilities(half, caps));
  CHECK(check_capabilities(require_value(PipelineCache::interface("transfer")), DeviceCaps{}));

  // ImageQuery (imageSize) ne demande rien ; une capacité que le socle ne connaît pas est
  // refusée, jamais supposée présente.
  CHECK(
      check_capabilities(ShaderInterface{.name = "taille", .capabilities = {kCapImageQuery}}, DeviceCaps{}));
  ShaderInterface int64{.name = "int64", .capabilities = {kCapInt64}};
  require_refusal(check_capabilities(int64, caps), RefusalCode::FeatureUnavailable);

  // Les opérations de sous-groupe suivent VkSubgroupFeatureFlagBits, bit par bit.
  ShaderInterface arithmetic{.name = "sous-groupes", .capabilities = {kCapGroupNonUniformArithmetic}};
  caps.subgroup_operations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT;
  require_refusal(check_capabilities(arithmetic, caps), RefusalCode::FeatureUnavailable);
  caps.subgroup_operations |= VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
  CHECK(check_capabilities(arithmetic, caps));
}
