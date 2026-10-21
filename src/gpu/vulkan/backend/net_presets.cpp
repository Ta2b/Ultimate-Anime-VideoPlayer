// Tables pré-réglées du moteur Vulkan (net_autotune.h : PresetTable), générées par
// tools/vk/presets_from_cache.py depuis le cache de l'autotuner d'un PC de référence : ne pas
// modifier à la main. Version des candidats : c5.
// NvidiaAda : 10de:2782:NVIDIA, 29 couches.
#include <array>
#include <span>

#include "gpu/vulkan/backend/net_autotune.h"

namespace ajp::gpu::vk::net {
namespace {

constexpr std::array<PresetEntry, 29> kNvidiaAda{{
    {0x02df5e51fbb34cea, "coopmat_acc16", "1,2"},   {0x2d60238f1885462c, "coopmat_acc16", "1,2"},
    {0x38a3b2ab83a347b7, "direct_f16p", "4,128,2"}, {0x398f2893afc4b72e, "direct_f16p", "4,64"},
    {0x51b7d89396cf1dc6, "direct_f16p", "4,64"},    {0x52d7406c8dec5d09, "direct_f16p", "2,128"},
    {0x55c9ef11a9354f39, "coopmat_acc16", "1,2"},   {0x5e97262aa12d11f5, "coopmat_acc16", "1,2"},
    {0x5f324bf1dcd95d31, "coopmat_acc16", "1,2"},   {0x6301a98d86a4421c, "coopmat_acc16", "1,2"},
    {0x6f477d08e0fea6e4, "coopmat_acc16", "1,4"},   {0x73b3d9282e7df442, "coopmat_acc16", "1,2"},
    {0x899dc7d028026bb3, "coopmat_acc16", "1,2"},   {0x89cfc2bc692c01c5, "coopmat_acc16", "2,2"},
    {0x932908c69d693fe9, "direct_f16p", "2,128"},   {0x9b6399d79b79fd42, "coopmat_acc16", "2,2"},
    {0x9c6ba39dfb0b1eb8, "coopmat_acc16", "1,2"},   {0x9cd8b3b436536905, "coopmat_acc16", "1,2"},
    {0x9fa61a2f5e5b0a5a, "direct_f16p", "2,64"},    {0xb7b58b13cb618d9c, "coopmat_acc16", "1,2"},
    {0xb833e64d1ed3ff4a, "coopmat_acc16", "1,2"},   {0xbbd6ad1e2b187f39, "coopmat_acc16", "1,2"},
    {0xc72c4ad83c8e8eda, "direct_f16p", "2,128"},   {0xc77cb709aa031116, "direct_f16p", "4,64"},
    {0xca919b1866135f4d, "coopmat_acc16", "1,2"},   {0xd40858c93e0c781c, "coopmat_acc16", "1,2"},
    {0xd9d61a890f2ca2bd, "coopmat", "1,2"},         {0xdad2897dd0c67f77, "coopmat_acc16", "2,8"},
    {0xecb130a6f9db696e, "coopmat_acc16", "1,8"},
}};

}  // namespace

std::span<const PresetTable> preset_tables() {
  static constexpr std::array<PresetTable, 1> kTables{{
      {.family = GpuFamily::NvidiaAda, .entries = kNvidiaAda},
  }};
  return kTables;
}

}  // namespace ajp::gpu::vk::net
