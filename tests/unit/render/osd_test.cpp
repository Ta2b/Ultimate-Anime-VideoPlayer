// OSD ImGui dessiné par libplacebo, et chargement des hooks du grade.
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include <imgui.h>

#include "render/compositor.h"
#include "render/grade.h"
#include "render/shaders.h"
#include "support.h"

namespace ajp::render {
namespace {

#define REQUIRE_DEVICE(device)                  \
  std::string skip_reason;                      \
  auto device = test::make_device(skip_reason); \
  if (!device) SKIP("pas d'appareil Vulkan : " << skip_reason)

std::size_t count(std::string_view text, std::string_view needle) {
  std::size_t n = 0;
  for (auto at = text.find(needle); at != std::string_view::npos; at = text.find(needle, at + 1)) ++n;
  return n;
}

// Un contexte ImGui de test, sans fenêtre : app possède le vrai.
struct ImGuiContextGuard {
  ImGuiContext* context = ImGui::CreateContext();
  ImGuiContextGuard() = default;
  ImGuiContextGuard(const ImGuiContextGuard&) = delete;
  ImGuiContextGuard& operator=(const ImGuiContextGuard&) = delete;
  ImGuiContextGuard(ImGuiContextGuard&&) = delete;
  ImGuiContextGuard& operator=(ImGuiContextGuard&&) = delete;
  ~ImGuiContextGuard() { ImGui::DestroyContext(context); }
};

}  // namespace

TEST_CASE("les hooks du grade sont embarqués tels quels", "[grade]") {
  // Neuf passes et le tampon d'état du lecteur actuel ; deux pour le vrai noir.
  CHECK(count(shaders::anime_hdr_grade, "//!HOOK OUTPUT") == 9);
  CHECK(count(shaders::anime_hdr_grade, "//!BUFFER AJG_STATE") == 1);
  CHECK(count(shaders::anime_hdr_grade, "//!PARAM ") == 17);
  CHECK(count(shaders::anime_hdr_black, "//!HOOK OUTPUT") == 2);
  CHECK(count(shaders::anime_hdr_black, "//!WHEN black_hi black_lo >") == 2);
  CHECK(count(shaders::anime_hdr_calibration, "//!HOOK OUTPUT") == 1);
}

TEST_CASE("les hooks du grade se chargent et prennent leurs paramètres", "[gpu]") {
  REQUIRE_DEVICE(device);
  auto grade = Grade::create(device->gpu());
  REQUIRE(grade);
  REQUIRE(grade->hooks().size() == 2);
  CHECK(grade->hooks()[0]->stages == PL_HOOK_OUTPUT);
  CHECK(grade->hooks()[1]->stages == PL_HOOK_OUTPUT);
  CHECK(grade->calibration() != nullptr);
  GradeParams params;
  params.peak = 750.0;
  params.black_hi = 0.5;
  grade->set_params(params);
  CHECK(grade->params() == params);
  const pl_hook& hook = *grade->hooks()[0];
  bool seen = false;
  for (int i = 0; i < hook.num_parameters; ++i) {
    if (std::string_view(hook.parameters[i].name) == "peak") {
      seen = true;
      CHECK(hook.parameters[i].data->f == 750.0F);
    }
  }
  CHECK(seen);
}

TEST_CASE("un ImDrawData simple dessine un rectangle au bon endroit", "[gpu]") {
  REQUIRE_DEVICE(device);
  constexpr int w = 64;
  constexpr int h = 48;
  const auto gray = test::solid(w, h, 0.5F, 0.5F, 0.5F);
  pl_tex slot = test::make_texture(device->gpu(), w, h, gray);
  REQUIRE(slot != nullptr);

  const ImGuiContextGuard guard;
  ImGuiIO& io = ImGui::GetIO();
  Compositor::configure_imgui(io);
  io.DisplaySize = ImVec2(static_cast<float>(w), static_cast<float>(h));
  io.DeltaTime = 1.0F / 60.0F;
  io.IniFilename = nullptr;
  ImGui::NewFrame();
  ImGui::GetForegroundDrawList()->AddRectFilled(ImVec2(8.0F, 6.0F), ImVec2(24.0F, 18.0F),
                                                IM_COL32(0, 0, 255, 255));
  ImGui::Render();

  auto compositor = Compositor::create(*device);
  auto target = OffscreenTarget::create(*device, {.width = w, .height = h, .display = test::sdr_display()});
  REQUIRE(compositor);
  REQUIRE(target);
  const SlotColor srgb{.primaries = Primaries::Bt709, .transfer = Transfer::Srgb};
  REQUIRE(compositor->compose(slot, srgb, target->frame(), {.osd = ImGui::GetDrawData()}));
  const auto pixels = test::read_back(*target);
  REQUIRE_FALSE(pixels.empty());
  CHECK(test::channel(pixels, w, 10, 8, 2) > 0.99F);
  CHECK(test::channel(pixels, w, 10, 8, 0) < 0.01F);
  CHECK(test::channel(pixels, w, 23, 17, 2) > 0.99F);
  CHECK(std::abs(test::channel(pixels, w, 30, 30, 0) - 0.5F) < 2e-3F);  // hors du rectangle
  CHECK(std::abs(test::channel(pixels, w, 7, 8, 2) - 0.5F) < 2e-3F);

  compositor->release_imgui_textures(ImGui::GetPlatformIO());
  pl_tex_destroy(device->gpu(), &slot);
}

}  // namespace ajp::render
