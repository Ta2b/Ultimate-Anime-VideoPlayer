#include "app/play/ui_fonts.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#if AJP_WITH_FONTCONFIG
#include <fontconfig/fontconfig.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

#include "app/play/subtitle_fonts.h"
#include "core/log.h"
#include "core/text.h"
#include "core/utf8_path.h"
#ifdef AJP_WITH_FREETYPE
#include "imgui_freetype.h"
#endif

namespace ajp::app::play {
namespace {

constexpr int kMaxFontBytes = 256 << 20;  // ImGui prend la taille en int

// Le fichier de police que fontconfig choisit pour pattern (« sans-serif », « sans-serif:lang=ja ») ;
// seulement TrueType ou OpenType, ce que lit ImGui. need_cjk : la police doit avoir de quoi écrire le
// japonais (あ et 日) ; fontconfig rend sinon la plus proche, parfois une police latine sans aucun
// idéogramme. need_hangul_hanzi : elle doit aussi avoir un hangul (한) et un caractère simplifié (汉), ce
// que les polices japonaises seules (IPAexGothic, TakaoGothic) n'ont pas (C24-10).
std::optional<FontFile> system_font(const char* pattern, bool need_cjk = false,
                                    bool need_hangul_hanzi = false) {
#if AJP_WITH_FONTCONFIG
  if (FcInit() == FcFalse) return std::nullopt;
  FcPattern* wanted = FcNameParse(reinterpret_cast<const FcChar8*>(pattern));
  if (wanted == nullptr) return std::nullopt;
  FcConfigSubstitute(nullptr, wanted, FcMatchPattern);
  FcDefaultSubstitute(wanted);
  FcResult result = FcResultNoMatch;
  FcPattern* match = FcFontMatch(nullptr, wanted, &result);
  std::optional<FontFile> found;
  FcChar8* file = nullptr;
  int index = 0;
  if (match != nullptr && FcPatternGetString(match, FC_FILE, 0, &file) == FcResultMatch) {
    FcPatternGetInteger(match, FC_INDEX, 0, &index);
    FcCharSet* charset = nullptr;
    const bool covers = !need_cjk || (FcPatternGetCharSet(match, FC_CHARSET, 0, &charset) == FcResultMatch &&
                                      FcCharSetHasChar(charset, 0x3042) != FcFalse &&
                                      FcCharSetHasChar(charset, 0x65E5) != FcFalse &&
                                      (!need_hangul_hanzi || (FcCharSetHasChar(charset, 0xD55C) != FcFalse &&
                                                              FcCharSetHasChar(charset, 0x6C49) != FcFalse)));
    std::string path(reinterpret_cast<const char*>(file));
    std::string ext = std::filesystem::path(path).extension().string();
    ext = core::ascii_lower(ext);
    if (covers && (ext == ".ttf" || ext == ".otf" || ext == ".ttc" || ext == ".otc"))
      found = FontFile{.path = std::move(path), .index = index};
  }
  if (match != nullptr) FcPatternDestroy(match);
  FcPatternDestroy(wanted);
  return found;
#else
  static_cast<void>(pattern);
  static_cast<void>(need_cjk);
  static_cast<void>(need_hangul_hanzi);
  return std::nullopt;
#endif
}

// Le fichier projeté dans font.data ; false s'il ne s'ouvre pas.
bool read_font(const std::filesystem::path& path, FontFile& font) {
  font.data = MappedFile::open(path, static_cast<std::size_t>(kMaxFontBytes));
  return font.data != nullptr;
}

#ifdef _WIN32
// Le dossier des polices de Windows (%WINDIR%\Fonts).
std::filesystem::path windows_fonts() {
  std::array<wchar_t, MAX_PATH> dir{};
  const UINT length = GetWindowsDirectoryW(dir.data(), static_cast<UINT>(dir.size()));
  const std::filesystem::path root = length > 0 && length < dir.size()
                                         ? std::filesystem::path(dir.data())
                                         : std::filesystem::path(L"C:\\Windows");
  return root / L"Fonts";
}

// La première des polices de ce groupe qui se lit dans le dossier de Windows.
std::optional<FontFile> first_font(const std::filesystem::path& dir, const WindowsFontGroup& group) {
  for (const wchar_t* name : group.files) {
    if (name == nullptr) break;
    const std::filesystem::path path = dir / name;
    FontFile font{.path = core::to_utf8(path), .index = 0};
    if (read_font(path, font)) return font;
  }
  return std::nullopt;
}

// Sans fontconfig, les polices que Windows 10 et 11 installent toujours (ui_fonts.h).
std::vector<FontFile> find_fonts() {
  std::vector<FontFile> fonts;
  const std::filesystem::path dir = windows_fonts();
  for (const WindowsFontGroup& group : kWindowsFonts) {
    std::optional<FontFile> font = first_font(dir, group);
    if (font) {
      fonts.push_back(std::move(*font));
    } else if (group.required) {
      return fonts;
    } else {
      log_channel("app")->warn("interface : aucune police pour le {} dans les polices de Windows",
                               group.role);
    }
  }
  return fonts;
}
#else
// Une police japonaise seule n'a ni hangul ni caractères simplifiés : si celle qu'on vient de choisir
// (trouvée par `found_with`, nullptr : la police du bureau) n'a pas les deux, on ajoute une seconde police
// de repli qui les a, pour les pistes coréennes et chinoises (C24-10).
void add_hangul_hanzi_fallback(std::vector<FontFile>& fonts, const char* found_with) {
  if (found_with != nullptr) {
    const auto again = system_font(found_with, true, true);
    if (again && again->path == fonts.back().path) return;
  } else if (const auto desktop = system_font("sans-serif", true, true);
             desktop && !fonts.empty() && desktop->path == fonts.front().path) {
    return;
  }
  for (const char* pattern : {"sans-serif:lang=ko", "sans-serif:lang=zh-cn", "Noto Sans CJK KR",
                              "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Droid Sans Fallback"}) {
    std::optional<FontFile> extra = system_font(pattern, true, true);
    if (!extra) continue;
    const bool known = std::ranges::any_of(fonts, [&](const FontFile& f) { return f.path == extra->path; });
    if (known || !read_font(std::filesystem::path(extra->path), *extra)) continue;
    log_channel("app")->info(
        "interface : seconde police CJK pour le hangul et les caractères simplifiés {} ({})", extra->path,
        pattern);
    fonts.push_back(std::move(*extra));
    return;
  }
  log_channel("app")->warn(
      "interface : aucune police avec hangul et caractères simplifiés, pistes en carrés possibles");
}

// Police du bureau (sans-serif), complétée par une police japonaise pour les titres de pistes.
std::vector<FontFile> find_fonts() {
  std::vector<FontFile> fonts;
  // La police du bureau d'abord : ce qui manquerait à Inter (autres écritures) vient d'elle.
  if (std::optional<FontFile> main = system_font("sans-serif");
      main && read_font(std::filesystem::path(main->path), *main))
    fonts.push_back(std::move(*main));
  // Titres japonais (et chinois, coréens) : la police CJK que fontconfig donne pour la langue, sinon les
  // familles courantes des distributions (Fedora : google-noto-sans-cjk-fonts ; Debian : fonts-noto-cjk ;
  // Arch : noto-fonts-cjk).
  bool main_covers = false;
  for (const char* pattern : {"sans-serif:lang=ja", "Noto Sans CJK JP", "Noto Sans JP", "Source Han Sans",
                              "IPAexGothic", "TakaoGothic", "VL Gothic", "WenQuanYi Zen Hei",
                              "Droid Sans Fallback", "sans-serif:lang=zh-cn", "sans-serif:lang=ko"}) {
    std::optional<FontFile> cjk = system_font(pattern, true);
    if (!cjk) continue;
    // La police du bureau est déjà celle que fontconfig donne pour cette langue : elle la couvre, il n'y a
    // rien à ajouter ni à signaler (A24-23).
    if (!fonts.empty() && cjk->path == fonts.front().path) {
      main_covers = true;
      continue;
    }
    if (!read_font(std::filesystem::path(cjk->path), *cjk)) continue;
    log_channel("app")->info("interface : police CJK {} ({})", cjk->path, pattern);
    fonts.push_back(std::move(*cjk));
    add_hangul_hanzi_fallback(fonts, pattern);
    return fonts;
  }
  if (main_covers) {
    log_channel("app")->info("interface : la police du bureau couvre le japonais, pas de police de repli");
    add_hangul_hanzi_fallback(fonts, nullptr);
    return fonts;
  }
  // Aucune dans le système : celle que le paquet Linux joint (fonts/ à côté du lecteur, subtitle_fonts.h).
  if (const std::filesystem::path bundled = bundled_cjk_font(); !bundled.empty()) {
    FontFile cjk{.path = core::to_utf8(bundled), .index = 0};
    if (read_font(bundled, cjk)) {
      log_channel("app")->info("interface : aucune police CJK dans le système, celle du paquet {}", cjk.path);
      fonts.push_back(std::move(cjk));
      return fonts;
    }
  }
  log_channel("app")->warn(
      "interface : aucune police CJK trouvée, les titres japonais s'afficheront en carrés (Noto Sans CJK)");
  return fonts;
}
#endif

// Une police de repli, fusionnée derrière la police embarquée.
bool add_fallback(ImFontAtlas& atlas, FontFile& file) {
  ImFontConfig config;
  config.MergeMode = true;
  // fontconfig range dans l'index d'une police variable le numéro de l'instance nommée (bits 16 et plus) :
  // seule la face compte.
  config.FontNo = static_cast<ImU32>(std::max(file.index, 0)) & 0xFFFFU;
  config.FontDataOwnedByAtlas = false;  // file.data vit dans PlayerUi
  return atlas.AddFontFromMemoryTTF(file.data->data(), static_cast<int>(file.data->size()), kUiFontSize,
                                    &config) != nullptr;
}

// La police embarquée ; flags : ceux du chargeur FreeType (gras), 0 sinon.
ImFont* add_embedded(ImFontAtlas& atlas, int flags) {
  ImFontConfig config;
  config.FontDataOwnedByAtlas = false;  // kUiFontTtf vit autant que le programme, ImGui ne l'écrit pas
  config.FontLoaderFlags = static_cast<unsigned int>(flags);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) : ImGui prend un void*, sans écrire dedans
  void* data = const_cast<char*>(kUiFontTtf.data());
  return atlas.AddFontFromMemoryTTF(data, static_cast<int>(kUiFontTtf.size()), kUiFontSize, &config);
}

}  // namespace

std::vector<FontFile> search_fonts() {
  return find_fonts();
}

// La police embarquée, puis les polices de repli de FontSearch. ImGui 1.92 rastérise à la taille demandée.
void load_fonts(ImFontAtlas& atlas, std::vector<FontFile>& fallbacks, ImFont*& bold) {
  const auto log = log_channel("app");
  // Un octet par pixel au lieu de quatre (A24-18) : l'atlas n'a que des couvertures, le dessin
  // (placebo_imgui.cpp) lit une texture r8 comme l'alpha d'un blanc, et chaque glyphe neuf renvoie l'atlas.
  atlas.TexDesiredFormat = ImTextureFormat_Alpha8;
#ifdef AJP_WITH_FREETYPE
  // FreeType lit les polices que stb_truetype ne lit pas : OpenType CFF2 et variables (Noto Sans CJK de
  // Fedora).
  atlas.SetFontLoader(ImGuiFreeType::GetFontLoader());
#endif
  if (add_embedded(atlas, 0) == nullptr) {
    atlas.AddFontDefaultVector();
    log->error("interface : police embarquée illisible, police vectorielle d'ImGui");
    return;
  }
  log->info("interface : police embarquée {}", kUiFontFamily);
  for (FontFile& file : fallbacks) {
    if (add_fallback(atlas, file)) {
      log->info("interface : police de repli {} chargée", file.path);
    } else {
      log->warn("interface : police de repli {} refusée par le chargeur de polices (format non lu)",
                file.path);
    }
  }
#ifdef AJP_WITH_FREETYPE
  // Les titres des infobulles du menu : la même police, épaissie par FreeType (panel_tip.cpp).
  bold = add_embedded(atlas, ImGuiFreeTypeLoaderFlags_Bold);
#else
  static_cast<void>(bold);
#endif
}

}  // namespace ajp::app::play
