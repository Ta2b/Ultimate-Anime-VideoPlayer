// Chemins et textes UTF-8 : ce que Windows lirait autrement dans sa page ANSI.
#include "prepare/utf8.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "prepare/identity.h"
#include "support.h"

using namespace ajp::prepare;

TEST_CASE("is_utf8 : formes valides et mal formées", "[prepare][utf8]") {
  CHECK(is_utf8(""));
  CHECK(is_utf8("ASCII"));
  CHECK(is_utf8("é ü 日本 \xf0\x9f\x8e\x9e"));  // 2, 3 et 4 octets
  CHECK(is_utf8("\xf4\x8f\xbf\xbf"));           // U+10FFFF, le dernier point de code
  CHECK_FALSE(is_utf8("\xff"));                 // octet jamais valide
  CHECK_FALSE(is_utf8("\x80"));                 // octet de suite isolé
  CHECK_FALSE(is_utf8("\xc3"));                 // séquence coupée
  CHECK_FALSE(is_utf8("\xc0\xaf"));             // forme trop longue de « / »
  CHECK_FALSE(is_utf8("\xe0\x80\xaf"));         // idem sur trois octets
  CHECK_FALSE(is_utf8("\xed\xa0\x80"));         // moitié de paire UTF-16
  CHECK_FALSE(is_utf8("\xf4\x90\x80\x80"));     // au-delà de U+10FFFF
  CHECK_FALSE(is_utf8("\xf5\x80\x80\x80"));     // octet de tête hors plage
}

TEST_CASE("utf8_path et path_from_utf8 : aller-retour sans perte", "[prepare][utf8]") {
  const std::string text = "moteurs/日本 é/aji-v3-x.engine";
  const auto path = path_from_utf8(text);
  CHECK(utf8_path(path) == text);
  CHECK(utf8_path(path.filename()) == "aji-v3-x.engine");
  CHECK(path.u8string() == std::u8string(u8"moteurs/日本 é/aji-v3-x.engine"));
}

TEST_CASE("onnx_source_key : emplacement de poids non ASCII, comme fs::u8path du fork",
          "[prepare][utf8][identity]") {
  // Valeur calculée par content_identity.h et model_identity.cpp:20-27 du fork
  // (118c272), avec fs::u8path pour l'emplacement.
  const test::TempDir dir;
  const std::string location = "poids/é 日本.bin";
  test::write_file(dir.path() / path_from_utf8(location), "poids É");
  const std::vector<std::string> locations{location};
  CHECK(onnx_source_key(test::as_bytes("graphe ONNX factice"), dir.path(), locations).value() ==
        "e6a98b327e65651e802f17f4f369dce23d081ecd208906c22d895d584bfd66a5");
}

TEST_CASE("ansi_safe_path : un chemin hors de la page ANSI prend son nom court (A05-11, A06-02)",
          "[prepare][utf8]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / fs::path(u8"ajp-ansi-Юля-テスト");
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path existing = dir / "model.onnx";
  {
    std::ofstream(existing) << "x";
  }
  const fs::path to_write = dir / "aji-v3-abc.engine";  // pas encore là : le nom reste, le dossier raccourcit
#ifdef _WIN32
  // La page ANSI du SYSTÈME (celle de trtexec), pas celle du processus : le manifeste UTF-8 d'ajp met CP_ACP
  // à 65001 quoi que dise le système (B34-01).
  DWORD system_page = 0;
  GetLocaleInfoEx(LOCALE_NAME_SYSTEM_DEFAULT, LOCALE_IDEFAULTANSICODEPAGE | LOCALE_RETURN_NUMBER,
                  reinterpret_cast<LPWSTR>(&system_page),
                  static_cast<int>(sizeof(system_page) / sizeof(wchar_t)));
  for (const fs::path& path : {existing, to_write}) {
    const auto safe = ajp::prepare::ansi_command_path(path);
    if (system_page == CP_UTF8) {
      // « Utiliser Unicode UTF-8 » : tout s'écrit, le chemin reste entier.
      REQUIRE(safe.has_value());
      CHECK(*safe == ajp::prepare::utf8_path(path));
      continue;
    }
    // Sans nom court sur ce volume (fsutil 8dot3name), aucune forme sûre : le refus se dit avant le
    // lancement.
    if (!safe) {
      WARN("pas de nom court 8.3 sur ce volume : forme sûre introuvable, dit par nullopt");
      CHECK(ajp::prepare::ansi_safe_path(path) == ajp::prepare::utf8_path(path));
      continue;
    }
    CHECK(std::ranges::all_of(*safe, [](char c) { return static_cast<unsigned char>(c) < 0x80; }));
    CHECK(fs::equivalent(ajp::prepare::path_from_utf8(*safe).parent_path(), dir));
    if (path == existing) {
      CHECK(fs::equivalent(ajp::prepare::path_from_utf8(*safe), existing));  // le fichier prend son nom court
    } else {
      CHECK(ajp::prepare::path_from_utf8(*safe).filename() == path.filename());
    }
  }
#else
  CHECK(ajp::prepare::ansi_command_path(existing) == ajp::prepare::utf8_path(existing));
  CHECK(ajp::prepare::ansi_safe_path(existing) == ajp::prepare::utf8_path(existing));
  CHECK(ajp::prepare::ansi_safe_path(to_write) == ajp::prepare::utf8_path(to_write));
#endif
  CHECK(ajp::prepare::ansi_safe_path(fs::path("C:/ascii/fichier.onnx")) == "C:/ascii/fichier.onnx");
  fs::remove_all(dir);
}

TEST_CASE("C04-07 : la page système illisible ne rend pas un chemin ASCII non représentable",
          "[prepare][utf8]") {
  using ajp::prepare::ansi_page_decides;
  // Page 0 : la lecture de la page système a échoué. L'ASCII passe, le reste n'est pas prouvé.
  CHECK(ansi_page_decides(0, u"C:/models/rife.onnx") == std::optional<bool>(true));
  CHECK(ansi_page_decides(0, u"C:/mod\u00e8les/rife.onnx") == std::optional<bool>(false));
  CHECK(ansi_page_decides(0, u"") == std::optional<bool>(true));
  // UTF-8 système : tout s'écrit.
  CHECK(ansi_page_decides(65001, u"C:/\u65e5\u672c/rife.onnx") == std::optional<bool>(true));
  // Une vraie page (1252) : seul Windows peut dire si le texte y tient.
  CHECK_FALSE(ansi_page_decides(1252, u"C:/mod\u00e8les").has_value());
}
