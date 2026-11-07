#include "app/play/ed25519.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "wycheproof_ed25519.h"

using namespace ajp::app::play;

namespace {

// Un vecteur produit par OpenSSL 3 (openssl pkeyutl -sign -rawin), message « ajp update manifest ».
constexpr const char* kKey = "023c519fd679f07cada015b990d1be35150f24fa582c7136aedcdde124a8ab47";
constexpr const char* kSig =
    "7d78ea645ec1ad57c74e3aec612b891802df3cfbab104ca0985f7754f34316d9617bd3d94521e84e3f48377e8ae76c9f65e08d70"
    "cc65fdeb74efc2d4336e5f01";

}  // namespace

TEST_CASE("Ed25519 : une signature d'OpenSSL se vérifie", "[ed25519]") {
  CHECK(ed25519_verify_hex(kKey, "ajp update manifest", kSig));
}

TEST_CASE("Ed25519 : RFC 8032, essai 1 (message vide)", "[ed25519]") {
  CHECK(ed25519_verify_hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
                           "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc"
                           "61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"));
}

TEST_CASE("Ed25519 : RFC 8032, essais 2 et 3 (un et deux octets)", "[ed25519]") {
  CHECK(
      ed25519_verify_hex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "\x72",
                         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458"
                         "f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"));
  CHECK(
      ed25519_verify_hex("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "\xaf\x82",
                         "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae6"
                         "7f760984dc6594a7c15e9716ed28dc027beceea1ec40a"));
}

TEST_CASE("Ed25519 : message, clé ou signature altérés sont refusés", "[ed25519]") {
  CHECK_FALSE(ed25519_verify_hex(kKey, "ajp update manifesT", kSig));
  CHECK_FALSE(ed25519_verify_hex(kKey, "", kSig));
  std::string bad_sig = kSig;
  bad_sig[10] = bad_sig[10] == '0' ? '1' : '0';
  CHECK_FALSE(ed25519_verify_hex(kKey, "ajp update manifest", bad_sig));
  std::string bad_key = kKey;
  bad_key[0] = '1';
  CHECK_FALSE(ed25519_verify_hex(bad_key, "ajp update manifest", kSig));
  // S non canonique (S + L) : refusé.
  CHECK_FALSE(ed25519_verify_hex(kKey, "ajp update manifest", std::string(128, 'f')));
}

TEST_CASE("Ed25519 : tailles et chiffres hexadécimaux faux", "[ed25519]") {
  CHECK_FALSE(ed25519_verify_hex("abcd", "x", kSig));
  CHECK_FALSE(ed25519_verify_hex(kKey, "x", "zz"));
  CHECK_FALSE(from_hex("abc"));
  CHECK(from_hex("00ff")->size() == 2);
}

// Un manifeste écrit et signé par tools/release/sign-manifest.sh (clé jetable) : le format se vérifie de bout
// en bout.
TEST_CASE("Ed25519 : le manifeste de sign-manifest.sh se vérifie et se lit", "[ed25519]") {
  const std::string manifest = R"MANIFEST(version = "0.5.0"
channel = "stable"
date = "2026-10-28"
notes_url = "https://github.com/Ta2b/ajp-releases/releases/tag/v0.5.0"
[linux-x64]
url = "https://github.com/Ta2b/ajp-releases/releases/download/v0.5.0/ajp-linux-x64.tar.gz"
sha256 = "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
size = 1
)MANIFEST";
  CHECK(ed25519_verify_hex("c438afee757ea9095f5e9d56380da968bfca8af4ff272c4773b51808bca6ce2d", manifest,
                           "6b7a50430e1ea3b88492ce9ec890270cce724b0488fe38dbec7a32eba55c0e8d0d3399bca608719c0"
                           "dd0aca127011750395b3463ceda2b4978ce52d548913400"));
  CHECK_FALSE(ed25519_verify_hex("c438afee757ea9095f5e9d56380da968bfca8af4ff272c4773b51808bca6ce2d",
                                 manifest + " ",
                                 "6b7a50430e1ea3b88492ce9ec890270cce724b0488fe38dbec7a32eba55c0e8d0d3399bca60"
                                 "8719c0dd0aca127011750395b3463ceda2b4978ce52d548913400"));
}

// Les 210 vecteurs de Wycheproof : points de petit ordre, S hors de [0, L), encodages non canoniques,
// signatures tronquées ou allongées. Un vecteur « acceptable » permet les deux réponses.
TEST_CASE("Ed25519 : vecteurs Wycheproof", "[ed25519]") {
  for (const wycheproof::Vector& v : wycheproof::kEd25519) {
    const auto message = from_hex(v.message);
    REQUIRE(message);
    const std::string text(message->begin(), message->end());
    const bool verified = ed25519_verify_hex(v.key, text, v.signature);
    INFO("tcId " << v.id);
    if (v.result >= 0) CHECK(verified == (v.result == 1));
  }
}
