#include "app/play/ed25519.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

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
