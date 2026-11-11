#!/usr/bin/env bash
# Écrit la clé privée Ed25519 des mises à jour (secret AJP_UPDATE_KEY, lu dans la variable UPDATE_KEY) dans un
# fichier PEM que lisent sign-file.sh et sign-manifest.sh, puis vérifie qu'elle correspond à une clé publique
# compilée dans le lecteur (AJP_UPDATE_PUBKEY ou AJP_UPDATE_PUBKEY_BACKUP de src/app/CMakeLists.txt).
#
#   UPDATE_KEY=<secret> tools/release/update-key.sh <fichier.pem>
#
# Le secret se colle à la main dans les réglages de GitHub : un copier-coller abîme facilement un PEM sur plusieurs
# lignes. Sont donc acceptés, en plus du PEM tel quel :
#   - fins de ligne CRLF, espaces ou guillemets autour, « \n » écrits en toutes lettres ;
#   - le fichier PEM entier encodé en base64 sur une ligne (« base64 -w0 update-ed25519.pem ») ;
#   - le corps base64 seul, sans les lignes BEGIN et END ;
#   - la graine de 32 octets en hexadécimal (64 caractères).
# En cas d'échec, le journal dit ce qui a été reçu (longueur, nombre de lignes, en-tête) sans jamais afficher la clé.
set -euo pipefail

[ $# -eq 1 ] || { sed -n '2,13p' "$0"; exit 2; }
out=$1
root="$(cd "$(dirname "$0")/../.." && pwd)"
raw=${UPDATE_KEY:-}
[ -n "$raw" ] || { echo "::error::secret AJP_UPDATE_KEY absent ou vide (docs/signature-des-releases.md)"; exit 1; }

umask 077
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Nettoyage : CR, « \n » littéraux, guillemets et espaces autour.
text="$(printf '%s' "$raw" | tr -d '\r' | sed 's/\\n/\n/g')"
text="$(printf '%s\n' "$text" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//' | sed '/^$/d')"
compact="$(printf '%s' "$text" | tr -d '[:space:]')"

pem_ok() { openssl pkey -in "$1" -noout >/dev/null 2>&1; }
wrap_pem() {  # corps base64 -> PEM PKCS#8
  { echo "-----BEGIN PRIVATE KEY-----"; printf '%s' "$1" | fold -w 64; echo; echo "-----END PRIVATE KEY-----"; } \
    | sed '/^$/d' > "$2"
}

found=""
# 1. Un PEM (éventuellement sur une seule ligne, les sauts de ligne perdus au collage).
if printf '%s' "$text" | grep -q -- '-----BEGIN'; then
  header="$(printf '%s' "$text" | grep -o -- '-----BEGIN [A-Z ]*-----' | head -1)"
  case "$header" in
    *"OPENSSH PRIVATE KEY"*) echo "::error::AJP_UPDATE_KEY est une clé OpenSSH (ssh-keygen) : il faut le fichier de « openssl genpkey -algorithm ed25519 »"; exit 1 ;;
    *"ENCRYPTED PRIVATE KEY"*) echo "::error::AJP_UPDATE_KEY est chiffrée par un mot de passe : il faut la clé en clair (openssl genpkey sans -aes)"; exit 1 ;;
    *"PUBLIC KEY"*) echo "::error::AJP_UPDATE_KEY contient la clé PUBLIQUE : il faut la clé privée (BEGIN PRIVATE KEY)"; exit 1 ;;
  esac
  printf '%s\n' "$text" > "$work/key.pem"
  if pem_ok "$work/key.pem"; then
    found=pem
  else
    body="$(printf '%s' "$compact" | sed -e 's/-----BEGIN[A-Z]*-----//' -e 's/-----END[A-Z]*-----//' \
      -e 's/-----BEGINPRIVATEKEY-----//' -e 's/-----ENDPRIVATEKEY-----//')"
    wrap_pem "$body" "$work/key.pem"
    pem_ok "$work/key.pem" && found="PEM aux sauts de ligne perdus"
  fi
# 2. La graine en hexadécimal.
elif printf '%s' "$compact" | grep -Eq '^[0-9a-fA-F]{64}$'; then
  # Préfixe DER PKCS#8 d'une clé Ed25519, puis la graine ; sans xxd (absent de certaines images).
  # shellcheck disable=SC2059
  printf "$(printf '302e020100300506032b657004220420%s' "$compact" | sed 's/../\\x&/g')" > "$work/key.der"
  openssl pkey -inform DER -in "$work/key.der" -out "$work/key.pem" 2>/dev/null && found="graine hexadécimale"
# 3. Du base64 : le PEM entier encodé, ou le corps seul.
elif printf '%s' "$compact" | grep -Eq '^[A-Za-z0-9+/=]+$'; then
  if printf '%s' "$compact" | base64 -d > "$work/decoded" 2>/dev/null && grep -q -- '-----BEGIN PRIVATE KEY' "$work/decoded"; then
    tr -d '\r' < "$work/decoded" > "$work/key.pem"
    pem_ok "$work/key.pem" && found="PEM encodé en base64"
  else
    wrap_pem "$compact" "$work/key.pem"
    pem_ok "$work/key.pem" && found="corps base64 sans en-têtes"
  fi
fi

if [ -z "$found" ]; then
  lines="$(printf '%s\n' "$raw" | wc -l)"
  first="$(printf '%s\n' "$raw" | head -1 | tr -d '\r')"
  case "$first" in -----*) shown="$first" ;; *) shown="(pas d'en-tête -----BEGIN)" ;; esac
  cr=non; printf '%s' "$raw" | grep -q $'\r' && cr=oui
  echo "::error::AJP_UPDATE_KEY illisible : ${#raw} caractères, $lines ligne(s), CR : $cr, première ligne : $shown. Attendu : le fichier de « openssl genpkey -algorithm ed25519 » (BEGIN PRIVATE KEY), ou sa version « base64 -w0 » (docs/signature-des-releases.md)."
  exit 1
fi

# Seules les clés Ed25519 conviennent.
openssl pkey -in "$work/key.pem" -noout -text 2>/dev/null | grep -qi 'ED25519' \
  || { echo "::error::AJP_UPDATE_KEY n'est pas une clé Ed25519"; exit 1; }

# La clé doit être une de celles dont le lecteur connaît la clé publique : sinon il refuserait tout ce qu'elle
# signe. AJP_UPDATE_PUBKEY posée : celle-là seule ; sinon l'active ou celle de secours de src/app/CMakeLists.txt
# (la secours signe le manifeste qui révoque une active volée, sign-manifest.sh --revoke).
pub="$(openssl pkey -in "$work/key.pem" -pubout -outform DER | tail -c 32 | od -An -tx1 -v | tr -d ' \n')"
cmake_key() { sed -n "s/^set($1 \"\([0-9a-f]*\)\".*/\1/p" "$root/src/app/CMakeLists.txt"; }
if [ -n "${AJP_UPDATE_PUBKEY:-}" ]; then
  known=("$AJP_UPDATE_PUBKEY")
else
  known=("$(cmake_key AJP_UPDATE_PUBKEY)" "$(cmake_key AJP_UPDATE_PUBKEY_BACKUP)")
fi
role=""
for i in "${!known[@]}"; do
  if [ -n "${known[$i]}" ] && [ "$pub" = "${known[$i]}" ]; then
    if [ "$i" = 0 ]; then role=active; else role=secours; fi
  fi
done
if [ -z "$role" ]; then
  echo "::error::AJP_UPDATE_KEY ne correspond à aucune clé publique du lecteur : elle donne $pub, le lecteur attend ${known[*]} (AJP_UPDATE_PUBKEY, AJP_UPDATE_PUBKEY_BACKUP)"
  exit 1
fi

cp "$work/key.pem" "$out"
echo "clé de signature lue ($found), clé publique $pub ($role)"
