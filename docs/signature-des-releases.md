# Publier et signer une release

Le lecteur ne croit un manifeste de mise à jour que s'il porte une signature Ed25519 valide pour la clé publique
compilée dans `AJP_UPDATE_PUBKEY` (`src/app/CMakeLists.txt`).

- **Clé publique** : `c09703350d91e8a6ebdb815bf27ec8b662e301c954aed52568da164258d45088` (valeur par défaut de `AJP_UPDATE_PUBKEY`), en place depuis le 10 novembre 2026 : l'ancienne clé privée n'a pas été retrouvée.
- **Clé privée** : dans le secret `AJP_UPDATE_KEY` de l'environnement protégé `release` du dépôt des sources (décision
  de Tatoub, 9 novembre 2026). Elle n'est **jamais** dans le dépôt ni dans un message. Une copie de sauvegarde reste
  chez Tatoub, chiffrée.

## Une release, de bout en bout

Une seule action : pousser l'étiquette `v<VERSION>` (ou `v<VERSION>-rc.N` pour une préversion), où `<VERSION>` est
celle de `project()` dans `CMakeLists.txt`. Aucune étiquette ne se pose sans la demande de Tatoub. Le workflow
`release.yml` fait le reste :

1. **Garde** (`release-guard.yml`) : le commit construit est celui de l'étiquette, l'étiquette dit la version de
   `CMakeLists.txt` (`tools/ci/check-version.sh <étiquette>`), et `ci.yml` est vert sur ce commit.
2. **Plateformes**, en parallèle (`release-linux.yml`, `release-windows.yml`, `release-macos.yml`) :
   - Linux et Windows construisent d'abord leur **paquet TensorRT** s'il manque pour l'empreinte de ses entrées
     (`tools/release/trt-fingerprint.sh`, `--windows`), et le publient dans la release fixe `trt-<empreinte>` ou
     `trt-win-<empreinte>`. Une release finale sans paquet TensorRT Linux ou Windows échoue ;
   - un paquet TensorRT publié est **immuable** : `attach-nvidia-*` refuse de réécrire une release `trt-*` qui a déjà
     son index (le SHA-256 du paquet est dans l'index signé des releases), et un seul paquet de chaque système se
     construit à la fois (`concurrency`) ; la liste blanche de `check-engine-archive.sh` n'admet dans l'archive que
     notre module et `trtexec` ;
   - avant de joindre quoi que ce soit, chacune **lance son paquet** tel que l'utilisateur le recevra
     (`smoke-linux.sh` : bibliothèques trouvées, `--help`, clip joué en Vulkan logiciel jusqu'à un instantané, AppImage ;
     `smoke-windows.ps1` : zip et installeur, `--help`, désinstallation) ; un échec arrête la release ;
   - chacune joint ses archives à la release de l'étiquette, qui reste une **préversion** pendant ce temps
     (`tools/release/attach-assets.sh`), et sort le SHA-256 de son archive comme sortie de job.
3. **Publier** (job final de `release.yml`, environnement `release`) :
   - `tools/release/check-built-sums.sh` compare les archives téléchargées aux SHA-256 sortis des jobs de
     construction : on ne signe que ce qui a été construit ;
   - `tools/release/check-release.sh` vérifie que les archives de Linux et de Windows sont là (celle de macOS, hors
     cible, est facultative : son échec ne bloque pas la publication), que les index de Linux et de Windows nomment
     TensorRT avec ses `[[vendor_file]]`, et que l'archive de chaque moteur existe avec la taille et le SHA-256 de
     l'index ;
   - l'index signé `engines-nvidia.lock` est écrit d'après les `engines.lock` embarqués dans les paquets (jamais un
     index déjà joint à l'étiquette), porte `tag = "<étiquette>"`, et `tools/release/check-nvidia-index.sh` prouve
     qu'il nomme les mêmes paquets TensorRT que ceux-ci ;
   - pour une release finale, `tools/release/sign-manifest.sh` écrit et signe `manifest.toml` (canal `stable`),
     `tools/release/verify-download.sh` le revérifie contre chaque archive, et les deux fichiers sont joints ;
   - la release devient finale et « latest », sauf si une release plus récente l'est déjà (relance sur une ancienne
     étiquette). Une préversion reste une préversion, sans manifeste.

Relancer une étiquette : Actions > Release > Run workflow, avec l'étiquette. Une plateforme seule se relance aussi à
la main (son workflow, avec l'étiquette) tant que la release est une préversion ; puis relancer Release pour publier.

Gardes à part : `vendor-watch.yml` (chaque semaine) relit la roue de NVIDIA que chaque index `nvidia-vendor` décrit
(`tools/release/check-vendor.py`) ; `resign-manifest.yml` (chaque semaine) re-signe le manifeste de la « latest » quand
il lui reste moins de 45 jours. Un échec de l'un ou l'autre ouvre une issue dans ce dépôt.

## Réglages du dépôt, une fois

Ce qui suit se fait sur GitHub, sans le PC :

1. **Environnement `release`** (Settings > Environments > New environment) : règle de déploiement « Selected branches
   and tags », avec le motif d'étiquette `v*` et la branche `main` (pour la re-signature mensuelle).
2. **Secret `AJP_UPDATE_KEY`** dans cet environnement : la clé privée sur une seule ligne, ce qui ne s'abîme pas au
   collage. Dans Git Bash ou un terminal : `base64 -w0 update-ed25519.pem` (sous Windows, `| clip` la copie), puis
   coller tel quel. Le fichier PEM entier (`-----BEGIN PRIVATE KEY-----` … `-----END PRIVATE KEY-----`) est accepté
   aussi. `tools/release/update-key.sh` lit le secret, accepte ces formes, et vérifie que la clé correspond à
   `AJP_UPDATE_PUBKEY` ; en cas d'échec, le journal dit ce qu'il a reçu (longueur, lignes, en-tête) sans afficher la
   clé.
3. **Secret `AJP_RELEASE_TOKEN`** (déjà en place) : un jeton qui peut écrire les releases du dépôt
   `AJP_RELEASE_REPO` de `tools/release/versions.env`.

Le dépôt des releases est écrit à un seul endroit, `tools/release/versions.env` (`AJP_RELEASE_REPO`) : les workflows,
les scripts de paquet et CMake le lisent.

**Expiration.** Le manifeste porte `expires`, 120 jours après sa signature : au-delà, le lecteur le refuse, pour
qu'un ancien manifeste signé ne se rejoue pas. Le workflow `resign-manifest.yml` re-signe chaque mois celui de la
« latest » quand il lui reste moins de 45 jours ; il se lance aussi à la main.

## Signer à la main (secours)

Si la CI ne peut pas signer, depuis un poste qui a la clé :

```bash
gh release download v0.11.0 -R Ta2b/ajp-releases -D rel -p 'ajp-*'
AJP_UPDATE_KEY=update-ed25519.pem tools/release/sign-manifest.sh 0.11.0 stable rel \
  https://github.com/Ta2b/ajp-releases/releases/download/v0.11.0 \
  https://github.com/Ta2b/ajp-releases/releases/tag/v0.11.0
gh release upload v0.11.0 -R Ta2b/ajp-releases --clobber rel/manifest.toml rel/manifest.toml.sig
```

## Changer ou perdre la clé

- Une nouvelle paire : `openssl genpkey -algorithm ed25519 -out update-ed25519.pem`, puis
  `openssl pkey -in update-ed25519.pem -pubout -outform DER | tail -c 32 | xxd -p -c 64` donne la clé publique à mettre
  dans `AJP_UPDATE_PUBKEY`. Les lecteurs déjà installés avec l'ancienne clé ne reconnaîtront plus les manifestes : il faut
  une release manuelle.
- Sauvegarder la clé privée (copie chiffrée chez Tatoub) : sans elle, plus de mise à jour. Une nouvelle clé active
  remplace le secret `AJP_UPDATE_KEY`.

### Clé de secours et révocation

Le lecteur croit deux clés : l'active (`AJP_UPDATE_PUBKEY`) et celle de secours (`AJP_UPDATE_PUBKEY_BACKUP`, vide
tant qu'elle n'est pas créée). Créer la paire de secours comme ci-dessus, compiler sa clé publique dans
`AJP_UPDATE_PUBKEY_BACKUP`, et garder la clé privée **hors ligne**, chez Tatoub, en sauvegarde chiffrée (décision du
9 novembre 2026) : jamais dans la CI, pour qu'un vol du secret `AJP_UPDATE_KEY` reste révocable.

Si l'active est perdue ou volée : signer le manifeste suivant avec la clé de secours
(`AJP_UPDATE_KEY=<secours>.pem tools/release/sign-manifest.sh …`) et y ajouter la ligne
`revoked_keys = ["<clé publique active>"]`. Les lecteurs le croient, retiennent la révocation dans `update.toml`, et ne
croient plus aucun manifeste signé par l'ancienne clé. La release suivante compile une nouvelle clé de secours.

Une clé ne révoque que celles qui la précèdent dans la liste « active, secours » (B26-02) : un manifeste signé par
l'active qui révoquerait la clé de secours est cru pour le reste, mais cette révocation-là est ignorée ; une clé qui se
révoque elle-même est refusée. Sans cela, qui tiendrait l'active pourrait neutraliser le secours. Le manifeste ne sert
que de ce qu'il porte (`version`, `expires`, `notes_url`, `revoked_keys`) : ses sections d'archive sont ignorées par le
lecteur. L'index des moteurs de la release (`engines-nvidia.lock`) porte `tag = "v<version>"` et n'est cru que dans la
release de cette étiquette (B51-06) ; `tools/release/nvidia-index.sh` l'écrit avant la signature.

## Pas de canal Préversions

Le canal Préversions est retiré (décision de Tatoub, 9 novembre 2026) : une préversion se publie sans manifeste, et
le lecteur ne lit que `https://github.com/<dépôt>/releases/latest/download/manifest.toml`. GitHub appelle « latest » la
dernière release qui n'est pas une préversion.

## Signature des exécutables et des archives (P7.1)

Les workflows de release savent signer ; rien ne change tant que les comptes n'existent pas (un avis
« signature ad hoc » ou « non signés » dans le journal de la CI). Décision de Tatoub pour la 0.10 : aucun compte
acheté pour l'instant ; les releases restent non signées sous Windows et macOS.

### Linux : vérifier une archive, sans compte

Le manifeste signé porte le SHA-256 et la taille de chaque archive : il sert de signature aux archives, sans
outil de plus. Pour vérifier un téléchargement :

```bash
tools/release/verify-download.sh manifest.toml manifest.toml.sig ajp-linux-x64.tar.gz
```

Le script vérifie la signature Ed25519 du manifeste pour la clé publique des releases, puis l'empreinte de
l'archive. Il ne demande qu'`openssl` et `sha256sum`.

### Windows : Azure Trusted Signing (environ 10 $ par mois)

`tools/release/windows/sign.ps1` signe `ajp.exe` et ses DLL avant le zip, puis l'installeur. Pour l'activer :

1. Créer un compte Trusted Signing et un profil de certificat « Public Trust » (validation d'identité par
   Microsoft, quelques jours).
2. Créer une application Entra ID avec un secret, et lui donner le rôle « Trusted Signing Certificate Profile
   Signer » sur le profil.
3. Dans les réglages du dépôt :
   - variables `AJP_TS_ENDPOINT` (`https://<région>.codesigning.azure.net`), `AJP_TS_ACCOUNT` et `AJP_TS_PROFILE` ;
   - secrets `AJP_AZURE_TENANT_ID`, `AJP_AZURE_CLIENT_ID` et `AJP_AZURE_CLIENT_SECRET`.

La signature est horodatée : elle reste valide après l'expiration du certificat.

### macOS : Developer ID et notarisation (99 $ par an)

`tools/release/package-macos.sh` signe avec le runtime durci, notarise l'app et l'image disque, puis y agrafe
le ticket. Pour l'activer, avec un compte Apple Developer :

1. Créer un certificat « Developer ID Application » et l'exporter en `.p12` avec un mot de passe.
2. Créer une clé d'API App Store Connect (rôle Developer) et télécharger son fichier `.p8`.
3. Créer ces secrets du dépôt :
   - `AJP_MACOS_CERT_P12` (le `.p12` en base64) et `AJP_MACOS_CERT_PASSWORD` ;
   - `AJP_MACOS_IDENTITY` (« Developer ID Application: Nom (ÉQUIPE) ») ;
   - `AJP_NOTARY_KEY_P8` (le contenu du `.p8`), `AJP_NOTARY_KEY_ID` et `AJP_NOTARY_ISSUER`.

Sans les trois derniers, l'app est signée mais pas notarisée : Gatekeeper demande encore une confirmation.
