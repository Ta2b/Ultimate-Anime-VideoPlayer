# Signer une release (mises à jour du lecteur)

Le lecteur ne croit un manifeste de mise à jour que s'il porte une signature Ed25519 valide pour la clé publique
compilée dans `AJP_UPDATE_PUBKEY` (`src/app/CMakeLists.txt`).

- **Clé publique** : `c50eeb3e63c327a087487cd3f485e66fbbf8ef2340a0b90d7ec99d4c2d3f0d58` (valeur par défaut de `AJP_UPDATE_PUBKEY`).
- **Clé privée** : `~/.config/ajp-release/update-ed25519.pem` sur le PC de Tatoub, droits 600. Elle n'est **jamais** dans le
  dépôt, ni dans un message, ni dans la CI : aucun secret CI n'existe pour l'instant, **la signature des releases se fait
  donc localement sur le PC de Tatoub**.

## Procédure, pour chaque release

1. Construire et empaqueter (`tools/release/package-*.sh`), poser les archives (`ajp-linux-x64.tar.gz`,
   `ajp-windows-x64.zip`, `ajp-macos-arm64.zip` selon ce qui existe) dans un dossier, par exemple `~/ajp-work/release/v0.5.0`.
2. Signer, depuis ce PC :

   ```bash
   AJP_UPDATE_KEY=~/.config/ajp-release/update-ed25519.pem \
     tools/release/sign-manifest.sh 0.5.0 stable ~/ajp-work/release/v0.5.0 \
     https://github.com/Ta2b/ajp-releases/releases/download/v0.5.0 \
     https://github.com/Ta2b/ajp-releases/releases/tag/v0.5.0
   ```

   Le script écrit `manifest.toml` (version, canal, adresses, SHA-256, tailles) et `manifest.toml.sig` (signature en
   hexadécimal) à côté des archives.
3. Joindre `manifest.toml` et `manifest.toml.sig` **à la release**, avec les archives. La version « stable » lue par le
   lecteur est la dernière release du dépôt ; une préversion se publie sous le tag fixe `preview`
   (`.../releases/download/preview/manifest.toml`), canal `preview`.
4. Vérifier : lancer `ajp` d'une version plus ancienne, Paramètres > Général > Mises à jour > Vérifier maintenant.

## Changer ou perdre la clé

- Une nouvelle paire : `openssl genpkey -algorithm ed25519 -out update-ed25519.pem`, puis
  `openssl pkey -in update-ed25519.pem -pubout -outform DER | tail -c 32 | xxd -p -c 64` donne la clé publique à mettre
  dans `AJP_UPDATE_PUBKEY`. Les lecteurs déjà installés avec l'ancienne clé ne reconnaîtront plus les manifestes : il faut
  une release manuelle.
- Sauvegarder la clé privée (gestionnaire de mots de passe de Tatoub, disque chiffré) : sans elle, plus de mise à jour.

## Canal « Préversions » et passage en release normale

Le lecteur lit son manifeste à deux adresses (`src/app/play/updater.cpp`, `manifest_url`) :

| Canal (Général > Mises à jour) | Adresse | Champ `channel` du manifeste |
|---|---|---|
| Stable (défaut) | `https://github.com/<dépôt>/releases/latest/download/manifest.toml` | `stable` |
| Préversions | `https://github.com/<dépôt>/releases/download/preview/manifest.toml` | `preview` |

GitHub appelle « latest » la dernière release **qui n'est pas une pré-version** : tant que la dernière version est une pré-version
(c'est le cas de la 0.5.0), l'adresse « Stable » ne trouve rien (404 : « aucune mise à jour »). Deux façons de les servir :

1. **Canal Préversions** : une release `preview` du dépôt des releases porte le manifeste du canal `preview` (même version, mêmes
   archives, `channel = "preview"`, signé à part : le champ est signé avec le reste). Un utilisateur qui choisit « Préversions » reçoit alors la 0.5.0.
   ```bash
   sed 's/^channel = "stable"$/channel = "preview"/' manifest.toml > preview/manifest.toml
   openssl pkeyutl -sign -inkey ~/.config/ajp-release/update-ed25519.pem -rawin -in preview/manifest.toml -out sig.bin
   xxd -p -c 256 sig.bin > preview/manifest.toml.sig
   gh release create preview -R Ta2b/ajp-releases --prerelease --title "Canal préversions (manifeste des mises à jour)" \
     --notes "Manifeste signé du canal Préversions." preview/manifest.toml preview/manifest.toml.sig
   # les fois suivantes : gh release upload preview -R Ta2b/ajp-releases --clobber preview/manifest.toml preview/manifest.toml.sig
   ```
2. **Release normale** (quand la version est validée) : retirer la marque « pré-version » de la release
   (`gh release edit v0.5.0 -R Ta2b/ajp-releases --prerelease=false --latest`) ; son `manifest.toml` (canal `stable`) est alors celui de
   « latest » et le canal Stable fonctionne.

Le canal du manifeste est vérifié par le lecteur : un manifeste `stable` servi à l'adresse `preview` (ou l'inverse) est refusé.
