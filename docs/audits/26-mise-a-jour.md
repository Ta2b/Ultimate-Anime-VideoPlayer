# Audit 26. Mise à jour automatique et signature

**Audit par zone. Priorité : haute.** 7 fichiers, 1 483 lignes ; tests rattachés : 4 fichiers, 688 lignes.

## Mission

La vérification, le téléchargement, la vérification de signature, l'application et le retour arrière des mises à jour.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Clé de secours et révocation (SEC-01).
- Garde de nouvelle version : ce qui compte comme un échec (SEC-06).
- Application atomique, coupure de courant, fichier tenu sous Windows (SEC-07).
- Accès concurrent à `update.toml` (SEC-09).
- Chemins tirés du manifeste et de `.ajp-files` (SEC-10, SEC-11).
- Manifeste introuvable affiché comme « à jour » (SEC-05).
- À réduire fortement si le chantier 00 passe à une simple notification.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (7 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (3 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (1 inclusion)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (1 inclusion)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)

Utilisé par :

- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| SEC-01 | Bloquant | Aucune clé de secours n'est compilée et la clé unique reste sur un PC personnel ; la révocation prévue par P1.14 est ignorée sans erreur. Un vol de clé donne l'exécution de code sur tout le parc. |
| SEC-05 | Majeur | Un 404 sur le manifeste veut dire « à jour ». Si « Latest » désigne une release sans manifeste (`models-interp-1`), tout le parc se croit à jour, sans erreur. |
| SEC-06 | Majeur | La garde confirme une version dès une sortie propre, même sans lecture. Une sortie sur erreur fatale ou une mort avant la garde ne compte jamais : une release cassée reste installée. |
| SEC-07 | Majeur | L'application copie fichier par fichier. Un fichier tenu ou une coupure laisse un mélange de deux versions et peut écraser la sauvegarde `previous`. |
| SEC-10 | Mineur | La version et le nom d'archive tirés du manifeste signé ne sont pas assainis. Une clé compromise permet d'écrire hors de `updates/`. |
| SEC-11 | Mineur | Sous Windows, `\Users\x\…` et `D:x` passent la garde de `.ajp-files`. Une liste altérée fait supprimer des fichiers hors de l'installation. |
| SEC-12 | Mineur | Sans `staged_sha256`, l'archive est extraite sans vérification. Elle peut aussi changer entre la vérification et l'extraction. |
| SEC-13 | Mineur | curl lit `.curlrc`, la taille du manifeste n'est pas bornée, et la vérification n'a ni proxy ni reprises. Derrière un proxy d'entreprise, elle échoue (voir WIN-05). |
| SEC-22 | Mineur | Les tests de révocation et de bout en bout ne testent pas ce qu'ils annoncent. L'application n'est jamais testée sous Windows. |
| SEC-23 | Mineur | `path::string()` reste dans l'application de la mise à jour, contre P5.12. Sous un chemin non ANSI, l'exception relance la copie à chaque lancement. |
| SEC-24 | Mineur | Sans re-signature en 120 jours, tout le parc affiche un échec. Un `expires` lointain est accepté sans borne. |
| SEC-26 | Mineur | L'installeur Inno Setup ignore les mises à jour : version affichée figée, fichiers ajoutés laissés à la désinstallation. `updates/previous` n'est jamais purgé. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/ed25519.cpp` | 346 |
| `src/app/play/ed25519.h` | 27 |
| `src/app/play/update_choice.h` | 39 |
| `src/app/play/updater.cpp` | 676 |
| `src/app/play/updater.h` | 158 |
| `src/core/update_manifest.cpp` | 168 |
| `src/core/update_manifest.h` | 69 |

## Tests rattachés

- `tests/unit/app/ed25519_test.cpp`
- `tests/unit/app/updater_test.cpp`
- `tests/unit/app/wycheproof_ed25519.h`
- `tests/unit/core/update_manifest_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
