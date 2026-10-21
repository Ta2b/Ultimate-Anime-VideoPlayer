# Architecture

Un seul cœur possède le profil, la politique et l'ordonnancement. Il résout un
`Plan` immuable ; les autres modules exécutent ce `Plan` et rien d'autre.

## Modules

Un dossier de `src/` = un module = une cible CMake `ajp::<nom>`, déclarée par
`ajp_add_module` avec ses dépendances. Le graphe est acyclique : `app` dépend de
tout, rien ne dépend de `app`. `kernels` est une feuille ; `media` et `subs` ne
dépendent que de `core_types`.

| Module | Responsabilité | Thread |
|---|---|---|
| `core` | Profil TOML v5, `resolve()`, `Plan`, choix des moteurs, `CostTable`, `Capabilities` | main, puis valeurs immuables |
| `core_types` | `types.h`, `inbox.h` : vocabulaire partagé, sans dépendance | aucun |
| `media` | Démultiplexage, décodage matériel, horodatage | source |
| `gpu` | Exécution du `Plan`, ring de sortie, backends chargés à l'exécution | gpu |
| `prepare` | Construction des moteurs, reçus de qualification | preparer |
| `vendor_hdr` | Interface `TrueHdr` : RTX Video SDK, Maxine VFX | appelé par gpu |
| `audio` | Décodage audio, sortie SDL3, horloge | audio |
| `subs` | Rendu libass en atlas bornés | subs |
| `render` | Composition libplacebo, grade, sous-titres, OSD | main |
| `present` | Swapchain, cadence, détection de l'écran | main |
| `export` | Encodage, mux, métadonnées HDR10 | export |
| `app` | Boucle SDL3, commandes, interface, threads | main |
| `kernels` | Spécification des noyaux, référence CPU FP32 (tests seulement) | aucun |

Chaque thread n'attend que sa boîte `Inbox`, bornée par type de message.

## Moteurs

Seul notre moteur Vulkan est livré partout (Metal en plus sous macOS). Les autres
sont des paquets facultatifs : l'écran « Moteurs » du premier lancement les
présélectionne selon le GPU détecté (`core/engines.h`), avec des badges
(Recommandé, NVIDIA uniquement, Compatibilité…). On y revient depuis les
paramètres. Un moteur n'est retiré que s'il n'a plus d'intérêt nulle part.

## Règles de code

- C++23 pour l'hôte, avec `expected`, `format`, `print`, `span`, `ranges`,
  `jthread` et initialiseurs désignés. Modules C++, `generator`, `stacktrace`,
  `flat_map`, `mdspan`, `move_only_function` interdits. `.cu` en C++20.
- Un fichier vise moins de 500 lignes ; la CI échoue au-delà de 800.
- Aucun `getenv` : toute option vient du profil ou de `DebugFlags`.
- Erreurs : `std::expected<T, Refusal>` ; une exception ne traverse jamais un
  module ni la boucle d'un thread. Une valeur hors plage est refusée, jamais
  corrigée en silence.
- Aucun global mutable ; chaque thread possède ses données.
- Journal : spdlog en `SPDLOG_USE_STD_FORMAT`, un canal par module.
- Tests : Catch2, un exécutable par module dans `tests/unit/<module>/`.
  Tests CPU sous ASan et UBSan, `Inbox` sous TSan.
- Code repris d'un dépôt existant : dépôt et commit d'origine en tête de fichier,
  et un test de parité avant sa première modification.

## Décisions

Les décisions d'architecture vivent dans `docs/adr/`, une par fichier.
