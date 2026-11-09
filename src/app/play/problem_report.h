// « Signaler un problème » (A39-01, A39-02, A39-03, A38-01) : ce que le bouton copie pour une issue publique.
// Un bloc de faits (version, système, carte, pilote, préréglage, moteur, refus de TensorRT) tenu pendant la
// séance et placé en tête, donc jamais perdu quand la fin du journal est coupée ; la fin de la séance d'avant
// quand elle s'est mal terminée ; le dernier minidump nommé ; et un nettoyage des chemins, qui ne laisse que
// le nom du fichier : le nom d'utilisateur et les dossiers des vidéos ne partent pas.
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ajp::app::play {

// Retient un fait pour le rapport (« Carte », « Pilote »…). Un même nom remplace sa valeur, l'ordre
// d'arrivée est gardé. Utilisable de tout thread.
void note_report_fact(std::string_view name, std::string_view value);
// Le moteur retenu et ceux qui ont refusé l'appareil avant lui, avec leurs raisons (vide : aucun refus).
void note_engine_facts(std::string_view engine, std::string_view passed_over);
std::vector<std::pair<std::string, std::string>> report_facts();
void clear_report_facts();

// Le système en une ligne : « Arch Linux · noyau 6.12 », « Windows 10.0.26100 », « macOS (Darwin 24.1) ».
std::string os_description();

// text sans les dossiers : tout chemin absolu (Windows « D:\… », UNC, POSIX « /… », « ~/… ») est réduit à
// « …/<nom du fichier> ». Le reste de la ligne après le nom est gardé. home sert aux chemins qui seraient
// encore à son nom (le journal les a déjà remplacés par « ~ »).
std::string sanitize_report_paths(std::string_view text, std::string_view home);

// Les derniers max_bytes d'un fichier, coupés à une fin de ligne (la première ligne n'est jamais tronquée
// au milieu). Vide si le fichier ne s'ouvre pas.
std::string read_tail(const std::filesystem::path& file, std::size_t max_bytes);

// Le nom du .dmp le plus récent de dir ; vide s'il n'y en a pas.
std::string latest_dump_name(const std::filesystem::path& dir);

struct ReportParts {
  std::vector<std::pair<std::string, std::string>> facts;
  std::string previous_log;  // la fin d'ajp.log.1, seulement si la séance d'avant s'est mal terminée
  std::string current_log;
  std::string dump_name;  // le dernier minidump du dossier crashes ; vide : aucun
  std::string home;
};

// Le texte à copier : faits, minidump, séance d'avant, séance en cours, le tout nettoyé et borné à max_bytes
// (un corps d'issue tient en 65 536 caractères) en rognant d'abord le début des journaux.
std::string compose_report(const ReportParts& parts, std::size_t max_bytes);

// --- La séance ouverte ---
// Une marque « session.open » posée au lancement et retirée à la sortie propre : trouvée au lancement
// suivant, la séance d'avant s'est arrêtée brutalement (plantage, coupure, arrêt forcé).
// Pose la marque dans dir ; vrai si elle y était déjà. Retient aussi dir pour end_session().
bool begin_session(const std::filesystem::path& dir);
void end_session();
// Le résultat de begin_session, relu pendant la séance.
bool previous_session_crashed();

// Le fichier du journal de cette séance (ajp.log, sous --data-dir quand il est donné) : le bouton le lit
// là, et son voisin « .1 » est la séance d'avant.
void set_report_log_file(const std::filesystem::path& file);
std::filesystem::path report_log_file();

// Le rapport complet pour ce lancement : lit les journaux et le dossier des minidumps.
std::string current_report(std::string_view home);

}  // namespace ajp::app::play
