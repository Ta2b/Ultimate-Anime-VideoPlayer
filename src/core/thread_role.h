// La place d'un thread face à la lecture : qui passe devant quand le processeur est plein.
//
// Les threads de la lecture (rendu, calcul GPU, audio) doivent tenir leur cadence même quand un travail de
// fond (la pré-analyse des plans, qui décode tout le fichier) occupe tous les cœurs. Chaque thread déclare
// son rôle au début de son corps ; le système en fait ce qu'il sait faire :
//   - Windows : MMCSS (« Playback », « Audio ») pour la lecture, mode de fond (processeur, E/S et mémoire)
//     pour le fond ; ni l'un ni l'autre ne passe aux threads créés ensuite, d'où un seul thread de décodage
//     pour la pré-analyse (media/shot_probe.cpp) ;
//   - Linux : nice 10 pour le fond (hérité par les threads qu'il crée ensuite, ceux de FFmpeg compris) ;
//     la lecture est relevée par un crochet que l'application installe (SDL_SetCurrentThreadPriority, qui
//     passe par rtkit quand le système l'a) ;
//   - macOS : classes de qualité de service (interactive, utilitaire).
// Jamais d'échec : un système qui refuse laisse le thread tel qu'il est.
#pragma once

namespace ajp {

enum class ThreadRole {
  Playback,    // rendu, calcul GPU de la lecture
  Audio,       // sortie audio
  Background,  // analyse, téléchargement : ne doit jamais retarder la lecture
};

// Relève la priorité du thread appelant ; rend vrai si le système l'a accordée. Le cœur ne dépend pas de SDL
// : l'application installe ce crochet après SDL_Init. Sans crochet, la lecture garde la priorité normale.
using PriorityRaiser = bool (*)() noexcept;
// Installe le crochet et relève aussitôt le thread appelant (le principal, dont le rôle était posé plus tôt).
// Rend vrai si le système l'a accordée.
bool set_thread_priority_raiser(PriorityRaiser raiser) noexcept;

// Applique le rôle au thread appelant, pour toute sa vie.
void set_thread_role(ThreadRole role) noexcept;

// Pose thread_namer (core/thread.h) : chaque start_thread donne ensuite son nom au système
// (pthread_setname_np, 15 octets sous Linux ; SetThreadDescription sous Windows), pour Nsight, gdb et les
// vidages de plantage (A20-11). À appeler une fois au démarrage, avant le premier thread.
void install_system_thread_names() noexcept;

}  // namespace ajp
