// AVFrame vers ajp::DecodedFrame : FrameData tient une référence AVFrame, la
// dernière copie relâchée rend le tampon à son pool. Interne au module.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "core/frame.h"
#include "core/types.h"
#include "media/av_util.h"

namespace ajp::media {

// Événement GPU à attendre avant de lire l'image (CUevent du flux de décodage), avec ce
// qui le garde en vie. handle 0 : rien à attendre. Une image Vulkan porte son propre
// sémaphore timeline : to_decoded_frame le lit dans l'AVVkFrame.
struct ReadyEvent {
  std::uintptr_t handle = 0;
  std::uint64_t value = 0;  // Vulkan : valeur du sémaphore timeline ; 0 pour un CUevent
  std::shared_ptr<const void> owner;
};

// Image décodée, sans identité ni époque : le thread source les pose.
// pts_us et duration_us viennent du décodeur, qui comble les trous ; kNoPts si inconnu.
// Refus : appareil ou format hors de core/frame.h, image retournée, image Vulkan à
// plusieurs VkImage (une attente par image, hors du contrat de FrameData).
// full_host_formats (DecoderOptions) : une image hôte 4:4:4 sort en Yuv444p16 et une 4:2:0 de plus de 10 bits
// en P016, au lieu du 4:2:0 8 ou 10 bits.
std::expected<DecodedFrame, Refusal> to_decoded_frame(FramePtr frame, std::int64_t pts_us,
                                                      std::int64_t duration_us, ReadyEvent ready = {},
                                                      bool full_host_formats = false);

// AVFrame tenu par une image de ce module, nullptr pour un autre FrameData.
// Pour les tests et le diagnostic : le chemin de lecture s'en tient à FrameData.
const AVFrame* av_frame_of(const FrameData& data);

}  // namespace ajp::media
