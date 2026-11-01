// Les octets des images d'exemple de la calibration du grade (data/calibration/*.jpg), embarqués par
// embed_text.cmake.
#pragma once

#include <string_view>

namespace ajp::app::play {

extern const std::string_view kCalibrationDark1;
extern const std::string_view kCalibrationDark2;
extern const std::string_view kCalibrationDark3;
extern const std::string_view kCalibrationDark4;
extern const std::string_view kCalibrationBright1;
extern const std::string_view kCalibrationBright2;
extern const std::string_view kCalibrationBright3;
extern const std::string_view kCalibrationBright4;

}  // namespace ajp::app::play
