# Réplique locale de .github/workflows/release-linux.yml (runner ubuntu-24.04), pour construire
# le paquet Linux avant de pousser sur release/linux. Voir run.sh à côté.
#
# Étages : base (paquets du workflow, g++-13, CMake 3.31 comme le runner), deps
# (tools/deps/linux-from-source.sh, en cache tant que le script ne change pas), openvino (roue
# PyPI), rocm (migraphx-dev et hip-dev du dépôt ROCm 6.4.1, comme le workflow), tests (Vulkan
# logiciel et Xvfb comme le job player de ci.yml, sur l'étage TEST_BASE : rocm ou openvino).
ARG TEST_BASE=rocm
FROM ubuntu:24.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
# Ce que l'image du runner fournit déjà (gcc-13, git, curl, file, pip) puis la liste du workflow.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl gpg git file xz-utils unzip sudo make \
      gcc g++ gcc-13 g++-13 python3 python3-pip python3-venv \
      ninja-build nasm autoconf automake libtool pkg-config \
      python3-jinja2 patchelf p7zip-full libvulkan-dev glslc glslang-tools spirv-tools \
      libfreetype-dev libharfbuzz-dev libfribidi-dev libfontconfig-dev libshaderc-dev glslang-dev \
      liblcms2-dev libdav1d-dev libasound2-dev libpulse-dev libcurl4-openssl-dev \
      libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxss-dev libxtst-dev \
      libxfixes-dev libxinerama-dev libxkbcommon-dev libxkbcommon-x11-dev libwayland-dev \
      wayland-protocols libdecor-0-dev libdrm-dev libgbm-dev libegl-dev libudev-dev libdbus-1-dev \
      libva-dev \
 && rm -rf /var/lib/apt/lists/*
# Le runner : utilisateur runner, pip --user autorisé, CMake 3.31.
RUN usermod -l runner -d /home/runner -m ubuntu && groupmod -n runner ubuntu \
 && echo 'runner ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/runner
USER runner
WORKDIR /home/runner
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PATH=/home/runner/.local/bin:$PATH CC=gcc-13 CXX=g++-13 \
    AJP_DEPS_PREFIX=/home/runner/ajp-deps
RUN pip install --user -q meson "cmake~=3.31.0"

FROM base AS deps
COPY --chown=runner tools/deps/linux-from-source.sh /home/runner/linux-from-source.sh
RUN bash /home/runner/linux-from-source.sh "$AJP_DEPS_PREFIX" /home/runner/deps-src \
 && rm -rf /home/runner/deps-src

FROM deps AS openvino
ARG OPENVINO_VERSION=2026.4.1
RUN pip install --user -q openvino==${OPENVINO_VERSION}

FROM openvino AS rocm
ARG ROCM_VERSION=6.4.1
RUN sudo mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
      | sudo tee /etc/apt/sources.list.d/rocm.list \
 && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' | sudo tee /etc/apt/preferences.d/rocm-pin-600 \
 && sudo apt-get update -qq \
 && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends migraphx-dev hip-dev \
 && sudo rm -rf /var/lib/apt/lists/*
ENV CMAKE_PREFIX_PATH=/opt/rocm

FROM ${TEST_BASE} AS tests
RUN sudo apt-get update -qq \
 && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      mesa-vulkan-drivers vulkan-tools xvfb xauth \
 && sudo rm -rf /var/lib/apt/lists/*
