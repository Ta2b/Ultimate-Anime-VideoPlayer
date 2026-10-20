# Réplique locale de .github/workflows/release-linux.yml (runner ubuntu-24.04), pour construire
# le paquet Linux et passer ses tests avant de pousser sur release/linux. Voir run.sh à côté.
#
# Étages : base (paquets de tools/release/versions.env, g++-13, CMake 3.31 comme le runner), deps
# (tools/deps/linux-from-source.sh, en cache tant que le script ne change pas), openvino (roue
# PyPI), rocm (migraphx-dev et hip-dev du dépôt ROCm, par tools/release/rocm-apt.sh), tests
# (Vulkan logiciel, couche de validation et Xvfb comme les tâches de tests de la CI, sur l'étage
# TEST_BASE : rocm ou openvino). Les versions et les listes de paquets viennent de versions.env,
# copié dans l'image : un changement y reconstruit les étages qui le lisent.
ARG TEST_BASE=rocm
FROM ubuntu:24.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
COPY tools/release/versions.env tools/release/rocm-apt.sh /ajp/
# Ce que l'image du runner fournit déjà (gcc-13, git, curl, file, pip) puis les listes du workflow.
RUN . /ajp/versions.env && apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl gpg git file xz-utils unzip sudo make \
      gcc g++ gcc-13 g++-13 python3 python3-pip python3-venv \
      $APT_BUILD $APT_PLAYER $APT_PACKAGE \
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
RUN . /ajp/versions.env && pip install --user -q "openvino==${OPENVINO_VERSION}"

FROM openvino AS rocm
RUN bash /ajp/rocm-apt.sh \
 && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends migraphx-dev hip-dev \
 && sudo rm -rf /var/lib/apt/lists/*
ENV CMAKE_PREFIX_PATH=/opt/rocm

FROM ${TEST_BASE} AS tests
RUN . /ajp/versions.env && sudo apt-get update -qq \
 && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $APT_TESTS \
 && sudo rm -rf /var/lib/apt/lists/*
