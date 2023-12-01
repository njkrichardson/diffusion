# syntax=docker/dockerfile:1

FROM ubuntu:22.04

RUN apt-get update && apt-get install --yes \
    build-essential \
    gdb \
    zsh \
    wget \
    curl \
    git \
    ninja-build \ 
    gettext \
    cmake \
    unzip \
    parallel \ 
    npm \
    python3-venv \
    vim 

# configure zsh 
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t af-magic \
    -p git \
    -p vi-mode 

RUN cd / && git clone https://github.com/njkrichardson/dots.git && parallel cp dots/.zshrc ::: /.zshrc ~/.zshrc && echo "export XDG_CONFIG_HOME=/.config" >> /.zshrc && echo "export XDG_CONFIG_HOME=/.config" >> ~/.zshrc

# configure neovim 
RUN git clone https://github.com/neovim/neovim && cd neovim && git checkout stable && make CMAKE_BUILD_TYPE=RelWithDebInfo && make install 
RUN mkdir /.config && cd / && git clone https://github.com/njkrichardson/nvimconfig.git && cp -r nvimconfig /.config/nvim 
RUN git clone --depth 1 https://github.com/wbthomason/packer.nvim ~/.local/share/nvim/site/pack/packer/start/packer.nvim

CMD ["/bin/zsh"]
