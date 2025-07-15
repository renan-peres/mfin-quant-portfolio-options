# Python base image from Docker Hub
FROM python:3.12.3

# Set up workspace
WORKDIR /workspace
COPY requirements.txt set_env_vars.py ./

# Update package lists, install system dependencies, and set up UV in a single layer
RUN apt-get update && apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    unixodbc-dev \
    ca-certificates \
    curl \
    apt-transport-https \
    lsb-release \
    gnupg \
    zsh \
    git \
    fonts-powerline && \
    curl https://install.duckdb.org | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Install UV package manager
    curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh && \
    # Install Oh My Zsh and Powerlevel10k theme
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k && \
    echo 'source ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k/powerlevel10k.zsh-theme' >> ~/.zshrc && \
    echo 'POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >> ~/.zshrc && \
    echo 'POWERLEVEL9K_LEFT_PROMPT_ELEMENTS=(dir vcs)' >> ~/.zshrc && \
    echo 'POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(status root_indicator background_jobs time)' >> ~/.zshrc && \
    # Add UV to path in zshrc
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.zshrc && \
    echo 'source /workspace/.venv/bin/activate' >> ~/.zshrc

# Set default shell to zsh
SHELL ["/bin/zsh", "-c"]

CMD ["zsh"]