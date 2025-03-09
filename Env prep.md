# Env prep
- HHKB
![](Env%20prep/%E6%88%AA%E5%B1%8F2025-02-19%20%E4%B8%8B%E5%8D%8811.26.55.png)
[hhkb_pro_jp_keylayout.pdf](Env%20prep/hhkb_pro_jp_keylayout.pdf)<!-- {"embed":"true", "preview":"true"} -->
- [‎Skitch - Snap. Mark up. Share.](https://apps.apple.com/us/app/skitch-snap-mark-up-share/id425955336?mt=12)
- history check fzf
  - [GitHub - unixorn/fzf-zsh-plugin: ZSH plugin to enable fzf searches of a lot more stuff - docker, tmux, homebrew and more.](https://github.com/unixorn/fzf-zsh-plugin)
  - https://github.com/jandamm/zgenom
  - https://github.com/zsh-users/antigen
  - https://github.com/junegunn/fzf/issues/1180
  - https://www.reddit.com/r/zsh/comments/y8j4af/fzfzshsource13/
```sh
# Add wisely, as too many plugins slow down shell startup.

git clone --depth 1 https://github.com/unixorn/fzf-zsh-plugin.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/fzf-zsh-plugin

plugins=(
	git
	zsh-autosuggestions
	zsh-syntax-highlighting
	fzf-zsh-plugin
)

source $ZSH/oh-my-zsh.sh
source ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source "${HOME}/.zgenom/zgenom.zsh"
#
```
## Iterminal
- https://zhuanlan.zhihu.com/p/364582751
```shell
brew install zsh
chsh -s /usr/local/bin/zsh

```
- Oh my zsh
```shell
export REMOTE=https://gitee.com/imirror/ohmyzsh.git
sh -c "$(wget -O- https://cdn.jsdelivr.net/gh/ohmyzsh/ohmyzsh/tools/install.sh)"
```
- Plugins
```
git clone https://gitee.com/imirror/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

git clone https://gitee.com/imirror/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

```
- .zshrc
```shell
ZSH_THEME="agnoster"

plugins=(git zsh-autosuggestions zsh-syntax-highlighting)

```
## Nvim
- [craftzdog/dotfiles-public: My personal dotfiles](https://github.com/craftzdog/dotfiles-public?tab=readme-ov-file)
- [Effective Neovim setup for web development towards 2024](https://www.devas.life/effective-neovim-setup-for-web-development-towards-2024/)
### Steps
- [neovim/INSTALL.md at master · neovim/neovim](https://github.com/neovim/neovim/blob/master/INSTALL.md)
- [🛠️ Installation](https://www.lazyvim.org/installation)
- https://www.nerdfonts.com/
- https://github.com/junegunn/fzf?tab=readme-ov-file#installation
- [使い込んで厳選したNeovimプラグインたちをご紹介します](https://zenn.dev/lighttiger2505/articles/6ff89ea53a10ac)
```shell
## nvim
brew install neovim

## LazyVim
# required
mv ~/.config/nvim{,.bak}

# optional but recommended
mv ~/.local/share/nvim{,.bak}
mv ~/.local/state/nvim{,.bak}
mv ~/.cache/nvim{,.bak}

git clone https://github.com/LazyVim/starter ~/.config/nvim
rm -rf ~/.config/nvim/.git

## Install optional tools
brew install ripgrep fd
brew install fzf

## Open an example project
cd ~/Developments/tutorials/
git clone https://github.com/craftzdog/craftzdog-homepage.git
cd craftzdog-homepage
vim package.json

## Theme - Solarized Osaka
# Create lua/plugins/colorscheme.lua
return {
    'craftzdog/solarized-osaka.nvim',
    lazy = false,
    priority = 1000,
    config = function()
      require("solarized-osaka").setup({
        styles = {
            floats = "transparent"
        },
      })
      vim.cmd[[colorscheme solarized-osaka]]
    end
}

## Cpp 
# https://github.com/williamboman/mason-lspconfig.nvim?tab=readme-ov-file

# prep lsp.lua tree.lua et.al

## put stdc++ into the llv folder: 
:MasonInstall clangd
:LspInstall clangd

# .zshrc
export C_INCLUDE_PATH=/usr/local/include
export CPLUS_INCLUDE_PATH=/usr/local/include
```
### Tips
- Press **Space L** and **U** to update the plugins.
- Press **Space f f** to launch telescope to search files.
- 