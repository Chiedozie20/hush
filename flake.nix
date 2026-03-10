{
  description = "ADLS project dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-stable,
    }:
    let
      devShellFor =
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          pkgs-stable = import nixpkgs-stable {
            inherit system;
            config.allowUnfree = true;
          };
          isLinux = pkgs.stdenv.isLinux;
        in
        pkgs.mkShell {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
            [
              pkgs.stdenv.cc.cc
              pkgs.zlib
              pkgs.glib
              pkgs.libGL
              pkgs.libxml2
              pkgs.ffmpeg
            ]
            ++ pkgs.lib.optionals isLinux [
              pkgs.cudaPackages.cudatoolkit
            ]
          );

          buildInputs = [
            pkgs.python311
            pkgs.git
            pkgs.gcc
            pkgs.gnumake
            pkgs.cmake
            pkgs.verilator
            pkgs.graphviz
            pkgs.surfer
            pkgs.ruff
            pkgs.just
            pkgs.zlib
            pkgs.glib
            pkgs.mesa
            pkgs.ffmpeg
          ]
          ++ pkgs.lib.optionals isLinux [
            pkgs-stable.verible
          ];

          shellHook = ''
            export VIRTUAL_ENV_DISABLE_PROMPT=1
            if [ ! -d ".venv" ]; then
              ${pkgs.python311}/bin/python -m venv .venv
              source .venv/bin/activate
              pip install --upgrade pip
              ${
                if isLinux then
                  "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
                else
                  "pip install torch torchvision torchaudio"
              }
              if [ -f requirements.txt ]; then
                pip install -r requirements.txt
              fi
              if [ -n "$MASE_PATH" ]; then
                pip install -e "$MASE_PATH"
              fi
              touch .venv/completed_init
            else
              source .venv/bin/activate
              if [ ! -f ".venv/completed_init" ]; then
                ${
                  if isLinux then
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
                  else
                    "pip install torch torchvision torchaudio"
                }
                if [ -f requirements.txt ]; then
                  pip install -r requirements.txt
                fi
                if [ -n "$MASE_PATH" ]; then
                  pip install -e "$MASE_PATH"
                fi
                touch .venv/completed_init
              fi
            fi
            ${if isLinux then "export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH" else ""}
          '';
        };
    in
    {
      devShells.x86_64-linux.default = devShellFor "x86_64-linux";
      devShells.aarch64-linux.default = devShellFor "aarch64-linux";
      devShells.x86_64-darwin.default = devShellFor "x86_64-darwin";
      devShells.aarch64-darwin.default = devShellFor "aarch64-darwin";
    };
}
