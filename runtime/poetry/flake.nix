{
  description = "DiffDisMatter-poetry";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    myPackages = {
      url = "git+ssh://git@github.com/Jonas-Finkler/nix-packages.git";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = { self, nixpkgs, flake-utils, myPackages, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system: 
      let 
        pkgs = import nixpkgs { 
          inherit system; 
          overlays = myPackages.overlays;
        };

        poetry2nix' = poetry2nix.lib.mkPoetry2Nix {
          inherit pkgs;
        };
        
        poetryEnv = poetry2nix'.mkPoetryEnv {
          python = pkgs.python311;
          projectDir = ./.;
          preferWheels = true; # avoid building from source
          editablePackageSources = {};
          overrides = poetry2nix'.defaultPoetryOverrides.extend (final: prev: {
            kimpy = prev.kimpy.overridePythonAttrs ( old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ 
                prev.setuptools
                prev.pybind11
                pkgs.kim-api
              ];
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ 
                pkgs.buildPackages.pkg-config
              ];
            });
            torch-scatter = prev.torch-scatter.overridePythonAttrs ( old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ 
                prev.setuptools
              ];
              propagatedBuildInputs = (old.buildInputs or [ ]) ++ [ 
                prev.torch
              ];
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ 
                pkgs.which
              ];
            });
            # sqnm = prev.sqnm.overridePythonAttrs ( old: {
            #   buildInputs = (old.buildInputs or [ ]) ++ [ 
            #     prev.setuptools
            #   ];
            # });
          });
        };

        devPkgs = with pkgs; [
          # poetry
          poetryEnv
          kim-api
        ];
        
      in {
    
        devShells.default = pkgs.mkShell {

          buildInputs = devPkgs;

          shellHook = ''
            export FLAKE="DiffDisMatter-poetry"
            # save dir
            export SAVE_ROOT_DIR='/home/jonas/sync/work/aau/code/DiffDisMatter/save_dir';
            # OMP
            export OMP_NUM_THREADS=8
            # back to zsh
            exec zsh
          '';

        };

        packages.default = pkgs.singularity-tools.buildImage {
          name = "DiffDisMatter";
          contents = devPkgs ++ (with pkgs; [
            # coreutils-full # provides ls, cat, ...
          ]);

          # the shadowSetup creates passwd and group files to prevent singularity from complaining
          runAsRoot = ''
            #!${pkgs.stdenv.shell}
            ${pkgs.dockerTools.shadowSetup} 
          '';
          # drop into shell by default
          runScript = ''
            #!${pkgs.stdenv.shell}
            exec /bin/sh $@"
          '';
          diskSize = 1024 * 40;
          memSize = 1024 * 8;
        };
      }
    );
}
