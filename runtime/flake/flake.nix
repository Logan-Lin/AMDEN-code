{
  description = "DiffDisMatter development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    myPackages = {
      url = "git+ssh://git@github.com/Jonas-Finkler/nix-packages.git";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  # nixConfig = {
  #   extra-substituters = [
  #     "https://cuda-maintainers.cachix.org"
  #   ];
  #   extra-trusted-public-keys = [
  #     "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
  #   ];
  # };

  outputs = { self, nixpkgs, nixpkgs-stable, flake-utils, myPackages}:
    flake-utils.lib.eachDefaultSystem (system: 
      let 
        pkgs = import nixpkgs { inherit system; };

        # NOTE: They don't work yet
        nixpkgs-patched = pkgs.applyPatches {
          name = "nixpkgs-patched";
          src = pkgs.path;
          patches = [ 
            # upgrades triton to 3.0.0
            (pkgs.fetchpatch {
              url = "https://patch-diff.githubusercontent.com/raw/NixOS/nixpkgs/pull/328247.diff";
              hash = "sha256-aphPF045n2OlN8ZhKJqAh0+SaC1Ms4GTMEo+vPyD5dc=";
            })
            # # the binary version
            # (pkgs.fetchpatch {
            #   url = "https://github.com/NixOS/nixpkgs/pull/329674.diff";
            #   hash = "sha256-DR8VWj428y4p8ea9JBF5fomBi0CpBuEVWcv70x/PtdQ=";
            # })

          ];
        };

        pkgs_c = with_cuda: (import nixpkgs { 
          inherit system;
          config = {
            allowUnfree = true; # because of cuda
            cudaSupport = with_cuda;
            # cudaCapabilities = [ "6.1" ]; # aare
            # cudaVersion = "12.6.0";
          };

          overlays = myPackages.packageOverlays ++ [
            (final: prev: {
              python311 = prev.python311.override {
                packageOverrides = python-final: python-prev: (
                    # NOTE: Python overrides don't compose (https://github.com/NixOS/nixpkgs/issues/44426)
                    # Therefore we enter the packages here like this
                    myPackages.pythonPackages final python-final
                  ) // {
                    # here, other packages could go

                    # NOTE: Using torch-bin for now to avoid long compile times
                    # torch = python-final.torch-bin;
                }; 
              };
            })
          ];
        });

        devPkgs = pkgs: (
          [
            (pkgs.python311.withPackages (ptPkgs: with ptPkgs; [
              numpy
              matplotlib

              # WARN: Neither torch/triton version currently works to compile 
              # Let's hope this will be fixed soon: https://github.com/NixOS/nixpkgs/pull/328247
              # With export TRITON_LIBCUDA_PATH="/lib/x86_64-linux-gnu/" it finds libcuda.so but then complains about compiler errors...
              # NOTE: This causes long compile times when cuda is enabled 
              torch 
              # the -bin version uses the upstream binaries and has cuda enabled 
              # torch-bin
              pyyaml
              ase
              scipy
              einops 
              tqdm
              torch-nl
              torch-geometric
              # torch-scatter
              # pytorch-lightning
              # pandas
              # scikit-learn
              # tables
              kimpy
            ]))
          ] ++ (with pkgs; [
              kim-api
          ])
        );
      in {

        packages = 
            let pkgs = pkgs_c true;
          in rec {
          default = singularityContainer;
          singularityContainer = pkgs.singularity-tools.buildImage {
            name = "DiffDisMatter";
            contents = devPkgs pkgs ++ (with pkgs; [
              coreutils-full # provides ls, cat, ...
              # glibc.bin # provides ldconfig needed for torch.compile (no help)
              # glibc
              # the compiler expects some cache file with shared objects, which do not exist in nix
              # maybe work around this once we actually need it or use an ubuntu singularity container
              # Does also not work
              # pkgs.cudaPackages.backendStdenv.cc
              # pkgs.cudaPackages.cuda_nvcc
            ]);
            # the shadowSetup creates passwd and group files to prevent singularity from complaining
            runAsRoot = ''
              #!${pkgs.stdenv.shell}
              ${pkgs.dockerTools.shadowSetup} 
            '';
            # drop into shell by default
            runScript = ''
              #!${pkgs.stdenv.shell}
              # export CC=${pkgs.cudaPackages.backendStdenv.cc}/bin/cc;
              # export CXX=${pkgs.cudaPackages.backendStdenv.cc}/bin/c++;
              exec /bin/sh $@"
            '';
            diskSize = 1024 * 40;
            memSize = 1024 * 8;
          };
        };
    
        devShells = let 
            mkDevShell = with_cuda: let
              pkgs = pkgs_c with_cuda;
            in pkgs.mkShell {

              buildInputs = devPkgs pkgs;

              shellHook = ''
                export FLAKE="DiffDisMatter"

                export SAVE_ROOT_DIR='/home/jonas/sync/work/aau/code/DiffDisMatter/save_dir';
                # OMP
                export OMP_NUM_THREADS=8
                # back to zsh
                exec zsh
              '';
            };
          in rec {
            default = withoutCuda;
            withCuda = mkDevShell true;
            withoutCuda = mkDevShell false;
        };
      }
    );
}
