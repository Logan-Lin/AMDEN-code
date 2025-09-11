{
  description = "Lammps with bmp potential";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    myPackages = {
      url = "github:Jonas-Finkler/nix-packages";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, myPackages }:
    flake-utils.lib.eachDefaultSystem (system: 
      let 
        pkgs = import nixpkgs { 
          inherit system; 
          overlays = myPackages.overlays;
        };
        
        python = pkgs.python311;
      
        bmp = pkgs.callPackage ./bmp {};
        bmpFixed = pkgs.callPackage ./bmp {fixDisconsts = true;};

        setupenv = with pkgs; let
          script = writeShellScript "builder.sh" ''
            #!/bin/sh
            export LOCALE_ARCHIVE=${glibcLocales}/lib/locale/locale-archive
            export BMP=${bmp}
            export BMP_FIXED=${bmpFixed}
            echo $BMP
          '';

        in stdenv.mkDerivation {
          pname = "setupenv";
          version = "0.0.1";
          phases = ["buildPhase" "installPhase"];
          buildInputs = [ bmp ];
          nativeBuildInputs = [ makeWrapper ];
          installPhase = ''
            mkdir -p $out/bin
            cp ${script} $out/bin/setupenv
            # wrapProgram $out/bin/github-downloader.sh \
            #   --prefix PATH : ${lib.makeBinPath [ bash subversion ]}
          '';
        };

        devPkgs = {wPostProcessing ? false}: [
          (python.withPackages (ptPkgs: with ptPkgs; [
            numpy
            ase
            # vitrum
            ## matplotlib
            # torch
            # einops
            # tqdm
            # einops
            # pytorch-lightning
            # pandas
            # tables
            scipy
            # kimpy
            sqnm
            (lammps.overrideAttrs {lammps = pkgs.lammps-mpi; })
            mpi4py
          ] ++ (if wPostProcessing then [
            scikit-learn
          ] else [])))
        ] ++ (with pkgs; [
          lammps-mpi
          mpi
          parallel
          gnutar
          xz
          bmp
          bmpFixed
          setupenv
        ]);
        
      in {
    
        devShells = let 
          shell = wPostProcessing: pkgs.mkShell {

            buildInputs = devPkgs { inherit wPostProcessing; };

            shellHook = ''
              export FLAKE="lammps-bmp"
              export OMP_NUM_THREADS=1
              export BMP=${bmp}
              export BMP_FIXED=${bmpFixed}

              # back to zsh
              exec zsh
            '';
          };
        in {
          default = shell false;
          postProcessing = shell true;
        };

        packages = rec {
          default = singularityContainer;

          singularityContainer = pkgs.singularity-tools.buildImage {
            name = "lammps-bmp";
            contents = devPkgs ++ (with pkgs; [
              coreutils-full # provides ls, cat, ... (assumed to be present by kim collection manager))
              bash
              glibcLocales
            ]);

            # the shadowSetup creates passwd and group files to prevent singularity from complaining
            runAsRoot = ''
              #!${pkgs.stdenv.shell}
              ${pkgs.dockerTools.shadowSetup} 
            '';
            # drop into shell by default
            # NOTE: Does not work
            runScript = ''
              #!${pkgs.stdenv.shell}
              exec /bin/sh $@"
            '';
            diskSize = 1024 * 40;
            memSize = 1024 * 8;
          };
        };
      }
    );
}
