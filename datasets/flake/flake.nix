{
  description = "Development environment to run scripts for analyzing the AMDEN datasets";

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
        tersoff = pkgs.callPackage ./tersoff {};

        setupenv = with pkgs; let
          script = writeShellScript "builder.sh" ''
            #!/bin/sh
            export LOCALE_ARCHIVE=${glibcLocales}/lib/locale/locale-archive
            export BMP=${bmp}
            export TERSOFF=${tersoff}
            # echo $BMP
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

        devPkgs = [
          (python.withPackages (ptPkgs: with ptPkgs; [
            numpy
            ase
            scipy
            kimpy
            sqnm
            (lammps.overrideAttrs {lammps = pkgs.lammps-mpi; })
            mpi4py
            scikit-learn
            numba
        ]))] ++ (with pkgs; [
          kim-api
          lammps-mpi
          mpi
          parallel
          gnutar
          xz
          bmp
          tersoff
          setupenv
        ]);
        
      in {
    
        devShells = {
          default= pkgs.mkShell {

            buildInputs = devPkgs;

            shellHook = ''
              export FLAKE="AMDEN"
              export OMP_NUM_THREADS=1
              export BMP=${bmp}
              export TERSOFF=${tersoff}

              # back to zsh
              exec zsh
            '';
          };
        };

        packages = rec {
          default = singularityContainer;

          singularityContainer = pkgs.singularity-tools.buildImage {
            name = "DDM paper";
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
            diskSize = 1024 * 40;
            memSize = 1024 * 8;
          };
        };
      }
    );
}
