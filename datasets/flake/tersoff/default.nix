{
  stdenv,
}: stdenv.mkDerivation {
  pname = "Tersoff_SiO2";
  version = "2007";
  src = ./2007_SiO.tersoff; 
  dontUnpack = true;
  installPhase = ''
    mkdir -p $out
    cp $src $out/2007_SiO.tersoff
  '';
    
      
    

}

