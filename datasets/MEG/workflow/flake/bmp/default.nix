{
  stdenv,
  unzip,
  fixDisconsts? false
}: stdenv.mkDerivation rec {
  pname = "BMP";
  version = "24-11-07";
  src = ./BMP.zip; # from https://sites.google.com/site/compmaterchem/downloads

  nativeBuildInputs = [ unzip ];

  installPhase = ''
    mkdir -p $out
    cp -r * $out
    # fix paths to table files
    substituteInPlace $out/in.BMP --replace-fail "/home/Table" $out
    # the B-O table needs to be constructed depending on composition
    # We remove it and will not use B
    sed -e '/Table_BMP-B-O.dat/s/^/#/g' -i $out/in.BMP

    # the pair coefficients for the 3b have only 37 id to element maps... assuming Mg or Eu got lost between V and B
    # Also added V to V IV (id 36), probably should use the same 3bp for both oxidation states?
    substituteInPlace $out/in.BMP --replace-fail \
      "pair_coeff * * nb3b/screened BMP.nb3b.shrm Si O NULL NULL NULL NULL NULL NULL P NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL V NULL B" \
      "pair_coeff * * nb3b/screened $out/BMP.nb3b.shrm Si O NULL NULL NULL NULL NULL NULL P NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL V V NULL NULL B" 

  '' + (if fixDisconsts then ''
    substituteInPlace $out/in.BMP --replace-fail \
      "variable	rvdw equal 7.0" \
      "variable	rvdw equal 7.95"
    substituteInPlace $out/BMP.nb3b.shrm --replace-fail \
      "   O     P     P  	32.5  109.47    1.00    2.00" \
      "   O     P     P  	32.5  109.47    1.00    9.00" 
    substituteInPlace $out/BMP.nb3b.shrm --replace-fail \
      "   O     P     Si 	60.0  109.47	1.00    2.00" \
      "   O     P     Si 	60.0  109.47	1.00    9.00" 
    substituteInPlace $out/BMP.nb3b.shrm --replace-fail \
      "   O     Si    Si       12.5  109.47    1.00    3.30" \
      "   O     Si    Si       12.5  109.47    1.00    9.00"
    substituteInPlace $out/BMP.nb3b.shrm --replace-fail \
      "   O     Si    P  	60.0  109.47    1.00    2.00" \
      "   O     Si    P  	60.0  109.47    1.00    9.00" 
    # WARN: B and V terms are not fixed
  '' else "");
    
      
    

}
