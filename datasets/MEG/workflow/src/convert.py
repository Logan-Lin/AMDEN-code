from lmpio import *
from ase.io import read, write
from sys import argv



def main():
    # get cmmdln args
    if len(argv) != 3:
        print("Usage: python convert.py inputfile outputfile")
        return

    inf = argv[1]
    outf = argv[2]

    inext = inf.split('.')[-1]
    outext = outf.split('.')[-1]

    if inext == 'lammpstrj' or inext == 'dump':
        ats = read_lammpstrj(inf)
    else:
        ats = read(inf, index=':')


    if outext == 'data':
        write_lammpsdata(outf, ats[-1])
    else:
        write(outf, ats)

     



if __name__ == '__main__':
    main()
