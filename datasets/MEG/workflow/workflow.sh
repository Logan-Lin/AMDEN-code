
date
t_start=$(date +%s)

id=$1  


# echo START $id

set -eu

# source setupenv

# --- Initialization ---

python3 src/main_init.py ${id}

echo INIT-DONE
date


# --- Evaporation ---

cd data/${id}
lmp -i ../../lammps-scripts/evap.lammps -sc none -l evap.log
cd ../..

python3 src/main_evap.py ${id} > data/${id}/T_HIGH.txt
export T_HIGH=$(cat data/${id}/T_HIGH.txt)
echo "T_HIGH=$T_HIGH"

echo EVAPORATION-DONE
date


# --- Melting ---

cd data/${id}
lmp -i ../../lammps-scripts/melt.lammps -sc none -l melt.log
cd ../..

python3 src/main_check_melted.py ${id} 

python3 src/main_check_no_evap.py ${id} melt

echo MELT-DONE
date


# --- Quenching ---

cd data/${id}
lmp -i ../../lammps-scripts/quench.lammps -sc none -l quench.log
cd ../..

python3 src/main_check_no_evap.py ${id} quench

echo QUENCH-DONE
date


# --- Thermal Expansion ---

cd data/${id}
lmp -i ../../lammps-scripts/thermal-expansion.lammps -sc none -l thermal-expansion.log
cd ../..

python3 src/main_thermal_expansion.py ${id} > data/${id}/thermal.txt

echo THERMAL-DONE
date


# --- Elastic ---

python3 src/main_elastic.py ${id} > data/${id}/elastic.txt

echo ELASTIC-DONE
date


# --- Compressing ---

cd data/${id}
tar -cvJ *.lammpstrj *.log heat.txt melt.txt quench.txt
rm *.lammpstrj *.log heat.txt melt.txt quench.txt
cd ../..

echo COMPRESS-DONE



# --- Final ---

echo ""
t_end=$(date +%s)
echo "Elapsed Time: $((t_end - t_start))"

# echo DONE $id
