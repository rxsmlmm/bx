#!/bin/sh
#PBS -N qsub_mpi
#PBS -e test_np32.e
#PBS -o test_np32.o
#PBS -l nodes=4:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)
# 注意把所有的ntt换成你的选题

for node in $NODES; do
    scp master_ubss1:/home/${USER}/gauss/main ${node}:/home/${USER} 1>&2
    scp -r master_ubss1:/home/${USER}/gauss/files ${node}:/home/${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np 32 -machinefile $PBS_NODEFILE /home/${USER}/main

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/gauss/ 2>&1
