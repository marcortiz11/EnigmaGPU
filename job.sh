#!/bin/bash
export PATH=/Soft/cuda/8.0.61/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N randm 
# Cambiar el shell
#$ -S /bin/bash

#nvprof --export-profile enigma.prof --unified-memory-profiling off ./bomba.exe
#nvprof --metrics sm_efficiency --unified-memory-profiling off ./bomba.exe
#nvprof  --query-metrics

./bomba.exe

#./kernel00.exe 50000 Y
#./kernel01.exe 50000 Y
#./kernel02.exe 50000 Y

