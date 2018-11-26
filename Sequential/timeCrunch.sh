#!/bin/bash
hostNo=$(hostname)
resultsLog="results/TDresults${hostNo}.log"
tempLog="temp/TDtemp${hostNo}.log" 
exportsLog="exports/timeDomainExport${hostNo}.csv"
module purge
module load gcc/6.3.0
module load libsndfile/intel/1.0.28
module load fftw/intel/3.3.6-pl2

#goes from 1 - 27
#file=$(ls ../Audio/96000/* | sed -n ${SLURM_ARRAY_TASK_ID}p)
#(time bin/seqConvolve.out -t -i $file ) 

(time bin/seqConvolve.out -t -i ../Audio/96000/2e04.wav ) >> $resultsLog 2>&1
tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog

(time bin/seqConvolve.out -t -i ../Audio/96000/2e05.wav ) >> $resultsLog 2>&1
tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog

(time bin/seqConvolve.out -t -i ../Audio/96000/2e06.wav ) >> $resultsLog 2>&1
tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog

(time bin/seqConvolve.out -t -i ../Audio/96000/2e07.wav ) >> $resultsLog 2>&1
tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog

(time bin/seqConvolve.out -t -i ../Audio/96000/2e08.wav ) >> $resultsLog 2>&1
tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog

(time bin/seqConvolve.out -t -i ../Audio/96000/2e09.wav ) >> $resultsLog 2>&1
tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog

for (( i=10; i<20; i=i+1 )); do 
    (time bin/seqConvolve.out -t -i ../Audio/96000/2e$i.wav ) >> $resultsLog 2>&1
    tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog
done

(time bin/seqConvolve.out -t -i ../Audio/96000/2e20.wav ) >> $resultsLog 2>&1
tail -3 $resultsLog | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog
#for (( i=21; i<31; i=i+1 )); do echo$(((2**${i} + 479999 + 2**23 + 48000 )* 4 / 1024 / 1024 * 4)); done
#total amount allocated in heap * sizeof(float) in MB * 4
#took approx 50 minutes to do 2^21. Overestimating to an hour & approximating that time doubles (as shown through past experiments)
sbatch -J=2e21.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=168   --time=01:00:00 timeTest.sh ../Audio/96000/2e21.wav
sbatch -J=2e22.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=200   --time=02:00:00 timeTest.sh ../Audio/96000/2e22.wav
sbatch -J=2e23.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=264   --time=04:00:00 timeTest.sh ../Audio/96000/2e23.wav
sbatch -J=2e24.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=392   --time=08:00:00 timeTest.sh ../Audio/96000/2e24.wav
sbatch -J=2e25.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=648   --time=16:00:00 timeTest.sh ../Audio/96000/2e25.wav
sbatch -J=2e26.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=1160  --time=32:00:00 timeTest.sh ../Audio/96000/2e26.wav
sbatch -J=2e27.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=2184  --time=64:00:00 timeTest.sh ../Audio/96000/2e27.wav
sbatch -J=2e28.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=4232  --time=128:00:00 timeTest.sh ../Audio/96000/2e28.wav
sbatch -J=2e29.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=8328  --time=256:00:00 timeTest.sh ../Audio/96000/2e29.wav
sbatch -J=2e30.log --mail-type=ALL --mail-user=ctb335@nyu.edu --mem=16520 --time=512:00:00 timeTest.sh ../Audio/96000/2e30.w64