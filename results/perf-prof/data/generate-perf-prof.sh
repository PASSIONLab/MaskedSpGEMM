#!/bin/bash
#rm *.png

python2.7 ../PerfProf.py tricnt-knl26-haswell.txt --style 1 --title "Triangle Counting - haswell - 1 phase vs 2 phases" --name="tricnt-knl26-haswell-1p2p" --lines "9,15,8,14,10,16,6,12,7,13,11,17" &
#python2.7 ../PerfProf.py tricnt-knl26-haswell.txt --style 2 --title "Triangle Counting - haswell" --name="tricnt-knl26-haswell-summary" --lines "9,8,10,3,2" &
#
#python2.7 ../PerfProf.py tricnt-knl26-knl.txt --style 1 --title "Triangle Counting - KNL - 1 phase vs 2 phases" --name="tricnt-knl26-knl-1p2p" --lines "9,15,8,14,10,16,6,12,7,13,11,17" &
#python2.7 ../PerfProf.py tricnt-knl26-knl.txt --style 2 --title "Triangle Counting - KNL" --name="tricnt-knl26-knl-summary" --lines "9,8,10,3,2,1" &
#
#python2.7 ../PerfProf.py ktruss-5-knl26-haswell.txt --style 1 --title "K-Truss - haswell - 1 phase vs 2 phases" --name="ktruss-knl26-haswell-1p2p" --lines "8,14,7,13,9,15,5,11,6,12,10,16" --xmax 1.9 &
#python2.7 ../PerfProf.py ktruss-5-knl26-haswell.txt --style 4 --title "K-Truss - haswell" --name="ktruss-knl26-haswell-summary" --lines "8,7,9,10,2,1" &
#
#python2.7 ../PerfProf.py ktruss-5-knl26-knl.txt --style 1 --title "K-Truss - KNL - 1 phase vs 2 phases" --name="ktruss-knl26-knl-1p2p" --lines "8,14,7,13,9,15,5,11,6,12,10,16" --xmax 1.9 &
#python2.7 ../PerfProf.py ktruss-5-knl26-knl.txt --style 4 --title "K-Truss - KNL" --name="ktruss-knl26-knl-summary" --lines "8,7,9,10,2,1" &
#
#python2.7 ../PerfProf.py bc-total-512-knl26-haswell.txt --style 3 --title "Betweenness centrality total - Haswell" --name="bc-512-knl26-knl-summary" --lines "5,4,7,6,1" --xmax 1.5
#python2.7 ../PerfProf.py bc-forward-512-knl26-haswell.txt --style 3 --title "Betweenness centrality forward phase - Haswell" --name="bc-512-knl26-knl-forward" --lines "5,4,7,6,1"
#python2.7 ../PerfProf.py bc-backward-512-knl26-haswell.txt --style 3 --title "Betweenness centrality backward phase - Haswell" --name="bc-512-knl26-knl-backward" --lines "5,4,7,6,1"

## Scaling
#python3.8 ../LinePlot.py --input tricnt-rmat-haswell-scaling-flops.txt &
#python3.8 ../LinePlot.py --input tricnt-rmat-knl-scaling-flops.txt &
#python3.8 ../LinePlot.py --input tricnt-rmat-haswell-scaling-teps.txt &
#python3.8 ../LinePlot.py --input tricnt-rmat-knl-scaling-teps.txt &
#
## Triangle counting RMAT
#python3.8 ../LinePlot.py --input tricnt-rmat-haswell-flops.txt &
#python3.8 ../LinePlot.py --input tricnt-rmat-knl-flops.txt &
#python3.8 ../LinePlot.py --input tricnt-rmat-haswell-teps.txt &
#python3.8 ../LinePlot.py --input tricnt-rmat-knl-teps.txt &
#
## KTruss RMAT
#python3.8 ../LinePlot.py --input ktruss-5-rmat-haswell-flops.txt &
#python3.8 ../LinePlot.py --input ktruss-5-rmat-knl-flops.txt &
#python3.8 ../LinePlot.py --input ktruss-5-rmat-haswell-teps.txt &
#python3.8 ../LinePlot.py --input ktruss-5-rmat-knl-teps.txt &
#
## Betweenness centrality RMAT
#python3.8 ../LinePlot.py --input bc-rmat-haswell-512.txt &
#python3.8 ../LinePlot.py --input bc-rmat-knl-512.txt

for f in $(ls *.p); do
  gnuplot

for f in `ls *.eps`; do
  echo $f
  epstopdf $f;
#  convert -density 300 $f -flatten ${f%.*}.png;
done
exit

# PNGs
if [ ! -d pdf ]; then mkdir pdf; fi
mv *.pdf pdf

cp pdf/* /home/sm108/projects/Masked-SpGEMM-paper/plots

# PNGs
if [ ! -d img ]; then mkdir img; fi
mv *.png img

# LaTeX
for f in `ls *.tex`; do
 :
# pdflatex $f
done

if [ ! -d tex ]; then mkdir tex; fi
mv *.tex tex

#rm -f *.p
#rm -f *.dat
#rm -f *.aux *.log
#rm -f *.gpi *.eps