#!/bin/bash
# Script to convert pdfs into jpegs

for pdffile in $(ls *.pdf)
do
   stem=$(echo $pdffile | cut -d"." -f 1) 
   echo $stem   
   gs -dSAFER -r600 -sDEVICE=pngalpha -o $stem.jpeg $stem.pdf	
done  
