#!/bin/bash

# Read filenames 
file1=$(sed -n '1p' partial-imagesChecked.txt)
file2=$(sed -n '2p' partial-imagesChecked.txt)

python3 code/hsv.py static/$file1 static/$file2 > PartialHSVresult.txt
python3 code/ssim.py ../static/$file1 ../static/$file2 > PartialSSIMresult.txt
python3 code/cnn.py static/$file1 static/$file2 > PartialCNNresult.txt