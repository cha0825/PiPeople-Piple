#!/bin/bash

# Read filenames 
file1=$(sed -n '1p' partial-imagesChecked.txt)
file2=$(sed -n '2p' partial-imagesChecked.txt)

python3 code/hsv_similarity.py ../static/$file1 ../static/$file2 > PartialHSVresult.txt
python3 code/ssim.py ../static/$file1 ../static/$file2 > PartialSSIMresult.txt