#!/bin/bash

# Read filenames from filenames.txt
file1=$(sed -n '1p' filenames.txt)
file2=$(sed -n '2p' filenames.txt)

# Generate a random directory name
random_dir=$(mktemp -d dir_XXXXXX)
touch imagefileName.txt
mv $random_dir static/

# Insert the random directory name into the third line of filenames.txt
echo $random_dir > imagefileName.txt

mv uploads/$file1 static/$random_dir/
mv uploads/$file2 static/$random_dir/

# Call the Python script to compare the images
python3 code/hsv.py static/$random_dir/$file1 static/$random_dir/$file2 > HSVresult.txt
python3 code/ssim.py ../static/$random_dir/$file1 ../static/$random_dir/$file2 > SSIMresult.txt
python3 code/cnn.py static/$random_dir/$file1 static/$random_dir/$file2 > CNNresult.txt
python3 code/yolo.py static/$random_dir/$file1 static/$random_dir/$file2
