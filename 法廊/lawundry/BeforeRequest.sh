#!/bin/bash

rm -r HSVresult.txt
rm -r SSIMresult.txt
rm -r YOLOresult1.txt
rm -r YOLOresult2.txt
rm -r CNNresult.txt

# Read filenames from filenames.txt
file1=$(sed -n '1p' filenames.txt)
file2=$(sed -n '2p' filenames.txt)
imagefile=$(sed -n '1p' imagefileName.txt)

# 确保 imagefile 不是空的不然會刪到static 資料夾
if [ -n "$imagefile" ] && [ -d "static/$imagefile" ]; then
  rm -r "static/$imagefile"
fi

rm -r filenames.txt
rm -r imagefileName.txt

# 清空 partpic
rm -rf static/partpic/*

rm -r partial-imagesChecked.txt
rm -r PartialHSVresult.txt
rm -r PartialSSIMresult.txt
rm -r PartialCNNresult.txt