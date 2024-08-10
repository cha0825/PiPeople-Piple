#!/bin/bash

rm -r HSVresult.txt
rm -r SSIMresult.txt
rm -r YOLOresult1.txt
rm -r YOLOresult2.txt

# Read filenames from filenames.txt
file1=$(sed -n '1p' filenames.txt)
file2=$(sed -n '2p' filenames.txt)
imagefile=$(sed -n '1p' imagefileName.txt)

# 突然想到不能這樣做 要改 不然會刪到別人的 所以還是用資料夾做更好
# 确保 imagefile 变量非空并且是一个目录
if [ -n "$imagefile" ] && [ -d "code/$imagefile" ]; then
  rm -r "code/$imagefile"
fi

rm -r filenames.txt
rm -r imagefileName.txt

# 清空partpic
rm -rf static/partpic/*

rm -r partial-imagesChecked.txt
rm -r PartialHSVresult.txt
rm -r PartialSSIMresult.txt