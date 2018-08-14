#! /bin/bash

# Run latexml on all the .tex files under argv[1] 
# Sends the Errors to the files latexml_errors_mess.txt
# and the processed file is stored in an xml file with 
#the same name as the original .tex file.

FILE_LIST=($(find $1 -name *.tex))

for f in ${FILE_LIST[@]}; do
     echo processing the file $f;
#    xpath=${f%.*}
#    echo $xpath
#    xbase=${f##*/}
#    xfext=${xbase##*.}
#    echo xbase is: $xbase
#    echo xfext is: $xfext
#    echo see if this works ${f%/*}
     #latexml $f  2>&1 > ${f%.*}.xml | echo  > ${f%/*}/latexml_errors_mess.txt
     latexml $f  2>${f%/*}/latexml_errors_mess.txt > ${f%.*}.xml 
done


