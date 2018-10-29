#! /bin/bash

# Run latexml on all the .tex files under argv[1] 
# Sends the Errors to the files latexml_errors_mess.txt
# and the processed file is stored in an xml file with 
#the same name as the original .tex file.

FILE_LIST=($(find $1 -iname *.tex))
#FILE_LIST=($(find  $1 -maxdepth 1 -type d '!' -exec test -e "{}/latexml_errors_mess.txt" ';' -print))

for f in ${FILE_LIST[@]}; do
     echo processing the file $f;
#    xpath=${f%.*}
#    echo $xpath
#Only the name of the file with extension
    xbase_with_extn=${f##*/}
    # Only the base name of the file without extension
    BSNM=${xbase_with_extn%.*}
#    xfext=${xbase##*.}
#    echo xbase is: $xbase
#    echo xfext is: $xfext
#    echo see if this works ${f%/*}
     #latexml $f  2>&1 > ${f%.*}.xml | echo  > ${f%/*}/ltxml_errs_mess_${f%.*}.txt
     /home/luis/Paquetes/LaTeXML/bin/latexml $f  2>${f%/*}/ltxml_err_mess_$BSNM.txt > ${f%.*}.xml 
done


