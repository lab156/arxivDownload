#! /bin/bash

# Run latexml on  the main .tex files under argv[1] 
# Sends the Errors to the files latexml_errors_mess.txt
# and the processed file is stored in an xml file with 
#the same name as the original .tex file.
#FILE_LIST=($(find  $1 -maxdepth 1 -type d '!' -exec test -e "{}/latexml_errors_mess.txt" ';' -print))

for article_dir in "$@"
do
    f=$(perl get_main_tex.pl $article_dir)
    echo "Running LaTeXML on the file " $f
    COMMENTARY_FILE=${f%/*}/commentary.txt 
    echo "main .tex file" $(basename $f) >> $COMMENTARY_FILE
    /home/luis/Paquetes/LaTeXML/bin/latexml $f  2>${f%/*}/latexml_errors_mess.txt > ${f%.*}.xml 
done


