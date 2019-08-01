#! /bin/bash

# Run latexml on  the main .tex files under argv[1] 
# Sends the Errors to the files latexml_errors_mess.txt
# and the processed file is stored in an xml file with 
#the same name as the original .tex file.
#FILE_LIST=($(find  $1 -maxdepth 1 -type d '!' -exec test -e "{}/latexml_errors_mess.txt" ';' -print))

#Define the LaTeXML binary file to run by sourcing the config file
# This define the variable latexml_bin
source <(grep ^LATEXML_BIN "$PWD/config.toml")
echo "latexml_bin file is: $PWD"

#The maximum amount of time in seconds that LaTeXML is
#allowed to run (seconds)
MAXT=1200

for article_dir in "$@"
do
    f=$(perl get_main_tex.pl $article_dir)
    if [ -z "$f" ]
    then
	    COMMENTARY_FILE=$article_dir/commentary.txt
	    echo "Could not find the main tex file. $COMMENTARY_FILE"
	    [ -f "$COMMENTARY_FILE" ] && echo "Main TeX file not found" >> $COMMENTARY_FILE
    else
	    echo "Running LaTeXML on the file " $f
	    COMMENTARY_FILE=${f%/*}/commentary.txt 
	    echo "main .tex file" $(basename $f) >> $COMMENTARY_FILE
	    timeout $MAXT  $LATEXML_BIN $f  2>${f%/*}/latexml_errors_mess.txt > ${f%.*}.xml 
	    if [ $? -eq 124 ]; then
		    echo "Timeout Occured with file $f"
		    echo "Timeout of $MAXT seconds occured" >> $COMMENTARY_FILE
	    else
		    echo "Finished in less than $MAXT seconds" >> $COMMENTARY_FILE
	    fi
    fi
done


