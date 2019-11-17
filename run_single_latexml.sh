#! /bin/bash

# Run LaTeXML on a specific .tex file
# Names 


#Define the LaTeXML binary file to run by sourcing the config file
# This define the variable latexml_bin
source <(grep ^LATEXML_BIN "$PWD/config.toml")
echo "latexml_bin file is: $PWD"

#The maximum amount of time in seconds that LaTeXML is
#allowed to run (seconds)
#source <(grep ^MAXT "$PWD/config.toml")
MAXT=4800

# Run on all the arguments of the command line
for TEX_FILE in "$@"
do
    COMMENTARY_FILE=${TEX_FILE%.*}"_commentary.txt"
    ERROR_MESS_FILE=${TEX_FILE%.*}"_errors_mess.txt"
    echo "Running LaTeXML on the file " $TEX_FILE
    echo "main .tex file" $(basename $TEX_FILE) >> $COMMENTARY_FILE
    #timeout $MAXT  $LATEXML_BIN $TEX_FILE  2>$ERROR_MESS_FILE > ${TEX_FILE%.*}.xml 

    # Remove the \pmmeta data
    timeout $MAXT   sed '/^\\usepackage{pmmeta}/,/^\\endmetadata/d' $TEX_FILE | latexml - 2>$ERROR_MESS_FILE > ${TEX_FILE%.*}.xml 

    if [ $? -eq 124 ]; then
        echo "Timeout Occured with file $TEX_FILE"
        echo "Timeout of $MAXT seconds occured" >> $COMMENTARY_FILE
    else
        echo "Finished in less than $MAXT seconds" >> $COMMENTARY_FILE
    fi
done
