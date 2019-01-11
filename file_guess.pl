#!/usr/bin/perl

use arXiv;
use arXiv::FileGuess qw(guess_file_type);
use LaTeXML::Util::Pack qw(unpack_source);
use LaTeXML::Util::Pathname qw(pathname_is_literaldata);
use LaTeXML qw(convert);

print "We have version ".$arXiv::VERSION." of the arXiv modules.\n";
$dir  = '/home/luis/MisDocumentos/arxivBulkDownload/math0303166.gz';
$dir1 = '/home/luis/MisDocumentos/arxivBulkDownload/1804.01883.zip';
$dir4 = '/home/luis/MisDocumentos/arxivBulkDownload/1804.01883.gz';
$dir2 = '/home/luis/MisDocumentos/arxivBulkDownload/example1/';
$dirfile = '/home/luis/MisDocumentos/arxivBulkDownload/example1/z_sys.tex';
$dir3 = '/home/luis/MisDocumentos/arxivBulkDownload/arXiv_src_0303_001.tar';
$dir5 = '/home/luis/MisDocumentos/arxivBulkDownload/math.0004134.zip';
$extdir = '/home/luis/MisDocumentos/arxivBulkDownload/ExtractionDir';

print "$dir\n";
@msg = guess_file_type($dirfile);
#print "@msg[0]\n";
#print unpack_source("literal:\\begin{document} hola \end{document}", $extdir);
#print pathname_is_literaldata('literal:dd');
$LaTeXML->convert($dirfile);
