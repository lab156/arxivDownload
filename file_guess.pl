#!/usr/bin/perl

use arXiv;
use arXiv::FileGuess qw(guess_file_type);
use LaTeXML::Util::Pack qw(unpack_source);

print "We have version ".$arXiv::VERSION." of the arXiv modules.\n";
$dir  = '/home/luis/MisDocumentos/arxivBulkDownload/math0303166.gz';
$dir1 = '/home/luis/MisDocumentos/arxivBulkDownload/1804.01883.zip';
$dir4 = '/home/luis/MisDocumentos/arxivBulkDownload/1804.01883.gz';
$dir2 = '/home/luis/MisDocumentos/arxivBulkDownload/example.zip';
$dir3 = '/home/luis/MisDocumentos/arxivBulkDownload/arXiv_src_0303_001.tar';
$extdir = '/home/luis/MisDocumentos/arxivBulkDownload/ExtractionDir';

print "$dir\n";
@msg = guess_file_type($dir2);
print "@msg[0]\n";
print unpack_source($dir2, $extdir);
