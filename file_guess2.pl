#!/usr/bin/perl

use arXiv;
use arXiv::FileGuess qw(guess_file_type);
use LaTeXML::Util::Pack qw(unpack_source);
use Data::Dumper qw(Dumper);

@msg = guess_file_type(@ARGV[0]);

print "@msg[0]\n";

