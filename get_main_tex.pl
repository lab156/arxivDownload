#!usr/bin/perl
#

use File::Spec::Functions qw(catfile);

sub get_main_file {
  my ($source_path, ) = @_;
  # This variable will hold the name of the main tex file
  my $main_source;
  #open directory 
  opendir my $source_dir, $source_path or die "Could not open directory $!";
  my @TeX_file_members = grep(/\.tex/, readdir($source_dir));
  if (!@TeX_file_members) { # No .tex file? Try files with no, or unusually long, extensions
    @TeX_file_members = grep {!/\./ || /\.[^.]{4,}$/} map { $_->fileName() } $zip_handle->members();
  }

  # Heuristically determine the input (borrowed from arXiv::FileGuess)
  my %Main_TeX_likelihood;
  my @vetoed = ();
  foreach my $tex_file (@TeX_file_members) {
    # Read in the content
    $tex_file = catfile($source_path, $tex_file);
    # Open file and read first few bytes to do magic sequence identification
    # note that file will be auto-closed when $FILE_TO_GUESS goes out of scope
    open(my $FILE_TO_GUESS, '<', $tex_file) ||
      (print STDERR "failed to open '$tex_file' to guess its format: $!. Continuing.\n");
    local $/ = "\n";
    my ($maybe_tex, $maybe_tex_priority, $maybe_tex_priority2);
  TEX_FILE_TRAVERSAL:
    while (<$FILE_TO_GUESS>) {
      if ((/\%auto-ignore/ && $. <= 10) ||    # Ignore
        ($. <= 10 && /\\input texinfo/) ||    # TeXInfo
        ($. <= 10 && /\%auto-include/))       # Auto-include
      { $Main_TeX_likelihood{$tex_file} = 0; last TEX_FILE_TRAVERSAL; }    # Not primary
      if ($. <= 12 && /^\r?%\&([^\s\n]+)/) {
        if ($1 eq 'latex209' || $1 eq 'biglatex' || $1 eq 'latex' || $1 eq 'LaTeX') {
          $Main_TeX_likelihood{$tex_file} = 3; last TEX_FILE_TRAVERSAL; }    # LaTeX
        else {
          $Main_TeX_likelihood{$tex_file} = 1; last TEX_FILE_TRAVERSAL; } }    # Mac TeX
          # All subsequent checks have lines with '%' in them chopped.
          #  if we need to look for a % then do it earlier!
      s/\%[^\r]*//;
      if (/(?:^|\r)\s*\\document(?:style|class)/) {
        $Main_TeX_likelihood{$tex_file} = 3; last TEX_FILE_TRAVERSAL; }    # LaTeX
      if (/(?:^|\r)\s*(?:\\font|\\magnification|\\input|\\def|\\special|\\baselineskip|\\begin)/) {
        $maybe_tex = 1; }
      if (/\\(?:input|include)(?:\s+|\{)([^ \}]+)/) {
        $maybe_tex = 1;
        # the argument of \input can't be the main file
        # (it could in very elaborate multi-target setups, but we DON'T support those)
        # so veto it.
        my $vetoed_file = $1;
        if ($vetoed_file eq 'amstex') { # TeX Priority
          $Main_TeX_likelihood{$tex_file} = 2; last TEX_FILE_TRAVERSAL; }
        if ($vetoed_file !~ /\./) {
          $vetoed_file .= '.tex';
        }
        my $base = $tex_file;
        $base =~ s/\/[^\/]+$//;
        $vetoed_file = "$base/$vetoed_file";
        push @vetoed, $vetoed_file; }
      if (/(?:^|\r)\s*\\(?:end|bye)(?:\s|$)/) {
        $maybe_tex_priority = 1; }
      if (/\\(?:end|bye)(?:\s|$)/) {
        $maybe_tex_priority2 = 1; }
      if (/\\input *(?:harv|lanl)mac/ || /\\input\s+phyzzx/) {
        $Main_TeX_likelihood{$tex_file} = 1; last TEX_FILE_TRAVERSAL; }        # Mac TeX
      if (/beginchar\(/) {
        $Main_TeX_likelihood{$tex_file} = 0; last TEX_FILE_TRAVERSAL; }        # MetaFont
      if (/(?:^|\r)\@(?:book|article|inbook|unpublished)\{/i) {
        $Main_TeX_likelihood{$tex_file} = 0; last TEX_FILE_TRAVERSAL; }        # BibTeX
      if (/^begin \d{1,4}\s+[^\s]+\r?$/) {
        if ($maybe_tex_priority) {
          $Main_TeX_likelihood{$tex_file} = 2; last TEX_FILE_TRAVERSAL; }      # TeX Priority
        if ($maybe_tex) {
          $Main_TeX_likelihood{$tex_file} = 1; last TEX_FILE_TRAVERSAL; }      # TeX
        $Main_TeX_likelihood{$tex_file} = 0; last TEX_FILE_TRAVERSAL; }        # UUEncoded or PC
      if (m/paper deliberately replaced by what little/) {
        $Main_TeX_likelihood{$tex_file} = 0; last TEX_FILE_TRAVERSAL; }
    }
    close $FILE_TO_GUESS || warn "couldn't close file: $!";
    if (!defined $Main_TeX_likelihood{$tex_file}) {
      if ($maybe_tex_priority) {
        $Main_TeX_likelihood{$tex_file} = 2; }
      elsif ($maybe_tex_priority2) {
        $Main_TeX_likelihood{$tex_file} = 1.5; }
      elsif ($maybe_tex) {
        $Main_TeX_likelihood{$tex_file} = 1; }
      else {
        $Main_TeX_likelihood{$tex_file} = 0; }
    }
  }
  # Veto files that were e.g. arguments of \input macros
  for my $filename(@vetoed) {
    delete $Main_TeX_likelihood{$filename};
  }
  # The highest likelihood (>0) file gets to be the main source.
  my @files_by_likelihood = sort { $Main_TeX_likelihood{$b} <=> $Main_TeX_likelihood{$a} } grep { $Main_TeX_likelihood{$_} > 0 } keys %Main_TeX_likelihood;
  if (@files_by_likelihood) {
   # If we have a tie for max score, grab the alphanumerically first file (to ensure deterministic runs)
    my $max_likelihood = $Main_TeX_likelihood{ $files_by_likelihood[0] };
    @files_by_likelihood = sort { $a cmp $b } grep { $Main_TeX_likelihood{$_} == $max_likelihood } @files_by_likelihood;
    $main_source = shift @files_by_likelihood; }

  # Return the main source from the unpacked files in the sandbox directory (or undef if failed)
  return $main_source;
}

print get_main_file('/home/luis/MisDocumentos/ODE2/hw1');
