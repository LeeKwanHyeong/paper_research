# GUI-launched editors may find latexmk by its absolute path while its child
# tools (pdflatex, bibtex, etc.) are absent from PATH. Use MacTeX's stable link.
# Other platforms retain their existing environment.
if ($^O eq 'darwin' && -d '/Library/TeX/texbin') {
    $ENV{'PATH'} = '/Library/TeX/texbin:' . ($ENV{'PATH'} // '');
}
