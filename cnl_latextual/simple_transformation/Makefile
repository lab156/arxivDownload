all: print sort delcol

print: print.ml
	ocamlfind ocamlc -o print -package pxp -linkpkg -custom -g \
		-predicates pxp_without_utf8 print.ml

sort: sort.ml
	ocamlfind ocamlc -o sort -package pxp -linkpkg -custom \
		-predicates pxp_without_utf8 sort.ml

delcol: delcol.ml
	ocamlfind ocamlc -annot -o delcol -package pxp -linkpkg -custom \
		-predicates pxp_without_utf8 delcol.ml

clean:
	rm -f *.cmo *.cma *.cmi *.cmxa *.a *.o

distclean: clean
	rm -f *~ print sort delcol

CLEAN: clean
