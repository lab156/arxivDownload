all: addone print sort delcol test_dtd

addone: addone.ml
	ocamlfind ocamlc -annot -o addone -package pxp -linkpkg -custom -g \
		-predicates pxp_without_utf8 addone.ml

addone: test_dtd.ml
	ocamlfind ocamlc -annot -o test_dtd -package pxp -linkpkg -custom -g \
		-predicates pxp_without_utf8 test_dtd.ml

print: print.ml
	ocamlfind ocamlc -annot -o print -package pxp -linkpkg -custom -g \
		-predicates pxp_without_utf8 print.ml

sort: sort.ml
	ocamlfind ocamlc -annot -o sort -package pxp -linkpkg -custom \
		-predicates pxp_without_utf8 sort.ml

delcol: delcol.ml
	ocamlfind ocamlc -annot -o delcol -package pxp -linkpkg -custom \
		-predicates pxp_without_utf8 delcol.ml

clean:
	rm -f *.cmo *.cma *.cmi *.cmxa *.a *.o

distclean: clean
	rm -f *~ print sort delcol addone

CLEAN: clean
