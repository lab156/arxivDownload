(* $Id$
 * ----------------------------------------------------------------------
 *
 *)

(* Add a par and print it as XML *)
open Pxp_types;;
open Pxp_document;;
open Pxp_tree_parser;;



let print tree =
  iter_tree
    ~pre:
      (fun n ->
	 match n # node_type with
	     T_element "cs" ->
	       print_endline ("Control Sequence is: " ^ n # data)
	   | T_element "math" ->
	       print_endline ("inline Math content: " ^ n # data)
	   | T_element "display" ->
	       print_endline ("display style math content: " ^ n # data)
	   | _ ->
	       ())
    ~post:
      (fun n ->
	 match n # node_type with
	     T_element "par" -> 
	       print_newline()
	   | _ ->
	       ())
    tree
;;

let main() =
  try
    let dtd = 
      Pxp_dtd_parser.parse_dtd_entity default_config (from_file "record.dtd") in
    let tree = 
      parse_content_entity default_config (from_file "sample.xml") dtd default_spec in
(*     let exemplar = new data_impl tree # extension in  *)
    let element_exemplar = new element_impl tree # extension in
    let data_exemplar    = new data_impl    tree # extension in
    let spec = make_spec_from_alist
             ~data_exemplar:data_exemplar
             ~default_element_exemplar:element_exemplar
             ~element_alist:[]
             () in
     let par1 = create_element_node spec dtd "par" [] in  
     let cs1 = create_element_node spec dtd "cs" [] in 
     let cs1 # data = "culero connor" 
     par1 # append_node cs1;
     tree # append_node par1;
    print tree
  with
      x ->
	prerr_endline(string_of_exn x);
	exit 1
;;


main();;
