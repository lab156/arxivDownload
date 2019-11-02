(* $Id$
 * ----------------------------------------------------------------------
 *
 *)

(* Read a record-list structure and print it *)
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
      parse_content_entity default_config (from_channel stdin) dtd default_spec in
    print tree
  with
      x ->
	prerr_endline(string_of_exn x);
	exit 1
;;


main();;

