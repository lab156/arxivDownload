(* $Id$
 * ----------------------------------------------------------------------
 *
 *)

(* Read a record-list, delete a column, and print it as XML *)
open Pxp_types;;
open Pxp_document;;
open Pxp_tree_parser;;

let delcol col tree =
  map_tree
    ~pre:
      (fun n -> 
	 match n # node_type with
	     T_element name when name = col ->
	       raise Skip
	   | _ -> n # orphaned_flat_clone)
    tree
;;


let main() =
  let column = ref "" in
  Arg.parse
      [ "-col", Arg.String (fun s -> column := s),
	    " (last-name|first-name|phone)";
      ]
      (fun _ -> raise (Arg.Bad "Bad usage"))
      "usage: sort [ options ]";
  if !column = "" then (
    prerr_endline "Column not specified!";
    exit 1;
  );
  if not(List.mem !column ["last-name"; "first-name"; "phone"]) then (
    prerr_endline ("Unknown column: " ^ !column);
    exit 1
  );
  try
    let dtd = 
      Pxp_dtd_parser.parse_dtd_entity default_config (from_file "record.dtd") in
    let tree = 
      parse_content_entity default_config (from_channel stdin) dtd default_spec
    in
    print_endline "<?xml encoding='ISO-8859-1'?>";
    (delcol !column tree) # write (`Out_channel stdout) `Enc_iso88591
  with
      x ->
	prerr_endline(string_of_exn x);
	exit 1
;;


main();;
