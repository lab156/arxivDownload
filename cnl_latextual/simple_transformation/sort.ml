(* $Id$
 * ----------------------------------------------------------------------
 *
 *)

(* Read a record-list, sort it, and print it as XML *)
open Pxp_types;;
open Pxp_document;;
open Pxp_tree_parser;;

let sort by tree =
  map_tree
    ~pre:
      (fun n -> n # orphaned_flat_clone)
    ~post:
      (fun n ->
	 match n # node_type with
	     T_element "record-list" ->
	       let l = n # sub_nodes in
	       let l' = List.sort
			  (fun a b ->
			     let a_string = 
			       try (find_element by a) # data 
			       with Not_found -> "" in
			     let b_string = 
			       try (find_element by b) # data 
			       with Not_found -> "" in
			     Pervasives.compare a_string b_string)
			  l in
	       n # set_nodes l';
	       n
	   | _ ->
	       n)
    tree
;;


let main() =
  let criterion = ref "last-name" in
  Arg.parse
      [ "-by", Arg.String (fun s -> criterion := s),
	    " (last-name|first-name|phone)";
      ]
      (fun _ -> raise (Arg.Bad "Bad usage"))
      "usage: sort [ options ]";
  if not(List.mem !criterion ["last-name"; "first-name"; "phone"]) then (
    prerr_endline ("Unknown criterion: " ^ !criterion);
    exit 1
  );
  try
    let dtd = 
      Pxp_dtd_parser.parse_dtd_entity default_config (from_file "record.dtd") in
    let tree = 
      parse_content_entity default_config (from_channel stdin) dtd default_spec
    in
    print_endline "<?xml encoding='ISO-8859-1'?>";
    (sort !criterion tree) # write (`Out_channel stdout) `Enc_iso88591
  with
      x ->
	prerr_endline(string_of_exn x);
	exit 1
;;


main();;

