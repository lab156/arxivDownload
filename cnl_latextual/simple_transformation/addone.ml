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
	   | T_element "par" ->
	       print_endline ("par: " ^ n # data)
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

let add_one_cs spec dtd att =
(*     cs is hardcoded because this is a helping function *)
     let cs1 = create_element_node spec dtd "cs" ["name", att] in 

      cs1
;;

let add_one_math spec dtd att =
(*     cs is hardcoded because this is a helping function *)
     let cs1 = create_element_node spec dtd "math" [] in 

      cs1
;;


let main() =
  try
      let dtd =   
        Pxp_dtd_parser.parse_dtd_entity default_config (from_file "record.dtd") in  
(*        let dtd = Pxp_dtd_parser.create_empty_dtd default_config in   *)
(*       dtd # allow_arbitrary; *)
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
     let par1dat = create_data_node spec dtd "par" in 
     par1dat # set_data "This is a very important content of par 1";
     let par2dat = create_data_node spec dtd "par" in 
     par2dat # set_data "This is the most important content of par 2";

     let cs1 = add_one_cs spec dtd "name of cs1" in
     let cs2 = add_one_cs spec dtd "name of cs2" in

     let math1 = add_one_cs spec dtd "name of math1" in

     let par1 = create_element_node spec dtd "par" [] in  
     par1 # append_node cs1;
     par1 # append_node par1dat;
     par1 # append_node cs2;
     par1 # append_node math1;
     par1 # append_node par2dat;
     tree # append_node par1;
(*     print tree *)
      tree # write (`Out_channel stdout) `Enc_utf8
  with
      x ->
	prerr_endline(string_of_exn x);
	exit 1
;;

main();;
