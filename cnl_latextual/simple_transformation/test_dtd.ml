open Pxp_types;;
open Pxp_document;;
open Pxp_tree_parser;;

let main() =
  try
      let dtd =   
        Pxp_dtd_parser.parse_dtd_entity default_config (from_file "record.dtd") in  
(*        let dtd = Pxp_dtd_parser.create_empty_dtd default_config in   *)
(*       dtd # allow_arbitrary; *)
    let tree = 
      parse_content_entity default_config (from_channel stdin) dtd default_spec in
(*     let exemplar = new data_impl tree # extension in  *)
(*     print tree *)
      tree # write (`Out_channel stdout) `Enc_utf8
  with
      x ->
	prerr_endline(string_of_exn x);
	exit 1
;;

main();;
