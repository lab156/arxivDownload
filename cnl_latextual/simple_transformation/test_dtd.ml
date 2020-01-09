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
let filen = "example.xml" in 
let oc = open_out filen in
tree # write (`Out_channel oc) `Enc_iso88591;
(* `Enc_iso88591 *)
close_out oc;
  with
      x ->
	prerr_string (string_of_exn x);
    exit 1
;;

main();;
