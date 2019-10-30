open Pxp_types
open Pxp_document
open Pxp_dtd.Entity

let exemplar_ext = None  in
let dtd = None  in

let element_exemplar = new element_impl exemplar_ext in
let data_exemplar    = new data_impl    exemplar_ext in

let spec = make_spec_from_alist 
             ~data_exemplar:data_exemplar
             ~default_element_exemplar:element_exemplar
             ~element_alist:[]
             () in

let a1 = create_element_node spec dtd "a" ["att", "apple"]
and b1 = create_element_node spec dtd "b" []
and c1 = create_element_node spec dtd "c" []
and a2 = create_element_node spec dtd "a" ["att", "orange"]
in

let cherries = create_data_node spec dtd "Cherries" in
let orange   = create_data_node spec dtd "An orange" in

a1 # append_node b1;
a1 # append_node c1;
b1 # append_node a2;
b1 # append_node cherries;
a2 # append_node orange;
