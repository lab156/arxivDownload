let convert_toks s = 
(*  let _ = print_endline s in   *)
  let toks = Latextual__Lexer.lex_string s in
(*  let _ = print_endline ("->"^(Latextual.tokens_to_string ". " toks)) in  *)
  toks;;

 Latextual.process_doc convert_toks  "../tex/currentversion-smallmasslimit.tex";; 
(*
let tt = convert_toks  "../tex/mini_file.tex";; 

let rec pp toks = 
    match toks with
    | [] -> None
    | t::rem -> print_endline (Latextual__Type.lex_token_to_string t);
    pp rem;;
pp tt;;
*)
