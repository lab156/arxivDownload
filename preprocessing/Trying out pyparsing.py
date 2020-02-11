# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from pyparsing import \
        Literal, Word, ZeroOrMore, OneOrMore, Group, Dict, Optional, \
        printables, ParseException, restOfLine, empty, \
        Combine, nums, alphanums, Suppress, SkipTo, Forward, printables, alphas, oneOf
import pprint
import glob
import os

# %load_ext autoreload
# %autoreload 2
import prepro as pp

#ssn ::= num+ '-' num+ '-' num+
#num ::= '0' | '1' | '2' etc
dash = '-'
ssn = Combine(Word(nums, exact=3) +
                 dash + Word(nums, exact=2) +
                 Suppress('-') + Word(nums, exact=4))
target = '123-45-6789'
result = ssn.parseString(target)
print(result)

cc = CommandCleaner('xymatrix')
cc.show_matches(short_example)[0][0]

example_ini = '''[DEFAULT]
ServerAliveInterval = 45
Compression = yes
CompressionLevel = 9
ForwardX11 = yes

[bitbucket.org]
User = hg

[topsecret.server.com]
Port = 50022
ForwardX11 = no'''

# +
inibnf = None
def inifile_BNF():
    global inibnf
    
    if not inibnf:

        # punctuation
        lbrack = Literal("[").suppress()
        rbrack = Literal("]").suppress()
        equals = Literal("=").suppress()
        semi   = Literal(";")
        
        comment = semi + Optional( restOfLine )
        
        nonrbrack = "".join( [ c for c in printables if c != "]" ] ) + " \t"
        nonequals = "".join( [ c for c in printables if c != "=" ] ) + " \t"
        
        sectionDef = lbrack + Word( nonrbrack ) + rbrack
        keyDef = ~lbrack + Word( nonequals ) + equals + empty + restOfLine
        # strip any leading or trailing blanks from key
        def stripKey(tokens):
            tokens[0] = tokens[0].strip()
        keyDef.setParseAction(stripKey)
        
        # using Dict will allow retrieval of named data fields as attributes of the parsed results
        inibnf = Dict( ZeroOrMore( Group( sectionDef + Dict( ZeroOrMore( Group( keyDef ) ) ) ) ) )
        
        inibnf.ignore( comment )
        
    return inibnf


pp = pprint.PrettyPrinter(2)

def test( strng ):
    print(strng)
    try:
        iniFile = open(strng)
        iniData = "".join( iniFile.readlines() )
        bnf = inifile_BNF()
        tokens = bnf.parseString( iniData )
        pp.pprint( tokens.asList() )

    except ParseException as err:
        print(err.line)
        print(" "*(err.column-1) + "^")
        print(err)
    
    iniFile.close()
    print()
    return tokens
test('../../example.ini')
# -

alphaword = Word(alphas)
integer = Word(nums)
sexp = Forward()
LPAREN = Suppress("(")
RPAREN = Suppress(")")
sexp << OneOrMore( alphaword | integer | ( LPAREN + ZeroOrMore(sexp) + RPAREN ))
tests = """\
 red
 100 ( hi )
 ( red 100 blue )
 ( green ( ( 1 2 ) mauve ) plaid () )""".splitlines()
for t in tests:
    print(t)
    print(sexp.parseString(t))
    print()

with open('../tests/tex_files/reinhardt/reinhardt-optimal-control.tex', 'r') as rein_file:
    rein = rein_file.read()
#with open('../tests/tex_files/short_xymatrix_example.tex') as xymatrix_file:
#    short_example = xymatrix_file.read()
#with open('../../stacks-tests/orig/perfect.tex') as xymatrix_file:
#    stacks_example = xymatrix_file.read()

# +
cstikzfig = oneOf(["\\tikzfig", "\\mathcal"]).suppress()
lbrace = Literal('{').suppress()
rbrace = Literal('}').suppress()
parens = Word("()%\\")
inside = SkipTo(rbrace)
allchars = Word(printables, excludeChars="{}")
inside = ZeroOrMore(allchars)
inside.setParseAction(lambda tok: " ".join(tok))
content = Forward()
content << OneOrMore(allchars|(lbrace + ZeroOrMore(content) + rbrace))
#content << (allchars + lbrace + ZeroOrMore(content) + rbrace)
content.setParseAction(lambda tok: " ".join(tok))
tikzfig = cstikzfig + lbrace + inside + rbrace + lbrace + inside + rbrace + lbrace + content + rbrace

csxymatrix = oneOf(["\\xymatrix","\\mathcal"]).suppress()
xymatrix = csxymatrix + lbrace + content + rbrace

search_res = tikzfig.searchString(rein)
search_res = xymatrix.searchString(short_example)

#tikzfig.setParseAction(lambda s: ' ')
#clean_str = tikzfig.transformString(rein)

xymatrix.setParseAction(lambda s: ' ')
clean_str = xymatrix.transformString(short_example)

#with open('../../stacks-tests/clean/perfect.tex','+w') as rein_file:
#    rein_file.write(clean_str)

for k,r in enumerate(search_res):
#    name, expl, text  = r
#    print(k,' ', name,' -- ', expl[:15],' -- ', text[:25], '...', text[-25:])
    #name, expl = r
    #print(k, ' ',name,' -- ', expl[:15],'...',expl[-15:])
    #name, expl, text  = r
    #print(k,' ', name,' -- ', expl[:15],' -- ', text[:25], '...', text[-25:])
    #name, expl = r #print(k, ' ',name,' -- ', expl[:15],'...',expl[-15:])
    print(r)
clean_str

# + jupyter={"outputs_hidden": true}
cc = pp.CommandCleaner('underline').del_matches(short_example)
print(cc)
# -

cleaner = pp.CommandCleaner('tikzfig')
clean_str = cleaner.del_matches(rein)
with open('../data/clean_rein_rm/reinhardt-optimal-control.tex', 'w') as r_file:
    r_file.write(clean_str)

# + jupyter={"outputs_hidden": true}
Cleaner = pp.CommandCleaner('xymatrix')
for tex_file in glob.glob('../../stacks-project/*.tex'):
    with open(tex_file, 'r') as tex_o_file:
        clean_str = Cleaner.del_matches(tex_o_file.read())
    basename = os.path.basename(tex_file)
    with open('../../stacks-clean/' + basename, 'w') as clean_file:
        print('Writing file: ' + basename)
        clean_file.write(clean_str)
