# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from pyparsing import \
        Literal, Word, ZeroOrMore, OneOrMore, Group, Dict, Optional, \
        printables, ParseException, restOfLine, empty, \
        Combine, nums, alphanums, Suppress, SkipTo, Forward, printables, alphas
import pprint

#ssn ::= num+ '-' num+ '-' num+
#num ::= '0' | '1' | '2' etc
dash = '-'
ssn = Combine(Word(nums, exact=3) +
                 dash + Word(nums, exact=2) +
                 Suppress('-') + Word(nums, exact=4))
target = '123-45-6789'
result = ssn.parseString(target)
print(result)


class PacKiller:
    def __init__(self, package, environments, standalones):
        '''
       *package* is the package name: ex "xy" 
       *environments* is a list of the environments provided by the package:
           [ "xyenvirons", ]
        *standalones* is a list of macros that the package also provides:
           [ "xymatrix" ]
        '''
        pass


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

# +
cstikzfig = Literal("\\tikzfig").suppress()
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
tikzfig = cstikzfig + lbrace + inside + rbrace + lbrace + inside + rbrace# + lbrace + content +rbrace

search_res = tikzfig.searchString(rein)

for k,r in enumerate(search_res):
    #name, expl, text  = r
    #print(k,' ', name,' -- ', expl[:15],' -- ', text[:25], '...', text[-25:])
    name, expl = r
    print(k, ' ',name,' -- ', expl[:15],'...',expl[-15:])
# -

print(rein[:10000])
