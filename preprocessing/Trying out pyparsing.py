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
        Literal, Word, ZeroOrMore, Group, Dict, Optional, \
        printables, ParseException, restOfLine, empty, \
        Combine, nums, Suppress
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

with open('../tests/tex_files/reinhardt/reinhardt-optimal-control.tex', 'r') as rein_file:
    rein = rein_file.read()

cstikzfig = '\\tikzfig'
lbrace = '{'
rbrace = '}'
tikzfig = Word(cstikzfig + lbrace  )
tikzfig.searchString(rein)


