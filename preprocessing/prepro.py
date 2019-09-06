from pyparsing import \
        Literal, Word, ZeroOrMore, OneOrMore, Group, Dict, Optional, \
        printables, ParseException, restOfLine, empty, \
        Combine, nums, alphanums, Suppress, SkipTo, Forward, printables, alphas

def patt(cs):
    '''
   Remove the cs with its arguments 
   with recursion on curly brackets
    '''
    cs_literal = Literal(cs).suppress()
    bslash = Literal('\\').suppress()
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

    return bslash + cs_literal + lbrace + content + rbrace

class CommandCleaner:
    def __init__(self, *xargs, **kwargs):
        '''
       *package* is the package name: ex "xy" 
       *environments* is a list of the environments provided by the package:
           [ "xyenvirons", ]
        *standalones* is a list of macros that the package also provides:
           [ "xymatrix" ]
        '''
        if xargs:
            self.pattern = patt(xargs[0])
    
    def show_matches(self, docum):
        '''
        Print the matches
        '''
        return self.pattern.searchString(docum)
    
    def del_matches(self, docum):
        '''
        Delete all matches
        '''
        self.pattern.setParseAction(lambda s: ' ')
        return self.pattern.transformString(stacks_example)
