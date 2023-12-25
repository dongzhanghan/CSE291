from asdl_gen import ADT

def generate_asdl_file():
    ADT("""
    module loma {
      stmt = Assign     ( string target, expr val )
           | Declare    ( string target, expr val )
           | Return     ( expr val )
           attributes   ( int? lineno, string? attr )

      expr = Var        ( string id )
           | ConstFloat ( float val )
           | Add        ( expr left, expr right )
           | Sub        ( expr left, expr right )
           | Mul        ( expr left, expr right )
           | Div        ( expr left, expr right )
           | Equal      ( expr left, expr right )
           attributes   ( int? lineno, string? attr )

      function = Function ( string name, string* args, stmt* body )
                 attributes( int? lineno, string? attr )


    }
    """,
    header= '',
    ext_types = {},
    memoize = ['Var'])
