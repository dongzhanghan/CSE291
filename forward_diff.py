import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            new_args = [loma_ir.Arg(arg.id,autodiff.type_to_diff_type(diff_structs, arg.t),arg.i) for arg in node.args]
            ret_dstruct = autodiff.type_to_diff_type(diff_structs, node.ret_type)
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            new_body = irmutator.flatten(new_body)
            return loma_ir.FunctionDef(\
                diff_func_id, new_args, new_body, node.is_simd, ret_dstruct, lineno = node.lineno)

        def mutate_return(self, node):
            val, dval = self.mutate_expr(node.val)
            match node.val.t:
                case loma_ir.Int():
                    return loma_ir.Return(\
                            val,
                            lineno = node.lineno)
                case loma_ir.Struct():
                    
                    return loma_ir.Return(\
                            node.val,
                            lineno = node.lineno)
                
            return loma_ir.Return(\
            loma_ir.Call('make__dfloat', [val, dval]),
            lineno = node.lineno)

        def mutate_declare(self, node):
            declare_dstruct = autodiff.type_to_diff_type(diff_structs, node.t)
            if node.val == None:               
                return loma_ir.Declare(\
                node.target,
                declare_dstruct,
                None,
                lineno = node.lineno)
            else:
                match node.t:
                    case loma_ir.Int():
                        return node
                    case loma_ir.Struct():
                        return loma_ir.Declare(\
                            node.target,
                            declare_dstruct,
                            node.val,
                            lineno = node.lineno)

                val, dval = self.mutate_expr(node.val)          
                return loma_ir.Declare(\
                node.target,
                declare_dstruct,
                loma_ir.Call('make__dfloat', [val, dval]),
                lineno = node.lineno)

        def mutate_assign(self, node):
            print(node.val)
            val, dval = self.mutate_expr(node.val)
            new_val = loma_ir.Call('make__dfloat', [val, dval])
            new_target = node.target
            match node.target:
                case loma_ir.ArrayAccess():
                    new_target = loma_ir.ArrayAccess(node.target.array, self.mutate_expr(node.target.index)[0], t= node.target.t)
            match node.val.t:             
                case loma_ir.Int():
                    new_val = val
                case loma_ir.Struct():
                    new_val = node.val
            return loma_ir.Assign(\
            new_target,
            new_val,
            lineno = node.lineno)

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            return node, loma_ir.ConstFloat(0.0)

        def mutate_const_int(self, node):
            return node, loma_ir.ConstFloat(0.0)

        def mutate_var(self, node):
            if (node.t == loma_ir.Int()):
                return node,loma_ir.ConstFloat(0.0)
            return loma_ir.StructAccess(node,'val', lineno = node.lineno,t= node.t),loma_ir.StructAccess(node,'dval', lineno = node.lineno,t= node.t)

        def mutate_array_access(self, node):
            new_node = loma_ir.ArrayAccess(\
                node.array,
                (self.mutate_expr(node.index))[0],
                lineno = node.lineno,
                t = node.t)   
            return loma_ir.StructAccess(new_node,'val',lineno = node.lineno,t= node.t), loma_ir.StructAccess(new_node,'dval',lineno = node.lineno,t= node.t)
            
            


        def mutate_struct_access(self, node):
            return loma_ir.StructAccess(\
            node,'val'),loma_ir.StructAccess(\
            node,'dval')

        def mutate_add(self, node):
            
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)
            left_plus_right_val = loma_ir.BinaryOp(\
                loma_ir.Add(),left_val, right_val)
            
            left_plus_right_dval = loma_ir.BinaryOp(\
                loma_ir.Add(),left_dval, right_dval)
            return left_plus_right_val, left_plus_right_dval

        def mutate_sub(self, node):
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)
            left_sub_right_val = loma_ir.BinaryOp(\
                loma_ir.Sub(),left_val, right_val, t=node.t)
            
            left_sub_right_dval = loma_ir.BinaryOp(\
                loma_ir.Sub(),left_dval, right_dval, t=node.t)
            return left_sub_right_val, left_sub_right_dval
        def mutate_mul(self, node):

            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            left_mul_right_val = loma_ir.BinaryOp(\
                loma_ir.Mul(),left_val, right_val, t=node.t)
            
            left_mul_right_dval = loma_ir.BinaryOp(loma_ir.Add(),
                loma_ir.BinaryOp(loma_ir.Mul(),left_dval, right_val),
                loma_ir.BinaryOp(loma_ir.Mul(),left_val, right_dval),t=node.t)

            return left_mul_right_val, left_mul_right_dval

        def mutate_div(self, node):
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            left_div_right_val = loma_ir.BinaryOp(\
                loma_ir.Div(),left_val, right_val)
            
            left_div_right_dval_num = loma_ir.BinaryOp(loma_ir.Sub(),
                loma_ir.BinaryOp(loma_ir.Mul(),left_dval, right_val),
                loma_ir.BinaryOp(loma_ir.Mul(),left_val, right_dval))
            left_div_right_dval_den = loma_ir.BinaryOp(loma_ir.Mul(),right_val, right_val)
            left_div_right_dval = loma_ir.BinaryOp(loma_ir.Div(),left_div_right_dval_num,left_div_right_dval_den)
            return left_div_right_val, left_div_right_dval

        def mutate_call(self, node):
            if (node.id == 'int2float'):
            
                dcall = loma_ir.ConstFloat(0.0)
                return node.args[0],dcall
            if (node.id == 'float2int'):

                call = loma_ir.Call(node.id,[self.mutate_expr(arg)[0] for arg in node.args],lineno = node.lineno,t = loma_ir.Int())
                dcall = loma_ir.ConstFloat(0.0)
                print(call)
                return call,dcall

            call = loma_ir.Call(node.id,[self.mutate_expr(arg)[0] for arg in node.args])
            if (node.id == 'sin'):
                dcall = loma_ir.BinaryOp(loma_ir.Mul(),
                loma_ir.Call('cos',[self.mutate_expr(arg)[0] for arg in node.args]),
                self.mutate_expr(node.args[0])[1])
            
            elif (node.id == 'cos'):
                dcall = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0),
                loma_ir.BinaryOp(loma_ir.Mul(),
                loma_ir.Call('sin',[self.mutate_expr(arg)[0] for arg in node.args]),
                self.mutate_expr(node.args[0])[1]))
            elif (node.id == 'sqrt'):
                dcall = loma_ir.BinaryOp(loma_ir.Div(),
                    loma_ir.BinaryOp(loma_ir.Div(),self.mutate_expr(node.args[0])[1],
                    call), loma_ir.ConstFloat(2.0))
            elif (node.id == 'log'):
                
                dcall = loma_ir.BinaryOp(loma_ir.Div(),self.mutate_expr(node.args[0])[1],
                    self.mutate_expr(node.args[0])[0])
            elif (node.id == 'exp'):
                dcall = loma_ir.BinaryOp(loma_ir.Mul(),call,self.mutate_expr(node.args[0])[1])
            elif (node.id == "pow"):
                left = loma_ir.BinaryOp(loma_ir.Mul(),self.mutate_expr(node.args[0])[1],
                        loma_ir.BinaryOp(loma_ir.Mul(),self.mutate_expr(node.args[1])[0],
                        loma_ir.Call('pow',[self.mutate_expr(node.args[0])[0],loma_ir.BinaryOp(loma_ir.Sub(),self.mutate_expr(node.args[1])[0],loma_ir.ConstFloat(1.0))])))
                right = loma_ir.BinaryOp(loma_ir.Mul(),self.mutate_expr(node.args[1])[1],
                        loma_ir.BinaryOp(loma_ir.Mul(),call,loma_ir.Call('log',[self.mutate_expr(node.args[0])[0]])))
                dcall = loma_ir.BinaryOp(loma_ir.Add(),left, right)
            return call, dcall


    return FwdDiffMutator().mutate_function_def(func)
