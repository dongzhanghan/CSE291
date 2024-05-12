import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

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
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def var_to_differential(expr, var_to_dvar):
        match expr:
            case loma_ir.Var():
                return loma_ir.Var(var_to_dvar[expr.id], t = expr.t)
            case loma_ir.ArrayAccess():
                return loma_ir.ArrayAccess(\
                    var_to_differential(expr.array, var_to_dvar),
                    expr.index,
                    t = expr.t)
            case loma_ir.StructAccess():
                return loma_ir.StructAccess(\
                    var_to_differential(expr.struct, var_to_dvar),
                    expr.member_id,
                    t = expr.t)
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Array():
                return []
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_arg(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_args)
            case _:
                assert False

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Assign(\
                target,
                val,
                lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    class ForwardPassMutator(irmutator.IRMutator):
        def __init__(self, output_args):
            self.out = []
            self.output_args = output_args
            self.cache_vars_list = {}
            self.var_to_dvar = {}
            self.type_cache_size = {}
            self.type_to_stack_and_ptr_names = {}

            self.while_loop_counter = 0
            self.current_while_loop_counter = 0
            self.max_iter = 0
            self.inloop = False

        def mutate_return(self, node):
            return []
        
        def mutate_while(self,node):
            self.inloop = True
            self.max_iter = node.max_iter
            self.while_loop_counter += 1
            self.current_while_loop_counter += 1
            cond = node.cond

            new_body = [self.mutate_stmt(stmt) for stmt in node.body]  
            new_body = irmutator.flatten(new_body)
            
            if self.while_loop_counter == 1:             
                var = loma_ir.Var("_loop_var_0_tmp", t=loma_ir.Int())
                new_body += [loma_ir.Assign(var,loma_ir.BinaryOp(loma_ir.Add(),var,loma_ir.ConstInt(1),t=loma_ir.Int()))]
            if self.while_loop_counter > 1:
                
                if self.while_loop_counter-self.current_while_loop_counter >0:  
                    i = self.current_while_loop_counter
                    var_tmp = loma_ir.Var("_loop_var_"+str(i)+"_tmp", t=loma_ir.Int()) 
                    var_tmp_pre = loma_ir.Var("_loop_var_"+str(i-1)+"_tmp", t=loma_ir.Int()) 
                    a = [loma_ir.Assign(var_tmp,loma_ir.ConstInt(0))]

                    var = loma_ir.Var("_loop_var_"+str(i-1), t=loma_ir.Int())
                    var_ptr = loma_ir.Var("_loop_var_"+str(i)+"_ptr", t=loma_ir.Int())
                    b = [loma_ir.Assign(var_ptr,loma_ir.BinaryOp(loma_ir.Add(),var_ptr,loma_ir.ConstInt(1),t=loma_ir.Int()))]
                    c = [loma_ir.Assign(var_tmp_pre,loma_ir.BinaryOp(loma_ir.Add(),var_tmp_pre,loma_ir.ConstInt(1),t=loma_ir.Int()))]
                    d = [loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var("_loop_var_"+str(i)),loma_ir.Var("_loop_var_"+str(i)+"_ptr"),t=loma_ir.Int()),var_tmp)]
                    new_body = a+new_body+d+b+c
                    self.inloop = False
                else:
                    i = self.current_while_loop_counter-1
                    var_tmp = loma_ir.Var("_loop_var_"+str(i)+"_tmp", t=loma_ir.Int())
                    a = [loma_ir.Assign(var_tmp,loma_ir.BinaryOp(loma_ir.Add(),var_tmp,loma_ir.ConstInt(1),t=loma_ir.Int()))]
                    new_body = new_body+a           
            self.current_while_loop_counter -= 1
            return loma_ir.While(\
                cond,
                node.max_iter,
                new_body,
                lineno = node.lineno)

        def mutate_declare(self, node):
            # For each declaration, add another declaration for the derivatives
            # except when it's an integer
            
            if node.t != loma_ir.Int():
                stmt = []
                dvar = '_d' + node.target + '_' + random_id_generator()
                self.var_to_dvar[node.target] = dvar
                return [node, loma_ir.Declare(\
                    dvar,
                    node.t,
                    lineno = node.lineno)]

            
            else:
                return node


        def mutate_assign(self, node):
            if check_lhs_is_output_arg(node.target, self.output_args):
                return []

            # y = f(x0, x1, ..., y)
            # we will use a temporary array _t to hold variable y for later use:

            # _t[stack_pos] = y
            # y = f(x0, x1, ..., y)
            assign_primal = loma_ir.Assign(\
                node.target,
                self.mutate_expr(node.val),
                lineno = node.lineno)
            # backup
            t_str = type_to_string(node.val.t)
            if t_str in self.type_to_stack_and_ptr_names:
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
            else:
                random_id = random_id_generator()
                stack_name = f'_t_{t_str}_{random_id}'
                stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                self.type_to_stack_and_ptr_names[t_str] = (stack_name, stack_ptr_name)
            
            stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
            cache_var_expr = loma_ir.ArrayAccess(
                loma_ir.Var(stack_name),
                stack_ptr_var,
                t = node.val.t)
            cache_primal = loma_ir.Assign(cache_var_expr, node.target)
            stack_advance = loma_ir.Assign(stack_ptr_var,
                loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))

            if node.val.t in self.cache_vars_list:
                self.cache_vars_list[node.val.t].append((cache_var_expr, node.target))
            else:
                self.cache_vars_list[node.val.t] = [(cache_var_expr, node.target)]
            if node.val.t in self.type_cache_size:
                if self.inloop:
                    self.type_cache_size[node.val.t] += pow(self.max_iter,self.current_while_loop_counter)
                else:
                    self.type_cache_size[node.val.t] += 1

            else:
                if self.inloop:
                    self.type_cache_size[node.val.t] = pow(self.max_iter,self.current_while_loop_counter)
                else:
                    self.type_cache_size[node.val.t] = 1
            return [cache_primal, stack_advance, assign_primal]
        
        def mutate_call_stmt(self, node):
            print(node)
            if node.call.id == "atomic_add":
                return [loma_ir.CallStmt(loma_ir.Call("atomic_add",node.call.args))]
            count = []
            args = funcs[node.call.id].args
            for i in range(len(args)):
                if args[i].i == loma_ir.Out():
                    count.append(i)
            
            for j in range(len(node.call.args)):
                arg = node.call.args[j]
                if j in count:
                    self.out.append(arg.id)
                    var = loma_ir.Var(arg.id, t= arg.t)

                    t_str = type_to_string(var.t)
                    match var.t:
                        case loma_ir.Array():
                            t_str = type_to_string(var.t.t)
                    if t_str in self.type_to_stack_and_ptr_names:
                        stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                    else:
                        random_id = random_id_generator()
                        stack_name = f'_t_{t_str}_{random_id}'
                        stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                        self.type_to_stack_and_ptr_names[t_str] = (stack_name, stack_ptr_name)
            
                    stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                    cache_var_expr = loma_ir.ArrayAccess(
                        loma_ir.Var(stack_name),
                        stack_ptr_var,
                        t = var.t)
                    cache_primal = loma_ir.Assign(cache_var_expr, var)
                    match var.t:
                        case loma_ir.Array():
                            cache_primal = loma_ir.Assign(cache_var_expr,loma_ir.ArrayAccess(var, loma_ir.ConstInt(0)))
                    stack_advance = loma_ir.Assign(stack_ptr_var,
                        loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))
                    print(var.t)
                    match var.t:
                        
                        case loma_ir.Array():
                            if var.t.t in self.cache_vars_list:
                                self.cache_vars_list[var.t.t].append((cache_var_expr, loma_ir.ArrayAccess(var, loma_ir.ConstInt(0))))
                            else:
                                self.cache_vars_list[var.t.t] = [(cache_var_expr, loma_ir.ArrayAccess(var, loma_ir.ConstInt(0)))]
                            if var.t.t in self.type_cache_size:
                                if self.inloop:
                                    self.type_cache_size[var.t.t] += pow(self.max_iter,self.current_while_loop_counter)
                                else:
                                    self.type_cache_size[var.t.t] += 1

                            else:
                                if self.inloop:
                                    self.type_cache_size[var.t.t] = pow(self.max_iter,self.current_while_loop_counter)
                                else:
                                    self.type_cache_size[var.t.t] = 1
                            return [cache_primal, stack_advance, node]

                    if var.t in self.cache_vars_list:
                        self.cache_vars_list[var.t].append((cache_var_expr, var))
                    else:
                        self.cache_vars_list[var.t] = [(cache_var_expr, var)]
                    if var.t in self.type_cache_size:
                        if self.inloop:
                            self.type_cache_size[var.t] += pow(self.max_iter,self.current_while_loop_counter)
                        else:
                            self.type_cache_size[var.t] += 1

                    else:
                        if self.inloop:
                            self.type_cache_size[var.t] = pow(self.max_iter,self.current_while_loop_counter)
                        else:
                            self.type_cache_size[var.t] = 1
                    return [cache_primal, stack_advance, node]

                

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.out = []
            cm = CallNormalizeMutator()
            node = cm.mutate_function_def(node)
            random.seed(hash(node.id))
            # Each input argument is followed by an output (the adjoint)
            # Each output is turned into an input
            # The return value turn into an input
            self.var_to_dvar = {}
            self.while_loop_counter = 0
            new_args = []
            self.output_args = set()
            self.current_while_loop_counter = 0
            
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    dvar_id = '_d' + arg.id + '_' + random_id_generator()
                    new_args.append(loma_ir.Arg(dvar_id, arg.t, i = loma_ir.Out()))
                    self.var_to_dvar[arg.id] = dvar_id
                else:
                    assert arg.i == loma_ir.Out()
                    self.output_args.add(arg.id)
                    new_args.append(loma_ir.Arg(arg.id, arg.t, i = loma_ir.In()))
                    self.var_to_dvar[arg.id] = arg.id
            if node.ret_type is not None:
                self.return_var_id = '_dreturn_' + random_id_generator()
                new_args.append(loma_ir.Arg(self.return_var_id, node.ret_type, i = loma_ir.In()))

            # Forward pass
            fm = ForwardPassMutator(self.output_args)
            forward_body = node.body
            mutated_forward = [fm.mutate_stmt(fwd_stmt) for fwd_stmt in forward_body]
            mutated_forward = irmutator.flatten(mutated_forward)
            self.var_to_dvar = self.var_to_dvar | fm.var_to_dvar

            self.cache_vars_list = fm.cache_vars_list
            self.type_cache_size = fm.type_cache_size
            self.type_to_stack_and_ptr_names = fm.type_to_stack_and_ptr_names
            self.while_loop_counter = fm.while_loop_counter
            self.max_iter = fm.max_iter

            self.out = fm.out

            tmp_declares = []
            for t, exprs in fm.cache_vars_list.items():
                t_str = type_to_string(t)
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                tmp_declares.append(loma_ir.Declare(stack_name,
                    loma_ir.Array(t, self.type_cache_size[t])))
                tmp_declares.append(loma_ir.Declare(stack_ptr_name,
                    loma_ir.Int(), loma_ir.ConstInt(0)))
            if (self.while_loop_counter ==1):
                tmp_declares.append(loma_ir.Declare("_loop_var_0_tmp",
                    loma_ir.Int(), loma_ir.ConstInt(0)))
            if (self.while_loop_counter >1):
                tmp_declares.append(loma_ir.Declare("_loop_var_0_tmp",
                    loma_ir.Int(), loma_ir.ConstInt(0)))
                for i in range(self.while_loop_counter-1):
                    tmp_declares.append(loma_ir.Declare("_loop_var_"+str(i+1),
                        loma_ir.Array(loma_ir.Int(), pow(self.max_iter,i+1))))
                    tmp_declares.append(loma_ir.Declare("_loop_var_"+str(i+1)+"_ptr",
                        loma_ir.Int(), loma_ir.ConstInt(0)))
                    tmp_declares.append(loma_ir.Declare("_loop_var_"+str(i+1)+"_tmp",
                        loma_ir.Int(), loma_ir.ConstInt(0)))
            mutated_forward = tmp_declares + mutated_forward

            # Reverse pass
            self.adj_count = 0
            self.in_assign = False
            self.adj_declaration = []
            reversed_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            reversed_body = irmutator.flatten(reversed_body)

            return loma_ir.FunctionDef(\
                diff_func_id,
                new_args,
                mutated_forward + self.adj_declaration + reversed_body,
                node.is_simd,
                ret_type = None,
                lineno = node.lineno)

        def mutate_return(self, node):
            # Propagate to each variable used in node.val
            self.adj = loma_ir.Var(self.return_var_id, t = node.val.t)
            return self.mutate_expr(node.val)

        def mutate_declare(self, node):
            if node.val is not None:
                if node.t == loma_ir.Int():
                    return []
                self.adj = loma_ir.Var(self.var_to_dvar[node.target])
                return self.mutate_expr(node.val)
            else:
                return []

        def mutate_assign(self, node):
            if node.val.t == loma_ir.Int():
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                    loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop()
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                return stmts

            self.adj = var_to_differential(node.target, self.var_to_dvar)
            if check_lhs_is_output_arg(node.target, self.output_args):
                # if the lhs is an output argument, then we can safely
                # treat this statement the same as "declare"
                return self.mutate_expr(node.val)
            else:
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                    loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop()
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                # First pass: accumulate
                self.in_assign = True
                self.adj_accum_stmts = []
                stmts += self.mutate_expr(node.val)
                self.in_assign = False

                # zero the target differential
                stmts += assign_zero(var_to_differential(node.target, self.var_to_dvar))

                # Accumulate the adjoints back to the target locations
                stmts += self.adj_accum_stmts
                return stmts

        def mutate_ifelse(self, node):
            cond = node.cond
            new_then_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.then_stmts)]
            new_else_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.else_stmts)]
            # Important: mutate_stmt can return a list of statements. We need to flatten the lists.
            new_then_stmts = irmutator.flatten(new_then_stmts)
            new_else_stmts = irmutator.flatten(new_else_stmts)
            return loma_ir.IfElse(\
                cond,
                new_then_stmts,
                new_else_stmts,
                lineno = node.lineno)

        def mutate_call_stmt(self, node):
            if node.call.id == "atomic_add":
                return [loma_ir.CallStmt(loma_ir.Call("atomic_add",node.call.args))]
            count = []
            args = funcs[node.call.id].args
            for i in range(len(args)):
                if args[i].i == loma_ir.Out():
                    count.append(i)
            
            stmts = [] 
            
            for j in range(len(node.call.args)):
                arg = node.call.args[j]
                if j in count:     
            
                    var = loma_ir.Var(arg.id, t= arg.t)
       
                    # restore the previous value of this assignment
                    t_str = type_to_string(var.t)
                    match var.t:
                        case loma_ir.Array():
                            t_str = type_to_string(var.t.t)
                    _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                    stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                    stmts.append(loma_ir.Assign(stack_ptr_var,
                        loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                    match var.t:
                        case loma_ir.Array():
                            cache_var_expr, cache_target = self.cache_vars_list[var.t.t].pop()
                        case loma_ir.Float():
                            cache_var_expr, cache_target = self.cache_vars_list[var.t].pop()
                    stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
             
            stmts += self.mutate_expr(node.call)
            for j in count:
                arg = node.call.args[j]
                dvar = self.var_to_dvar[arg.id]
                print(arg.t)
                if arg.t != loma_ir.Array:
                    stmts += assign_zero(loma_ir.Var(dvar, t= arg.t))
            print(stmts)
            return stmts

        def mutate_while(self, node):
            self.current_while_loop_counter += 1

            new_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
                # Important: mutate_stmt can return a list of statements. We need to flatten the list.
            new_body = irmutator.flatten(new_body)
            
            if self.while_loop_counter == 1:
                var = loma_ir.Var("_loop_var_0_tmp", t=loma_ir.Int())
                cond = loma_ir.BinaryOp(loma_ir.Greater(),var,loma_ir.ConstInt(0))                
                new_body += [loma_ir.Assign(var,loma_ir.BinaryOp(loma_ir.Sub(),var,loma_ir.ConstInt(1),t=loma_ir.Int()))]
                self.while_loop_counter -= 1
                
            if self.while_loop_counter > 1:
                 
                if self.while_loop_counter-self.current_while_loop_counter >0:
                    i = self.current_while_loop_counter
                    var_tmp = loma_ir.Var("_loop_var_"+str(i)+"_tmp", t=loma_ir.Int())
                    var_tmp_pre = loma_ir.Var("_loop_var_"+str(i-1)+"_tmp", t=loma_ir.Int())
                    a = [loma_ir.Assign(var_tmp,loma_ir.ConstInt(0))]

                    var = loma_ir.Var("_loop_var_"+str(i-1), t=loma_ir.Int())
                    cond = loma_ir.BinaryOp(loma_ir.Greater(),var_tmp_pre,loma_ir.ConstInt(0))  
                    var_ptr = loma_ir.Var("_loop_var_"+str(i)+"_ptr", t=loma_ir.Int())
                    b = [loma_ir.Assign(var_ptr,loma_ir.BinaryOp(loma_ir.Sub(),var_ptr,loma_ir.ConstInt(1),t=loma_ir.Int()))]
                    c = [loma_ir.Assign(var_tmp_pre,loma_ir.BinaryOp(loma_ir.Sub(),var_tmp_pre,loma_ir.ConstInt(1),t=loma_ir.Int()))]
                    d = [loma_ir.Assign(var_tmp,loma_ir.ArrayAccess(loma_ir.Var("_loop_var_"+str(i)),loma_ir.Var("_loop_var_"+str(i)+"_ptr"),t=loma_ir.Int()))]
                    new_body = b+d+new_body+c
                    
                else:
                    i = self.current_while_loop_counter-1
                    var_tmp = loma_ir.Var("_loop_var_"+str(i)+"_tmp", t=loma_ir.Int())
                    cond = loma_ir.BinaryOp(loma_ir.Greater(),var_tmp,loma_ir.ConstInt(0))  
                    a = [loma_ir.Assign(var_tmp,loma_ir.BinaryOp(loma_ir.Sub(),var_tmp,loma_ir.ConstInt(1),t=loma_ir.Int()))]
                    new_body = new_body+a
            self.current_while_loop_counter -= 1

            return loma_ir.While(\
                    cond,
                    node.max_iter,
                    new_body,
                    lineno = node.lineno)


        def mutate_var(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False)
                
                return [accum_deriv(target_expr, self.adj, overwrite = True)]
            else:
                return [loma_ir.CallStmt(loma_ir.Call("atomic_add",[var_to_differential(node, self.var_to_dvar),
                    self.adj]))]

        def mutate_const_float(self, node):
            return []

        def mutate_const_int(self, node):
            return []

        def mutate_array_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False)
                return [accum_deriv(target_expr, self.adj, overwrite = True)]
            else:
                return [loma_ir.CallStmt(loma_ir.Call("atomic_add",[var_to_differential(node, self.var_to_dvar),
                    self.adj]))]

        def mutate_struct_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False)
                return [accum_deriv(target_expr, self.adj, overwrite = True)]
            else:
                return [loma_ir.CallStmt(loma_ir.Call("atomic_add",[var_to_differential(node, self.var_to_dvar),
                    self.adj]))]

        def mutate_add(self, node):
            left = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)
            return left + right

        def mutate_sub(self, node):
            old_adj = self.adj
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                loma_ir.ConstFloat(0.0), old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_mul(self, node):
            # z = x * y
            # dz/dx = dz * y
            # dz/dy = dz * x
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                node.right, old_adj)
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                node.left, old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_div(self, node):
            # z = x / y
            # dz/dx = dz / y
            # dz/dy = - dz * x / y^2
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                old_adj, node.right)
            left = self.mutate_expr(node.left)
            # - dz
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                loma_ir.ConstFloat(0.0), old_adj)
            # - dz * x
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                self.adj, node.left)
            # - dz * x / y^2
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                self.adj, loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right))
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_call(self, node):
            match node.id:
                case 'sin':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(\
                            'cos',
                            node.args,
                            lineno = node.lineno,
                            t = node.t),
                        old_adj,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0]) 
                    self.adj = old_adj
                    return ret
                case 'cos':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.Call(\
                                'sin',
                                node.args,
                                lineno = node.lineno,
                                t = node.t),
                            self.adj,
                            lineno = node.lineno),
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0]) 
                    self.adj = old_adj
                    return ret
                case 'sqrt':
                    assert len(node.args) == 1
                    # y = sqrt(x)
                    # dx = (1/2) * dy / y
                    old_adj = self.adj
                    sqrt = loma_ir.Call(\
                        'sqrt',
                        node.args,
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(0.5), self.adj,
                        lineno = node.lineno)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        self.adj, sqrt,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'pow':
                    assert len(node.args) == 2
                    # y = pow(x0, x1)
                    # dx0 = dy * x1 * pow(x0, x1 - 1)
                    # dx1 = dy * pow(x0, x1) * log(x0)
                    base_expr = node.args[0]
                    exp_expr = node.args[1]

                    old_adj = self.adj
                    # base term
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, exp_expr,
                        lineno = node.lineno)
                    exp_minus_1 = loma_ir.BinaryOp(\
                        loma_ir.Sub(),
                        exp_expr, loma_ir.ConstFloat(1.0),
                        lineno = node.lineno)
                    pow_exp_minus_1 = loma_ir.Call(\
                        'pow',
                        [base_expr, exp_minus_1],
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, pow_exp_minus_1,
                        lineno = node.lineno)
                    base_stmts = self.mutate_expr(base_expr)
                    self.adj = old_adj

                    # exp term
                    pow_expr = loma_ir.Call(\
                        'pow',
                        [base_expr, exp_expr],
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, pow_expr,
                        lineno = node.lineno)
                    log = loma_ir.Call(\
                        'log',
                        [base_expr],
                        lineno = node.lineno)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, log,
                        lineno = node.lineno)
                    exp_stmts = self.mutate_expr(exp_expr)
                    self.adj = old_adj
                    return base_stmts + exp_stmts
                case 'exp':
                    assert len(node.args) == 1
                    exp = loma_ir.Call(\
                        'exp',
                        node.args,
                        lineno = node.lineno,
                        t = node.t)
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, exp,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'log':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Div(),
                        self.adj, node.args[0])
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'int2float':
                    # don't propagate the derivatives
                    return []
                case 'float2int':
                    # don't propagate the derivatives
                    return []

                case _:
                    new_args = []
                    if self.in_assign:
                        target = f'_adj_{str(self.adj_count)}'
                        self.adj_count += 1
                        self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                        target_expr = loma_ir.Var(target, t=node.t)
                        
                        
                        for arg in node.args: 
                            new_args.append(arg)
                            dvar_id = self.var_to_dvar[arg.id]
                            new_args.append(loma_ir.Var(dvar_id, t=arg.t))
                        new_args.append(target_expr)
                        fw_func_id = func_to_rev[node.id]
                        self.adj_accum_stmts += \
                                [loma_ir.CallStmt(\
                        loma_ir.Call(fw_func_id,new_args),
                        lineno = node.lineno)]
                        return [accum_deriv(target_expr, self.adj, overwrite = True)]
                    
                    args = funcs[node.id].args
                    no_out = True
                    count = []
                    for j in range(len(args)):  
                        if args[j].i == loma_ir.Out():
                            count.append(j)
                    
                    for j in range(len(node.args)): 
                        arg = node.args[j]
                        if j not in count:
                            new_args.append(arg)
                            dvar_id = self.var_to_dvar[arg.id]
                            new_args.append(loma_ir.Var(dvar_id, t=arg.t))
                        else:
                            no_out = False
                            dvar_id = self.var_to_dvar[arg.id]
                            new_args.append(loma_ir.Var(dvar_id, t=arg.t))
                    if no_out:
                        new_args.append(self.adj)
                    fw_func_id = func_to_rev[node.id]
                    print(new_args)
                    return [loma_ir.CallStmt(\
                    loma_ir.Call(fw_func_id,new_args),
                    lineno = node.lineno)]



    return RevDiffMutator().mutate_function_def(func)