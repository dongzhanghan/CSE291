def rev_call_stmt_side_effects2(x : In[float], _dx_rbnDSC : Out[float], y : In[float], _dy_bbGfPS : Out[float], _dreturn_bhZk9L : In[float]) -> void:
        _t_float_V1WaLi : Array[float, 1]
        _stack_ptr_float_V1WaLi : int = (int)(0)
        _t_int_PiIF9v : Array[int, 1]
        _stack_ptr_int_PiIF9v : int = (int)(0)
        _call_t_0_n8biN1 : int
        y0 : float = ((float)(0.5)) * (y)
        _dy0_Cvx85S : float
        z : float = ((y0) * (y0)) * (x)
        _dz_LzajaL : float
        (_t_float_V1WaLi)[_stack_ptr_float_V1WaLi] = y0
        _stack_ptr_float_V1WaLi = (_stack_ptr_float_V1WaLi) + ((int)(1))
        foo(x,y0)
        (_t_int_PiIF9v)[_stack_ptr_int_PiIF9v] = _call_t_0_n8biN1
        _stack_ptr_int_PiIF9v = (_stack_ptr_int_PiIF9v) + ((int)(1))
        _call_t_0_n8biN1 = (int)(2)
        _dy0_Cvx85S = (_dy0_Cvx85S) + ((int2float(_call_t_0_n8biN1)) * (_dreturn_bhZk9L))
        _dz_LzajaL = (_dz_LzajaL) + (_dreturn_bhZk9L)
        _stack_ptr_int_PiIF9v = (_stack_ptr_int_PiIF9v) - ((int)(1))
        _call_t_0_n8biN1 = (_t_int_PiIF9v)[_stack_ptr_int_PiIF9v]
        _stack_ptr_float_V1WaLi = (_stack_ptr_float_V1WaLi) - ((int)(1))
        y0 = (_t_float_V1WaLi)[_stack_ptr_float_V1WaLi]
        _d_rev_foo(x,_dx_rbnDSC,_dy0_Cvx85S)
        _dy0_Cvx85S = (_dy0_Cvx85S) + ((y0) * ((x) * (_dz_LzajaL)))
        _dy0_Cvx85S = (_dy0_Cvx85S) + ((y0) * ((x) * (_dz_LzajaL)))
        _dx_rbnDSC = (_dx_rbnDSC) + (((y0) * (y0)) * (_dz_LzajaL))
        _dy_bbGfPS = (_dy_bbGfPS) + (((float)(0.5)) * (_dy0_Cvx85S))