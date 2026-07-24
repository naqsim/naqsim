from enum import Enum, auto
import math
import numpy as np
from qreg_plane import *
from macro import *


class latency_functions:
    def __init__(self,
                 hw_cfg: dict,
                 cx_type: PhyOpType,
                 meas_type: PhyOpType,
                 move_type: MoveType
                 ):
        self.hw_cfg = hw_cfg
        self.cx_type = cx_type
        self.meas_type = meas_type
        self.move_type = move_type
        self._shuttle_tnorm_cache = {}
        #
        self.setup_param()

    def setup_param(self):
        # qubit
        q_cfg = self.hw_cfg['qubit']
        if self.cx_type == PhyOpType.ZONE:
            self.L_um = q_cfg['atom_spacing_um_zone']
        elif self.cx_type == PhyOpType.SELECT:
            self.L_um = q_cfg['atom_spacing_um_select']
        else:
            raise Exception()

        # operation
        op_cfg = self.hw_cfg['operation']
        ##
        self.sq_us = op_cfg['sq_gate']['local']['latency_us']
        ##
        if self.cx_type == PhyOpType.ZONE:
            self.tq_us = op_cfg['tq_gate']['global']['latency_us']
        elif self.cx_type == PhyOpType.SELECT:
            self.tq_us = op_cfg['tq_gate']['local']['latency_us']
            self.tq_us += op_cfg['tq_gate']['local']['switching_us']
        else:
            raise Exception()
        ##
        if self.meas_type == PhyOpType.ZONE:
            self.dm_us = op_cfg['d_measure']['global']['latency_us']
            self.ndm_us = op_cfg['nd_measure']['global']['latency_us']
            self.rst_us = op_cfg['reset']['global']['latency_us']
        elif self.meas_type == PhyOpType.SELECT:
            self.dm_us = op_cfg['d_measure']['local']['latency_us']
            self.ndm_us = op_cfg['nd_measure']['local']['latency_us']
            self.rst_us = op_cfg['reset']['local']['latency_us']
        else:
            raise Exception()
        ##
        self.trf_us = op_cfg['transfer']['latency_us']
        ##
        self.slm_reconf_us = op_cfg['slm_reconf']['latency_us']
        self.slm_onoff_us = op_cfg['slm_onoff']['latency_us']

        return

    def shuttle_us(self, l_um):
        # Assume STA
        if self.move_type == MoveType.STA:
            mass_yb171 = 2.839e-25 # kg
            hbar = 1.054571817e-34 # kg m^2 s^-1
            w0 = 2*math.pi*1e5 # s^-1
            delta_n = 0.1
            sta_coeff = ((((3600*mass_yb171)/(hbar * (w0**5) * delta_n))) ** (1/6))*1e4
            return round(sta_coeff * (l_um ** (1/3)), 2)
        else:
            raise Exception()

    def shuttle_tnorm_at_xnorm(self, shuttle_us, x_norm, n_intervals=200):
        if self.move_type == MoveType.STA:
            if x_norm == 1:
                return 1
            cache_key = (self.move_type, float(shuttle_us), float(x_norm), n_intervals)
            if cache_key in self._shuttle_tnorm_cache:
                return self._shuttle_tnorm_cache[cache_key]
            ###
            w0 = 2*math.pi*1e5
            scale = 1 / ((w0**2) * ((shuttle_us*1e-6)**2))
            ##
            def shuttle_xnorm_at_tnorm(t_norm):
                return (10*(t_norm**3) - 15*(t_norm**4) + 6*(t_norm**5)) + (60*scale*t_norm - 180*scale*(t_norm**2) + 120*scale*(t_norm**3))
            ##
            from scipy.optimize import brentq
            tol = 1e-12
            ##
            xs = np.linspace(0, 1, n_intervals+1)
            roots = []
            vals = shuttle_xnorm_at_tnorm(xs) - x_norm
            #print(vals)
            for i in range(n_intervals):
                v1, v2 = vals[i], vals[i+1]
                if np.isnan(v1) or np.isnan(v2):
                    continue
                if v1 == 0:
                    roots.append(xs[i])
                if v1*v2 < 0:
                    try:
                        r = brentq(lambda t: shuttle_xnorm_at_tnorm(t)-x_norm, xs[i], xs[i+1], xtol=tol)
                        roots.append(r)
                    except ValueError:
                        pass
            roots_unique = []
            for r in roots:
                if not any(abs(r - ru) < 1e-8 for ru in roots_unique):
                    roots_unique.append(r)
            assert len(roots_unique) == 1
            [t_norm] = roots_unique
            self._shuttle_tnorm_cache[cache_key] = t_norm
            return t_norm
        else:
            raise Exception()