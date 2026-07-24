from qreg_plane import *
from macro import *
#
from enum import Enum, auto
from typing import List
from itertools import product
from copy import deepcopy
import math, random

def intersects(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])

def gen_interval(interval_type, begin, latency):
    return (interval_type, [begin, begin+latency])

def xy_to_dq(x, y, y_max):
    return (x+y, y_max-1-x+y)

def num_unique_cols(pos_list):
    pos_dict = {}
    for (r, c) in pos_list:
        if not r in pos_dict.keys():
            pos_dict[r] = []
        pos_dict[r].append(c)
    ###
    target_col_lists = list(pos_dict.values())
    seen = set()
    unique_col_lists = []
    for col_list in target_col_lists:
        t = tuple(col_list)
        if t not in seen:
            seen.add(t)
            unique_col_lists.append(col_list)
    return len(unique_col_lists)


#
## instruction definition
class InstType(Enum):
    INIT_Z = auto()
    REQ_MY = auto()
    REQ_Y = auto()
    MEAS_Z = auto()
    MEAS_XorZ = auto()
    MEAS_RESET_Z = auto()
    PAULI_X = auto()
    PAULI_Y = auto()
    PAULI_Z = auto()
    TRANS_CX = auto()
    #
    CLR_M = auto()
    CLR_Y = auto()
    #
    TRANS_H_ROT = auto()
    TRANS_H = auto()
    ROTATION = auto()
    #
    REQ_M = auto()
    TRANS_S = auto() # dummy

class InstBase:
    def __init__(
            self,
            inst_name: str,
            inst_type: InstType,
            qreg_names: List[str],
            creg_names_cond: List[str],
            creg_name_out: str
    ):
        self.inst_name = inst_name
        self.inst_type = inst_type
        self.qreg_names = qreg_names
        self.creg_names_cond = creg_names_cond
        self.creg_name_out = creg_name_out
        #
        self.is_scheduled = False

class InstInitZ(InstBase):
    def __init__(self, qreg_names):
        super().__init__(
            InstType.INIT_Z.name,
            InstType.INIT_Z,
            qreg_names,
            [],
            ""
        )

class InstReqMY(InstBase):
    def __init__(self, qreg_names):
        super().__init__(
            InstType.REQ_MY.name,
            InstType.REQ_MY,
            qreg_names,
            [],
            ""
        )

class InstReqY(InstBase):
    def __init__(self, qreg_names):
        super().__init__(
            InstType.REQ_Y.name,
            InstType.REQ_Y,
            qreg_names,
            [],
            ""
        )
class InstReqM(InstBase):
    def __init__(self, qreg_names):
        super().__init__(
            InstType.REQ_M.name,
            InstType.REQ_M,
            qreg_names,
            [],
            ""
        )

class InstMeasZ(InstBase):
    def __init__(self, qreg_names, creg_name_out):
        super().__init__(
            InstType.MEAS_Z.name,
            InstType.MEAS_Z,
            qreg_names,
            [],
            creg_name_out
        )


class InstMeasXorZ(InstBase):
    def __init__(self, qreg_names, creg_names_cond, creg_name_out):
        super().__init__(
            InstType.MEAS_XorZ.name,
            InstType.MEAS_XorZ,
            qreg_names,
            creg_names_cond,
            creg_name_out
        )

class InstMeasResetZ(InstBase):
    def __init__(self, qreg_names, creg_name_out):
        super().__init__(
            InstType.MEAS_RESET_Z.name,
            InstType.MEAS_RESET_Z,
            qreg_names,
            [],
            creg_name_out
        )

class InstPauliX(InstBase):
    def __init__(self, qreg_names, creg_names_cond):
        super().__init__(
            InstType.PAULI_X.name,
            InstType.PAULI_X,
            qreg_names,
            creg_names_cond,
            ""
        )

class InstPauliY(InstBase):
    def __init__(self, qreg_names, creg_names_cond):
        super().__init__(
            InstType.PAULI_Y.name,
            InstType.PAULI_Y,
            qreg_names,
            creg_names_cond,
            ""
        )

class InstPauliZ(InstBase):
    def __init__(self, qreg_names, creg_names_cond):
        super().__init__(
            InstType.PAULI_Z.name,
            InstType.PAULI_Z,
            qreg_names,
            creg_names_cond,
            ""
        )

class InstTransH(InstBase):
    def __init__(self, qreg_names):
        super().__init__(
            InstType.TRANS_H.name,
            InstType.TRANS_H,
            qreg_names,
            [],
            ""
        )

class InstTransS(InstBase): # dummy
    def __init__(self, qreg_names, creg_names_cond):
        super().__init__(
            InstType.TRANS_S.name,
            InstType.TRANS_S,
            qreg_names,
            creg_names_cond,
            ""
        )

class InstTransHRot(InstBase):
    def __init__(self, qreg_names):
        super().__init__(
            InstType.TRANS_H_ROT.name,
            InstType.TRANS_H_ROT,
            qreg_names,
            [],
            ""
        )
        self.visited = False
        self.pred_movs = []
        self.succ_movs = []

class InstRotation(InstBase):
    def __init__(self, qreg_names):
        super().__init__(
            InstType.ROTATION.name,
            InstType.ROTATION,
            qreg_names,
            [],
            ""
        )
        self.visited = False
        self.pred_movs = []
        self.succ_movs = []

class InstTransCX(InstBase):
    def __init__(self, qreg_names, oreg_names, cx_type):
        super().__init__(
            InstType.TRANS_CX.name,
            InstType.TRANS_CX,
            qreg_names,
            [],
            ""
        )
        self.oreg_names = oreg_names
        self.cx_type = cx_type
        self.qreg_statuses = []


##########################################

class UopType(Enum):
    PATCH_H = auto()
    PATCH_RH = auto()
    PATCH_RD = auto()
    PATCH_DIR = auto()
    GROUP_DIR = auto()
    PATCH_CX = auto()
    PATCH_MEAS = auto()
    MOVE = auto()
    ESM = auto()
    #
###
class UopBase:
    def __init__(self,
                 uop_name,
                 uop_type,
                 qreg_names):
        self.uop_name = uop_name
        self.uop_type = uop_type
        self.qreg_names = qreg_names

###
class UopESM(UopBase):
    def __init__(self, cx_type, meas_type):
        super().__init__(UopType.ESM.name,
                         UopType.ESM,
                         [])
        self.cx_type = cx_type
        self.meas_type = meas_type

    def get_intervals_rst(self, offset, lftn):
        intervals_rst = []
        intervals_aod = []
        curr_time = offset
        #
        if self.meas_type == PhyOpType.SELECT:
            rst_us = lftn.rst_us
            intervals_rst.append(gen_interval(IntervalType.QOP, curr_time, rst_us))
            curr_time += rst_us
        else:
            raise Exception()
        #
        self.finish_us = curr_time
        #
        return intervals_rst, intervals_aod

    def get_intervals_body(self, offset, plane, lftn):
        intervals_h = []
        intervals_cx = []
        intervals_aod = []
        curr_time = offset
        #
        if self.cx_type == PhyOpType.SELECT and self.meas_type == PhyOpType.SELECT:
            h_us = lftn.sq_us
            cx_us = lftn.tq_us * 4
            #
            for _ in range(4):
                intervals_h.append(gen_interval(IntervalType.QOP, curr_time, h_us))
                curr_time += h_us
                intervals_cx.append(gen_interval(IntervalType.QOP, curr_time, cx_us))
                curr_time += cx_us
            intervals_h.append(gen_interval(IntervalType.QOP, curr_time, h_us))
            curr_time += h_us
        else:
            raise Exception()
        #
        self.finish_us = curr_time
        return intervals_h, intervals_cx, intervals_aod

    def get_intervals_meas(self, offset, lftn):
        intervals_meas = []
        intervals_aod = []
        curr_time = offset
        #
        if self.meas_type == PhyOpType.SELECT:
            meas_us = lftn.ndm_us
            intervals_meas.append(gen_interval(IntervalType.QOP, curr_time, meas_us))
            curr_time += meas_us
        else:
            raise Exception()
        #
        self.finish_us = curr_time
        return intervals_meas, intervals_aod

###
class UopMove(UopBase):
    def __init__(self, qreg_names, src_dst_list):
        super().__init__(UopType.MOVE.name,
                         UopType.MOVE,
                         qreg_names)
        self.src_dst_list = src_dst_list

    def update(self, plane, qregs, temp=False):
        for qn, (src, dst) in zip(self.qreg_names, self.src_dst_list):
            # update qregs
            qr = qregs[qn]
            old_pos = qr.pos
            assert src == old_pos
            new_pos = dst
            qr.pos = new_pos

            # update plane
            ## move the cells
            old_r, old_c = old_pos
            old_cell =  plane.field[old_r][old_c]
            new_r, new_c = new_pos
            new_cell = plane.field[new_r][new_c]
            ### position
            old_cell.occupants.remove(qn)
            new_cell.occupants.append(qn)
            ### status
            for cell in [old_cell, new_cell]:
                if temp:
                    pass
                else:
                    assert len(cell.occupants) <= 2
                if len(cell.occupants) == 0:
                    cell.status_q = CellStatusQ.FREE
                elif len(cell.occupants) == 1:
                    cell.status_q = CellStatusQ.OCCUPIED_Q
                elif len(cell.occupants) == 2:
                    cell.status_q = CellStatusQ.OVERLAP_Q
                else:
                    if temp:
                        cell.status_q == CellStatusQ.TEMPOVER_Q
                    else:
                        raise Exception()
        #
        if not temp:
            def get_pos_list(move_dir, pos_s, pos_e):
                (src_r, src_c) = pos_s
                (dst_r, dst_c) = pos_e
                if move_dir == MoveDir.UP_Y:
                    assert src_c == dst_c
                    r_range = range(src_r, dst_r-1, -1)
                    c_range = [src_c]
                elif move_dir == MoveDir.DOWN_Y:
                    assert src_c == dst_c
                    r_range = range(src_r, dst_r+1, 1)
                    c_range = [src_c]
                elif move_dir == MoveDir.LEFT_X:
                    assert src_r == dst_r
                    r_range = [src_r]
                    c_range = range(src_c, dst_c-1, -1)
                elif move_dir == MoveDir.RIGHT_X:
                    assert src_r == dst_r
                    r_range = [src_r]
                    c_range = range(src_c, dst_c+1, 1)
                elif move_dir == MoveDir.UP_Q:
                    r_range = range(src_r, dst_r-1, -1)
                    c_range = range(src_c, dst_c-1, -1)
                elif move_dir == MoveDir.DOWN_Q:
                    r_range = range(src_r, dst_r+1, +1)
                    c_range = range(src_c, dst_c+1, +1)
                elif move_dir == MoveDir.LEFT_D:
                    r_range = range(src_r, dst_r+1, +1)
                    c_range = range(src_c, dst_c-1, -1)
                elif move_dir == MoveDir.RIGHT_D:
                    r_range = range(src_r, dst_r-1, -1)
                    c_range = range(src_c, dst_c+1, +1)
                else:
                    raise Exception()
                #
                if move_dir in [MoveDir.UP_Y, MoveDir.DOWN_Y, MoveDir.LEFT_X, MoveDir.RIGHT_X]:
                    pos_list = [(r, c) for (r, c) in product(r_range, c_range)]
                elif move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q, MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                    pos_list = [(r, c) for (r, c) in zip(r_range, c_range)]
                else:
                    raise Exception()
                #
                return pos_list
            #
            for mov_dir, src, dst in self.first_move:
                pos_list = get_pos_list(mov_dir, src, dst)
                for cell in [plane.field[r][c] for (r, c) in pos_list]:
                    if cell.pos == src:
                        cell.is_src.append(True)
                    else:
                        cell.is_src.append(False)
                    #
                    if cell.pos == dst:
                        cell.is_dst.append(True)
                    else:
                        cell.is_dst.append(False)
                    #
                    cell.route_dir.append(mov_dir)
                    cell.status_op = CellStatusOp.MOV
            #
            for mov_dir, src, dst in self.second_move:
                pos_list = get_pos_list(mov_dir, src, dst)
                for cell in [plane.field[r][c] for (r, c) in pos_list]:
                    if cell.pos == src:
                        cell.is_src.append(True)
                    else:
                        cell.is_src.append(False)
                    #
                    if cell.pos == dst:
                        cell.is_dst.append(True)
                    else:
                        cell.is_dst.append(False)
                    #
                    cell.route_dir.append(mov_dir)
                    cell.status_op = CellStatusOp.MOV
        return


    # preprocessing
    def get_move_step(self, plane, src, dst):
        src_r, src_c = src
        dst_r, dst_c = dst
        #
        ## HV
        onestep_h = (src_r == dst_r)
        onestep_v = (src_c == dst_c)
        ## DQ
        src_r_dq, src_c_dq = xy_to_dq(src_r, src_c, plane.h)
        dst_r_dq, dst_c_dq = xy_to_dq(dst_r, dst_c, plane.h)
        onestep_d = (src_r_dq == dst_r_dq)
        onestep_q = (src_c_dq == dst_c_dq)
        ###
        if onestep_h or onestep_v:
            return MoveStep.ONE_STEP_XY
        elif onestep_d or onestep_q:
            return MoveStep.ONE_STEP_DQ
        else:
            return MoveStep.TWO_STEP

    def check_move_step(self, plane):
        move_step_list = []
        for src, dst in self.src_dst_list:
            move_step = self.get_move_step(plane, src, dst)
            move_step_list.append(move_step)
        #
        if any([move_step == MoveStep.TWO_STEP for move_step in move_step_list]):
            self.is_twostep = True
            self.is_onestep = False
        else:
            self.is_twostep = False
            self.is_onestep = True
        return

    def split_move_step(self, plane, twostep_order):
        def move_dir(src, dst):
            src_r, src_c = src
            dst_r, dst_c = dst
            src_r_dq, src_c_dq = xy_to_dq(src_r, src_c, plane.h)
            dst_r_dq, dst_c_dq = xy_to_dq(dst_r, dst_c, plane.h)
            #
            if src_r == dst_r:
                if src_c < dst_c:
                    move_dir = MoveDir.RIGHT_X
                else:
                    move_dir = MoveDir.LEFT_X
            elif src_c == dst_c:
                if src_r < dst_r:
                    move_dir = MoveDir.DOWN_Y
                else:
                    move_dir = MoveDir.UP_Y
            elif src_r_dq == dst_r_dq:
                if src_c_dq < dst_c_dq:
                    move_dir = MoveDir.RIGHT_D
                else:
                    move_dir = MoveDir.LEFT_D
            elif src_c_dq == dst_c_dq:
                if src_r_dq < dst_r_dq:
                    move_dir = MoveDir.DOWN_Q
                else:
                    move_dir = MoveDir.UP_Q
            else:
                raise Exception
            return move_dir

        def get_midpoint(src, dst, order):
            ##
            src_r, src_c = src
            dst_r, dst_c = dst
            #
            if src_r > dst_r and src_c < dst_c: ## upper right
                if order == 'dq_xy':
                   mid_r = src_r
                   mid_c = src_c
                   while True:
                       mid_r -= 1
                       mid_c += 1
                       if mid_r == dst_r or mid_c == dst_c:
                           break
                elif order == 'xy_dq':
                    mid_r = dst_r
                    mid_c = dst_c
                    while True:
                        mid_r += 1
                        mid_c -= 1
                        if mid_r == src_r or mid_c == src_c:
                           break
                elif order == 'xy_hv':
                    mid_r = src_r
                    mid_c = dst_c
                elif order == 'xy_vh':
                    mid_r = dst_r
                    mid_c = src_c
                else:
                    raise Exception()
            elif src_r < dst_r and src_c < dst_c: ## bottom right
                if order == 'dq_xy':
                   mid_r = src_r
                   mid_c = src_c
                   while True:
                       mid_r += 1
                       mid_c += 1
                       if mid_r == dst_r or mid_c == dst_c:
                           break
                elif order == 'xy_dq':
                    mid_r = dst_r
                    mid_c = dst_c
                    while True:
                        mid_r -= 1
                        mid_c -= 1
                        if mid_r == src_r or mid_c == src_c:
                           break
                elif order == 'xy_hv':
                    mid_r = src_r
                    mid_c = dst_c
                elif order == 'xy_vh':
                    mid_r = dst_r
                    mid_c = src_c
                else:
                    print(src, dst)
                    raise Exception()
            elif src_r > dst_r and src_c > dst_c: ## upper left
                if order == 'dq_xy':
                   mid_r = src_r
                   mid_c = src_c
                   while True:
                       mid_r -= 1
                       mid_c -= 1
                       if mid_r == dst_r or mid_c == dst_c:
                           break
                elif order == 'xy_dq':
                    mid_r = dst_r
                    mid_c = dst_c
                    while True:
                        mid_r += 1
                        mid_c += 1
                        if mid_r == src_r or mid_c == src_c:
                           break
                elif order == 'xy_hv':
                    mid_r = src_r
                    mid_c = dst_c
                elif order == 'xy_vh':
                    mid_r = dst_r
                    mid_c = src_c
                else:
                    raise Exception()
            elif src_r < dst_r  and src_c > dst_c: ## bottom left
                if order == 'dq_xy':
                   mid_r = src_r
                   mid_c = src_c
                   while True:
                       mid_r += 1
                       mid_c -= 1
                       if mid_r == dst_r or mid_c == dst_c:
                           break
                elif order == 'xy_dq':
                    mid_r = dst_r
                    mid_c = dst_c
                    while True:
                        mid_r -= 1
                        mid_c += 1
                        if mid_r == src_r or mid_c == src_c:
                           break
                elif order == 'xy_hv':
                    mid_r = src_r
                    mid_c = dst_c
                elif order == 'xy_vh':
                    mid_r = dst_r
                    mid_c = src_c
                else:
                    raise Exception()
            else:
                    raise Exception()
            return (mid_r, mid_c)

        if self.is_onestep:
            self.first_move = [(move_dir(src, dst), src, dst) for (src, dst) in self.src_dst_list]
            self.second_move = []
            pass
        elif self.is_twostep:
            xy_onestep = []
            dq_onestep = []
            xy_twostep = []
            dq_twostep = []
            for (src, dst) in self.src_dst_list:
                move_step = self.get_move_step(plane, src, dst)
                if move_step == MoveStep.ONE_STEP_XY:
                    xy_onestep.append((move_dir(src, dst), src, dst))
                elif move_step == MoveStep.ONE_STEP_DQ:
                    dq_onestep.append((move_dir(src, dst), src, dst))
                elif move_step == MoveStep.TWO_STEP:
                    mid = get_midpoint(src, dst, twostep_order)
                    first_move_step = self.get_move_step(plane, src, mid)
                    second_move_step = self.get_move_step(plane, mid, dst)
                    if twostep_order == 'dq_xy':
                        assert first_move_step == MoveStep.ONE_STEP_DQ
                        assert second_move_step == MoveStep.ONE_STEP_XY
                        dq_twostep.append((move_dir(src, mid), src, mid))
                        xy_twostep.append((move_dir(mid, dst), mid, dst))
                    elif twostep_order == 'xy_dq':
                        assert first_move_step == MoveStep.ONE_STEP_XY
                        assert second_move_step == MoveStep.ONE_STEP_DQ
                        xy_twostep.append((move_dir(src, mid), src, mid))
                        dq_twostep.append((move_dir(mid, dst), mid, dst))
                    else:
                        raise Exception()
                else:
                    raise Exception()
            #
            if twostep_order == 'dq_xy':
                self.first_move = dq_onestep + dq_twostep
                self.second_move = xy_onestep + xy_twostep
            elif twostep_order == 'xy_dq':
                self.first_move = xy_onestep + xy_twostep
                self.second_move = dq_onestep + dq_twostep
            else:
                raise Exception()
        else:
            raise Exception()
        return

    ###
    def set_aod_type(self, aod_type, num_aodh_max, num_aodd_max, num_aodr_max):
        if aod_type == AodType.AODH:
            assert num_aodh_max >= 1
        elif aod_type == AodType.AODD:
            raise Exception()
        else:
            raise Exception()
        self.aod_type = aod_type
        self.num_use_aod = 1
        return

    def check_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        if self.aod_type == AodType.AODH:
            return len(aodh_free_list) >= 1
        elif self.aod_type == AodType.AODD:
            raise Exception()
        else:
            raise Exception()

    def get_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        aod_list = []
        if self.aod_type == AodType.AODH:
           for _ in range(self.num_use_aod):
               aod_list.append(aodh_free_list.pop(0))
        elif self.aod_type == AodType.AODD:
            for _ in range(self.num_use_aod):
               aod_list.append(aodd_free_list.pop(0))
        else:
            raise Exception()
        return aod_list

    ####
    def calculate_latency(self, plane, qregs, lftn, code_dist):
        # pick/drop
        ## first pick
        fp_xy_pos_list = [cell.pos for cell in self.first_pick_cells]
        first_pick_sf = num_unique_cols(fp_xy_pos_list)
        self.first_pick_us = first_pick_sf * lftn.trf_us
        ## first drop
        fd_xy_pos_list = [cell.pos for cell in self.first_drop_cells]
        first_drop_sf = num_unique_cols(fd_xy_pos_list)
        self.first_drop_us = first_drop_sf * lftn.trf_us
        ## second pick
        sp_xy_pos_list = [cell.pos for cell in self.second_pick_cells]
        second_pick_sf = num_unique_cols(sp_xy_pos_list)
        self.second_pick_us = second_pick_sf * lftn.trf_us
        ## second drop
        sd_xy_pos_list = [cell.pos for cell in self.second_drop_cells]
        second_drop_sf = num_unique_cols(sd_xy_pos_list)
        self.second_drop_us = second_drop_sf * lftn.trf_us

        # shuttle
        def cal_shuttle_latency(move_list):
            max_route_len = max([math.sqrt((src_r-dst_r)**2 + (src_c-dst_c)**2) for _, (src_r, src_c), (dst_r, dst_c) in move_list])
            #
            if plane.cell_size == CellSize.SMALLEST:
                shuttle_um = max_route_len * (code_dist * lftn.L_um)
            elif plane.cell_size == CellSize.DOUBLE_TE:
                shuttle_um = 1.5 * max_route_len * (code_dist * lftn.L_um)
            elif plane.cell_size == CellSize.DOUBLE_DIR:
                shuttle_um = math.sqrt(2) * max_route_len * (code_dist * lftn.L_um)
            else:
                raise Exception()
            shuttle_us = lftn.shuttle_us(shuttle_um)
            return shuttle_us
        ##
        assert self.first_move
        self.first_shuttle_us = cal_shuttle_latency(self.first_move)
        if self.second_move:
            self.second_shuttle_us = cal_shuttle_latency(self.second_move)
        else:
            self.second_shuttle_us = 0
        return

    ###
    def build_route(self, plane, route, pass_target, shuttle_us, xnorm_to_tnorm):

        def set_cell_tnorm(cell, cell_part, enter_x, exit_x, len_x, shuttle_us, xnorm_to_tnorm):
            if (enter_x is not None) and (exit_x is not None):
                enter_xnorm = enter_x / len_x
                exit_xnorm = exit_x / len_x
                enter_tnorm = xnorm_to_tnorm(shuttle_us, enter_xnorm)
                exit_tnorm = xnorm_to_tnorm(shuttle_us, exit_xnorm)
                #
                if cell_part == CellPart.C:
                    cell.enter_tnorm_c = enter_tnorm
                    cell.exit_tnorm_c = exit_tnorm
                elif cell_part == CellPart.E:
                    cell.enter_tnorm_e = enter_tnorm
                    cell.exit_tnorm_e = exit_tnorm
                elif cell_part == CellPart.S:
                    cell.enter_tnorm_s = enter_tnorm
                    cell.exit_tnorm_s = exit_tnorm
                elif cell_part == CellPart.SE:
                    cell.enter_tnorm_se = enter_tnorm
                    cell.exit_tnorm_se = exit_tnorm
                else:
                    raise Exception()
            else:
                if cell_part == CellPart.C:
                    cell.enter_tnorm_c = None
                    cell.exit_tnorm_c = None
                elif cell_part == CellPart.E:
                    cell.enter_tnorm_e = None
                    cell.exit_tnorm_e = None
                elif cell_part == CellPart.S:
                    cell.enter_tnorm_s = None
                    cell.exit_tnorm_s = None
                elif cell_part == CellPart.SE:
                    cell.enter_tnorm_se = None
                    cell.exit_tnorm_se = None
                else:
                    raise Exception()
            return

        def append_cell_to_route(pass_target, cell, cell_part, lock_type):
            if cell_part == CellPart.C:
                cond = (cell.enter_tnorm_c is not None) and (cell.exit_tnorm_c is not None)
            elif cell_part == CellPart.E:
                cond = (cell.enter_tnorm_e is not None) and (cell.exit_tnorm_e is not None)
            elif cell_part == CellPart.S:
                cond = (cell.enter_tnorm_s is not None) and (cell.exit_tnorm_s is not None)
            elif cell_part == CellPart.SE:
                cond = (cell.enter_tnorm_se is not None) and (cell.exit_tnorm_se is not None)
            else:
                raise Exception()
            #
            if cond:
                if not lock_type in pass_target.keys():
                    pass_target[lock_type] = []
                pass_target[lock_type].append(cell)
            else:
                pass
            return


        #########################
        move_dir, (src_r, src_c), (dst_r, dst_c) = route
        if move_dir == MoveDir.UP_Y:
            assert src_c == dst_c
            r_range = range(src_r, dst_r-1, -1)
            c_range = [src_c]
        elif move_dir == MoveDir.DOWN_Y:
            assert src_c == dst_c
            r_range = range(src_r, dst_r+1, 1)
            c_range = [src_c]
        elif move_dir == MoveDir.LEFT_X:
            assert src_r == dst_r
            r_range = [src_r]
            c_range = range(src_c, dst_c-1, -1)
        elif move_dir == MoveDir.RIGHT_X:
            assert src_r == dst_r
            r_range = [src_r]
            c_range = range(src_c, dst_c+1, 1)
        elif move_dir == MoveDir.UP_Q:
            r_range = range(src_r, dst_r-1, -1)
            c_range = range(src_c, dst_c-1, -1)
        elif move_dir == MoveDir.DOWN_Q:
            r_range = range(src_r, dst_r+1, +1)
            c_range = range(src_c, dst_c+1, +1)
        elif move_dir == MoveDir.LEFT_D:
            r_range = range(src_r, dst_r+1, +1)
            c_range = range(src_c, dst_c-1, -1)
        elif move_dir == MoveDir.RIGHT_D:
            r_range = range(src_r, dst_r-1, -1)
            c_range = range(src_c, dst_c+1, +1)
        else:
            raise Exception()

        ##
        if self.aod_type == AodType.AODH:
            # AODH - Move in XY axis
            if move_dir in [MoveDir.UP_Y, MoveDir.DOWN_Y, MoveDir.LEFT_X, MoveDir.RIGHT_X]:
                pos_list = [(r, c) for (r, c) in product(r_range, c_range)]
                ##
                for idx, (r, c) in enumerate(pos_list):   ### AODH-XY-SMALLEST ###
                    if plane.cell_size in [CellSize.SMALLEST, CellSize.DOUBLE_DIR]:
                        len_x = len(pos_list)-1

                        # cell on the route
                        cell = plane.field[r][c]
                        cell.init_tnorm()

                        ## center
                        ### set tnorm values
                        #### x values
                        if idx == 0:
                            enter_x_c = 0
                        else:
                            enter_x_c = (idx-1)
                        if idx == len(pos_list)-1:
                            exit_x_c = len(pos_list)-1
                        else:
                            exit_x_c = idx+1
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.C, LockType.SQR)

                    ### AODH-XY-DOUBLE ###
                    elif plane.cell_size == CellSize.DOUBLE_TE:
                        len_x = 1.5*(len(pos_list)-1)

                        # cell on the route
                        cell = plane.field[r][c]
                        cell.init_tnorm()

                        ## center
                        ### set tnorm values
                        #### x values
                        if idx == 0:
                            enter_x_c = 0
                        else:
                            enter_x_c = (1.5*idx) - 1
                        if idx == len(pos_list)-1:
                            exit_x_c = (1.5*idx)
                        else:
                            exit_x_c = (1.5*idx) + 1
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.C, LockType.SQR)

                        ## east
                        ### set tnorm values
                        #### x values
                        if move_dir == MoveDir.LEFT_X and idx != 0:
                            enter_x_e = 1.5 * (idx-1)
                            exit_x_e = 1.5 * (idx)
                        elif move_dir == MoveDir.RIGHT_X and idx != len(pos_list)-1:
                            enter_x_e = 1.5 * idx
                            exit_x_e = 1.5 * (idx+1)
                        else:
                            enter_x_e = None
                            exit_x_e = None
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.E, enter_x_e, exit_x_e, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.E, LockType.E)

                        ## south
                        ### set tnorm values
                        #### x values
                        if move_dir == MoveDir.UP_Y and idx != 0:
                            enter_x_s = 1.5 * (idx-1)
                            exit_x_s = 1.5 * (idx)
                        elif move_dir == MoveDir.DOWN_Y and idx != len(pos_list)-1:
                            enter_x_s = 1.5 * (idx)
                            exit_x_s = 1.5 * (idx+1)
                        else:
                            enter_x_s = None
                            exit_x_s = None
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.S, enter_x_s, exit_x_s, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.S, LockType.S)

                        ## south-east
                        ### pass

                    ### AODH-XY-OTHERS -> EXCEPTION ###
                    else:
                        raise Exception()

            # AODH - Move in DQ axis
            elif move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q, MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                pos_list = [(r, c) for (r, c) in zip(r_range, c_range)]
                ##
                for idx, (r, c) in enumerate(pos_list):
                    ### AODH-DQ-SMALLEST ###
                    if plane.cell_size in [CellSize.SMALLEST, CellSize.DOUBLE_DIR]:
                        len_x = len(pos_list)-1

                        ############################
                        # cell on the route
                        cell = plane.field[r][c]
                        cell.init_tnorm()

                        ## center
                        ### set tnorm values
                        #### x values
                        if idx == 0:
                            enter_x_c = 0
                        else:
                            enter_x_c = (idx-1)
                        if idx == len(pos_list)-1:
                            exit_x_c = len(pos_list)-1
                        else:
                            exit_x_c = idx+1
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.C, LockType.SQR)

                        ############################
                        # cell that right/left portion touched
                        if idx == len(pos_list) - 1:
                            continue
                        #
                        if move_dir == MoveDir.UP_Q:
                            cell_r = plane.field[r][c-1]
                            cell_l = plane.field[r-1][c]
                        elif move_dir == MoveDir.DOWN_Q:
                            cell_r = plane.field[r+1][c]
                            cell_l = plane.field[r][c+1]
                        elif move_dir == MoveDir.LEFT_D:
                            cell_r = plane.field[r][c-1]
                            cell_l = plane.field[r+1][c]
                        elif move_dir == MoveDir.RIGHT_D:
                            cell_r = plane.field[r-1][c]
                            cell_l = plane.field[r][c+1]
                        else:
                            raise Exception()
                        cell_r.init_tnorm()
                        cell_l.init_tnorm()

                        ############################
                        # cell right portion touched
                        ## center
                        ### set tnorm values
                        #### x values
                        enter_x_c = idx
                        exit_x_c = idx+1
                        #### tnorm values
                        set_cell_tnorm(cell_r, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        if move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q]:
                            lock_type = LockType.QR
                        elif move_dir in [MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                            lock_type = LockType.DR
                        else:
                            raise Exception()
                        append_cell_to_route(pass_target, cell_r, CellPart.C, lock_type)

                        ############################
                        # cell left portion touched
                        ## center
                        ### set tnorm values
                        #### x values
                        enter_x_c = idx
                        exit_x_c = idx+1
                        #### tnorm values
                        set_cell_tnorm(cell_l, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        if move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q]:
                            lock_type = LockType.QL
                        elif move_dir in [MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                            lock_type = LockType.DL
                        else:
                            raise Exception()
                        append_cell_to_route(pass_target, cell_l, CellPart.C, lock_type)

                    ### AODH-DQ-DOUBLE ###
                    elif plane.cell_size == CellSize.DOUBLE_TE:
                        len_x = 1.5 * (len(pos_list)-1)

                        ############################
                        # cell on the route
                        cell = plane.field[r][c]
                        cell.init_tnorm()

                        ## center
                        ### set tnorm values
                        #### x values
                        if idx == 0:
                            enter_x_c = 0
                        else:
                            enter_x_c = (1.5*idx)-1
                        if idx == len(pos_list)-1:
                            exit_x_c = (1.5*idx)
                        else:
                            exit_x_c = (1.5*idx)+1
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.C, LockType.SQR)

                        ## east
                        ### set tnorm values
                        #### x values
                        if move_dir == MoveDir.UP_Q and idx != 0:
                            enter_x_e = (1.5*idx) - 1
                            exit_x_e = (1.5*idx)
                        elif move_dir == MoveDir.DOWN_Q and idx != len(pos_list)-1:
                            enter_x_e = (1.5*idx)
                            exit_x_e = (1.5*idx) + 1
                        elif move_dir == MoveDir.LEFT_D and idx != 0:
                            enter_x_e = (1.5*idx) - 1
                            exit_x_e = (1.5*idx)
                        elif move_dir == MoveDir.RIGHT_D and idx != len(pos_list)-1:
                            enter_x_e = (1.5*idx)
                            exit_x_e = (1.5*idx) + 1
                        else:
                            enter_x_e = None
                            exit_x_e = None
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.E, enter_x_e, exit_x_e, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.E, LockType.E)

                        ## south
                        ### set tnorm values
                        #### x values
                        if move_dir == MoveDir.UP_Q and idx != 0:
                            enter_x_s = (1.5*idx) - 1
                            exit_x_s = (1.5*idx)
                        elif move_dir == MoveDir.DOWN_Q and idx != len(pos_list)-1:
                            enter_x_s = (1.5*idx)
                            exit_x_s = (1.5*idx) + 1
                        elif move_dir == MoveDir.LEFT_D and idx != len(pos_list)-1:
                            enter_x_s = (1.5*idx)
                            exit_x_s = (1.5*idx) + 1
                        elif move_dir == MoveDir.RIGHT_D and idx != 0:
                            enter_x_s = (1.5*idx) - 1
                            exit_x_s = (1.5*idx)
                        else:
                            enter_x_s = None
                            exit_x_s = None
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.S, enter_x_s, exit_x_s, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.S, LockType.S)

                        ## south-east
                        ### set tnorm values
                        #### x values
                        if move_dir == MoveDir.UP_Q and idx != 0:
                            enter_x_se = 1.5 * (idx-1)
                            exit_x_se = 1.5 * (idx)
                        elif move_dir == MoveDir.DOWN_Q and idx != len(pos_list)-1:
                            enter_x_se = 1.5 * (idx)
                            exit_x_se = 1.5 * (idx+1)
                        else:
                            enter_x_se = None
                            exit_x_se = None
                        #### tnorm values
                        set_cell_tnorm(cell, CellPart.SE, enter_x_se, exit_x_se, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        append_cell_to_route(pass_target, cell, CellPart.SE, LockType.SE)

                        ############################
                        # cell that right/left portion touched
                        if idx == len(pos_list) - 1:
                            continue
                        #
                        if move_dir == MoveDir.UP_Q:
                            cell_r = plane.field[r][c-1]
                            cell_l = plane.field[r-1][c]
                        elif move_dir == MoveDir.DOWN_Q:
                            cell_r = plane.field[r+1][c]
                            cell_l = plane.field[r][c+1]
                        elif move_dir == MoveDir.LEFT_D:
                            cell_r = plane.field[r][c-1]
                            cell_l = plane.field[r+1][c]
                        elif move_dir == MoveDir.RIGHT_D:
                            cell_r = plane.field[r-1][c]
                            cell_l = plane.field[r][c+1]
                        else:
                            raise Exception()
                        cell_r.init_tnorm()
                        cell_l.init_tnorm()

                        ############################
                        # cell right portion touched
                        ## center
                        ### set tnorm values
                        #### x values
                        enter_x_c = (1.5*idx) + 0.5
                        exit_x_c = (1.5*idx) + 1
                        #### tnorm values
                        set_cell_tnorm(cell_r, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        if move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q]:
                            lock_type = LockType.QRR
                        elif move_dir in [MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                            lock_type = LockType.DRR
                        else:
                            raise Exception()
                        append_cell_to_route(pass_target, cell_r, CellPart.C, lock_type)

                        ## east
                        ### set tnorm values
                        #### x values
                        if move_dir in [MoveDir.UP_Q, MoveDir.LEFT_D]:
                            enter_x_e = (1.5*idx)
                            exit_x_e = (1.5*idx) + 1
                        elif move_dir in [MoveDir.DOWN_Q, MoveDir.RIGHT_D]:
                            enter_x_e = (1.5*idx) + 0.5
                            exit_x_e = (1.5*idx) + 1.5
                        else:
                            raise Exception()
                        #### tnorm values
                        set_cell_tnorm(cell_r, CellPart.E, enter_x_e, exit_x_e, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        lock_type = LockType.E
                        append_cell_to_route(pass_target, cell_r, CellPart.E, lock_type)

                        ## south
                        ### set tnorm values
                        #### x values
                        if move_dir == MoveDir.LEFT_D:
                            enter_x_s = (1.5*idx) + 0.5
                            exit_x_s = (1.5*idx) + 1.5
                        elif move_dir == MoveDir.RIGHT_D:
                            enter_x_s = (1.5*idx)
                            exit_x_s = (1.5*idx) + 1
                        elif move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q]:
                            enter_x_s = None
                            exit_x_s = None
                        else:
                            raise Exception()
                        #### tnorm values
                        set_cell_tnorm(cell_r, CellPart.S, enter_x_s, exit_x_s, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        lock_type = LockType.S
                        append_cell_to_route(pass_target, cell_r, CellPart.S, lock_type)

                        ## south-east
                        ### set tnorm values
                        #### x values
                        if move_dir in [MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                            enter_x_se = (1.5*idx)
                            exit_x_se = (1.5*idx) + 1.5
                        elif move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q]:
                            enter_x_se = None
                            exit_x_se = None
                        else:
                            raise Exception()
                        #### tnorm values
                        set_cell_tnorm(cell_r, CellPart.SE, enter_x_se, exit_x_se, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        lock_type = LockType.SE
                        append_cell_to_route(pass_target, cell_r, CellPart.SE, lock_type)

                        ############################
                        # cell left portion touched
                        ## center
                        ### set tnorm values
                        #### x values
                        enter_x_c = (1.5*idx) + 0.5
                        exit_x_c = (1.5*idx) + 1
                        #### tnorm values
                        set_cell_tnorm(cell_l, CellPart.C, enter_x_c, exit_x_c, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        if move_dir in [MoveDir.UP_Q, MoveDir.DOWN_Q]:
                            lock_type = LockType.QLL
                        elif move_dir in [MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                            lock_type = LockType.DLL
                        else:
                            raise Exception()
                        append_cell_to_route(pass_target, cell_l, CellPart.C, lock_type)

                        ## east
                        ### pass

                        ## south
                        ### set tnorm values
                        #### x values
                        if move_dir == MoveDir.UP_Q:
                            enter_x_s = 1.5*idx
                            exit_x_s = 1.5*idx + 1
                        elif move_dir == MoveDir.DOWN_Q:
                            enter_x_s = 1.5*idx + 0.5
                            exit_x_s = 1.5*idx + 1.5
                        elif move_dir in [MoveDir.LEFT_D, MoveDir.RIGHT_D]:
                            enter_x_s = None
                            exit_x_s = None
                        else:
                            raise Exception()
                        #### tnorm values
                        set_cell_tnorm(cell_l, CellPart.S, enter_x_s, exit_x_s, len_x, shuttle_us, xnorm_to_tnorm)
                        ### track lock_type and related cells
                        lock_type = LockType.S
                        append_cell_to_route(pass_target, cell_l, CellPart.S, lock_type)

                        ## south-east
                        ### pass

                    ### AODH-DQ-OTHERS -> EXCEPTION ###
                    else:
                        raise Exception()

        elif self.aod_type == AodType.AODD:
            raise Exception()
        else:
            raise Exception()

        return



    ###
    def inspect_delay(self, offset, plane, qregs, lftn, code_dist):
        # Calculate delay
        def cal_pick_delay(target_cells, first_shuttle_us, pick_us, offset):
            if self.aod_type == AodType.AODH:
                ret_delay_us = 0
                #
                while True:
                    curr_delay_us = 0

                    # AODH - Calculate pick delay
                    ## cells to pick
                    for cell in target_cells:
                        cell_delay_us = 0
                        #
                        enter_t = ret_delay_us + offset
                        exit_t = enter_t + (pick_us + cell.exit_tnorm_c*first_shuttle_us)
                        #
                        for lock_type, lock_intervals in cell.lock_dict.items():
                            ## center
                            if lock_type in LOCK_TYPES_C:
                                lock_intervals.sort(key=lambda x: x[0])
                                for (lock_s, lock_e) in lock_intervals:
                                    if intersects((enter_t, exit_t), (lock_s, lock_e)):
                                        assert lock_e > enter_t
                                        gap = lock_e - enter_t
                                        cell_delay_us += gap
                                        enter_t += gap
                                        exit_t += gap
                                    else:
                                        pass
                            else:
                                pass
                        if cell_delay_us > 1e-9:
                            pass
                        curr_delay_us = max(curr_delay_us, cell_delay_us)
                    ##
                    if curr_delay_us > 1e-9:
                        ret_delay_us += curr_delay_us
                    else:
                        break
                ##
                return ret_delay_us


            elif self.aod_type == AodType.AODD:
                raise Exception()
            else:
                raise Exception()

        # shuttle delay
        def cal_shuttle_delay(pass_target, offset, shuttle_us):
            ret_delay_us = 0
            iter = 0
            #
            while True:
                curr_delay_us = 0
                for pass_type, cell_list in pass_target.items():
                    ##
                    if plane.cell_size in [CellSize.SMALLEST, CellSize.DOUBLE_DIR]:
                        assert pass_type in LOCK_TYPES_C
                        pass_part = CellPart.C
                        ##
                        if pass_type == LockType.SQR:
                            conflict_types = LOCK_TYPES_C
                        elif pass_type == LockType.QR:
                            conflict_types = [lt for lt in LOCK_TYPES_C if not lt in [LockType.QL]]
                        elif pass_type == LockType.QL:
                            conflict_types = [lt for lt in LOCK_TYPES_C if not lt in [LockType.QR]]
                        elif pass_type == LockType.DR:
                            conflict_types = [lt for lt in LOCK_TYPES_C if not lt in [LockType.DL]]
                        elif pass_type == LockType.DL:
                            conflict_types = [lt for lt in LOCK_TYPES_C if not lt in [LockType.DR]]
                        else:
                            raise Exception()
                    ##
                    elif plane.cell_size == CellSize.DOUBLE_TE:
                        if pass_type in LOCK_TYPES_C:
                            pass_part = CellPart.C
                            ##
                            compatible_types = []
                            if pass_type == LockType.SQR:
                                pass
                            elif pass_type in [LockType.QRR, LockType.QLL, LockType.DRR, LockType.DLL]:
                                compatible_types += [lt for lt in [LockType.QRR, LockType.QLL, LockType.DRR, LockType.DLL] if lt != pass_type]
                            else:
                                raise Exception()
                            ##
                            conflict_types = [lt for lt in LOCK_TYPES_C if not lt in compatible_types]
                        elif pass_type in LOCK_TYPES_E:
                            pass_part = CellPart.E
                            conflict_types = LOCK_TYPES_E
                        elif pass_type in LOCK_TYPES_S:
                            pass_part = CellPart.S
                            conflict_types = LOCK_TYPES_S
                        elif pass_type in LOCK_TYPES_SE:
                            pass_part = CellPart.SE
                            conflict_types = LOCK_TYPES_SE
                        else:
                            raise Exception()
                    ##
                    else:
                        raise Exception()

                    ## check each cell's delay
                    for cell_idx, cell in enumerate(cell_list):
                        cell_delay_us = 0
                        #
                        if pass_part == CellPart.C:
                            enter_tnorm = cell.enter_tnorm_c
                            exit_tnorm = cell.exit_tnorm_c
                        elif pass_part == CellPart.E:
                            enter_tnorm = cell.enter_tnorm_e
                            exit_tnorm = cell.exit_tnorm_e
                        elif pass_part == CellPart.S:
                            enter_tnorm = cell.enter_tnorm_s
                            exit_tnorm = cell.exit_tnorm_s
                        elif pass_part == CellPart.SE:
                            enter_tnorm = cell.enter_tnorm_se
                            exit_tnorm = cell.exit_tnorm_se
                        else:
                            raise Exception()
                        #
                        enter_t = ret_delay_us + offset + shuttle_us * enter_tnorm
                        exit_t = ret_delay_us + offset + shuttle_us * exit_tnorm

                        ###
                        for lock_type, lock_intervals in cell.lock_dict.items():
                            if (lock_type in conflict_types) or (lock_type == LockType.SOL and cell_idx == len(cell_list)-1):
                                lock_intervals.sort(key=lambda x: x[0])
                                for (lock_s, lock_e) in lock_intervals:
                                    if intersects((enter_t, exit_t), (lock_s, lock_e)):
                                        assert lock_e > enter_t
                                        gap = lock_e - enter_t
                                        cell_delay_us += gap
                                        enter_t += gap
                                        exit_t += gap
                            else:
                                pass
                        ###
                        if cell_delay_us > 1e-9:
                            pass
                        curr_delay_us = max(curr_delay_us, cell_delay_us)
                ##
                if curr_delay_us > 1e-9:
                    ret_delay_us += curr_delay_us
                    iter += 1
                else:
                    break
            return ret_delay_us

        ######
        qrs = [qregs[qn] for qn in self.qreg_names]
        cells = [plane.field[r][c] for
        (r, c) in [qr.pos for qr in qrs]]
        ##
        first_src_list = []
        first_dst_list = []
        for route in self.first_move:
            _, first_src, first_dst = route
            first_src_list.append(first_src)
            first_dst_list.append(first_dst)
        second_src_list = []
        second_dst_list = []
        for route in self.second_move:
            _, second_src, second_dst = route
            second_src_list.append(second_src)
            second_dst_list.append(second_dst)
        ##
        self.first_pick_cells = [plane.field[r][c] for (r, c) in first_src_list]
        self.first_drop_cells = [plane.field[r][c] for (r, c) in (list(set(first_dst_list)-set(second_src_list)))]
        self.second_pick_cells = [plane.field[r][c] for (r, c) in (list(set(second_src_list)-set(first_dst_list)))]
        self.second_drop_cells = [plane.field[r][c] for (r, c) in second_dst_list]

        ###
        # Calculate latencies
        self.calculate_latency(plane, qregs, lftn, code_dist)
        ##
        self.first_routes = dict()
        for route in self.first_move:
            self.build_route(plane, route, self.first_routes, self.first_shuttle_us, lftn.shuttle_tnorm_at_xnorm)

        ####

        curr_time = offset
        ## First pick: delay
        self.first_pick_delay = cal_pick_delay(self.first_pick_cells, self.first_shuttle_us, self.first_pick_us, curr_time)
        curr_time += self.first_pick_delay
        ## First pick
        curr_time += self.first_pick_us

        ## First shuttle: delay
        self.first_shuttle_delay = cal_shuttle_delay(self.first_routes, curr_time, self.first_shuttle_us)
        curr_time += self.first_shuttle_delay
        ## First shuttle
        curr_time += self.first_shuttle_us

        ## First drop
        #### conservative serialize
        curr_time += self.first_drop_us

        self.second_routes = dict()
        for route in self.second_move:
            self.build_route(plane, route, self.second_routes, self.second_shuttle_us, lftn.shuttle_tnorm_at_xnorm)

        ## Second pick: delay
        self.second_pick_delay = cal_pick_delay(self.second_pick_cells, self.second_shuttle_us, self.second_pick_us, curr_time)
        curr_time += self.second_pick_delay
        ## Second pick
        curr_time += self.second_pick_us

        ## Second shuttle: delay
        self.second_shuttle_delay = cal_shuttle_delay(self.second_routes, curr_time, self.second_shuttle_us)
        curr_time += self.second_shuttle_delay
        ## Second shuttle
        curr_time += self.second_shuttle_us

        ## Second drop
        curr_time += self.second_drop_us
        #
        return

    def inspect_finish(self, offset, ignore_rotation):
        finish_us = offset

        ## First pick: delay
        if self.first_pick_delay > 1e-9:
            finish_us += self.first_pick_delay

        ## First pick
        finish_us += self.first_pick_us

        ## First shuttle: delay
        if self.first_shuttle_delay > 1e-9:
            finish_us += self.first_shuttle_delay

        ## First shuttle
        finish_us += self.first_shuttle_us

        ## First drop
        #### conservative serialize
        finish_us += self.first_drop_us

        ## Second pick: delay
        if self.second_pick_delay > 1e-9:
            finish_us += self.second_pick_delay

        ## Second pick
        finish_us += self.second_pick_us

        ## Second shuttle: delay
        if self.second_shuttle_delay > 1e-9:
            finish_us += self.second_shuttle_delay

        ## Second shuttle
        finish_us += self.second_shuttle_us

        ## Second drop
        finish_us += self.second_drop_us

        return finish_us


    def get_intervals(self, offset, plane, qregs, lftn, code_dist, ignore_rotation):
        intervals = []
        curr_time = offset
        #
        ## First pick: delay
        if self.first_pick_delay > 1e-6:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.first_pick_delay))
            curr_time += self.first_pick_delay
        ## First shuttle: delay
        if self.first_shuttle_delay > 1e-6:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.first_shuttle_delay))
            curr_time += self.first_shuttle_delay

        ## First pick
        self.begin_first_pick = curr_time
        if self.first_pick_us > 1e-6:
            intervals.append(gen_interval(IntervalType.PICK, curr_time, self.first_pick_us))
            curr_time += self.first_pick_us
        self.end_first_pick = curr_time

        ## First shuttle
        self.begin_first_shuttle = curr_time
        if self.first_shuttle_us > 1e-6:
            intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, self.first_shuttle_us))
            curr_time += self.first_shuttle_us
        self.end_first_shuttle = curr_time

        ## First drop
        self.begin_first_drop = curr_time
        if self.first_drop_us > 1e-6:
            intervals.append(gen_interval(IntervalType.DROP, curr_time, self.first_drop_us))
            curr_time += self.first_drop_us
        self.end_first_drop = curr_time

        ## Second pick: delay
        if self.second_pick_delay > 1e-6:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.second_pick_delay))
            curr_time += self.second_pick_delay
        ## Second shuttle: delay
        if self.second_shuttle_delay > 1e-6:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.second_shuttle_delay))
            curr_time += self.second_shuttle_delay

        ## Second pick
        self.begin_second_pick = curr_time
        if self.second_pick_us > 1e-6:
            intervals.append(gen_interval(IntervalType.PICK, curr_time, self.second_pick_us))
            curr_time += self.second_pick_us
        self.end_second_pick = curr_time

        ## Second shuttle
        self.begin_second_shuttle = curr_time
        if self.second_shuttle_us > 1e-6:
            intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, self.second_shuttle_us))
            curr_time += self.second_shuttle_us
        self.end_second_shuttle = curr_time

        ## Second drop
        self.begin_second_drop = curr_time
        if self.second_drop_us > 1e-6:
            intervals.append(gen_interval(IntervalType.DROP, curr_time, self.second_drop_us))
            curr_time += self.second_drop_us
        self.end_second_drop = curr_time

        ###
        self.finish_us = curr_time
        return intervals

    def lock_plane_cells(self, plane, qregs, lftn, ignore_path_conflict):
        if ignore_path_conflict:
            return

        def lock_shuttle_routes(routes, begin_shuttle, shuttle_us):
            for lock_type, cells in routes.items():
                if lock_type in LOCK_TYPES_C:
                    lock_part = CellPart.C
                elif lock_type in LOCK_TYPES_E:
                    lock_part = CellPart.E
                elif lock_type in LOCK_TYPES_S:
                    lock_part = CellPart.S
                elif lock_type in LOCK_TYPES_SE:
                    lock_part = CellPart.SE
                else:
                    raise Exception()
                ##
                for cell in cells:
                    if lock_part == CellPart.C:
                        enter_tnorm = cell.enter_tnorm_c
                        exit_tnorm = cell.exit_tnorm_c
                    elif lock_part == CellPart.E:
                        enter_tnorm = cell.enter_tnorm_e
                        exit_tnorm = cell.exit_tnorm_e
                    elif lock_part == CellPart.S:
                        enter_tnorm = cell.enter_tnorm_s
                        exit_tnorm = cell.exit_tnorm_s
                    elif lock_part == CellPart.SE:
                        enter_tnorm = cell.enter_tnorm_se
                        exit_tnorm = cell.exit_tnorm_se
                    else:
                        raise Exception()
                    begin = begin_shuttle + enter_tnorm*shuttle_us
                    end = begin_shuttle + exit_tnorm*shuttle_us
                    if not lock_type in cell.lock_dict.keys():
                        cell.lock_dict[lock_type] = []
                    cell.lock_dict[lock_type].append((begin, end))
            return

        # First pick: Lock
        if self.aod_type == AodType.AODH:
            for cell in self.first_pick_cells:
                lock_type = LockType.SQR
                begin = self.begin_first_pick
                end = self.end_first_pick
                #
                if not lock_type in cell.lock_dict.keys():
                    cell.lock_dict[lock_type] = []
                cell.lock_dict[lock_type].append((begin, end))
        elif self.aod_type == AodType.AODD:
            raise Exception()
        else:
            raise Exception()

        # First shuttle: Lock
        self.first_routes = dict()
        for route in self.first_move:
            self.build_route(plane, route, self.first_routes, self.first_shuttle_us, lftn.shuttle_tnorm_at_xnorm)
        lock_shuttle_routes(self.first_routes, self.begin_first_shuttle, self.first_shuttle_us)

        # First Drop: Lock
        ## waiting cells too, so not drop but pick
        if self.aod_type == AodType.AODH:
            for cell in self.first_pick_cells:
                lock_type = LockType.SQR
                begin = self.begin_first_drop
                end = self.end_first_drop
                #
                if not lock_type in cell.lock_dict.keys():
                    cell.lock_dict[lock_type] = []
                cell.lock_dict[lock_type].append((begin, end))
                #
        elif self.aod_type == AodType.AODD:
            raise Exception()
        else:
            raise Exception()

        # Second pick: Lock
        ## waiting cells too: so, not pick but drop
        if self.aod_type == AodType.AODH:
            for cell in self.second_drop_cells:
                lock_type = LockType.SQR
                begin = self.begin_second_pick
                end = self.end_second_pick
                #
                if not lock_type in cell.lock_dict.keys():
                    cell.lock_dict[lock_type] = []
                cell.lock_dict[lock_type].append((begin, end))
        elif self.aod_type == AodType.AODD:
            raise Exception()
        else:
            raise Exception()

        # Second shuttle: Lock
        self.second_routes = dict()
        for route in self.second_move:
            self.build_route(plane, route, self.second_routes, self.second_shuttle_us, lftn.shuttle_tnorm_at_xnorm)
        lock_shuttle_routes(self.second_routes, self.begin_second_shuttle, self.second_shuttle_us)

        # Second Drop: Lock
        if self.aod_type == AodType.AODH:
            for cell in self.second_drop_cells:
                lock_type = LockType.SQR
                begin = self.begin_second_drop
                end = self.end_second_drop
                #
                if not lock_type in cell.lock_dict.keys():
                    cell.lock_dict[lock_type] = []
                cell.lock_dict[lock_type].append((begin, end))
                #
        elif self.aod_type == AodType.AODD:
            raise Exception()
        else:
            raise Exception()

        return

###
class UopPatchH(UopBase):
    def __init__(self, qreg_names):
        super().__init__(UopType.PATCH_H.name,
                         UopType.PATCH_H,
                         qreg_names)

    def get_intervals(self, offset, lftn):
        intervals = []

        # qop - single-qubit gate
        op_us = lftn.sq_us
        ##
        interval_type = IntervalType.QOP
        begin = offset
        end = begin + op_us
        interval = [begin, end]
        intervals.append((interval_type, interval))

        return intervals

    def update(self, plane, qregs):
        # update qregs
        qrs = [qregs[qn] for qn in self.qreg_names]
        for qr in qrs:
            if qr.status == QregStatus.ACTIVE_A:
                qr.status = QregStatus.ACTIVE_B
            elif qr.status == QregStatus.ACTIVE_B:
                qr.status = QregStatus.ACTIVE_A
            elif qr.status == QregStatus.ACTIVE_C:
                qr.status = QregStatus.ACTIVE_D
            elif qr.status == QregStatus.ACTIVE_D:
                qr.status = QregStatus.ACTIVE_C
            else:
                raise Exception()


class UopPatchDIR(UopBase):
    def __init__(self, qreg_names, is_ideal):
        super().__init__(UopType.PATCH_DIR.name,
                         UopType.PATCH_DIR,
                         qreg_names)
        self.is_ideal = is_ideal
        #
    def update(self, plane, qregs):
        qrs = [qregs[qn] for qn in self.qreg_names]
        for qr in qrs:
            if qr.status == QregStatus.ACTIVE_A:
                qr.status = QregStatus.ACTIVE_B
            elif qr.status == QregStatus.ACTIVE_B:
                qr.status = QregStatus.ACTIVE_A
            elif qr.status == QregStatus.ACTIVE_C:
                qr.status = QregStatus.ACTIVE_D
            elif qr.status == QregStatus.ACTIVE_D:
                qr.status = QregStatus.ACTIVE_C
            else:
                raise Exception()
        # update plane
        pass

    def set_aod_type(self, aod_type, num_aodh_max, num_aodd_max, num_aodr_max):
        assert aod_type == AodType.AODR
        assert num_aodr_max >= 1
        #
        self.aod_type = AodType.AODR
        self.num_use_aod = 1
        return

    def check_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        assert self.aod_type == AodType.AODR
        return len(aodr_free_list) >= self.num_use_aod

    def get_free_aod(self, aodh_free_list,
    aodd_free_list, aodr_free_list):
        aod_list = []
        assert self.aod_type == AodType.AODR
        for _ in range(self.num_use_aod):
            aod_list.append(aodr_free_list.pop(0))
        return aod_list

    def cal_min_latency(self, code_dist, lftn):
        latency_us = 0
        #
        pick_us = drop_us = lftn.trf_us
        #
        shuttle_um = (math.pi*math.sqrt(2)/4) * (code_dist * lftn.L_um)
        shuttle_us = lftn.shuttle_us(shuttle_um)
        #
        self.slm_reconf_us = lftn.slm_reconf_us
        #
        if self.is_ideal:
            latency_us = pick_us + shuttle_us + drop_us
        else:
            latency_us = pick_us + shuttle_us + drop_us + 2*lftn.slm_reconf_us
        return latency_us

    def calculate_latency(self, plane, qregs, lftn, code_dist):
        qrs = [qregs[qn] for qn in self.qreg_names]
        cells = [plane.field[r][c] for (r, c) in [qr.pos
        for qr in qrs]]
        assert len(cells) == 1

        # pick/drop
        self.pick_us = self.drop_us = lftn.trf_us

        ## rotation (shuttle)
        shuttle_um = (math.pi*math.sqrt(2)/4) * (code_dist * lftn.L_um)
        self.shuttle_us = lftn.shuttle_us(shuttle_um)

        #
        self.slm_reconf_us = lftn.slm_reconf_us
        return

    def inspect_delay(self, offset, plane, qregs, lftn, code_dist):
        if plane.cell_size in [CellSize.SMALLEST, CellSize.DOUBLE_DIR]:
            ret_delay_us = 0
            #
            self.calculate_latency(plane, qregs, lftn, code_dist)
            #
            [pos] = [qregs[qn].pos for qn in self.
            qreg_names]
            (r, c) = pos
            #
            while True:
                #
                cell_rot = plane.field[r][c]
                #
                curr_delay_us = 0
                begin = ret_delay_us + offset
                if self.is_ideal:
                    end = begin + (self.pick_us + self.shuttle_us + self.drop_us)
                else:
                    end = begin + (self.pick_us + lftn.slm_reconf_us + self.shuttle_us + lftn.slm_reconf_us + self.drop_us)
                #
                conflict_types = LOCK_TYPES_C
                #
                for lock_type, lock_intervals in cell_rot.lock_dict.items():
                    if lock_type in conflict_types:
                        lock_intervals.sort(key=lambda x: x[0])
                        for (lock_s, lock_e) in lock_intervals:
                            if intersects((begin, end), (lock_s, lock_e)):
                                assert lock_e > begin
                                gap = lock_e - begin
                                curr_delay_us += gap
                                begin += gap
                                end += gap
                            else:
                                pass
                #
                if curr_delay_us > 1e-9:
                    ret_delay_us += curr_delay_us
                else:
                    break
            #
            if ret_delay_us > 1e-9:
                self.pick_delay_us = ret_delay_us
            else:
                self.pick_delay_us = 0
        else:
            raise Exception()

    def inspect_finish(self, offset, ignore_rotation):
        if ignore_rotation:
            return offset

        finish_us = offset

        # delay
        if self.pick_delay_us > 1e-9:
            finish_us += self.pick_delay_us

        # pick
        finish_us += self.pick_us

        # slm pattern change
        if self.is_ideal:
            pass
        else:
            finish_us += self.slm_reconf_us

        # shuttle
        finish_us += self.shuttle_us

        # slm pattern change
        if self.is_ideal:
            pass
        else:
            finish_us += self.slm_reconf_us

        # drop
        finish_us += self.drop_us

        return finish_us

    def get_intervals(self, offset, plane, qregs, lftn, code_dist, ignore_rotation):
        if ignore_rotation:
            self.scheduled_time = offset
            self.begin_time = offset
            self.end_time = offset
            self.finish_us = offset
            return []
        ##
        self.scheduled_time = offset
        #
        intervals = []
        curr_time = offset
        #
        # delay
        if self.pick_delay_us > 1e-9:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.pick_delay_us))
            curr_time += self.pick_delay_us
        #
        self.begin_time = curr_time
        # pick
        intervals.append(gen_interval(IntervalType.PICK, curr_time, self.pick_us))
        curr_time += self.pick_us
        # SLM pattern change (reconfig)
        if self.is_ideal:
            pass
        else:
            curr_time += lftn.slm_reconf_us

        # shuttle
        intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, self.shuttle_us))
        curr_time += self.shuttle_us

        # SLM patter change (reconfig)
        if self.is_ideal:
            pass
        else:
            curr_time += lftn.slm_reconf_us
        # drop
        intervals.append(gen_interval(IntervalType.DROP, curr_time, self.drop_us))
        curr_time += self.drop_us
        #
        self.end_time = curr_time
        self.finish_us = curr_time
        #
        return intervals

    def lock_plane_cells(self, plane, qregs, lftn, ignore_path_conflict):
        if ignore_path_conflict:
            return

        if plane.cell_size in [CellSize.SMALLEST, CellSize.DOUBLE_DIR]:
            [pos] = [qregs[qn].pos for qn in self.
            qreg_names]
            (r, c) = pos
            ##
            cell_rot = plane.field[r][c]
            #
            lock_type = LockType.SQR
            if not lock_type in cell_rot.lock_dict.keys():
                cell_rot.lock_dict[lock_type] = []
            cell_rot.lock_dict[lock_type].append((self.begin_time, self.end_time))
            #
            lock_type = LockType.SOL
            if not lock_type in cell_rot.lock_dict.keys():
                cell_rot.lock_dict[lock_type] = []
            cell_rot.lock_dict[lock_type].append((self.scheduled_time, self.end_time))
        else:
            raise Exception()
        return


class UopGroupDIR(UopBase):
    def __init__(self, qreg_names, rotpch_pos_list, is_aod_infinite):
        super().__init__(
            UopType.GROUP_DIR.name,
            UopType.GROUP_DIR,
            qreg_names
        )
        self.rotpch_pos_list = rotpch_pos_list
        self.is_aod_infinite = is_aod_infinite
        assert len(qreg_names) <= len(rotpch_pos_list)

    def update(self, plane, qregs):
        qrs = [qregs[qn] for qn in self.qreg_names]
        pos_list = [qr.pos for qr in qrs]
        cell_list = [plane.field[r][c] for (r, c) in pos_list]

        assert all([cell.status_q == CellStatusQ.OCCUPIED_Q for cell in cell_list])

        for qr in qrs:
            if qr.status == QregStatus.ACTIVE_A:
                qr.status = QregStatus.ACTIVE_B
            elif qr.status == QregStatus.ACTIVE_B:
                qr.status = QregStatus.ACTIVE_A
            elif qr.status == QregStatus.ACTIVE_C:
                qr.status = QregStatus.ACTIVE_D
            elif qr.status == QregStatus.ACTIVE_D:
                qr.status = QregStatus.ACTIVE_C
            else:
                raise Exception()

    def set_aod_type(self, aod_type, num_aodh_max, num_aodd_max, num_aodr_max):
        assert aod_type == AodType.AODR
        self.aod_type = AodType.AODR

        ####
        if self.is_aod_infinite:
            if num_aodh_max > (len(self.rotpch_pos_list) - len(self.qreg_names)):
                self.num_use_aodh = len(self.rotpch_pos_list) - len(self.qreg_names)
            else:
                raise Exception()
            if num_aodr_max > len(self.qreg_names):
                self.num_use_aodr = len(self.qreg_names)
            else:
                raise Exception()
        else: # finite AOD
            # Fix the num_use_aodh to 1
            self.num_use_aodh = 1
            #
            self.num_use_aodr = min(len(self.qreg_names), num_aodr_max)

    def check_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        aodr_cond = len(aodr_free_list) >= self.num_use_aodr
        aodh_cond = len(aodh_free_list) >= self.num_use_aodh
        return (aodr_cond and aodh_cond)

    def get_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        aod_list = []
        for _ in range(self.num_use_aodr):
            aod_list.append(aodr_free_list.pop(0))
        for _ in range(self.num_use_aodh):
            aod_list.append(aodh_free_list.pop(0))
        return aod_list

    def calculate_latency(self, plane, qregs, lftn, code_dist):
        # pick/drop non-rotating patches
        if self.is_aod_infinite:
            togl_pick_sf = 1
            togl_drop_sf = 1
            self.num_rot_iter = 1
        else: # finite aod
            ## determine the rotation positions to pick first
            rot_pos_list = [qregs[qn].pos for qn in self.qreg_names]
            if self.num_use_aodr >= len(rot_pos_list):
                self.rot_pos_pick = rot_pos_list
                self.rot_pos_drop = rot_pos_list
                self.num_rot_iter = 1
            else: # i.e., num_aodr is not sufficient
                self.num_rot_iter = math.ceil(len(self.qreg_names) / self.num_use_aodr)
                ##
                num_drop = len(self.qreg_names) % self.num_use_aodr
                if num_drop == 0:
                    num_drop = self.num_use_aodr
                self.rot_pos_pick = random.sample(rot_pos_list, k=self.num_use_aodr)
                self.rot_pos_drop = random.sample(list(set(rot_pos_list)-set(self.rot_pos_pick)), k=num_drop)
            #
            togl_pick_pos_list = [pos for pos in self.rotpch_pos_list if not pos in self.rot_pos_pick]
            togl_pick_sf = num_unique_cols(togl_pick_pos_list)
            ###
            togl_drop_pos_list = [pos for pos in self.rotpch_pos_list if not pos in self.rot_pos_drop]
            togl_drop_sf = num_unique_cols(togl_drop_pos_list)
            ###
        ###
        self.togl_pick_us = togl_pick_sf * lftn.trf_us
        self.togl_drop_us = togl_drop_sf * lftn.trf_us
        ###

        # pick/drop rotating patches (one by one per AOD)
        self.rot_pick_us = self.rot_drop_us = lftn.trf_us

        # rotation
        shuttle_um = (math.pi*math.sqrt(2)/4) * (code_dist * lftn.L_um)
        self.shuttle_us = lftn.shuttle_us(shuttle_um)

        # lateny calculation
        latency_us = 0
        if self.is_aod_infinite:
            # AOD PICK - Others + ROT
            latency_us += max(self.togl_pick_us, self.rot_pick_us)
            # SLM turn off
            latency_us += lftn.slm_onoff_us
            # Rotation
            assert self.num_rot_iter == 1
            latency_us += self.shuttle_us
            # SLM turn on
            latency_us += lftn.slm_onoff_us
            # AOD DROP - Others + ROT
            latency_us += max(self.togl_drop_us, self.rot_drop_us)
            ##
            self.latency_us = latency_us
        else: # finite aod
            # AOD PICK - Others + first rot pick
            latency_us += max(self.togl_pick_us, self.rot_pick_us)
            # SLM turn off
            latency_us += lftn.slm_onoff_us
            # Rotation
            latency_us -= self.rot_pick_us
            for _ in range(self.num_rot_iter):
                latency_us += self.rot_pick_us
                latency_us += self.shuttle_us
                latency_us += self.rot_drop_us
            latency_us -= self.rot_drop_us
            # SLM turn on
            latency_us += lftn.slm_onoff_us
            # AOD DROP - Others + last rot drop
            latency_us += max(self.togl_drop_us, self.rot_drop_us)
            ##
            self.latency_us = latency_us
        return

    def inspect_delay(self, offset, plane, qregs, lftn, code_dist):
        if plane.cell_size in [CellSize.SMALLEST, CellSize.DOUBLE_DIR]:
            ret_delay_us = 0
            #
            self.calculate_latency(plane, qregs, lftn, code_dist)
            #
            conflict_types = LOCK_TYPES_C
            wait_cells = [plane.field[r][c] for (r, c) in self.rotpch_pos_list]
            #
            while True:
                curr_delay_us = 0
                for cell in wait_cells:
                    cell_delay_us = 0
                    begin = ret_delay_us + offset
                    end = begin + self.latency_us
                    ## check single cell's delay
                    for lock_type, lock_intervals in cell.lock_dict.items():
                        if lock_type in conflict_types:
                            lock_intervals.sort(key=lambda x: x[0])
                            for (lock_s, lock_e) in lock_intervals:
                                if intersects((begin, end), (lock_s, lock_e)):
                                    assert lock_e > begin
                                    gap = lock_e - begin
                                    cell_delay_us += gap
                                    begin += gap
                                    end += gap
                                else:
                                    pass
                        else:
                            pass
                    ##
                    curr_delay_us = max(curr_delay_us, cell_delay_us)
                ##
                if curr_delay_us > 1e-9:
                    ret_delay_us += curr_delay_us
                else:
                    break
            ##
            self.pick_delay_us = ret_delay_us
            ##
            return ret_delay_us

        else: # other cell sizes
            raise Exception("Not support")

    def inspect_finish(self, offset, ignore_rotation):
        if ignore_rotation:
            return offset

        finish_us = offset

        # delay
        if self.pick_delay_us > 1e-9:
            finish_us += self.pick_delay_us

        finish_us += self.latency_us

        return finish_us


    def get_intervals(self, offset, plane, qregs, lftn, code_dist, ignore_rotation):
        if ignore_rotation:
            self.scheduled_time = offset
            self.begin_time = offset
            self.end_time = offset
            self.finish_us = offset
            return []
        ###
        self.scheduled_time = offset
        ###
        intervals = []
        curr_time = offset
        ###
        # delay
        if self.pick_delay_us > 1e-9:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.pick_delay_us))
            curr_time += self.pick_delay_us

        # start
        self.begin_time = curr_time
        ##
        if self.is_aod_infinite:
            # AOD PICK - Others + ROT
            intervals.append(gen_interval(IntervalType.PICK, curr_time, max(self.togl_pick_us, self.rot_pick_us)))
            curr_time += max(self.togl_pick_us, self.rot_pick_us)
            # SLM turn off
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, lftn.slm_onoff_us))
            curr_time += lftn.slm_onoff_us
            # Rotation
            assert self.num_rot_iter == 1
            intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, self.shuttle_us))
            curr_time += self.shuttle_us
            # SLM turn on
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, lftn.slm_onoff_us))
            curr_time += lftn.slm_onoff_us
            # AOD DROP - Others + ROT
            intervals.append(gen_interval(IntervalType.DROP, curr_time, max(self.togl_drop_us, self.rot_drop_us)))
            curr_time += max(self.togl_drop_us, self.rot_drop_us)
        else:
            # AOD PICK - Others + ROT
            intervals.append(gen_interval(IntervalType.PICK, curr_time, max(self.togl_pick_us, self.rot_pick_us)))
            curr_time += max(self.togl_pick_us, self.rot_pick_us)
            # SLM turn off
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, lftn.slm_onoff_us))
            curr_time += lftn.slm_onoff_us
            # Rotation
            for iter in range(self.num_rot_iter):
                # intermeidate pick
                if iter == 0:
                    pass
                else:
                    intervals.append(gen_interval(IntervalType.PICK, curr_time, self.rot_pick_us))
                    curr_time += self.rot_pick_us
                # shuttle
                intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, self.shuttle_us))
                curr_time += self.shuttle_us
                # intermediate drop
                if iter == self.num_rot_iter-1:
                    pass
                else:
                    intervals.append(gen_interval(IntervalType.DROP, curr_time, self.rot_drop_us))
                    curr_time += self.rot_drop_us
            # SLM turn on
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, lftn.slm_onoff_us))
            curr_time += lftn.slm_onoff_us
            # AOD DROP - Others + ROT
            intervals.append(gen_interval(IntervalType.DROP, curr_time, max(self.togl_drop_us, self.rot_drop_us)))
            curr_time += max(self.togl_drop_us, self.rot_drop_us)
        ##
        self.end_time = curr_time
        self.finish_us = curr_time

        return intervals

    def lock_plane_cells(self, plane, qregs, lftn, ignore_path_conflict):
        if ignore_path_conflict:
            return
        if plane.cell_size in [CellSize.SMALLEST, CellSize.DOUBLE_DIR]:
            if (self.end_time - self.begin_time) < 1e-9:
                return
            ##
            target_cells = [plane.field[r][c] for (r, c) in self.rotpch_pos_list]
            ##
            lock_type = LockType.SQR
            ##
            for cell in target_cells:
                if not lock_type in cell.lock_dict.keys():
                    cell.lock_dict[lock_type] = []
                cell.lock_dict[lock_type].append((self.begin_time, self.end_time))
            ##
            lock_type = LockType.SOL
            ##
            qrs = [qregs[qn] for qn in self.qreg_names]
            rot_cells = [plane.field[r][c] for (r, c) in [qr.pos for qr in qrs]]
            for cell in rot_cells:
                if not lock_type in cell.lock_dict.keys():
                    cell.lock_dict[lock_type] = []
                cell.lock_dict[lock_type].append((self.scheduled_time, self.end_time))
            return
        else: # other cell sizes
            raise Exception()


###
class UopPatchRH(UopBase):
    def __init__(self, qreg_names, refl_type):
        super().__init__(UopType.PATCH_RH.name,
                         UopType.PATCH_RH,
                         qreg_names)
        self.refl_type = refl_type
        #
    def update(self, plane, qregs):
        # update qregs
        qrs = [qregs[qn] for qn in self.qreg_names]
        for qr in qrs:
            if qr.status == QregStatus.ACTIVE_A:
                qr.status = QregStatus.ACTIVE_D
            elif qr.status == QregStatus.ACTIVE_B:
                qr.status = QregStatus.ACTIVE_C
            elif qr.status == QregStatus.ACTIVE_C:
                qr.status = QregStatus.ACTIVE_B
            elif qr.status == QregStatus.ACTIVE_D:
                qr.status = QregStatus.ACTIVE_A
            else:
                raise Exception()
        # update plane
        cells = [plane.field[r][c] for (r,c) in [qr.pos for qr in qrs]]
        for cell in cells:
            if cell.status_op in [CellStatusOp.IDLE, CellStatusOp.MOV]:
                cell.status_op = CellStatusOp.RH
            elif cell.status_op == CellStatusOp.RD:
                cell.status_op = CellStatusOp.RD_RH
            else:
                print(cell.status_op)
                raise Exception()

    def set_aod_type(self, aod_type, num_aodh_max, num_aodd_max, num_aodr_max):
        assert aod_type == AodType.AODH
        assert num_aodh_max >= 1
        self.aod_type = AodType.AODH
        ##
        if num_aodh_max >= 2:
            self.num_use_aod = 2
        else:
            self.num_use_aod = 1
        return

    def check_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        assert self.aod_type == AodType.AODH
        return len(aodh_free_list) >= self.num_use_aod

    def get_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        aod_list = []
        assert self.aod_type == AodType.AODH
        for _ in range(self.num_use_aod):
            aod_list.append(aodh_free_list.pop(0))
        return aod_list

    def inspect_delay(self, offset, plane, qregs, lftn, code_dist):
        self.min_latency_us = self.cal_min_latency(code_dist, lftn)
        ##
        curr_time = offset
        ##
        delay_us = 0
        # Pick delay
        qrs = [qregs[qn] for qn in self.qreg_names]
        cells = [plane.field[r][c] for (r, c) in [qr.pos for qr in qrs]]
        conflict_types = LOCK_TYPES_C
        for cell in cells:
            for lock_type, lock_intervals in cell.lock_dict.items():
                if lock_type in conflict_types:
                    lock_until = max([end for (begin, end) in lock_intervals])
                    delay_us = max(delay_us, lock_until-curr_time)
                else:
                    pass
        ##
        if delay_us > 1e-9:
            self.pick_delay_us = delay_us
        else:
            self.pick_delay_us = 0
        return

    def inspect_finish(self, offset, ignore_rotation):
        if ignore_rotation:
            return offset

        finish_us = offset

        # delay
        if self.pick_delay_us > 1e-9:
            finish_us += self.pick_delay_us

        #
        # latency
        ## infinite aod only
        ## then, min latnecy (iterative pick shuttle drop with min latency)
        finish_us += self.min_latency_us

        return finish_us

    def cal_min_latency(self, code_dist, lftn):
        ## Iterative moves
        ###
        if self.num_use_aod == 2:
            num_iter = 1
        elif self.num_use_aod == 1:
            num_iter = 2
        else:
            raise Exception()
        ###
        latency_us = 0
        assert code_dist % 2 == 1
        ###
        if self.refl_type == ReflType.STATIC_TE:
            num_rows = code_dist
            while num_rows > 1:
                for _ in range(num_iter):
                    ### Pick
                    latency_us += lftn.trf_us
                    ### Shuttle
                    shuttle_um = math.ceil(num_rows/2) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    latency_us += shuttle_us
                    ### Drop
                    latency_us += lftn.trf_us
                ### update for the next iteration
                num_rows = math.floor(num_rows/2)
        elif self.refl_type == ReflType.STATIC_SE:
            num_rows = code_dist-1
            while num_rows > 0:
                for _ in range(num_iter):
                    ### Pick
                    latency_us += lftn.trf_us
                    ### Shuttle
                    shuttle_um = math.ceil(num_rows) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    latency_us += shuttle_us
                    ### Drop
                    latency_us += lftn.trf_us
                ### update for the next iteration
                num_rows -= 2
        else:
            raise Exception()
        #
        return latency_us

    def get_intervals(self, offset, plane, qregs, lftn, code_dist, ignore_rotation):
        if ignore_rotation:
            self.begin_time = offset
            self.end_time = offset
            self.finish_us = offset
            return []

        intervals = []
        curr_time = offset
        #
        qrs = [qregs[qn] for qn in self.qreg_names]
        cells = [plane.field[r][c] for (r, c) in [qr.pos for qr in qrs]]
        #
        xy_pos_list = [qr.pos for qr in qrs]
        pick_drop_sf = num_unique_cols(xy_pos_list)
        pick_us = drop_us = pick_drop_sf * lftn.trf_us
        #

        ## DELAY
        if self.pick_delay_us > 1e-9:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.pick_delay_us))
            curr_time += self.pick_delay_us

        ## Iterative moves
        ###
        if self.num_use_aod == 2:
            num_iter = 1
        elif self.num_use_aod == 1:
            num_iter = 2
        else:
            raise Exception()
        ###
        self.begin_time = curr_time
        assert code_dist % 2 == 1
        ###
        if self.refl_type == ReflType.STATIC_TE:
            num_rows = code_dist
            while num_rows > 1:
                for _ in range(num_iter):
                    ### Pick
                    intervals.append(gen_interval(IntervalType.PICK, curr_time, pick_us))
                    curr_time += pick_us
                    ### Shuttle
                    shuttle_um = math.ceil(num_rows/2) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, shuttle_us))
                    curr_time += shuttle_us
                    ### Drop
                    intervals.append(gen_interval(IntervalType.DROP, curr_time, drop_us))
                    curr_time += drop_us
                ### update for the next iteration
                num_rows = math.floor(num_rows/2)
        elif self.refl_type == ReflType.STATIC_SE:
            num_rows = code_dist-1
            while num_rows > 0:
                for _ in range(num_iter):
                    ### Pick
                    intervals.append(gen_interval(IntervalType.PICK, curr_time, pick_us))
                    curr_time += pick_us
                    ### Shuttle
                    shuttle_um = math.ceil(num_rows) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, shuttle_us))
                    curr_time += shuttle_us
                    ### Drop
                    intervals.append(gen_interval(IntervalType.DROP, curr_time, drop_us))
                    curr_time += drop_us
                ### update for the next iteration
                num_rows -= 2
        else:
            raise Exception()
        ###
        self.end_time = curr_time
        self.finish_us = curr_time
        ###
        return intervals

    def lock_plane_cells(self, plane, qregs, lftn, ignore_path_conflict):
        if ignore_path_conflict:
            return
        pos_list = [qregs[qn].pos for qn in self.qreg_names]

        for cell in [plane.field[r][c] for (r, c) in pos_list]:
            # Target cells: SQR
            lock_type = LockType.SQR
            if not lock_type in cell.lock_dict.keys():
                cell.lock_dict[lock_type] = []
            cell.lock_dict[lock_type].append((self.begin_time, self.end_time))
        return

class UopPatchRD(UopBase):
    def __init__(self, qreg_names, refl_type):
        super().__init__(UopType.PATCH_RD.name,
                         UopType.PATCH_RD,
                         qreg_names)
        self.refl_type = refl_type

    def update(self, plane, qregs):
        # update qregs
        qrs = [qregs[qn] for qn in self.qreg_names]
        for qr in qrs:
            if qr.status == QregStatus.ACTIVE_A:
                qr.status = QregStatus.ACTIVE_C
            elif qr.status == QregStatus.ACTIVE_B:
                qr.status = QregStatus.ACTIVE_D
            elif qr.status == QregStatus.ACTIVE_C:
                qr.status = QregStatus.ACTIVE_A
            elif qr.status == QregStatus.ACTIVE_D:
                qr.status = QregStatus.ACTIVE_B
            else:
                raise Exception()
        # update plane
        cells = [plane.field[r][c] for (r,c) in [qr.pos for qr in qrs]]
        for cell in cells:
            if cell.status_op in [CellStatusOp.IDLE, CellStatusOp.MOV]:
                cell.status_op = CellStatusOp.RD
            elif cell.status_op == CellStatusOp.RH:
                cell.status_op = CellStatusOp.RH_RD
            else:
                raise Exception()

    def set_aod_type(self, aod_type, num_aodh_max, num_aodd_max, num_aodr_max):
        assert aod_type == AodType.AODD
        assert num_aodd_max >= 1
        self.aod_type = aod_type
        ##
        if num_aodd_max >= 4:
            self.num_use_aod = 4
        elif num_aodd_max >= 2:
            self.num_use_aod = 2
        else:
            self.num_use_aod = 1
        return

    def check_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        assert self.aod_type == AodType.AODD
        return len(aodd_free_list) >= self.num_use_aod

    def get_free_aod(self, aodh_free_list, aodd_free_list, aodr_free_list):
        aod_list = []
        assert self.aod_type == AodType.AODD
        for _ in range(self.num_use_aod):
            aod_list.append(aodd_free_list.pop(0))
        return aod_list

    def cal_min_latency(self, code_dist, lftn):
        latency_us = 0
        #
        pick_us = drop_us = lftn.trf_us
        ## iterative moves
        ##
        if self.refl_type == ReflType.STATIC_SE:
            if self.num_use_aod >= 2:
                num_iter = 1
            else:
                num_iter = 2
            #
            # First diag lattice
            num_rows = code_dist-1
            while num_rows > 0:
                for _ in range(num_iter):
                    ### Pick
                    latency_us += pick_us
                    ### Shuttle
                    shuttle_um = math.sqrt(2) * math.ceil(num_rows) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    latency_us += shuttle_us
                    ### Drop
                    latency_us += drop_us
                num_rows -= 2
            # Second diag lattice
            if self.num_use_aod >= 4:
                pass
            else:
                num_rows = code_dist-2
                while num_rows > 0:
                    for _ in range(num_iter):
                        latency_us += pick_us
                        ### Shuttle
                        shuttle_um = math.sqrt(2) * math.ceil(num_rows) * lftn.L_um
                        shuttle_us = lftn.shuttle_us(shuttle_um)
                        latency_us += shuttle_us
                        ### Drop
                        latency_us += drop_us
                    num_rows -= 2
            ##
        ###
        elif self.refl_type == ReflType.STATIC_TE:
            ## 1. Upper left to bottom right, and vice versa
            if self.num_use_aod == 4:
                num_iter = 1
            elif self.num_use_aod == 2:
                num_iter = 2
            elif self.num_use_aod == 1:
                num_iter = 4
            else:
                raise Exception()
            #
            for _ in range(num_iter):
                #### Pick
                latency_us += pick_us
                #### Shuttle
                shuttle_um = math.sqrt(2) * math.ceil(code_dist/2) * lftn.L_um
                shuttle_us = lftn.shuttle_us(shuttle_um)
                latency_us += shuttle_us
                #### Drop
                latency_us += drop_us

            ## 2. Iterative shuttling (log)
            if self.num_use_aod >= 2:
                num_iter = 1
            elif self.num_use_aod == 1:
                num_iter = 2
            else:
                raise Exception()
            num_rows = math.floor(code_dist/2)
            while num_rows > 1:
                for _ in range(num_iter):
                    #### Pick
                    latency_us += pick_us
                    #### Shuttle
                    shuttle_um = math.sqrt(2) * math.ceil(num_rows/2) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    latency_us += shuttle_us
                    #### Drop
                    latency_us += drop_us
                ## update for the next iteration
                num_rows = math.floor(num_rows/2)

            ## 3. Last alignment (one SLM spacing)
            #### Pick
            latency_us += pick_us
            #### Shuttle
            shuttle_um = lftn.L_um
            shuttle_us = lftn.shuttle_us(shuttle_um)
            latency_us += shuttle_us
            #### Drop
            latency_us += drop_us
        ###
        else:
            raise Exception()
        #
        return latency_us


    def inspect_delay(self, offset, plane, qregs, lftn, code_dist):
        #
        self.min_latency_us = self.cal_min_latency(code_dist, lftn)
        #
        curr_time = offset
        #
        delay_us = 0
        # Pick delay

        if self.refl_type == ReflType.STATIC_SE:
            qrs = [qregs[qn] for qn in self.qreg_names]
            cells = [plane.field[r][c] for (r, c) in [qr.pos for qr in qrs]]
            conflict_types = LOCK_TYPES_C
            #
            for cell in cells:
                for lock_type, lock_intervals in cell.lock_dict.items():
                    if lock_type in conflict_types:
                        lock_until = max([end for (begin, end) in lock_intervals])
                        delay_us = max(delay_us, lock_until-curr_time)
                    else:
                        pass
        elif self.refl_type == ReflType.STATIC_TE:
            #
            if plane.cell_size == CellSize.SMALLEST:
                raise Exception()
            #
            elif plane.cell_size == CellSize.DOUBLE_TE:
                qrs = [qregs[qn] for qn in self.qreg_names]
                ##
                cells_rot = [plane.field[r][c] for (r, c) in [qr.pos for qr in qrs]]
                #
                for cell in cells_rot:
                    conflict_types = LOCK_TYPES_C + LOCK_TYPES_E + LOCK_TYPES_S
                    ##
                    for lock_type, lock_intervals in cell.lock_dict.items():
                        if lock_type in conflict_types:
                            lock_until = max([end for (begin, end) in lock_intervals])
                            delay_us = max(delay_us, lock_until-curr_time)
                        else:
                            pass
                ##
                cells_n = [plane.field[r-1][c] for (r, c) in [qr.pos for qr in qrs] if r != 0]
                for cell in cells_n:
                    conflict_types = LOCK_TYPES_S
                    ##
                    for lock_type, lock_intervals in cell.lock_dict.items():
                        if lock_type in conflict_types:
                            lock_until = max([end for (begin, end) in lock_intervals])
                            delay_us = max(delay_us, lock_until-curr_time)
                        else:
                            pass
                ##
                cells_w = [plane.field[r][c-1] for (r, c) in [qr.pos for qr in qrs] if c != 0]
                for cell in cells_w:
                    conflict_types = LOCK_TYPES_E
                    ##
                    for lock_type, lock_intervals in cell.lock_dict.items():
                        if lock_type in conflict_types:
                            lock_until = max([end for (begin, end) in lock_intervals])
                            delay_us = max(delay_us, lock_until-curr_time)
                        else:
                            pass
            #
            else:
                raise Exception()
        else:
            raise Exception()
        ##############
        ##
        if delay_us > 1e-9:
            self.pick_delay_us = delay_us
        else:
            self.pick_delay_us = 0

        return

    def inspect_finish(self, offset, ignore_rotation):
        if ignore_rotation:
            return offset

        finish_us = offset

        # delay
        if self.pick_delay_us > 1e-9:
            finish_us += self.pick_delay_us

        # latency
        ## infinite aod only
        ## then, min latnecy (iterative pick shuttle drop)
        finish_us += self.min_latency_us
        #
        return finish_us

    def get_intervals(self, offset, plane, qregs, lftn, code_dist, ignore_rotation):
        if ignore_rotation:
            self.begin_time = offset
            self.end_time = offset
            self.finish_us = offset
            return []

        intervals = []
        curr_time = offset
        #
        qrs = [qregs[qn] for qn in self.qreg_names]
        cells = [plane.field[r][c] for (r, c) in [qr.pos for qr in qrs]]
        ## check pick/drop serialization
        xy_pos_list = [qr.pos for qr in qrs]
        dq_pos_list = [xy_to_dq(r, c, plane.h) for (r, c) in xy_pos_list]
        pick_drop_sf = num_unique_cols(dq_pos_list)
        pick_us = drop_us = pick_drop_sf * lftn.trf_us

        ## DELAY
        if self.pick_delay_us > 1e-9:
            intervals.append(gen_interval(IntervalType.DELAY, curr_time, self.pick_delay_us))
            curr_time += self.pick_delay_us

        ## iterative moves
        self.begin_time = curr_time
        ##
        if self.refl_type == ReflType.STATIC_SE:
            if self.num_use_aod >= 2:
                num_iter = 1
            else:
                num_iter = 2

            # First diag lattice
            num_rows = code_dist-1
            while num_rows > 0:
                for _ in range(num_iter):
                    ### Pick
                    intervals.append(gen_interval(IntervalType.PICK, curr_time, pick_us))
                    curr_time += pick_us
                    ### Shuttle
                    shuttle_um = math.sqrt(2) * math.ceil(num_rows) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, shuttle_us))
                    curr_time += shuttle_us
                    ### Drop
                    intervals.append(gen_interval(IntervalType.DROP, curr_time, drop_us))
                    curr_time += drop_us
                num_rows -= 2
            # Second diag lattice
            if self.num_use_aod >= 4:
                pass
            else:
                num_rows = code_dist-2
                while num_rows > 0:
                    for _ in range(num_iter):
                        intervals.append(gen_interval(IntervalType.PICK, curr_time, pick_us))
                        curr_time += pick_us
                        ### Shuttle
                        shuttle_um = math.sqrt(2) * math.ceil(num_rows) * lftn.L_um
                        shuttle_us = lftn.shuttle_us(shuttle_um)
                        intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, shuttle_us))
                        curr_time += shuttle_us
                        ### Drop
                        intervals.append(gen_interval(IntervalType.DROP, curr_time, drop_us))
                        curr_time += drop_us
                    num_rows -= 2
            ##
        ###
        elif self.refl_type == ReflType.STATIC_TE:
            ## 1. Upper left to bottom right, and vice versa
            if self.num_use_aod == 4:
                num_iter = 1
            elif self.num_use_aod == 2:
                num_iter = 2
            elif self.num_use_aod == 1:
                num_iter = 4
            else:
                raise Exception()
            #
            for _ in range(num_iter):
                #### Pick
                intervals.append(gen_interval(IntervalType.PICK, curr_time, pick_us))
                curr_time += pick_us
                #### Shuttle
                shuttle_um = math.sqrt(2) * math.ceil(code_dist/2) * lftn.L_um
                shuttle_us = lftn.shuttle_us(shuttle_um)
                intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, shuttle_us))
                curr_time += shuttle_us
                #### Drop
                intervals.append(gen_interval(IntervalType.DROP, curr_time, drop_us))
                curr_time += drop_us

            ## 2. Iterative shuttling (log)
            if self.num_use_aod >= 2:
                num_iter = 1
            elif self.num_use_aod == 1:
                num_iter = 2
            else:
                raise Exception()
            num_rows = math.floor(code_dist/2)
            while num_rows > 1:
                for _ in range(num_iter):
                    #### Pick
                    intervals.append(gen_interval(IntervalType.PICK, curr_time, pick_us))
                    curr_time += pick_us
                    #### Shuttle
                    shuttle_um = math.sqrt(2) * math.ceil(num_rows/2) * lftn.L_um
                    shuttle_us = lftn.shuttle_us(shuttle_um)
                    intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, shuttle_us))
                    curr_time += shuttle_us
                    #### Drop
                    intervals.append(gen_interval(IntervalType.DROP, curr_time, drop_us))
                    curr_time += drop_us
                ## update for the next iteration
                num_rows = math.floor(num_rows/2)

            ## 3. Last alignment (one SLM spacing)
            #### Pick
            intervals.append(gen_interval(IntervalType.PICK, curr_time, pick_us))
            curr_time += pick_us
            #### Shuttle
            shuttle_um = lftn.L_um
            shuttle_us = lftn.shuttle_us(shuttle_um)
            intervals.append(gen_interval(IntervalType.SHUTTLE, curr_time, shuttle_us))
            curr_time += shuttle_us
            #### Drop
            intervals.append(gen_interval(IntervalType.DROP, curr_time, drop_us))
            curr_time += drop_us
        ###
        else:
            raise Exception()
        #
        self.end_time = curr_time
        self.finish_us = curr_time
        #
        return intervals

    def lock_plane_cells(self, plane, qregs, lftn, ignore_path_conflict):
        if ignore_path_conflict:
            return

        pos_list = [qregs[qn].pos for qn in self.qreg_names]
        ##

        if self.refl_type == ReflType.STATIC_SE:
            cells = [plane.field[r][c] for (r, c) in pos_list]
            for cell in cells:
                lock_type = LockType.SQR
                if not lock_type in cell.lock_dict.keys():
                    cell.lock_dict[lock_type] = []
                cell.lock_dict[lock_type].append((self.begin_time, self.end_time))

        elif self.refl_type == ReflType.STATIC_TE:
            if plane.cell_size == CellSize.SMALLEST:
                raise Exception()
            #
            elif plane.cell_size == CellSize.DOUBLE_TE:
                ##
                cells_rot = [plane.field[r][c] for (r, c) in pos_list]
                for cell in cells_rot:
                    for lock_type in [LockType.SQR, LockType.E, LockType.S]:
                        if not lock_type in cell.lock_dict.keys():
                            cell.lock_dict[lock_type] = []
                        cell.lock_dict[lock_type].append((self.begin_time, self.end_time))
                ##
                cells_n = [plane.field[r-1][c] for (r, c) in pos_list if r != 0]
                for cell in cells_n:
                    lock_type = LockType.S
                    if not lock_type in cell.lock_dict.keys():
                        cell.lock_dict[lock_type] = []
                    cell.lock_dict[lock_type].append((self.begin_time, self.end_time))
                ##
                cells_w = [plane.field[r][c-1] for (r, c) in pos_list if c != 0]
                for cell in cells_w:
                    lock_type = LockType.E
                    if not lock_type in cell.lock_dict.keys():
                        cell.lock_dict[lock_type] = []
                    cell.lock_dict[lock_type].append((self.begin_time, self.end_time))
            #
            else:
                raise Exception()
        else:
            raise Exception()
        #
        return

class UopPatchCX(UopBase):
    def __init__(self, qreg_names):
        super().__init__(UopType.PATCH_CX.name,
                        UopType.PATCH_CX,
                        qreg_names)

    def get_intervals(self, offset, lftn):
        intervals = []

        if lftn.meas_type == PhyOpType.SELECT:
            # qop - CX (four steps)
            op_us = (lftn.tq_us+lftn.sq_us) * 4
            begin = offset
            end = begin + op_us
            interval_type = IntervalType.QOP
            interval = [begin, end]
            intervals.append((interval_type, interval))
        elif lftn.meas_type == PhyOpType.ZONE:
            # TODO
            raise Exception()
        else:
            raise Exception()
        ##
        return intervals

    def check_ready(self, plane, qregs):
        for i in range(0, len(self.qreg_names), 2):
            qubit_pair = self.qreg_names[i:i+2]
            # same position
            ##
            pos_pair = [qregs[qn].pos for qn in qubit_pair]
            assert pos_pair[0] == pos_pair[1]
            ##
            (r, c) = pos_pair[0]
            cell = plane.field[r][c]
            assert set(qubit_pair) == set(cell.occupants)

            # same boundary
            bd_pair = [qregs[qn].status for qn in qubit_pair]
            if bd_pair[0] != bd_pair[1]:
                print(f"DIFFERENT BOUNDARY: {qubit_pair}")
            assert bd_pair[0] == bd_pair[1]
        return

class UopPatchMeas(UopBase):
    def __init__(self, qreg_names):
        super().__init__(UopType.PATCH_MEAS.name,
                        UopType.PATCH_MEAS,
                        qreg_names)

    def get_intervals(self, offset, lftn):
        intervals = []

        if lftn.meas_type == PhyOpType.SELECT:
            # qop - measurement (destructive)
            op_us = lftn.dm_us
            begin = offset
            end = begin + op_us
            interval_type = IntervalType.QOP
            interval = [begin, end]
            intervals.append((interval_type, interval))
        elif lftn.meas_type == PhyOpType.ZONE:
            # TODO
            raise Exception()
        else:
            raise Exception()
        ##
        return intervals

    def update(self, plane, qregs):
        # update plane
        for qn in self.qreg_names:
            qr = qregs[qn]
            pos = qr.pos
            r, c = pos
            cell = plane.field[r][c]
            assert qn in cell.occupants
            #
            if qn in self.reset_qubits:
                qr.status = QregStatus.ACTIVE_A
            else:
                cell.occupants.remove(qn)
                if len(cell.occupants) == 0:
                    cell.status_q = CellStatusQ.FREE
                elif len(cell.occupants) == 1:
                    cell.status_q = CellStatusQ.OCCUPIED_Q
                else:
                    raise Exception()
                #
                del qregs[qn]