from typing import Dict
from tsc_instructions import IntervalType, UopType, UopBase
from config import experiment_config
import random
from macro import *
seed=227
random.seed(seed)


def calc_group_dir_pick_drop_num(lqwise_schedule, cfg_in: experiment_config)->Dict[str, int]:
    qreg_names = lqwise_schedule.keys()
    pick_drop_num_dict = {qn: 0 for qn in qreg_names}
    if cfg_in.rot_type not in [RotType.DIR_TOGL, RotType.DIR_IDEAL]:
        return pick_drop_num_dict

    remaining_qubits = list(lqwise_schedule.keys())
    for qn in qreg_names:
        if qn not in remaining_qubits:
            continue
        uop_schedule = lqwise_schedule[qn]
        for index, uop_tuple in enumerate(uop_schedule):
            uop, interval_type, interval = uop_tuple
            if uop.uop_type == UopType.GROUP_DIR:
                if interval_type in [IntervalType.DROP]:
                    if cfg_in.rot_plane_opt != RotPlaneOpt.DEDICATED_ROT:
                        for target_qreg in lqwise_schedule.keys():
                            if target_qreg not in uop.qreg_names:
                                pick_drop_num_dict[target_qreg] += 2

                        for operand_qreg in uop.qreg_names:
                            if operand_qreg in remaining_qubits:
                                remaining_qubits.remove(operand_qreg)

                    else:
                        target_num = (len(uop.rotpch_pos_list) - len(uop.qreg_names))*2
                        # randomly choose target_num*2 logical qubit except for operand qubits
                        set_operand = set(uop.qreg_names)
                        candidate_list = list(set(lqwise_schedule.keys()) - set_operand)
                        choosed_qn_list = random.sample(candidate_list, target_num)
                        for choosed_qn in choosed_qn_list:
                            pick_drop_num_dict[choosed_qn] += 2

                        for operand_qreg in uop.qreg_names:
                            if operand_qreg in remaining_qubits:
                                remaining_qubits.remove(operand_qreg)

    return pick_drop_num_dict


def sq_logical_error(qn, lqwise_schedule, pick_drop_offset, cfg_in, code_dist):
    uop_schedule = lqwise_schedule[qn]
    pick_drop_num = pick_drop_offset

    se_rd_once = True
    se_rh_once = True

    for index, uop_tuple in enumerate(uop_schedule):
        uop, interval_type, interval = uop_tuple
        if interval_type in [IntervalType.DROP, IntervalType.PICK]:
            if cfg_in.rot_type == RotType.REFL and cfg_in.refl_type_h == ReflType.STATIC_SE:
                if uop.uop_type == UopType.PATCH_RD:
                    if se_rd_once:
                        pick_drop_num += 2 ### pick and drop for space efficient RD
                        se_rd_once = False
                elif uop.uop_type == UopType.PATCH_RH:
                    if se_rh_once:
                        pick_drop_num += 2 ### pick and drop for space efficient RH
                        se_rh_once = False
                elif uop.uop_type == UopType.MOVE:
                    pick_drop_num += 1
                else:
                    print("Unknown type of pick/drop", uop.uop_type)
                    assert(0)
            elif cfg_in.rot_type == RotType.DIR_TOGL:
                if uop.uop_type == UopType.MOVE:
                    pick_drop_num += 1
                elif uop.uop_type == UopType.GROUP_DIR:
                    pass

            else:
                pick_drop_num += 1

    ### CZ 1q small
    A    = 3.982611e-02
    C1    = 9.541261e-03
    C0    = 3.594541e-02
    p_eff_scaled = C1 * pick_drop_num + C0
    logical_error = A * (p_eff_scaled ** (code_dist / 2.0))
    ### CZ 1q small

    return logical_error

def tq_logical_error(qn_ctrl, qn_target, lqwise_schedule, pick_drop_offset_c, pick_drop_offset_t, cfg_in, code_dist):
    uop_schedule_c = lqwise_schedule[qn_ctrl]
    uop_schedule_t = lqwise_schedule[qn_target]
    ##

    pick_drop_num_c = pick_drop_offset_c
    pick_drop_num_t = pick_drop_offset_t

    c_se_rd_once = True
    c_se_rh_once = True

    for index, uop_tuple in enumerate(uop_schedule_c):
        uop, interval_type, interval = uop_tuple
        if interval_type in [IntervalType.DROP, IntervalType.PICK]:
            if cfg_in.rot_type == RotType.REFL and cfg_in.refl_type_h == ReflType.STATIC_SE:
                if uop.uop_type == UopType.PATCH_RD:
                    if c_se_rd_once:
                        pick_drop_num_c += 2 ### pick and drop for space efficient RD
                        c_se_rd_once = False
                elif uop.uop_type == UopType.PATCH_RH:
                    if c_se_rh_once:
                        pick_drop_num_c += 2 ### pick and drop for space efficient RH
                        c_se_rh_once = False
                elif uop.uop_type == UopType.MOVE:
                    pick_drop_num_c += 1
                else:
                    print("Unknown type of pick/drop", uop.uop_type)
                    assert(0)
            elif cfg_in.rot_type == RotType.DIR_TOGL:
                if uop.uop_type == UopType.MOVE:
                    pick_drop_num_c += 1
                elif uop.uop_type == UopType.GROUP_DIR:
                    pass

            else:
                pick_drop_num_c += 1

    t_se_rd_once = True
    t_se_rh_once = True
    for index, uop_tuple in enumerate(uop_schedule_t):
        uop, interval_type, interval = uop_tuple
        if interval_type in [IntervalType.DROP, IntervalType.PICK]:
            if cfg_in.rot_type == RotType.REFL and cfg_in.refl_type_h == ReflType.STATIC_SE:
                if uop.uop_type == UopType.PATCH_RD:
                    if t_se_rd_once:
                        pick_drop_num_t += 2 ### pick and drop for space efficient RD
                        t_se_rd_once = False
                elif uop.uop_type == UopType.PATCH_RH:
                    if t_se_rh_once:
                        pick_drop_num_t += 2 ### pick and drop for space efficient RH
                        t_se_rh_once = False
                elif uop.uop_type == UopType.MOVE:
                    pick_drop_num_t += 1
                else:
                    print("Unknown type of pick/drop", uop.uop_type)
                    assert(0)
            elif cfg_in.rot_type == RotType.DIR_TOGL:
                if uop.uop_type == UopType.MOVE:
                    pick_drop_num_t += 1
                elif uop.uop_type == UopType.GROUP_DIR:
                    pass

            else:
                pick_drop_num_t += 1

    ### CZ, small 1Q
    A  = 0.03631249635532327
    c1 = 0.0032295551857353124
    c2 = 0.012020782808911909
    c0 = 0.0561688642202154
    ### CZ, small 1Q

    p_eff_scaled = c1 * pick_drop_num_c + c2 * pick_drop_num_t + c0

    logical_error = A * (p_eff_scaled ** (code_dist / 2.0))

    return logical_error
