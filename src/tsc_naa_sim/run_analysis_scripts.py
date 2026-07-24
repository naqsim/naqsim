import os, sys
#
import pickle
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
#
from config import *
from macro import *
#
from tsc_instructions import UopType, InstType
from tsc_inst_translator import *
from plane_initializer import *
from tsc_inst_scheduler import *
from tsc_qubit_mapper_jit import *
from tsc_uop_scheduler_comptime import *
#
import zstandard as zstd
cctx = zstd.ZstdCompressor(level=10)
################################

def resolve_qmap_in(qmap_in, qc_name=None, cfg_name=None):
    if qmap_in is None:
        return None
    if callable(qmap_in):
        return qmap_in(qc_name, cfg_name)
    if not isinstance(qmap_in, dict):
        return qmap_in

    tuple_key = (qc_name, cfg_name)
    if tuple_key in qmap_in:
        return qmap_in[tuple_key]
    if qc_name in qmap_in:
        qc_maps = qmap_in[qc_name]
        if isinstance(qc_maps, dict) and cfg_name in qc_maps:
            return qc_maps[cfg_name]
    if cfg_name in qmap_in:
        return qmap_in[cfg_name]
    return qmap_in


class suppress_stdout:
    def __enter__(self):
        self.stdout = sys.stdout
        self.devnull = open(os.devnull, 'w')
        sys.stdout = self.devnull

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.devnull.close()


def dump_zst(obj, file_path):
    with open(f'{file_path}.zst', 'wb') as f:
        f.write(cctx.compress(pickle.dumps(obj)))


def load_zst(file_path):
    with open(f'{file_path}.zst', 'rb') as f:
        return pickle.loads(zstd.ZstdDecompressor().decompress(f.read()))


def compilation_stage(
    qc_in: qiskit.QuantumCircuit,
    cfg_in: experiment_config,
    qmap_in: dict = None,
    propagate_creg_names_cond: bool = True,
):
    #### instcution translator ####
    inst_translator = tsc_inst_translator(
        qc_qiskit_in=qc_in,
        rot_trans_opt=cfg_in.rot_trans_opt,
        #
        s_trans_opt=cfg_in.s_trans_opt,
    )
    inst_translator.run()
    inst_dag = deepcopy(inst_translator.sc_dag)

    #### plane initializer ####
    plane_init = plane_initializer(
        num_lq=qc_in.num_qubits,
        plane_type=cfg_in.plane_type
    )
    plane_init.run()
    plane_char = plane_init.plane_char

    #### instruction scheduler ####
    inst_scheduler = tsc_inst_scheduler(
        sc_dag_in=deepcopy(inst_dag),
        plane_char_in=plane_char,
        inst_sched_opt=cfg_in.inst_sched_opt,
        rot_sched_opt=cfg_in.rot_sched_opt,
        s_trans_opt=cfg_in.s_trans_opt,
        skip_h=cfg_in.skip_h,
        propagate_creg_names_cond=propagate_creg_names_cond,
    )
    inst_scheduler.run()
    inst_schedule_trace = inst_scheduler.inst_schedule_trace
    req_schedule_trace = inst_scheduler.req_schedule_trace

    #### qubit mapper ####
    if qmap_in is None:
        qubit_mapper = sa_mapper(
            num_lq=qc_in.num_qubits,
            inst_schedule=inst_schedule_trace,
            plane_char=plane_char,
        )
        qubit_mapper.run()
        qmap_init = qubit_mapper.sa_mapping
    #
    else:
        qmap_init = qmap_in

    return (inst_dag, plane_char, inst_schedule_trace, req_schedule_trace, qmap_init)


def execution_stage(
    comp_out,
    cfg_in,
    run_opt,
):
    inst_dag, plane_char, inst_schedule_trace, req_schedule_trace, qmap_init = comp_out
    ###
    def init_uop_scheduler():
        uop_scheduler = tsc_uop_scheduler_comptime(
                            plane_char=plane_char,
                            qmap_init=qmap_init,
                            inst_schedule_trace=inst_schedule_trace,
                            req_schedule_trace=req_schedule_trace,
                            #
                            lattice_bound=cfg_in.lattice_bound,
                            cell_size=cfg_in.cell_size,
                            plane_type=cfg_in.plane_type,
                            #
                            move_type=cfg_in.move_type,
                            cx_type=cfg_in.cx_type,
                            meas_type=cfg_in.meas_type,
                            hw_cfg=cfg_in.hw_cfg,
                            #
                            skip_uop_grouping=cfg_in.skip_uop_grouping,
                            #
                            is_aod_infinite=cfg_in.is_aod_infinite,
                            num_aodh_max=cfg_in.num_aodh_max,
                            num_aodd_max=cfg_in.num_aodd_max,
                            num_aodr_max=cfg_in.num_aodr_max,
                            #
                            rot_type=cfg_in.rot_type,
                            refl_type_h=cfg_in.refl_type_h,
                            refl_type_d=cfg_in.refl_type_d,
                            rot_plane_opt=cfg_in.rot_plane_opt,
                            num_rot_cell=cfg_in.num_rot_cell,
                            #
                            aod_sched_opt=cfg_in.aod_sched_opt,
                            mov_twostep_opt=cfg_in.mov_twostep_opt,
                            #
                            code_dist=cfg_in.code_dist,
                            rounds=cfg_in.rounds,
                            #
                            s_trans_opt=cfg_in.s_trans_opt,
                            skip_h=cfg_in.skip_h,
                            )
        return uop_scheduler
    ###
    uop_scheduler = init_uop_scheduler()
    if run_opt == RunOpt.IGNORE_NONE:
        uop_scheduler.ignore_path_conflict = False
        uop_scheduler.ignore_rotation = False
    elif run_opt == RunOpt.IGNORE_PC_ROT:
        uop_scheduler.ignore_path_conflict = True
        uop_scheduler.ignore_rotation = True
    elif run_opt == RunOpt.IGNORE_ROT:
        uop_scheduler.ignore_path_conflict = False
        uop_scheduler.ignore_rotation = True
    elif run_opt == RunOpt.IGNORE_PC:
        uop_scheduler.ignore_path_conflict = True
        uop_scheduler.ignore_rotation = False
    else:
        raise Exception()


    ##
    uop_scheduler.run()
    qregs_trace = uop_scheduler.qregs_trace
    plane_trace = uop_scheduler.plane_trace
    uop_schedule_trace = uop_scheduler.uop_schedule_trace
    ##
    exec_out = (uop_schedule_trace, plane_trace, qregs_trace)
    #
    return exec_out

####################

def load_exec_out_dict(outdir, run_opts):
    return {
        run_opt: load_zst(os.path.join(outdir, f"exec_out_{run_opt.name}"))
        for run_opt in run_opts
    }


def get_esm_latency_from_uop_schedule(uop_schedule):
    esm_us = 0
    for intervals in uop_schedule.values():
        for uop, _, (begin, end) in intervals:
            if uop.uop_type == UopType.ESM:
                esm_us += end - begin
    return esm_us


def get_finish_us_from_uop_schedule(uop_schedule):
    finish_us = 0
    for intervals in uop_schedule.values():
        for uop, _, (_, end) in intervals:
            if uop.uop_type == UopType.ESM:
                finish_us = max(finish_us, end)
    return finish_us


def get_aod_latency_from_uop_schedule(uop_schedule):
    main_aod_begin = None
    main_aod_end = None
    for laser_name, intervals in uop_schedule.items():
        if "AOD" not in laser_name:
            continue
        for _, _, (begin, end) in intervals:
            main_aod_begin = begin if main_aod_begin is None else min(main_aod_begin, begin)
            main_aod_end = end if main_aod_end is None else max(main_aod_end, end)
    if main_aod_begin is None and main_aod_end is None:
        return 0
    return main_aod_end - main_aod_begin


def analyze_qc_cfg_from_exec_out_dict(qc_name, exec_out_dict, draw_graph=False):
    uop_schedule_trace_no_pc_rot, _, _ = exec_out_dict[RunOpt.IGNORE_PC_ROT]
    uop_schedule_trace_no_rot, _, _ = exec_out_dict[RunOpt.IGNORE_ROT]
    uop_schedule_trace_no_pc, _, _ = exec_out_dict[RunOpt.IGNORE_PC]
    uop_schedule_trace_all, _, _ = exec_out_dict[RunOpt.IGNORE_NONE]
    assert len(uop_schedule_trace_no_pc_rot) == len(uop_schedule_trace_no_rot) == len(uop_schedule_trace_all)

    label_list = []
    esm_list = []
    others_list = []
    move_list = []
    pc_mov_list = []
    rot_list = []
    pc_rot_list = []
    sum_list = []

    esm_latency = None
    for idx, (
        uop_schedule_no_pc_rot,
        uop_schedule_no_rot,
        uop_schedule_no_pc,
        uop_schedule_all,
    ) in enumerate(
        zip(
            uop_schedule_trace_no_pc_rot,
            uop_schedule_trace_no_rot,
            uop_schedule_trace_no_pc,
            uop_schedule_trace_all,
        )
    ):
        finish_no_pc_rot = get_finish_us_from_uop_schedule(uop_schedule_no_pc_rot)
        finish_no_rot = get_finish_us_from_uop_schedule(uop_schedule_no_rot)
        finish_no_pc = get_finish_us_from_uop_schedule(uop_schedule_no_pc)
        finish_all = get_finish_us_from_uop_schedule(uop_schedule_all)

        if esm_latency is None:
            esm_latency = get_esm_latency_from_uop_schedule(uop_schedule_no_pc_rot)
        move_latency = get_aod_latency_from_uop_schedule(uop_schedule_no_pc_rot)
        others_latency = finish_no_pc_rot - move_latency - esm_latency
        rot_latency = finish_no_pc - finish_no_pc_rot
        pc_latency = finish_all - finish_no_pc
        pc_mov_latency = finish_no_rot - finish_no_pc_rot
        pc_rot_latency = pc_latency - pc_mov_latency
        if pc_rot_latency < 0:
            pc_mov_latency = pc_latency
            pc_rot_latency = 0

        label_list.append(f"L{idx}")
        esm_list.append(esm_latency)
        others_list.append(others_latency)
        move_list.append(move_latency)
        pc_mov_list.append(pc_mov_latency)
        rot_list.append(rot_latency)
        pc_rot_list.append(pc_rot_latency)
        sum_list.append(finish_all)

    qc_cfg_res = {
        "Label": label_list,
        "ESM": esm_list,
        "Others": others_list,
        "Move": move_list,
        "Route_Conflict (Move)": pc_mov_list,
        "Route_Conflict (Rot)": pc_rot_list,
        "Rotation": rot_list,
        "Sum": sum_list,
    }

    if draw_graph:
        df = pd.DataFrame(qc_cfg_res)
        df = df.sort_values(by='Sum').reset_index(drop=True)
        num_cols = df.select_dtypes(include="number")
        df[num_cols.columns] = num_cols.clip(lower=0)
        ax = df.plot(
            x='Label',
            y=[col for col in df.columns if not col in ['Label', 'Sum']],
            kind='area',
            stacked=True,
            linewidth=0,
        )
        plt.title(f'qc: {qc_name}')
        ax.set_ylabel('Latency (us)')
        plt.xlabel('Layer idx')
        plt.tight_layout()
        plt.show()

    return qc_cfg_res


def error_qc_cfg_from_outputs(qc_name, cfg_in, exec_out, comp_out, code_dist, draw_graph=False):
    uop_schedule_trace, _, qregs_trace = exec_out
    assert len(uop_schedule_trace) == len(qregs_trace)

    lqwise_schedule_trace = []
    for uop_schedule, qregs in zip(uop_schedule_trace, qregs_trace):
        live_lq_list = list(qregs.keys())
        lqwise_schedule = {qn: [] for qn in live_lq_list}
        seen_intervals = {qn: set() for qn in live_lq_list}

        def append_unique(qn, uop, interval_type, interval):
            interval_key = tuple(interval)
            seen = seen_intervals[qn]
            if interval_key in seen:
                return
            seen.add(interval_key)
            lqwise_schedule[qn].append((uop, interval_type, interval))

        for uop_interval_list in uop_schedule.values():
            for uop, interval_type, interval in uop_interval_list:
                if uop.uop_type == UopType.ESM:
                    for qn in live_lq_list:
                        append_unique(qn, uop, interval_type, interval)
                elif uop.uop_type == UopType.PATCH_MEAS:
                    continue
                else:
                    for qn in uop.qreg_names:
                        append_unique(qn, uop, interval_type, interval)
        lqwise_schedule_trace.append(lqwise_schedule)

    from logical_error_model import sq_logical_error
    from logical_error_model import tq_logical_error
    from logical_error_model import calc_group_dir_pick_drop_num

    _, _, inst_schedule_trace, req_schedule_trace, _ = comp_out

    y_dist_error = 0
    m_dist_error = 0

    label_list = []
    layer_logical_errors = []
    for idx, (lqwise_schedule, inst_schedule, req_schedule) in enumerate(
        zip(lqwise_schedule_trace, inst_schedule_trace, req_schedule_trace)
    ):
        layer_logical_error = 0

        if y_dist_error or m_dist_error:
            qn_req_y = set()
            qn_req_m = set()
            for req in req_schedule:
                for qn in req.qreg_names:
                    if "Y" in qn:
                        qn_req_y.add(qn)
                    elif "M" in qn:
                        qn_req_m.add(qn)
                    else:
                        raise Exception()
            layer_logical_error += len(qn_req_y) * y_dist_error
            layer_logical_error += len(qn_req_m) * m_dist_error

        pick_drop_num_dict = calc_group_dir_pick_drop_num(lqwise_schedule, cfg_in)
        remaining_qubits = set(lqwise_schedule.keys())

        print("idx: ", idx, "total offset", sum([pick_drop_num_dict[key] for key in pick_drop_num_dict.keys()]))

        for cx in (inst for inst in inst_schedule if inst.inst_type == InstType.TRANS_CX):
            qn_ctrl, qn_target = cx.qreg_names
            tq_error = tq_logical_error(
                qn_ctrl=qn_ctrl,
                qn_target=qn_target,
                lqwise_schedule=lqwise_schedule,
                pick_drop_offset_c=pick_drop_num_dict[qn_ctrl],
                pick_drop_offset_t=pick_drop_num_dict[qn_target],
                cfg_in=cfg_in,
                code_dist=code_dist,
            )
            layer_logical_error += tq_error
            remaining_qubits.discard(qn_ctrl)
            remaining_qubits.discard(qn_target)

        layer_tq_error = deepcopy(layer_logical_error)
        for qn in remaining_qubits:
            sq_error = sq_logical_error(
                qn=qn,
                lqwise_schedule=lqwise_schedule,
                pick_drop_offset=pick_drop_num_dict[qn],
                cfg_in=cfg_in,
                code_dist=code_dist,
            )
            layer_logical_error += sq_error

        layer_sq_error = deepcopy(layer_logical_error) - layer_tq_error
        layer_logical_errors.append(layer_logical_error)
        label_list.append(f"L{idx}")
        print("idx: ", idx, "1Q: ", layer_sq_error, "2Q: ", layer_tq_error)
        print("idx: ", idx, "total pick drop", sum([pick_drop_num_dict[key] for key in pick_drop_num_dict.keys()]))

    qc_cfg_err = {
        "Label": label_list,
        "Logical_Error": layer_logical_errors,
    }

    if draw_graph:
        df = pd.DataFrame(qc_cfg_err)
        ax = df.plot(
            x='Label',
            y=[col for col in df.columns if not col in ['Label']],
            kind='line',
            marker='o',
            color='black',
            linewidth=1.0,
            markersize=1
        )
        plt.title(f'qc: {qc_name}')
        ax.set_ylabel('Logical error rate')
        plt.xlabel('Layer idx')
        plt.tight_layout()
        plt.show()

    return qc_cfg_err
