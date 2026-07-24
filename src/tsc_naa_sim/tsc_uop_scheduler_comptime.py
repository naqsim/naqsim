from latency_functions import latency_functions
from macro import *
from qreg_plane import *
from tsc_instructions import *
#
from bisect import bisect_left
from collections import deque
from copy import deepcopy
from itertools import chain, combinations, permutations, product
import numpy as np
import rustworkx as rx
import sys
from scipy.optimize import linear_sum_assignment

global LaserType

class tsc_uop_scheduler_comptime:
    def __init__(self,
                 plane_char,
                 qmap_init,
                 inst_schedule_trace,
                 req_schedule_trace,
                 #
                 lattice_bound,
                 cell_size,
                 plane_type,
                 #
                 move_type,
                 cx_type,
                 meas_type,
                 hw_cfg,
                 #
                 skip_uop_grouping,
                 is_aod_infinite,
                 num_aodh_max,
                 num_aodd_max,
                 num_aodr_max,
                 #
                 rot_type,
                 refl_type_h,
                 refl_type_d,
                 rot_plane_opt,
                 num_rot_cell,
                 #
                 aod_sched_opt,
                 mov_twostep_opt,
                 #
                 code_dist,
                 rounds,
                 #
                 s_trans_opt,
                 skip_h,
                 #
                 ignore_path_conflict=False,
                 ignore_rotation=False,
                 #
                 debug_print=False,
                 #
                 ):
        # debug
        self.debug_print = debug_print
        # input
        self.rot_plane_opt = rot_plane_opt
        self.num_rot_cell = num_rot_cell
        self.aod_sched_opt = aod_sched_opt
        self.mov_twostep_opt = mov_twostep_opt
        ##
        self.lattice_bound = lattice_bound
        self.cell_size = cell_size
        self.plane = None
        self.init_plane(plane_char, lattice_bound, cell_size, plane_type)
        self.qregs = None
        self.init_qregs(qmap_init)
        self.can_overlap = True
        self.cannot_overlap = False
        ##
        self.inst_schedule_trace = inst_schedule_trace
        self.req_schedule_trace = req_schedule_trace
        ##
        self.plane_type = plane_type
        self.move_type = move_type
        self.cx_type = cx_type
        self.meas_type = meas_type
        self.lftn = latency_functions(hw_cfg,
                                      cx_type,
                                      meas_type,
                                      move_type)
        ##
        self.rot_type = rot_type
        if self.rot_type == RotType.REFL:
            self.refl_type_h = refl_type_h
            self.refl_type_d = refl_type_d
        else:
            self.refl_type_h = None
            self.refl_type_d = None
        ##
        self.code_dist = code_dist
        self.rounds = rounds

        ##
        self.is_aod_infinite = is_aod_infinite
        if self.is_aod_infinite:
            large_num = 500
            self.num_aodh_max = large_num
            self.num_aodd_max = large_num
            self.num_aodr_max = large_num
        else:
            self.num_aodh_max = num_aodh_max
            self.num_aodd_max = num_aodd_max
            self.num_aodr_max = num_aodr_max
        self.init_lasers()
        ##
        self.skip_uop_grouping = skip_uop_grouping
        ##
        self.ignore_path_conflict = ignore_path_conflict
        self.ignore_rotation = ignore_rotation
        ##
        self.s_trans_opt = s_trans_opt
        self.skip_h = skip_h

        # output
        self.uop_schedule_trace = []
        self.plane_trace = []
        self.qregs_trace = []
        #
        self.uop_dag_trace = []

        # intermediate data structures
        ## preprocessing
        ### dynamically changing qubit names at each layer
        self.qubit_names_trace = []
        ### next CX events indexed by current qubit name
        self.next_cx_events_by_qn = dict()
        ##
        self.layer_time = 0
        self.layer_latency = 0
        self.uop_schedule = dict()
        ##
        self.aodh_free_list = []
        self.aodd_free_list = []
        self.aodr_free_list = []
        self.aod_inuse_track = dict()
        self.qn_inuse_track = dict()
        self.reset_aod()

    def reset_aod(self):
        self.aodh_free_list = [LaserType[f'AODH_{i}'] for i in range(self.num_aodh_max)]
        self.aodd_free_list = [LaserType[f'AODD_{i}'] for i in range(self.num_aodd_max)]
        self.aodr_free_list = [LaserType[f'AODR_{i}'] for i in range(self.num_aodr_max)]
        self.aod_inuse_track = dict()
        #
        self.qn_inuse_track = dict()

    def moving_cost(self, pos1, pos2, layer_gap):
        # Calculate the moving_cost
        ## Calculate the cost with the current positions
        ## But, real movement will happen later the layer_gap
        #
        distance = self.moving_distance_rev(pos1, pos2)

        # can adjust the weight function
        max_level = 5
        list_weight = [1-0.2*l for l in range(max_level)]
        if layer_gap < max_level:
            weight = list_weight[layer_gap]
        else:
            weight = list_weight[-1]
        #
        cost = weight * distance

        return cost

    def init_lasers(self):
        base_lasers = {
            'RMN_L': auto(),
            'RYD_L': auto(),
            'IMG_L': auto(),
        }
        aod_d_lasers = {f'AODD_{i}': auto() for i in range(0, self.num_aodd_max)}
        aod_h_lasers = {f'AODH_{i}': auto() for i in range(0, self.num_aodh_max)}
        aod_r_lasers = {f'AODR_{i}': auto() for i in range(0, self.num_aodr_max)}
        all_lasers = {**base_lasers, **aod_d_lasers, **aod_h_lasers, **aod_r_lasers}

        global LaserType
        LaserType = Enum('LaserType', all_lasers)

    def init_plane(self, plane_char, lattice_bound, cell_size, plane_type):
        height = len(plane_char)
        width = len(plane_char[0])
        field = []
        for r in range(height):
            field_row = []
            for c in range(width):
                ct_char = plane_char[r][c]
                if ct_char == 'M':
                    ct = CellType.PORT_M
                elif ct_char == 'Y':
                    ct = CellType.PORT_Y
                elif ct_char == 'N':
                    ct = CellType.NORMAL
                else:
                    ct = CellType.INVALID
                cp = (r, c)
                cs_q = CellStatusQ.FREE
                cs_op = CellStatusOp.IDLE
                cell = Cell(cp, ct, cs_q, cs_op)
                field_row.append(cell)
            field.append(field_row)
        self.plane = QubitPlane(width, height, field, lattice_bound, cell_size, plane_type)
        ##
        if self.rot_plane_opt == RotPlaneOpt.DEDICATED_ROT:
            def centers(n, m, k):
                r = int(math.sqrt(k))
                while k % r != 0:
                    r -= 1
                c = k // r
                #
                h = n // r
                w = m // c
                #
                xs = [h//2 + h*i for i in range(r)]
                ys = [w//2 + w*j for j in range(c)]

                return [(x,y) for x in xs for y in ys]
            #
            self.rotpch_pos_list = [(r+2, c) for (r, c) in centers(self.plane.h-2, self.plane.w, self.num_rot_cell)]
            for (r, c) in product(range(self.plane.h), range(self.plane.w)):
                cell = self.plane.field[r][c]
                if (r, c) in self.rotpch_pos_list:
                    cell.can_rot = True
                else:
                    cell.can_rot = False
        elif self.rot_plane_opt == RotPlaneOpt.ALL_ROT:
            self.rotpch_pos_list = []
            for (r, c) in product(range(self.plane.h), range(self.plane.w)):
                self.rotpch_pos_list.append((r, c))
                cell = self.plane.field[r][c]
                cell.can_rot = True
        else:
            raise Exception()
        return

    def init_qregs(self, qmap_init):
        self.qregs = dict()
        for qn, pos in qmap_init.items():
            self.allocate_qreg(qn, pos)
        return

    def allocate_qreg(self, qn, pos):
        assert qn not in self.qregs.keys()
        self.qregs[qn] = Qreg(qn, QregStatus.ACTIVE_A, pos)
        #
        (r, c) = pos
        cell = self.plane.field[r][c]
        cell.status_q = CellStatusQ.OCCUPIED_Q
        cell.occupants.append(qn)
        return

    def trace_qubit_names(self):
        # qubit name trace
        qubit_names_trace = []
        qubit_names = dict()
        for qn in self.qregs.keys():
            qubit_names[qn] = qn
        # two-layer timing difference
        ## CX for resource state teleport
        ## -> Teleport fini/hsed (i.e., name changed)
        qubit_names_trace.append(deepcopy(qubit_names))
        qubit_names_trace.append(deepcopy(qubit_names))

        for inst_schedule in self.inst_schedule_trace:
            for inst in inst_schedule:
                if inst.inst_type == InstType.TRANS_CX:
                    [qn_0, qn_1] = inst.oreg_names
                    if ("M" in qn_0 or "Y" in qn_0) and not ("M" in qn_1 or "Y" in qn_1):
                        new_qn = qn_0
                        ori_qn = qn_1
                    elif ("M" in qn_1 or "Y" in qn_1) and not ("M" in qn_0 or "Y" in qn_0):
                        new_qn = qn_1
                        ori_qn = qn_0
                    elif ("M" in qn_0 and "Y" in qn_1) or ("M" in qn_1 and "Y" in qn_0):
                        continue
                    else:
                        continue
                    qubit_names[ori_qn] = new_qn
            qubit_names_trace.append(deepcopy(qubit_names))
        self.qubit_names_trace = qubit_names_trace[:-2]
        return

    def trace_next_cx(self):
        # Precompute future CX events for each current qubit name. The pair name is
        # resolved at query time because Q-origin names depend on start_idx.
        events_by_qn = dict()
        for layer_idx, inst_schedule in enumerate(self.inst_schedule_trace):
            for inst in inst_schedule:
                if inst.inst_type != InstType.TRANS_CX:
                    continue
                for qn in inst.qreg_names:
                    next_pair_on = None
                    for curr_qn, orig_qn in zip(inst.qreg_names, inst.oreg_names):
                        if curr_qn != qn:
                            next_pair_on = orig_qn
                            break
                    if next_pair_on is None:
                        continue
                    events_by_qn.setdefault(qn, []).append((layer_idx, next_pair_on))

        self.next_cx_events_by_qn = dict()
        for qn, events in events_by_qn.items():
            layers = [layer_idx for layer_idx, _ in events]
            pair_ons = [next_pair_on for _, next_pair_on in events]
            self.next_cx_events_by_qn[qn] = (layers, pair_ons)
        return

    def find_next_cx(self, start_idx, qn):
        # Find the next upcoming CX of the given qn
        # Return
        ## layer_idx of the next CX
        next_layer_idx = None
        ## pair qubit's name at the next CX
        ### the qubit name at the timing of start_idx
        next_pair_qn = None
        ###
        events = self.next_cx_events_by_qn.get(qn)
        if events is None:
            return next_layer_idx, next_pair_qn

        layers, pair_ons = events
        event_idx = bisect_left(layers, start_idx)
        if event_idx == len(layers):
            return next_layer_idx, next_pair_qn

        next_layer_idx = layers[event_idx]
        next_pair_on = pair_ons[event_idx]
        if "Q" in next_pair_on:
            next_pair_qn = self.qubit_names_trace[start_idx][next_pair_on]
        else:
            next_pair_qn = next_pair_on
        return next_layer_idx, next_pair_qn

    def req_placing_cost(self, layer_idx, req, pos):
        # Find the next upcoming CX
        ##
        if req.inst_type in [InstType.REQ_MY, InstType.REQ_M]:
            start_idx = layer_idx+1
            [qn] = [qn for qn in req.qreg_names if "M" in qn]
        elif req.inst_type == InstType.REQ_Y:
            start_idx = layer_idx
            [qn] = req.qreg_names
        else:
            raise Exception()
        ##
        next_layer_idx, next_pair_qn = self.find_next_cx(start_idx, qn)
        ### If pair qreg is not yet assigned, return cost of zero
        if next_layer_idx is None or not next_pair_qn or next_pair_qn not in self.qregs:
            return 0
        layer_gap = (next_layer_idx - layer_idx)

        # Calculate the cost
        pair_pos = self.qregs[next_pair_qn].pos
        #
        cost = self.moving_cost(pos, pair_pos, layer_gap)
        return cost

    def req_placement(self, layer_idx, req_schedule):
        if not req_schedule:
            return
        ### Minimum-weight full matching ###
        # Left: MY requests
        left = req_schedule
        # Right: candidate Y port cells
        ## Assumption: M will be placed nearest cell to the Y
        if self.s_trans_opt == STransOpt.GATE_TEL:
            right = self.plane.get_free_pos(CellType.PORT_Y)
        elif self.s_trans_opt == STransOpt.TRANS_S:
            right = self.plane.get_free_pos(CellType.PORT_M)
        else:
            raise Exception()

        # Build cost matrix
        cost_matrix = np.full((len(left), len(right)), sys.maxsize, dtype=float)
        for i, req in enumerate(left):
            for j, pos in enumerate(right):
                cost_matrix[i][j] = self.req_placing_cost(layer_idx, req, pos)
        # Solve the minimum-weight full matching
        left_ids, right_ids = linear_sum_assignment(cost_matrix)

        # Allocate M & Y following the determeined positions
        for left_id, right_id in zip(left_ids, right_ids):
            req = left[left_id]
            pos_y = right[right_id]
            ## REQ_Y
            if req.inst_type == InstType.REQ_Y:
                ### Allocate Y
                [qn_y] = req.qreg_names
                self.allocate_qreg(qn_y, pos_y)
            ## REQ_M
            elif req.inst_type == InstType.REQ_MY:
                ### Allocate Y
                [qn_y] = [qn for qn in req.qreg_names if "Y" in qn]
                self.allocate_qreg(qn_y, pos_y)
                ### Allocate M
                [qn_m] = [qn for qn in req.qreg_names if "M" in qn]
                #### Find the nearest PORT_M with the Y
                pos_m = None
                min_dist_m = None
                for pos in self.plane.get_free_pos(CellType.PORT_M):
                    dist_m = self.moving_distance_rev(pos, pos_y)
                    if (not pos_m) or (dist_m < min_dist_m):
                        pos_m = pos
                        min_dist_m = dist_m
                assert pos_m
                self.allocate_qreg(qn_m, pos_m)
            ##
            elif req.inst_type == InstType.REQ_M:
                ### Allocate M
                [qn_m] = req.qreg_names
                self.allocate_qreg(qn_m, pos_y) #pos_m
            else:
                raise Exception()
        return


    def schedule_nonaod_before(self, inst_schedule):
        # SYNCHRONOUS
        phase_time = self.layer_time
        finish_time = phase_time
        phase_uops = []

        ## PatchH - RMN_L
        if self.skip_h:
            h_scheduled = []
        else:
            h_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H, InstType.TRANS_H_ROT]]
        if h_scheduled:
            qubits = []
            for h in h_scheduled:
                qubits += h.qreg_names
            #
            uop = UopPatchH(qubits)
            phase_uops.append(uop)
            #
            laser = LaserType.RMN_L
            #
            if laser.name not in self.uop_schedule.keys():
                self.uop_schedule[laser.name] = []
            #
            intervals = uop.get_intervals(phase_time, self.lftn)
            for (interval_type, interval_us) in intervals:
                self.uop_schedule[laser.name].append((uop, interval_type, interval_us))
                finish_time = max(finish_time, interval_us[-1])

        ## PatchMeas - IMG_L in SELECT
        meas_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.MEAS_Z, InstType.MEAS_XorZ, InstType.MEAS_RESET_Z]]
        if meas_scheduled and self.meas_type == PhyOpType.SELECT:
            meas_qubits = []
            reset_qubits = []
            for meas in meas_scheduled:
                meas_qubits += meas.qreg_names
                if meas.inst_type == InstType.MEAS_RESET_Z:
                    reset_qubits += meas.qreg_names
            #
            uop = UopPatchMeas(meas_qubits)
            uop.reset_qubits = reset_qubits
            phase_uops.append(uop)
            ##
            laser = LaserType.IMG_L
            if laser.name not in self.uop_schedule.keys():
                self.uop_schedule[laser.name] = []
            ##
            intervals = uop.get_intervals(phase_time, self.lftn)
            for (interval_type, interval_us) in intervals:
                self.uop_schedule[laser.name].append((uop, interval_type, interval_us))
                finish_time = max(finish_time, interval_us[-1])
        ##
        self.layer_time = finish_time
        #
        for uop in phase_uops:
            uop.update(self.plane, self.qregs)
        return



    def schedule_main_aod(self, layer_idx, inst_schedule):
        phase_time = self.layer_time

        ##### DECODE #####
        ### Move & Rotation generation
        uop_dag = self.generate_aod_uops(layer_idx, inst_schedule)
        self.uop_dag_trace.append(deepcopy(uop_dag))
        ##
        if len(uop_dag) == 0:
            return

        if self.debug_print:
            print("GENERATED UOPS")
            for uop in uop_dag.nodes():
                if uop.uop_type == UopType.MOVE:
                    print(uop.uop_name, uop.qreg_names, uop.src_dst_list)
                else:
                    print(uop.uop_name, uop.qreg_names)
            print()

        ##### Issue until drain #####
        self.reset_aod()
        #
        mov_drain = True
        rot_drain = False
        #
        while len(uop_dag) != 0:
            front_indices = [i for i in uop_dag.node_indices() if uop_dag.in_degree(i) == 0]
            front_uops = [uop_dag[i] for i in front_indices]
            #
            if self.is_aod_infinite:
                assert self.skip_uop_grouping
                ###
                group_dict = dict()
                for idx, uop in zip(front_indices, front_uops):
                    if uop.uop_type == UopType.MOVE:
                        # check move step
                        uop.check_move_step(self.plane)
                        # split move
                        if uop.is_twostep:
                            uop_1 = deepcopy(uop)
                            uop_1.split_move_step(self.plane, 'dq_xy')
                            # skip grouping
                            uop_1.set_aod_type(AodType.AODH, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            group_dict[uop_1] = [idx]
                            #
                            uop_2 = deepcopy(uop)
                            uop_2.split_move_step(self.plane, 'xy_dq')
                            # skip grouping
                            uop_2.set_aod_type(AodType.AODH, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            group_dict[uop_2] = [idx]
                        else:
                            uop.split_move_step(self.plane, '')
                            # skip grouping
                            uop.set_aod_type(AodType.AODH, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            group_dict[uop] = [idx]
                    elif uop.uop_type == UopType.PATCH_RH:
                        uop.set_aod_type(AodType.AODH, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                        # skip grouping
                        group_dict[uop] = [idx]
                    elif uop.uop_type == UopType.PATCH_RD:
                        uop.set_aod_type(AodType.AODD, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                        # skip grouping
                        group_dict[uop] = [idx]
                    elif uop.uop_type == UopType.PATCH_DIR:
                        uop.set_aod_type(AodType.AODR, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                        # skip grouping
                        group_dict[uop] = [idx]
                    elif uop.uop_type == UopType.GROUP_DIR:
                        uop.set_aod_type(AodType.AODR, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                        group_dict[uop] = [idx]
                    else:
                        raise Exception()

            else: # finite aods
                ##### Grouping #####
                ## Operation dependency
                ### Check only the front_layer (i.e., without dependency)
                ### Might miss some grouping opportunities in a naive approach
                if not self.skip_uop_grouping:
                    # Step 1. Classify uops based on the uop_type
                    move_idx_map = dict()  # uop:idx
                    rh_idx_map = dict()    # uop:idx
                    rd_idx_map = dict()    # uop:idx
                    #
                    grp_idx_map = dict()
                    for idx, uop in zip(front_indices, front_uops):
                        # Move
                        if uop.uop_type == UopType.MOVE:
                            move_idx_map[uop] = idx
                        # Rotation: PatchRH
                        elif uop.uop_type == UopType.PATCH_RH:
                            rh_idx_map[uop] = idx
                        # Rotation: PatchRD
                        elif uop.uop_type == UopType.PATCH_RD:
                            rd_idx_map[uop] = idx
                        # Rotation: GroupDIR
                        elif uop.uop_type == UopType.GROUP_DIR:
                            grp_idx_map[uop] = idx
                        else:
                            raise Exception()

                    # Step 2. Generate grouped uops for each type
                    group_dict = dict()
                    ## Move
                    ### (1) Build conflict graph
                    def aod_conflict(mov1, mov2):
                        cond_list = []
                        for src_dst_1, src_dst_2 in product(mov1.src_dst_list, mov2.src_dst_list):
                            ((src_r_1, src_c_1), (dst_r_1, dst_c_1)) = src_dst_1
                            ((src_r_2, src_c_2), (dst_r_2, dst_c_2)) = src_dst_2

                            # Cond 1. same src diff dst
                            cond1_r = (src_r_1 == src_r_2) and (dst_r_1 != dst_r_2)
                            cond1_c = (src_c_1 == src_c_2) and (dst_c_1 != dst_c_2)
                            cond1 = (cond1_r or cond1_c)

                            # Cond 2. same dst diff src
                            cond2_r = (src_r_1 != src_r_2) and (dst_r_1 == dst_r_2)
                            cond2_c = (src_c_1 != src_c_2) and (dst_c_1 == dst_c_2)
                            cond2 = (cond2_r or cond2_c)

                            # Cond 3. order violation
                            cond3_r = (src_r_1 - src_r_2) * (dst_r_1 - dst_r_2) < 0
                            cond3_c = (src_c_1 - src_c_2) * (dst_c_1 - dst_c_2) < 0
                            cond3 = (cond3_r or cond3_c)

                            #
                            cond = (cond1 or cond2 or cond3)
                            cond_list.append(cond)

                        return any(cond_list)
                    #####
                    # key: uop_mov
                    # val: set of uop_movs in aod_conflict
                    conflict_graph = dict()
                    for uop in move_idx_map.keys():
                        conflict_graph[uop] = set()
                    # aod conflicts
                    for uop_1, uop_2 in combinations(move_idx_map.keys(), 2):
                        if aod_conflict(uop_1, uop_2):
                            conflict_graph[uop_1].add(uop_2)
                            conflict_graph[uop_2].add(uop_1)
                    # debug
                    if self.debug_print:
                        print("Conflict graph")
                        for uop, adj in conflict_graph.items():
                            print(f"{uop.uop_name} {uop.qreg_names}: ", end='')
                            for uop_adj in adj:
                                print(f"{uop_adj.uop_name} {uop_adj.qreg_names} ", end='')
                            print()
                        print()

                    ### (2) Iterate - Find a maximal independent set
                    def maximal_independet_set(graph):
                        ## sort
                        remaining = list(graph.keys())
                        dist_list = []
                        for uop in remaining:
                            [(src, dst)] = uop.src_dst_list
                            dist = self.moving_distance_rev(src, dst)
                            dist_list.append(dist)
                        ###
                        remaining_sorted = [uop for _, uop in sorted(zip(dist_list, remaining), key=lambda x: x[0])]
                        remaining = list(remaining_sorted)


                        selected = set()
                        ##
                        while remaining:
                            v = remaining[0]
                            selected.add(v)
                            #
                            forbidden = graph[v] | {v}
                            remaining = [u for u in remaining if u not in forbidden]
                        ##
                        return selected
                    #####
                    graph = conflict_graph
                    groups = []
                    while graph:
                        indep = maximal_independet_set(graph)
                        groups.append(indep)
                        to_delete = set(indep)
                        ## delete from the graph
                        for uop in to_delete:
                            if uop not in graph:
                                continue
                            for nbr in graph[uop]:
                                if nbr in graph:
                                    graph[nbr].discard(uop)
                            del graph[uop]
                    # got move groups
                    for group in groups:
                        src_dst_list = []
                        qreg_names = []
                        idx_list = []
                        for uop in group:
                            src_dst_list += uop.src_dst_list
                            qreg_names += uop.qreg_names
                            idx_list.append(move_idx_map[uop])
                        uop_mov = UopMove(qreg_names, src_dst_list)
                        ##
                        uop_mov.check_move_step(self.plane)
                        if uop_mov.is_twostep:
                            uop_mov_1 = deepcopy(uop_mov)
                            uop_mov_1.split_move_step(self.plane, 'dq_xy')
                            uop_mov_1.set_aod_type(AodType.AODH, self.num_aodh_max, None, None)
                            group_dict[uop_mov_1] = idx_list
                            ##
                            uop_mov_2 = deepcopy(uop_mov)
                            uop_mov_2.split_move_step(self.plane, 'xy_dq')
                            uop_mov_2.set_aod_type(AodType.AODH, self.num_aodh_max, None, None)
                            group_dict[uop_mov_2] = idx_list

                        else:
                            uop_mov.split_move_step(self.plane, '')
                            uop_mov.set_aod_type(AodType.AODH, self.num_aodh_max, None, None)
                            group_dict[uop_mov] = idx_list

                    ## PatchRH - Group to single uop
                    rh_idx_list = []
                    rh_qreg_names = []
                    for uop in rh_idx_map.keys():
                        rh_qreg_names += uop.qreg_names
                        rh_idx_list.append(rh_idx_map[uop])
                    if rh_qreg_names:
                        uop_rh = UopPatchRH(rh_qreg_names, self.refl_type_h)
                        uop_rh.set_aod_type(AodType.AODH, self.num_aodh_max, None, None)
                        group_dict[uop_rh] = rh_idx_list

                    ## PatchRD - Group to one or two uop
                    ### NOTE: we cannot group cells at different diagonal planes
                    odd_idx_list = []
                    odd_qreg_names = []
                    even_idx_list = []
                    even_qreg_names = []
                    ####
                    for uop in rd_idx_map.keys():
                        [qn] = uop.qreg_names
                        qr = self.qregs[qn]
                        (xy_r, xy_c) = qr.pos
                        dq_r, dq_c = xy_to_dq(xy_r, xy_c, self.plane.h)
                        if dq_r % 2 == 0: # even rows
                            even_qreg_names.append(qn)
                            even_idx_list.append(rd_idx_map[uop])
                        else: # odd rows
                            odd_qreg_names.append(qn)
                            odd_idx_list.append(rd_idx_map[uop])
                    if even_qreg_names:
                        uop_rd = UopPatchRD(even_qreg_names, self.refl_type_d)
                        uop_rd.set_aod_type(AodType.AODD, None, self.num_aodd_max, None)
                        group_dict[uop_rd] = even_idx_list
                    if odd_qreg_names:
                        uop_rd = UopPatchRD(odd_qreg_names, self.refl_type_d)
                        uop_rd.set_aod_type(AodType.AODD, None, self.num_aodd_max, None)
                        group_dict[uop_rd] = odd_idx_list

                    ## GroupDIR: already grouped uop
                    for uop, idx in grp_idx_map.items():
                        uop.set_aod_type(AodType.AODR, self.num_aodh_max, None, self.num_aodr_max)
                        group_dict[uop] = [idx]

                else: # finite aods but skip_uop_grouping
                    group_dict = dict()
                    for idx, uop in zip(front_indices, front_uops):
                        if uop.uop_type == UopType.MOVE:
                            # check move step
                            uop.check_move_step(self.plane)
                            uop.split_move_step(self.plane, '')
                            # skip grouping
                            uop.set_aod_type(AodType.AODH, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            group_dict[uop] = [idx]
                        elif uop.uop_type == UopType.PATCH_RH:
                            uop.set_aod_type(AodType.AODH, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            # skip grouping
                            group_dict[uop] = [idx]
                        elif uop.uop_type == UopType.PATCH_RD:
                            uop.set_aod_type(AodType.AODD, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            # skip grouping
                            group_dict[uop] = [idx]
                        elif uop.uop_type == UopType.PATCH_DIR:
                            uop.set_aod_type(AodType.AODR, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            # skip grouping
                            group_dict[uop] = [idx]
                        elif uop.uop_type == UopType.GROUP_DIR:
                            uop.set_aod_type(AodType.AODR, self.num_aodh_max, self.num_aodd_max, self.num_aodr_max)
                            group_dict[uop] = [idx]
                        else:
                            raise Exception()


            ##### Wake-up #####
            if self.debug_print:
                for uop in group_dict.keys():
                    print(f"GROUPED: {uop.uop_name} {uop.qreg_names}")
            ## AOD & Qn & Trap availibility
            ready_uops = []
            for uop in group_dict.keys():
                aod_cond = uop.check_free_aod(self.aodh_free_list, self.aodd_free_list, self.aodr_free_list)
                ###
                qn_cond = True
                for qn in uop.qreg_names:
                    if not qn in self.qn_inuse_track.keys():
                        continue
                    ##
                    qn_finish_us = self.qn_inuse_track[qn]
                    if (qn_finish_us - phase_time) > 1e-9:
                        qn_cond = False
                    ##
                ###
                trap_cond = True
                if uop.uop_type == UopType.MOVE:
                    dst_cells = [self.plane.field[r][c]for _, (r, c) in uop.src_dst_list]
                    if any([cell.status_q == CellStatusQ.OVERLAP_Q for cell in dst_cells]):
                        if self.debug_print:
                            print(f"BLOCK... MOVE {uop.qreg_names}")
                        trap_cond = False
                elif uop.uop_type in [UopType.GROUP_DIR]:
                    rot_cells = [self.plane.field[r][c] for (r, c) in uop.rot_pos_list]
                    if any([cell.status_q != CellStatusQ.OCCUPIED_Q for cell in rot_cells]):
                        if self.debug_print:
                            print(f"BLOCK... GROUP_DIR {uop.qreg_names}")
                            for cell in rot_cells:
                                print(f"{cell.pos} {cell.occupants}")
                        trap_cond = False
                else:
                    pass
                ###
                if aod_cond and qn_cond and trap_cond:
                    ready_uops.append(uop)

            ## Proceed the phase time & release AODs
            if not ready_uops:
                if self.debug_print:
                    print("NOT READY UOPS")
                if not self.aod_inuse_track and self._bypass_full_move_intermediate(
                    uop_dag, front_indices
                ):
                    continue
                ## find the earliest AODs to finish
                release_time = min(self.aod_inuse_track.values())
                release_lasers = []
                for laser, time in self.aod_inuse_track.items():
                    if (time - release_time) < 1e-9: # same
                        release_lasers.append(laser)
                ## free the found AOD
                for release_laser in release_lasers:
                    del self.aod_inuse_track[release_laser]
                    if "AODH" in release_laser.name:
                        self.aodh_free_list.append(release_laser)
                    elif "AODD" in release_laser.name:
                        self.aodd_free_list.append(release_laser)
                    elif "AODR" in release_laser.name:
                        self.aodr_free_list.append(release_laser)
                    else:
                        raise Exception()
                ##
                release_qns = []
                for qn, time in self.qn_inuse_track.items():
                    if (time - release_time) < 1e-9:
                        release_qns.append(qn)
                ##
                for release_qn in release_qns:
                    del self.qn_inuse_track[release_qn]
                ##
                if self.debug_print:
                    print(f"Phase time: {phase_time} -> {release_time}")
                phase_time = release_time

                continue

            ###### Select #####
            ### Naive: Drain Move -> Drain Rot -> Drain Move
            if self.aod_sched_opt == AodSchedOpt.NAIVE_DRAIN:
                selected_uop = None
                #
                if (not selected_uop) and mov_drain:
                    for uop in ready_uops:
                        if uop.uop_type == UopType.MOVE:
                            selected_uop = uop
                            break
                if (not selected_uop) and mov_drain:
                    mov_drain = False
                    rot_drain = True
                #
                if (not selected_uop) and rot_drain:
                    for uop in ready_uops:
                        if uop.uop_type in [UopType.PATCH_RD, UopType.PATCH_RH, UopType.PATCH_DIR, UopType.GROUP_DIR]:
                            selected_uop = uop
                            break
                if (not selected_uop) and rot_drain:
                    mov_drain = True
                    rot_drain = False
                #
                if (not selected_uop) and mov_drain:
                    for uop in ready_uops:
                        if uop.uop_type == UopType.MOVE:
                            selected_uop = uop
                            break
                ##
            elif self.aod_sched_opt == AodSchedOpt.LAST_FINISH:
                max_finish_us = None
                selected_uop = None
                for uop in ready_uops:
                    uop.inspect_delay(phase_time, self.plane, self.qregs, self.lftn, self.code_dist)
                    finish_us = uop.inspect_finish(phase_time, self.ignore_rotation)
                    if (max_finish_us is None) or (max_finish_us < finish_us):
                        max_finish_us = finish_us
                        selected_uop = uop

            elif self.aod_sched_opt == AodSchedOpt.FIRST_FINISH:
                min_finish_us = None
                selected_uop = None
                for uop in ready_uops:
                    uop.inspect_delay(phase_time, self.plane, self.qregs, self.lftn, self.code_dist)
                    finish_us = uop.inspect_finish(phase_time, self.ignore_rotation)
                    if (min_finish_us is None) or (min_finish_us > finish_us):
                        min_finish_us = finish_us
                        selected_uop = uop
            else:
                raise Exception()
            ##
            assert selected_uop
            selected_indices = group_dict[selected_uop]

            if self.debug_print:
                print(f"SELECTED: {selected_uop.uop_name} {selected_uop.qreg_names} {selected_uop.aod_type.name}", end="")
                if selected_uop.uop_type == UopType.MOVE:
                    print(selected_uop.src_dst_list)
                else:
                    print()

            ##### Issue #####
            ## Get intervals
            delay_us = selected_uop.inspect_delay(phase_time, self.plane, self.qregs, self.lftn, self.code_dist)
            intervals = selected_uop.get_intervals(phase_time, self.plane, self.qregs, self.lftn, self.code_dist, self.ignore_rotation)

            ## Lock cells
            selected_uop.lock_plane_cells(self.plane, self.qregs, self.lftn, self.ignore_path_conflict)

            ## Get & Lock AOD
            lasers = selected_uop.get_free_aod(self.aodh_free_list, self.aodd_free_list, self.aodr_free_list)
            for laser in lasers:
                self.aod_inuse_track[laser] = selected_uop.finish_us
            ## Lock Qregs
            for qn in selected_uop.qreg_names:
                self.qn_inuse_track[qn] = selected_uop.finish_us

            ## update schedule
            for (interval_type, interval_us) in intervals:
                for laser in lasers:
                    if laser.name not in self.uop_schedule.keys():
                        self.uop_schedule[laser.name] = []
                    self.uop_schedule[laser.name].append((selected_uop, interval_type, interval_us))

            for idx in selected_indices:
                uop_dag.remove_node(idx)
            selected_uop.update(self.plane, self.qregs)
        ##
        finish_time = None
        for time in self.aod_inuse_track.values():
            if not finish_time:
                finish_time = time
            else:
                if finish_time < time:
                    finish_time = time
        ##
        self.layer_time = finish_time
        ##
        return

    def _bypass_full_move_intermediate(self, uop_dag, front_indices):
        """Collapse a serial move pair whose intermediate cell is already full.

        Placement may emit ``src -> intermediate -> dst`` for one qubit when
        the first move expels it from a rotation cell and the second places it
        for a CX in the same layer.  Other moves in that layer can fill the
        intermediate before this pair reaches the DAG front.  With no AOD in
        flight there is then nothing to wait for, but the two endpoints can be
        connected directly.  Only collapse when the second move depends solely
        on the blocked first move, so no unrelated dependency is bypassed.
        """

        for first_idx in front_indices:
            first = uop_dag[first_idx]
            if first.uop_type != UopType.MOVE:
                continue
            if len(first.qreg_names) != 1 or len(first.src_dst_list) != 1:
                continue

            [qn] = first.qreg_names
            [(src, intermediate)] = first.src_dst_list
            cell = self.plane.field[intermediate[0]][intermediate[1]]
            if cell.status_q != CellStatusQ.OVERLAP_Q or qn in cell.occupants:
                continue

            serial_successors = []
            for successor_idx in uop_dag.successor_indices(first_idx):
                successor = uop_dag[successor_idx]
                if successor.uop_type != UopType.MOVE:
                    continue
                if successor.qreg_names != [qn] or len(successor.src_dst_list) != 1:
                    continue
                [(successor_src, _)] = successor.src_dst_list
                if successor_src == intermediate:
                    serial_successors.append(successor_idx)

            if len(serial_successors) != 1:
                continue
            [second_idx] = serial_successors
            if uop_dag.in_degree(second_idx) != 1:
                continue

            second = uop_dag[second_idx]
            [(_, dst)] = second.src_dst_list
            merged = UopMove([qn], [(src, dst)])
            merged_idx = uop_dag.add_node(merged)

            successors = set(uop_dag.successor_indices(first_idx))
            successors.discard(second_idx)
            successors.update(uop_dag.successor_indices(second_idx))
            for successor_idx in sorted(successors):
                uop_dag.add_edge(merged_idx, successor_idx, None)

            uop_dag.remove_node(first_idx)
            uop_dag.remove_node(second_idx)
            if self.debug_print:
                print(
                    f"BYPASS FULL MOVE INTERMEDIATE: {qn} "
                    f"{src}->{intermediate}->{dst}"
                )
            return True

        return False

    def schedule_nonaod_after(self, inst_schedule):
        phase_time = self.layer_time
        finish_time = phase_time
        phase_uops = []

        ## PatchCX (SYNCHRONOUS)
        cx_scheduled = [inst for inst in inst_schedule if inst.inst_type == InstType.TRANS_CX]
        if cx_scheduled:
            qubits = []
            for cx in cx_scheduled:
                qubits += cx.qreg_names
            uop = UopPatchCX(qubits)
            uop.check_ready(self.plane, self.qregs)
            phase_uops.append(uop)
            ##
            if self.cx_type == PhyOpType.SELECT:
                laser = LaserType.RYD_L
            elif self.cx_type == PhyOpType.ZONE:
                raise Exception()
            else:
                raise Exception()
            if laser.name not in self.uop_schedule.keys():
                self.uop_schedule[laser.name] = []
            ##
            intervals = uop.get_intervals(phase_time, self.lftn)
            for (interval_type, interval_us) in intervals:
                self.uop_schedule[laser.name].append((uop, interval_type, interval_us))
                finish_time = max(finish_time, interval_us[-1])
        #
        self.layer_time = finish_time
        #
        return

    def schedule_esm_all(self):
        phase_time = self.layer_time
        #
        self.reset_aod()
        #
        uop = UopESM(self.cx_type, self.meas_type)

        for _ in range(self.rounds):
            # RESET AQ (and bring it to DQ)
            if self.meas_type == PhyOpType.SELECT:
                laser_rst = LaserType.IMG_L
                laser_aod = None
                if laser_rst.name not in self.uop_schedule.keys():
                    self.uop_schedule[laser_rst.name] = []
            else:
                raise Exception()
            intervals_rst, intervals_aod = uop.get_intervals_rst(phase_time, self.lftn)
            ###
            for (interval_type, interval_us) in intervals_rst:
                self.uop_schedule[laser_rst.name].append((uop, interval_type, interval_us))
            ###
            if laser_aod:
                assert intervals_aod
                for (interval_type, interval_us) in intervals_aod:
                    self.uop_schedule[laser_aod.name].append((uop, interval_type, interval_us))
            phase_time = uop.finish_us

            # BODY H-CZ-H-CZ-H-CZ-H-CZ-H
            laser_h = LaserType.RMN_L
            if laser_h.name not in self.uop_schedule.keys():
                self.uop_schedule[laser_h.name] = []
            if self.cx_type == PhyOpType.SELECT and self.meas_type == PhyOpType.SELECT:
                laser_cx = LaserType.RYD_L
                laser_aod = None
                if laser_cx.name not in self.uop_schedule.keys():
                    self.uop_schedule[laser_cx.name] = []
            else:
                raise Exception()
            intervals_h, intervals_cx, intervals_aod = uop.get_intervals_body(phase_time, self.plane, self.lftn)
            ###
            for (interval_type, interval_us) in intervals_h:
                self.uop_schedule[laser_h.name].append((uop, interval_type, interval_us))
            ###
            for (interval_type, interval_us) in intervals_cx:
                self.uop_schedule[laser_cx.name].append((uop, interval_type, interval_us))
            ###
            if laser_aod:
                assert intervals_aod
                for (interval_type, interval_us) in intervals_aod:
                    self.uop_schedule[laser_aod.name].append((uop, interval_type, interval_us))
            phase_time = uop.finish_us

            # MEAS AQ (do not consider AQ removing latency)
            if self.meas_type == PhyOpType.SELECT:
                laser_meas = LaserType.IMG_L
                laser_aod = None
                if laser_meas.name not in self.uop_schedule.keys():
                    self.uop_schedule[laser_meas.name] = []
            else:
                raise Exception()

            ###
            intervals_meas, intervals_aod = uop.get_intervals_meas(phase_time, self.lftn)
            for (interval_type, interval_us) in intervals_meas:
                self.uop_schedule[laser_meas.name].append((uop, interval_type, interval_us))
            ###
            if laser_aod:
                assert intervals_aod
                for (interval_type, interval_us) in intervals_aod:
                    self.uop_schedule[laser_aod.name].append((uop, interval_type, interval_us))
            phase_time = uop.finish_us
        #
        self.layer_time = phase_time
        return

    ######
    def generate_rot_uops(self, inst_schedule):
        if self.skip_h:
            rot_scheduled = []
        else:
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]
        if not rot_scheduled:
            return []
        #
        if self.rot_type == RotType.REFL:
            # No grouping
            uop_rot_list = []
            for rot in rot_scheduled:
                [qn] = rot.qreg_names
                uop_rh = UopPatchRH([qn], self.refl_type_h)
                uop_rh.succ_movs = rot.succ_movs
                uop_rh.pred_movs = rot.pred_movs
                uop_rot_list.append(uop_rh)
                #
                uop_rd = UopPatchRD([qn], self.refl_type_d)
                uop_rd.succ_movs = rot.succ_movs
                uop_rd.pred_movs = rot.pred_movs
                uop_rot_list.append(uop_rd)
            return uop_rot_list
        ##
        elif self.rot_type in [RotType.DIR_IDEAL, RotType.DIR_CHANGE]:
            # No grouping
            uop_rot_list = []
            for rot in rot_scheduled:
                [qn] = rot.qreg_names
                if self.rot_type == RotType.DIR_IDEAL:
                    is_ideal = True
                elif self.rot_type == RotType.DIR_CHANGE:
                    is_ideal = False
                uop_dir = UopPatchDIR([qn], is_ideal)
                uop_dir.succ_movs = rot.succ_movs
                uop_dir.pred_movs = rot.pred_movs
                uop_rot_list.append(uop_dir)
            return uop_rot_list
        ##
        elif self.rot_type == RotType.DIR_TOGL:
            # Grouping for synchronous rot
            qreg_names = []
            succ_movs = []
            pred_movs = []
            rot_pos_list = []
            #
            for rot in rot_scheduled:
                [qn] = rot.qreg_names
                qreg_names.append(qn)
                succ_movs += rot.succ_movs
                pred_movs += rot.pred_movs
                if self.rot_plane_opt == RotPlaneOpt.DEDICATED_ROT:
                    rot_pos_list.append(rot.rot_pos)
            #
            uop_rot = UopGroupDIR(qreg_names, self.rotpch_pos_list, self.is_aod_infinite)
            uop_rot.succ_movs = succ_movs
            uop_rot.pred_movs = pred_movs
            uop_rot.rot_pos_list = rot_pos_list
            #
            return [uop_rot]
        else:
            raise Exception()


    def build_aod_uops_dag(self, uop_mov_list, uop_rot_list):
        def check_serial_moves(uop_1, uop_2):
            assert len(uop_1.qreg_names) == 1
            assert len(uop_2.qreg_names) == 1
            [qn_1] = uop_1.qreg_names
            [(_, dst_1)] = uop_1.src_dst_list
            [qn_2] = uop_2.qreg_names
            [(src_2, _)] = uop_2.src_dst_list
            ##
            return (qn_1 == qn_2) and (dst_1 == src_2)

        def check_existing_dep(dag, idx_1, idx_2):
            return rx.has_path(dag, idx_1, idx_2)

        ################
        uop_dag = rx.PyDiGraph()
        node_map = dict()
        for uop in uop_rot_list:
            node_map[uop] = uop_dag.add_node(uop)
        for uop in uop_mov_list:
            node_map[uop] = uop_dag.add_node(uop)

        # All moves handled by rev4 are single-qubit, single-segment moves.
        # Index them once for rotation lookup, serial-move lookup, and the
        # per-cell dependency cases below.
        moves_by_route = dict()
        moves_by_qn_src = dict()
        moves_by_src = dict()
        moves_by_dst = dict()
        move_indices = dict()
        for move_i, mov in enumerate(uop_mov_list):
            [qn] = mov.qreg_names
            [(src, dst)] = mov.src_dst_list
            moves_by_route.setdefault((qn, src, dst), []).append(mov)
            moves_by_qn_src.setdefault((qn, src), []).append(mov)
            moves_by_src.setdefault(src, []).append(mov)
            moves_by_dst.setdefault(dst, []).append(mov)
            move_indices[mov] = move_i

        ## Rotation
        ### move dependency
        rot_pred_movs = []
        rot_succ_movs = []
        for uop in uop_rot_list:
            for qn, src, dst in uop.pred_movs:
                [pred_mov] = moves_by_route[(qn, src, dst)]
                assert pred_mov
                uop_dag.add_edge(node_map[pred_mov], node_map[uop], None)
                rot_pred_movs.append(pred_mov)
            ###
            for qn, src, dst in uop.succ_movs:
                [succ_mov] = moves_by_route[(qn, src, dst)]
                assert succ_mov
                uop_dag.add_edge(node_map[uop], node_map[succ_mov], None)
                rot_succ_movs.append(succ_mov)
        if self.debug_print:
            for mov in rot_pred_movs:
                print('rot_pred_movs: Move', mov.qreg_names)
            for mov in rot_succ_movs:
                print('rot_succ_movs: Move', mov.qreg_names)
        ## Move
        ## special case handling - consecutive moves
        serial_pair_indices = set()
        for move_i, uop_i in enumerate(uop_mov_list):
            [qn_i] = uop_i.qreg_names
            [(_, dst_i)] = uop_i.src_dst_list
            for uop_j in moves_by_qn_src.get((qn_i, dst_i), []):
                move_j = move_indices[uop_j]
                if move_i != move_j:
                    serial_pair_indices.add(tuple(sorted((move_i, move_j))))

        # Sorting retains the order of the corresponding filtered
        # combinations(uop_mov_list, 2) traversal.
        for move_i, move_j in sorted(serial_pair_indices):
            uop_i = uop_mov_list[move_i]
            uop_j = uop_mov_list[move_j]
            if check_serial_moves(uop_i, uop_j):
                if check_existing_dep(uop_dag, node_map[uop_j], node_map[uop_i]):
                    continue
                uop_dag.add_edge(node_map[uop_i], node_map[uop_j], None)
            if check_serial_moves(uop_j, uop_i):
                if check_existing_dep(uop_dag, node_map[uop_i], node_map[uop_j]):
                    continue
                uop_dag.add_edge(node_map[uop_j],
                node_map[uop_i], None)
        ####

        ## Move
        edge_cand_dict = dict()
        for (r, c) in product(range(self.plane.h), range(self.plane.w)):
            cell = self.plane.field[r][c]
            pos = (r, c)
            assert cell.pos == pos
            if cell.type != CellType.NORMAL:
                continue
            ##
            children = list(moves_by_dst.get(pos, ()))
            parents = list(moves_by_src.get(pos, ()))
            ##
            if len(children) == 0 or len(parents) == 0:
                continue

            # Move - Case #1. for occ cell, one-out two-in
            if cell.status_q == CellStatusQ.OCCUPIED_Q:
                if len(children) == 2 and len(parents) == 1:
                    [parent] = parents
                    [child_1, child_2] = children
                    # rot case
                    if pos in self.rotpch_pos_list and parent in rot_pred_movs:
                        assert check_existing_dep(uop_dag, node_map[parent], node_map[child_1]) or check_existing_dep(uop_dag, node_map[parent], node_map[child_2])
                        # then no problem -> pass
                    # genral case
                    else:
                        if check_serial_moves(child_1, parent):
                            uop_dag.add_edge(node_map[parent], node_map[child_2], None)
                            if self.debug_print:
                                print(f"MOVE CASE #1-OCC_1P2C at {pos}: Move {parent.qreg_names}-> Move {child_2.qreg_names} fixed")
                        elif check_serial_moves(child_2, parent):
                            uop_dag.add_edge(node_map[parent], node_map[child_1], None)
                            if self.debug_print:
                                print(f"MOVE CASE #1-OCC_1P2C at {pos}: Move {parent.qreg_names}-> Move {child_1.qreg_names} fixed")
                        else: # general - either one
                            # parent -> child_1 or
                            # parent -> child_2
                            edge_cand_dict[pos] = []
                            edge_cand_dict[pos].append([(node_map[parent], node_map[child_1])])
                            edge_cand_dict[pos].append([(node_map[parent], node_map[child_2])])
                            if self.debug_print:
                                print(f"MOVE CASE #1-OCC_1P2C at {pos}: Move {parent.qreg_names}-> Move {child_1.qreg_names} or {child_2.qreg_names}")
                else:
                    continue
            # Case #2, 3, 4 for overalpped cell
            elif cell.status_q == CellStatusQ.OVERLAP_Q:
                ## Case #2
                if len(children) == 1 and len(parents) == 1:
                    # should not be the consecutive moves
                    # should not be the rotation
                    [parent] = parents
                    [child] = children
                    ###
                    ### special case: moving to rot patch and come back...
                    if check_existing_dep(uop_dag, node_map[parent], node_map[child]):
                        if self.debug_print:
                            print(f"MOVE CASE #2-OVERLAP_1P1C at pos {pos}: Move {parent.qreg_names} out for rot and back after rot")
                        continue
                    #
                    if check_serial_moves(child, parent):
                        if self.debug_print:
                            print(f"pos: {pos}, Move {child.qreg_names} {child.src_dst_list} -> {parent.qreg_names} {parent.src_dst_list}..?")
                    assert not check_serial_moves(child, parent)
                    if (pos in self.rotpch_pos_list and child in rot_pred_movs):
                        if self.debug_print:
                            print(f"pos: {pos}, Move {parent.qreg_names} {parent.src_dst_list} -> {child.qreg_names} {child.src_dst_list}..?")
                    assert not (pos in self.rotpch_pos_list and child in rot_pred_movs)
                    ###
                    uop_dag.add_edge(node_map[parent], node_map[child], None)
                    ###
                    if self.debug_print:
                        print(f"MOVE CASE #2-OVERLAP_1P1C at {pos}: Move {parent.qreg_names}-> Move {child.qreg_names}")
                ## Case #3.
                elif len(children) == 1 and len(parents) == 2:
                    [parent_1, parent_2] = parents
                    [child] = children
                    ###
                    if check_serial_moves(child, parent_1):
                        uop_dag.add_edge(node_map[parent_2], node_map[child], None)
                        if self.debug_print:
                            print(f"MOVE CASE #3-OVERLAP_2P1C at {pos} (serial): Move {parent_2.qreg_names}-> Move {child.qreg_names}")
                    elif check_serial_moves(child, parent_2):
                        uop_dag.add_edge(node_map[parent_1], node_map[child], None)
                        if self.debug_print:
                            print(f"MOVE CASE #3-OVERLAP_2P1C at {pos} (serial): Move {parent_1.qreg_names}-> Move {child.qreg_names}")
                    else:
                        edge_cand_dict[pos] = []
                        edge_cand_dict[pos].append([(node_map[parent_1], node_map[child])])
                        edge_cand_dict[pos].append([(node_map[parent_2], node_map[child])])
                        if self.debug_print:
                            print(f"MOVE CASE #3-OVERLAP_2P1C at {pos} (gen): Move {parent_1.qreg_names} or {parent_2.qreg_names} -> Move {child.qreg_names}")
                ## Case #4
                elif len(children) == 2 and len(parents) == 2:
                    [child_1, child_2] = children
                    [parent_1, parent_2] = parents
                    # rot case
                    if (pos in self.rotpch_pos_list) and\
                    (set(parents) <= set(rot_pred_movs))and \
                    ((child_1 in rot_succ_movs and child_2 in rot_pred_movs) or (child_2 in rot_succ_movs and child_1 in rot_pred_movs)):
                        if check_existing_dep(uop_dag, node_map[parent_1], node_map[child_1]) and check_existing_dep(uop_dag, node_map[parent_2], node_map[child_1]):
                            edge_cand_dict[pos] = []
                            edge_cand_dict[pos].append([(node_map[parent_1], node_map[child_2])])
                            edge_cand_dict[pos].append([(node_map[parent_2], node_map[child_2])])
                            if self.debug_print:
                                print(f"MOVE CASE #4-OVERLAP_2P2C at {pos} (rot): Move {parent_1.qreg_names} or {parent_2.qreg_names} -> Move {child_2.qreg_names}")
                        elif check_existing_dep(uop_dag, node_map[parent_1], node_map[child_2]) and check_existing_dep(uop_dag, node_map[parent_2], node_map[child_2]):
                            edge_cand_dict[pos] = []
                            edge_cand_dict[pos].append([(node_map[parent_1], node_map[child_1])])
                            edge_cand_dict[pos].append([(node_map[parent_2], node_map[child_1])])
                            if self.debug_print:
                                print(f"MOVE CASE #4-OVERLAP_2P2C at {pos} (rot): Move {parent_1.qreg_names} or {parent_2.qreg_names} -> Move {child_1.qreg_names}")
                        else:
                            # invalid
                            raise Exception()
                    # general case
                    else:
                        edge_cand_dict[pos] = []
                        edge_cand_dict[pos].append([(node_map[parent_1], node_map[child_1]), (node_map[parent_2], node_map[child_2])])
                        edge_cand_dict[pos].append([(node_map[parent_1], node_map[child_2]), (node_map[parent_2], node_map[child_1])])
                        if self.debug_print:
                            print(f"MOVE CASE #4-OVERLAP_2P2C at {pos} (gen): (Move {parent_1.qreg_names} -> Move {child_1.qreg_names} and Move {parent_2.qreg_names} -> Move {child_2.qreg_names}) or (Move {parent_1.qreg_names} -> Move {child_2.qreg_names} and Move {parent_2.qreg_names} -> Move {child_1.qreg_names})")
                ## Case 5.
                elif len(children) == 1 and len(parents) == 3:
                    [child] = children
                    assert sum(check_serial_moves(child, parent) for parent in parents) == 1
                    for parent in parents:
                        if check_serial_moves(child, parent):
                            parents.remove(parent)
                    [parent_1, parent_2] = parents
                    #
                    # rotation cell that rot target moves to other cell after rot
                    assert pos in self.rotpch_pos_list and \
                        child in rot_pred_movs and \
                        set(parents) <= set(rot_pred_movs)
                    ##
                    edge_cand_dict[pos] = []
                    edge_cand_dict[pos].append([(node_map[parent_1], node_map[child])])
                    edge_cand_dict[pos].append([(node_map[parent_2], node_map[child])])
                    if self.debug_print:
                        print(f"MOVE CASE #5-OVERLAP_3P1C at {pos} (rot): Move {parent_1.qreg_names} or {parent_2.qreg_names} -> Move {child.qreg_names}")
                else:
                    # no other cases
                    if self.debug_print:
                        print(f"pos: {pos}, # child: {len(children)}, # parent: {len(parents)}")
                        for child in children:
                            print(f"child: Move {child.qreg_names} {child.src_dst_list}")
                        for parent in parents:
                            print(f"parent: Move {parent.qreg_names} {parent.src_dst_list}")
                    raise Exception()
            else:
                continue

        ################
        # add edges from candidates
        node_indices = list(uop_dag.node_indices())
        base_indegree = {idx: uop_dag.in_degree(idx) for idx in node_indices}
        base_successors = {
            idx: list(uop_dag.successor_indices(idx)) for idx in node_indices
        }

        def depth_with_candidate_edges(edge_comb):
            """Return DAG depth, or None if the candidate edges form a cycle."""

            indegree = base_indegree.copy()
            candidate_successors = dict()
            for edges in edge_comb:
                for parent, child in edges:
                    candidate_successors.setdefault(parent, []).append(child)
                    indegree[child] += 1

            ready = deque(idx for idx in node_indices if indegree[idx] == 0)
            depth = {idx: 0 for idx in node_indices}
            visited = 0
            while ready:
                parent = ready.popleft()
                visited += 1
                for child in chain(
                    base_successors[parent],
                    candidate_successors.get(parent, ()),
                ):
                    depth[child] = max(depth[child], depth[parent] + 1)
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        ready.append(child)

            if visited != len(node_indices):
                return None
            return max(depth.values(), default=0)

        selected_edge_comb = None
        min_depth = None
        for edge_comb in product(*edge_cand_dict.values()):
            depth = depth_with_candidate_edges(edge_comb)
            if depth is None:
                if self.debug_print:
                    print(f"Cycle with {edge_comb}")
                if self.debug_print:
                    temp_dag = deepcopy(uop_dag)
                    for edges in edge_comb:
                        for (p, c) in edges:
                            temp_dag.add_edge(p, c, None)
                    cycles = list(rx.simple_cycles(temp_dag))
                    for cycle in cycles:
                        for idx in cycle:
                            print(f"{temp_dag[idx].uop_type.name} {temp_dag[idx].qreg_names}->", end="")
                        print()
                continue
            if selected_edge_comb is None or min_depth > depth:
                selected_edge_comb = edge_comb
                min_depth = depth
                if self.debug_print:
                    print(f"edge_comb: {edge_comb}, depth: {depth}")
        #
        assert selected_edge_comb is not None
        ret_dag = deepcopy(uop_dag)
        for edges in selected_edge_comb:
            for parent, child in edges:
                ret_dag.add_edge(parent, child, None)
        #
        return ret_dag


    def moving_distance_rev(self, pos1, pos2):
        # shortest dsitance with the diagonal movements
        (r1, c1), (r2, c2) = pos1, pos2
        is_onestep = self.check_move_onestep(pos1, pos2)
        #
        if is_onestep:
            dist_g = math.sqrt((r1-r2)**2 + (c1-c2)**2)
            #
            if self.move_type == MoveType.STA:
                dist_m = dist_g**(1/3)
            elif self.move_type == MoveType.CJ:
                dist_m = dist_g**(1/2)
            elif self.move_type == MoveType.CV:
                dist_m = dist_g
            else:
                raise Exception()
        else: # two step
            if self.mov_twostep_opt == MoveTwostepOpt.DQ_XY:
                r_diff, c_diff = abs(r1-r2), abs(c1-c2)
                shorter = min(r_diff, c_diff)
                longer = max(r_diff, c_diff)
                #
                dist_g1 = math.sqrt(2) * shorter
                dist_g2 = (longer-shorter)
            elif self.mov_twostep_opt == MoveTwostepOpt.XY_XY:
                r_diff, c_diff = abs(r1-r2), abs(c1-c2)
                dist_g1 = r_diff
                dist_g2 = c_diff
            else:
                raise Exception()
            #
            if self.move_type == MoveType.STA:
                dist_m = dist_g1**(1/3) + dist_g2**(1/3)
            elif self.move_type == MoveType.CJ:
                dist_m = dist_g1**(1/2) + dist_g2**(1/2)
            elif self.move_type == MoveType.CV:
                dist_m = dist_g1 + dist_g2
            else:
                raise Exception()

        return dist_m

    ######
    def next_cx_cost_reduction(self, qn, prev, next, curr_layer, start_layer, qregs):
        #
        next_layer, next_pair_qn = self.find_next_cx(start_layer, qn)
        if not next_pair_qn or not next_pair_qn in qregs.keys():
            return 0
        #
        pair_pos = qregs[next_pair_qn].pos
        layer_gap = (next_layer - curr_layer)

        # distance
        ## before
        dist_before = self.moving_distance_rev(prev, pair_pos)
        ## after
        dist_after = self.moving_distance_rev(next, pair_pos)
        ##
        dist_reduction = (dist_before - dist_after)

        # weight
        ## linear with max 5 layers
        max_level = 5
        scaling = 0.2
        list_weight = [1-scaling*l for l in range(max_level)]
        if layer_gap < max_level:
            weight = list_weight[layer_gap]
        else:
            weight = 0

        # cost reduction
        cost_reduction = dist_reduction * weight

        return cost_reduction


    ######
    def rotation_placement_ded(self, plane, qregs, layer_idx, inst_schedule):
        #
        cx_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_CX]]
        cx_qn_list = []
        for cx in cx_scheduled:
            cx_qn_list += cx.qreg_names
        #
        if self.skip_h:
            rot_scheduled = []
        else:
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]
        rot_qn_list = []
        for rot in rot_scheduled:
            rot_qn_list += rot.qreg_names
        ##
        free_normal_pos = plane.get_free_pos(CellType.NORMAL)
        occ_normal_pos = plane.get_occ_pos(CellType.NORMAL)
        dst_pos_list = [p for p in (free_normal_pos+occ_normal_pos) if not p in self.rotpch_pos_list]
        #
        # 1. List up all the next mapping with rot mov
        rot_mappings = [dict(zip(rot_scheduled, pos)) for pos in permutations(self.rotpch_pos_list, len(rot_scheduled))]

        # 2. Generate all the next mappings with expel
        per_mapping_mov_lists = []
        per_mapping_qreg_planes = []
        for idx, mapping in enumerate(rot_mappings):
            temp_plane = deepcopy(plane)
            temp_qregs = deepcopy(qregs)
            ##
            if self.debug_print:
                print(f"Maping: {idx}")
            mov_list = []
            # consider rotating qubits new positions
            for rot, dst in mapping.items():
                [qn] = rot.qreg_names
                src = qregs[qn].pos
                if self.debug_print:
                    print(f"ROT {qn} {src}->{dst}")
                #
                mov_list.append((qn, src, dst))
                #
                uop_mov = UopMove([qn], [(src, dst)])
                uop_mov.update(temp_plane, temp_qregs, temp=True)
            # derive best expel movs for each mapping
            ## construct expel_dict
            expel_dict = dict()
            for pos in mapping.values():
                (r, c) = pos
                cell = plane.field[r][c]
                for occ in cell.occupants:
                    if occ in rot_qn_list:
                        continue
                    expel_dict[occ] = []
                    for dst in dst_pos_list:
                        moving_cost = self.moving_distance_rev(pos, dst)
                        cost_reduction = self.next_cx_cost_reduction(occ, pos, dst, layer_idx, layer_idx, temp_qregs)
                        cost = moving_cost - cost_reduction
                        ###
                        expel_dict[occ].append((dst, cost))
            ## prune expel_dict
            num_expel = len(expel_dict.keys())
            for k, v in expel_dict.items():
                expel_dict[k] = sorted(expel_dict[k], key=lambda x: x[1])[:num_expel]
            ##
            left = []
            right = set()
            qn_dst_cost = dict()
            for qn, dst_cost_list in expel_dict.items():
                left.append(qn)
                qn_dst_cost[qn] = dict()
                for (dst, _) in dst_cost_list:
                    right.add(dst)
                    qn_dst_cost[qn][dst] = cost
            right = list(right)
            ##
            cost_matrix = np.full((len(left), len(right)), sys.maxsize, dtype=float)
            for i, qn in enumerate(left):
                for j, pos in enumerate(right):
                    try:
                        cost_matrix[i][j] = qn_dst_cost[qn][pos]
                    except:
                        pass
            ##
            left_ids, right_ids = linear_sum_assignment(cost_matrix)
            for left_id, right_id in zip(left_ids, right_ids):
                qn = left[left_id]
                src = qregs[qn].pos
                dst = right[right_id]
                assert src != dst
                mov = (qn, src, dst)
                mov_list.append(mov)
                if self.debug_print:
                    print(f"EXPEL {qn}: {src} -> {dst}")
                #
                uop_mov = UopMove([qn], [(src, dst)])
                uop_mov.update(temp_plane, temp_qregs, temp=True)
            ##
            per_mapping_mov_lists.append(mov_list)
            per_mapping_qreg_planes.append((temp_qregs, temp_plane))
        ##

        # 3. Calculate cost of all the next mapping
        per_mapping_costs = []
        for (mapping, mov_list, (temp_qregs, temp_plane)) in zip(rot_mappings, per_mapping_mov_lists, per_mapping_qreg_planes):
            # moving cost for the mapping
            moving_cost = 0
            for mov in mov_list:
                qn, src, dst = mov
                moving_cost += self.moving_distance_rev(src, dst)

            # next_cost reduction
            cost_reduction = 0
            for mov in mov_list:
                qn, src, dst = mov
                ###
                next_layer, next_pair_qn = self.find_next_cx(layer_idx, qn)
                if not next_pair_qn or not next_pair_qn in qregs.keys():
                    cost_reduction = 0
                else:
                    layer_gap = (next_layer - layer_idx)
                    pair_pos_old = qregs[next_pair_qn].pos
                    pair_pos_new = temp_qregs[next_pair_qn].pos
                    #
                    dist_before = self.moving_distance_rev(src, pair_pos_old)
                    #
                    dist_after = self.moving_distance_rev(dst, pair_pos_new)
                    #
                    dist_reduction = (dist_before - dist_after)
                    # weight
                    ## linear with max 5 layers
                    max_level = 5
                    scaling = 0.2
                    list_weight = [1-scaling*l for l in range(max_level)]
                    if layer_gap < max_level:
                        weight = list_weight[layer_gap]
                    else:
                        weight = 0
                    # cost reduction
                    cost_reduction += dist_reduction * weight
            #
            mapping_cost = moving_cost - cost_reduction
            per_mapping_costs.append(mapping_cost)

        # 4. Pick the next mapping and moves with minimum cost
        ret_idx = per_mapping_costs.index(min(per_mapping_costs))
        ret_mapping = rot_mappings[ret_idx]
        ret_mov_list = per_mapping_mov_lists[ret_idx]

        # 5. Track dependency
        for rot, pos in ret_mapping.items():
            rot.rot_pos = pos
            # rot target's move toward pos
            [rot_mov] = [(q, src, dst) for (q, src, dst) in ret_mov_list if dst == pos]
            _, rm_src, rm_dst = rot_mov
            if rm_src == rm_dst:
                ret_mov_list.remove(rot_mov)
            else:
                rot.pred_movs.append(rot_mov)
            # expel others from the pos
            expel_movs = [(q, src, dst) for (q, src, dst) in ret_mov_list if src == pos]
            rot.pred_movs += expel_movs
        # CHECK
        if self.debug_print:
            print(f"Chosen - Mapping: {ret_idx}")
        ##
        rot_pos_list = ret_mapping.values()
        ##
        return ret_mov_list, rot_pos_list

    ######
    def rotation_placement_all(self, plane, qregs, layer_idx, inst_schedule):
        mov_list = []
        #
        cx_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_CX]]
        cx_qn_list = []
        for cx in cx_scheduled:
            cx_qn_list += cx.qreg_names
        #
        if self.skip_h:
            rot_scheduled = []
        else:
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]
        rot_qn_list = []
        for rot in rot_scheduled:
            rot_qn_list += rot.qreg_names
        ##

        # determine the rotations need movement
        rot_targets = []
        pos_used = []
        for rot in rot_scheduled:
            [target_qn] = rot.qreg_names
            r, c = pos = qregs[target_qn].pos
            cell = plane.field[r][c]
            ##
            if cell.status_q == CellStatusQ.OCCUPIED_Q:
                continue
            ##
            assert cell.status_q == CellStatusQ.OVERLAP_Q
            ##
            ## Handle special case: mate is on the reusable CX
            [mate_qn] = [occ for occ in cell.occupants if occ != target_qn]
            ###
            if (not mate_qn in rot_qn_list) and (mate_qn in cx_qn_list):
                # check reusable CXs
                mate_cx = next(cx for cx in cx_scheduled if mate_qn in cx.qreg_names)
                [pair_qn] = [q for q in mate_cx.qreg_names if q != mate_qn]
                rp, cp = pair_pos = qregs[pair_qn].pos
                pair_cell = plane.field[rp][cp]
                #
                pair_reuse = (pair_cell.status_q == CellStatusQ.OCCUPIED_Q) and not ((mate_cx.cx_type == CXType.MY and pair_cell.type == CellType.PORT_M) or (mate_cx.cx_type == CXType.YQ and pair_cell.type == CellType.PORT_Y))
                #
                if pair_reuse and (not pair_qn in rot_qn_list):
                    mov = (mate_qn, pos, pair_pos)
                    mov_list.append(mov)
                    pos_used.append(pair_pos)
                    continue
                #
            rot_targets.append(rot)
        ##
        rot_target_pos = set()
        rot_pos_visited = []
        for rot in rot_targets:
            [target_qn] = rot.qreg_names
            pos = qregs[target_qn].pos

            if pos in rot_target_pos:
                rot_pos_visited.append(rot)
            else:
                rot_target_pos.add(pos)
        #
        rot_targets = [rot for rot in rot_targets if not rot in rot_pos_visited]

        # Step 1. Set pos candidates and costs for each mov
        ## general rotation placement
        free_normal_pos = plane.get_free_pos(CellType.NORMAL)
        occ_normal_pos = plane.get_occ_pos(CellType.NORMAL)
        ##

        for rot in rot_targets:
            # key: pos
            # value: cost
            rot_cost_dict = dict()
            # key: pos
            # value: qn
            rot_qn_dict = dict()
            #
            [target_qn] = rot.qreg_names
            r, c = pos = qregs[target_qn].pos
            cell = plane.field[r][c]
            #
            [mate_qn] = [occ for occ in cell.occupants if occ != target_qn]
            #
            # Free cells
            ## both can move
            for dst in free_normal_pos:
                #
                moving_cost = self.moving_distance_rev(pos, dst)
                #
                cost_reduction_t = self.next_cx_cost_reduction(target_qn, pos, dst, layer_idx, layer_idx, qregs)
                cost_reduction_m = self.next_cx_cost_reduction(mate_qn, pos, dst, layer_idx, layer_idx, qregs)
                #
                if cost_reduction_t > cost_reduction_m:
                    cost = moving_cost - cost_reduction_t
                    rot_cost_dict[dst] = cost
                    rot_qn_dict[dst] = target_qn
                else:
                    cost = moving_cost - cost_reduction_m
                    rot_cost_dict[dst] = cost
                    rot_qn_dict[dst] = mate_qn

            # OCC cells without rotation
            ## not-rotating mate can move
            ## for now, prohibt future CX to be overlapped
            if mate_qn in rot_qn_list or mate_qn in cx_qn_list:
                pass
            else:
                for dst in occ_normal_pos:
                    if dst in pos_used:
                        continue
                    rd, rc = dst
                    dst_cell = plane.field[rd][rc]
                    [dst_qn] = dst_cell.occupants
                    if dst_qn in rot_qn_list:
                        continue
                    ##
                    moving_cost = self.moving_distance_rev(pos, dst)
                    cost_reduction = self.next_cx_cost_reduction(mate_qn, pos, dst, layer_idx, layer_idx, qregs)
                    #
                    cost = moving_cost - cost_reduction
                    #
                    rot_cost_dict[dst] = cost
                    rot_qn_dict[dst] = mate_qn

            #
            # Maximum # candidates = len(rot_targets)
            keep_pos_list = sorted(rot_cost_dict, key=lambda k: rot_cost_dict[k])[:len(rot_targets)]
            rot.rot_cost_dict = {k: rot_cost_dict[k] for k in keep_pos_list}
            rot.rot_qn_dict = {k: rot_qn_dict[k] for k in keep_pos_list}
            ##
            for k, v in rot.rot_cost_dict.items():
                qn = rot.rot_qn_dict[k]
            ##

        # Step 2. Build cost matrix & Solve minimum-weight matching
        ## build
        left = rot_targets
        right = set()
        for rot in rot_targets:
            right |= set(rot.rot_cost_dict.keys())
        right = list(right)
        ## solve
        cost_matrix = np.full((len(left), len(right)), sys.maxsize, dtype=float)
        for i, rot in enumerate(left):
            for j, pos in enumerate(right):
                if pos in rot.rot_cost_dict.keys():
                    cost = rot.rot_cost_dict[pos]
                    cost_matrix[i][j] = cost
                else:
                    pass
        ## solve
        left_ids, right_ids = linear_sum_assignment(cost_matrix)

        # Step 3. Generate determined moves
        rot_pos_list = []
        for lid, rid in zip(left_ids, right_ids):
            rot = left[lid]
            [rot_qn] = rot.qreg_names
            src = qregs[rot_qn].pos
            ##
            dst = right[rid]
            mov_qn = rot.rot_qn_dict[dst]
            ##
            mov = (mov_qn, src, dst)
            mov_list.append(mov)
            ##
            if rot_qn == mov_qn:
                rot_pos_list.append(dst)
            else:
                rot_pos_list.append(src)

        # mark rotation dependency
        for rot in rot_scheduled:
            [qn] = rot.qreg_names
            pos = qregs[qn].pos
            pred_movs = [(q, src, dst) for (q, src, dst) in mov_list if q == qn or src == pos]
            rot.pred_movs += pred_movs

        return mov_list, rot_pos_list

    ######
    def cx_reuse_placement(self, plane, qregs, layer_idx, inst_schedule):
        # Purpose: Prioritize the scenario that we can move only one patch to the other
        ## i.e., one (or both) of the CX targets are OCCUPIED
        #
        cx_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_CX]]
        if self.skip_h:
            rot_scheduled = []
        else:
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]

        # Generate mov_list
        mov_list = []
        cx_targets = []
        for cx in cx_scheduled:
            [q0, q1] = cx.qreg_names
            r0, c0 = pos0 = qregs[q0].pos
            r1, c1 = pos1 = qregs[q1].pos
            if pos0 == pos1:
                continue
            cell0 = plane.field[r0][c0]
            cell1 = plane.field[r1][c1]
            ## check condition
            cond0 = (cell0.status_q == CellStatusQ.OCCUPIED_Q) and not ((cx.cx_type == CXType.MY and cell0.type == CellType.PORT_M) or (cx.cx_type == CXType.YQ and cell0.type == CellType.PORT_Y)or (cx.cx_type == CXType.MQ and cell0.type == CellType.PORT_M))
            cond1 = (cell1.status_q == CellStatusQ.OCCUPIED_Q) and not ((cx.cx_type == CXType.MY and cell1.type == CellType.PORT_M) or (cx.cx_type == CXType.YQ and cell1.type == CellType.PORT_Y)
            or (cx.cx_type == CXType.MQ and cell1.type == CellType.PORT_M))
            #
            if not cond0 and not cond1:
                continue
            elif cond0 and cond1:
                cost_reduction0 = self.next_cx_cost_reduction(q0, pos0, pos1, layer_idx, layer_idx+1, qregs)
                cost_reduction1 = self.next_cx_cost_reduction(q1, pos1, pos0, layer_idx, layer_idx+1, qregs)
                #
                if cost_reduction0 > cost_reduction1:
                    mov = (q0, pos0, pos1)
                else:
                    mov = (q1, pos1, pos0)
                mov_list.append(mov)
            elif cond0:
                mov = (q1, pos1, pos0)
                mov_list.append(mov)
            elif cond1:
                mov = (q0, pos0, pos1)
                mov_list.append(mov)
            else:
                raise Exception()

            # mark rotation dependency
            assert mov
            for rot in rot_scheduled:
                [rot_qn] = rot.qreg_names
                if rot_qn in [q0, q1]:
                    rot.succ_movs.append(mov)

        return mov_list

    ######
    def cx_general_placement(self, plane, qregs, layer_idx, inst_schedule, rot_pos_list):
        #
        mov_list = []

        # check
        cx_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_CX]]
        cx_qn_list = []
        for cx in cx_scheduled:
            cx_qn_list += cx.qreg_names
        #
        if self.skip_h:
            rot_scheduled = []
        else:
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]
        rot_qn_list = []
        for rot in rot_scheduled:
            rot_qn_list += rot.qreg_names

        # Need at least TWO moves
        ## except for special case:
        ### one mate goes out to another cell (will be handled here)
        ###
        gen_targets = []
        for cx in cx_scheduled:
            [q0, q1] = cx.qreg_names
            r0, c0 = pos0 = qregs[q0].pos
            r1, c1 = pos1 = qregs[q1].pos
            cell0 = plane.field[r0][c0]
            cell1 = plane.field[r1][c1]
            if pos0 != pos1:
                assert (cell0.status_q == CellStatusQ.OVERLAP_Q) or  (cx.cx_type == CXType.MY and cell0.type == CellType.PORT_M) or (cx.cx_type == CXType.YQ and cell0.type == CellType.PORT_Y) or (cx.cx_type == CXType.MQ and cell0.type == CellType.PORT_M)
                assert (cell1.status_q == CellStatusQ.OVERLAP_Q) or  (cx.cx_type == CXType.MY and cell1.type == CellType.PORT_M) or (cx.cx_type == CXType.YQ and cell1.type == CellType.PORT_Y) or (cx.cx_type == CXType.MQ and cell1.type == CellType.PORT_M)
                gen_targets.append(cx)

        # Step 1. Set pos candidates and costs for each cx
        free_normal_pos = plane.get_free_pos(CellType.NORMAL)
        occ_normal_pos = plane.get_occ_pos(CellType.NORMAL)
        #
        gen_qn_list = []
        for cx in gen_targets:
            gen_qn_list += cx.qreg_names
        #
        for cx in gen_targets:
            # key: (cx_pos, push_pos)
            # value: cost
            cx_cost_dict = dict()
            push_cost_dict = dict()
            #
            # Type 1. Pos1 or Pos2
            for qn in cx.qreg_names:
                r, c = cx_pos = qregs[qn].pos
                cell = plane.field[r][c]
                if not cell.status_q == CellStatusQ.OVERLAP_Q or not cell.type == CellType.NORMAL:
                    continue

                # cell is overlap
                [mate_qn] = [occ for occ in cell.occupants if not occ in cx.qreg_names]
                #
                [pair_qn] = [q for q in cx.qreg_names if q != qn]
                pair_pos = qregs[pair_qn].pos
                #
                pair_moving_cost = self.moving_distance_rev(pair_pos, cx_pos)
                pair_cost_reduction = self.next_cx_cost_reduction(pair_qn, pair_pos, cx_pos, layer_idx, layer_idx+1, qregs)
                ##
                cx_cost = (pair_moving_cost - pair_cost_reduction)
                if mate_qn in gen_qn_list: # mate is on CX
                    ## Move: Q1->pos2 or Q2->pos1
                    ## No additional cost - One move
                    ##
                    cost = cx_cost
                    cx_cost_dict[cx_pos] = cost

                else: # mate is not on CX
                    ## Move: Q1->pos2 or Q2->pos1
                    ## Move: Mate -> push_pos

                    # list push pos candidates
                    push_pos_list = []
                    ## free normal cells
                    for pos in free_normal_pos:
                        if not pos in rot_pos_list:
                            push_pos_list.append(pos)
                    ## occupied cells wihtout aod operation
                    for pos in occ_normal_pos:
                        r, c = pos
                        cell = plane.field[r][c]
                        [occ_q] = cell.occupants
                        #
                        assert not occ_q in cx_qn_list
                        if not occ_q in rot_qn_list:
                            push_pos_list.append(pos)
                    #
                    assert push_pos_list
                    push_cost_dict[cx_pos] = dict()
                    min_push_cost = sys.maxsize
                    #
                    for push_pos in push_pos_list:
                        mate_moving_cost = self.moving_distance_rev(cx_pos, push_pos)
                        mate_cost_reduction = self.next_cx_cost_reduction(mate_qn, cx_pos, push_pos, layer_idx, layer_idx+1, qregs)
                        ##
                        push_cost = (mate_moving_cost - mate_cost_reduction)
                        ##
                        push_cost_dict[cx_pos][push_pos] = push_cost
                        ##
                        min_push_cost = min(min_push_cost, push_cost)
                        #
                    cost = cx_cost + min_push_cost
                    cx_cost_dict[cx_pos] = cost
            ##
            # Type 2. Free cells
            ## Move: Q1 -> Free cell
            ## Move: Q2 -> Free cell
            for cx_pos in free_normal_pos:
                if cx_pos in rot_pos_list:
                    continue
                # no mate to push
                push_pos = None
                #
                # Calcluate cost
                cost = 0
                for qn in cx.qreg_names:
                    # moving cost
                    curr_pos = qregs[qn].pos
                    moving_cost = self.moving_distance_rev(curr_pos, cx_pos)
                    # next cost reduction
                    cost_reduction = self.next_cx_cost_reduction(qn, curr_pos, cx_pos, layer_idx, layer_idx+1, qregs)
                    cost += (moving_cost - cost_reduction)
                #
                #pos_cost_dict[(cx_pos, push_pos)] = cost
                cx_cost_dict[cx_pos] = cost
            ##
            cx.cx_cost_dict = cx_cost_dict
            cx.push_cost_dict = push_cost_dict

        # Prohibit the SWAP between two overlapped cells
        for cx_i, cx_j in combinations(gen_targets, 2):
            pos_pair_i = [qregs[qn].pos for qn in cx_i.qreg_names]
            pos_pair_j = [qregs[qn].pos for qn in cx_j.qreg_names]
            if set(pos_pair_i) == set(pos_pair_j):
                # allows only one cx to utilize pos_pair
                min_pos_i = None
                min_others_i = None
                for pos, cost in cx_i.cx_cost_dict.items():
                    if pos in pos_pair_i:
                        if min_pos_i is None:
                            min_pos_i = cost
                        else:
                            min_pos_i = min(min_pos_i, cost)
                    else:
                        if min_others_i is None:
                            min_others_i = cost
                        else:
                            min_others_i = min(min_others_i, cost)
                #
                min_pos_j = None
                min_others_j = None
                for pos, cost in cx_j.cx_cost_dict.items():
                    if pos in pos_pair_j:
                        if min_pos_j is None:
                            min_pos_j = cost
                        else:
                            min_pos_j = min(min_pos_j, cost)
                    else:
                        if min_others_j is None:
                            min_others_j = cost
                        else:
                            min_others_j = min(min_others_j, cost)

                if (min_pos_i + min_others_j) < (min_pos_j + min_others_i):
                    del cx_j.cx_cost_dict[pos_pair_j[0]]
                    del cx_j.cx_cost_dict[pos_pair_j[1]]
                else:
                    del cx_i.cx_cost_dict[pos_pair_i[0]]
                    del cx_i.cx_cost_dict[pos_pair_i[1]]
            else:
                pass
        #
        # pruning cx_cost_dict
        for cx in gen_targets:
            # Maximum # candidates = len(gen_targets)
            keep_pos_list = sorted(cx.cx_cost_dict, key=lambda k: cx.cx_cost_dict[k])[:len(gen_targets)]
            cx.cx_cost_dict = {k: cx.cx_cost_dict[k] for k in keep_pos_list}

        #
        # Step 2. Build cost matrix & Solve minimum-weight matching
        ## build
        left = gen_targets
        right = set()
        for cx in gen_targets:
            right |= set(cx.cx_cost_dict.keys())
        right = list(right)
        ##
        cost_matrix = np.full((len(left), len(right)), sys.maxsize, dtype=float)
        for i, cx in enumerate(left):
            for j, pos in enumerate(right):
                if pos in cx.cx_cost_dict.keys():
                    cost = cx.cx_cost_dict[pos]
                    cost_matrix[i][j] = cost
                else:
                    pass
        ## solve
        left_ids, right_ids = linear_sum_assignment(cost_matrix)

        # Step 3. Generate determined moves
        mov_list = []
        #
        cx_pos_list = [right[i] for i in right_ids]
        #
        selected_push_pos = set()
        for lid, rid in zip(left_ids, right_ids):
            cx = left[lid]
            cx_pos = right[rid]
            cx_pos_list.append(cx_pos)
            # cx_pos
            for qn in cx.qreg_names:
                pos = qregs[qn].pos
                if pos != cx_pos:
                    mov = (qn, pos, cx_pos)
                    mov_list.append(mov)
                else:
                    pass
            #
            push_pos = None
            if cx_pos in cx.push_cost_dict.keys():
                d = cx.push_cost_dict[cx_pos]
                #
                push_pos_list = sorted(d, key=d.get)[:len(gen_targets)]
                #
                push_pos = next(pos for pos in push_pos_list if (not pos in cx_pos_list and not pos in selected_push_pos))

            # push_pos
            if push_pos:
                selected_push_pos.add(push_pos)
                r, c = cx_pos
                cell = plane.field[r][c]
                [mate_qn] = [occ for occ in cell.occupants if not occ in cx.qreg_names]
                assert mate_qn
                ##
                mov = (mate_qn, cx_pos, push_pos)
                mov_list.append(mov)
            else:
                pass

        # Step 4. Mark preceding rotation
        ## currently dst cannot be a rotating cell
        ## Why? This step is for the CX targets at the overlapped cells

        return mov_list


    def check_move_onestep(self, src, dst):
        # shortest dsitance with the diagonal movements
        (r1, c1), (r2, c2) = src, dst
        #
        onestep_h = (r1 == r2)
        onestep_v = (c1 == c2)
        #
        r1_dq, c1_dq = (r1+c1, self.plane.h-1-r1+c1)
        r2_dq, c2_dq = (r2+c2, self.plane.h-1-r2+c2)
        onestep_d = (r1_dq == r2_dq)
        onestep_q = (c1_dq == c2_dq)

        return any([onestep_h, onestep_v, onestep_d, onestep_q])


    ######
    def generate_aod_uops(self, layer_idx, inst_schedule):
        temp_plane = deepcopy(self.plane)
        temp_qregs = deepcopy(self.qregs)

        # 1. Rotation placement
        # Generate moves for the rotation
        #
        if self.rot_plane_opt == RotPlaneOpt.ALL_ROT:
            mov_rot_list, rot_pos_list = self.rotation_placement_all(temp_plane, temp_qregs, layer_idx, inst_schedule)
        elif self.rot_plane_opt == RotPlaneOpt.DEDICATED_ROT:
            mov_rot_list, rot_pos_list = self.rotation_placement_ded(temp_plane, temp_qregs, layer_idx, inst_schedule)
        else:
            raise Exception()
        uop_mov_rot_list = [UopMove([qn], [(src, dst)]) for (qn, src, dst) in mov_rot_list]
        for uop in uop_mov_rot_list:
            uop.update(temp_plane, temp_qregs, temp=True)
        if self.debug_print:
            self.debug_plane_rot = deepcopy(temp_plane)
            self.debug_qregs_rot = deepcopy(temp_qregs)
        # rot check
        if self.skip_h:
            rot_scheduled = []
        else:
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]
        rot_qn_list = []
        for rot in rot_scheduled:
            rot_qn_list += rot.qreg_names
        rot_cells = [temp_plane.field[r][c] for (r, c) in [temp_qregs[qn].pos for qn in rot_qn_list]]
        assert all([cell.status_q == CellStatusQ.OCCUPIED_Q for cell in rot_cells])

        # 2. CX placement
        # Generate moves for the CX
        # Goal: CX targets should be OVERLAP at the same cell
        # REUSE
        mov_reuse_list = []
        while True:
            temp_mov_list = self.cx_reuse_placement(temp_plane, temp_qregs, layer_idx, inst_schedule)
            if not temp_mov_list:
                break
            else:
                uop_mov_temp_list = [UopMove([qn], [(src, dst)]) for (qn, src, dst) in temp_mov_list]
                for uop in uop_mov_temp_list:
                    uop.update(temp_plane, temp_qregs, temp=True)
                mov_reuse_list += temp_mov_list
        uop_mov_reuse_list = [UopMove([qn], [(src, dst)]) for (qn, src, dst) in mov_reuse_list]
        if self.debug_print:
            self.debug_plane_reus = deepcopy(temp_plane)
            self.debug_qregs_reus = deepcopy(temp_qregs)

        # GENERAL
        mov_gen_list = self.cx_general_placement(temp_plane, temp_qregs, layer_idx, inst_schedule, rot_pos_list)
        uop_mov_gen_list = [UopMove([qn], [(src, dst)]) for (qn, src, dst) in mov_gen_list]
        for uop in uop_mov_gen_list:
            #uop.update(temp_lane, temp_qregs)
            pass
        if self.debug_print:
            self.debug_plane_gen = deepcopy(temp_plane)
            self.debug_qregs_gen = deepcopy(temp_qregs)

        # 3. Generate individal Uops
        ## Move
        uop_mov_list = uop_mov_rot_list + uop_mov_reuse_list + uop_mov_gen_list
        ## Rotation
        uop_rot_list = self.generate_rot_uops(inst_schedule)

        # 4. Build Uop DAG wihout grouping
        # Grouping should be t the issue/select stages
        # to consider AOD avalability at runtime
        uop_dag = self.build_aod_uops_dag(uop_mov_list, uop_rot_list)

        return uop_dag


    ###############################

    def run(self):
        assert len(self.req_schedule_trace) == len(self.inst_schedule_trace)

        if self.debug_print:
            self.initial_plane = deepcopy(self.plane)
            self.initial_qregs = deepcopy(self.qregs)
        # preprocessing
        self.trace_qubit_names()
        self.trace_next_cx()

        for idx, (req_schedule, inst_schedule) in enumerate(zip(self.req_schedule_trace, self.inst_schedule_trace)):
            ##### Start LayerUop Scheduling #####
            if self.debug_print:
                print(f"UOP SCHEDLE START - Layer {idx}")
            # Initialization
            self.layer_time =0
            self.layer_latenc = 0
            self.uop_schedule = dict()
            #
            self.plane.reset_status_op()
            for cell in [self.plane.field[r][c] for (r, c) in product(range(self.plane.h), range(self.plane.w))]:
                cell.lock_dict = dict()
            ###
            # PHASE 0 - M/Y REQ PLACEMENT
            self.req_placement(idx, req_schedule)
            # PHASE 1 - Logical single-qubit uop
            self.schedule_nonaod_before(inst_schedule)

            # PHASE 2 - AOD shuttling uop
            self.schedule_main_aod(idx, inst_schedule)

            # PHASE 3 - Logical two-qubit uop
            self.schedule_nonaod_after(inst_schedule)

            # PHASE 4 - ESM rounds
            self.schedule_esm_all()

            ##### Finish Layer Uop Scheduling #####
            self.uop_schedule_trace.append(deepcopy(self.uop_schedule))
            self.plane_trace.append(deepcopy(self.plane))
            self.qregs_trace.append(deepcopy(self.qregs))
            ###
            if self.debug_print:
                print(f"UOP SCHEDULE FINISH - Layer {idx}")
                print()

        if self.debug_print:
            print("UOP SCHEDULING SUCCESSFULLY FINISHED!")
