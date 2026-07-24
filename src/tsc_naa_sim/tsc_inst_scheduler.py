from tsc_instructions import *
from qreg_plane import *
from macro import *
#
from collections import deque, Counter
import rustworkx as rx

class tsc_inst_scheduler:
    def __init__(self,
                 sc_dag_in,
                 plane_char_in,
                 #
                 inst_sched_opt,
                 rot_sched_opt,
                 s_trans_opt=STransOpt.GATE_TEL,
                 skip_h=False,
                 propagate_creg_names_cond=True,
                 debug_print=False,
                 ):
        # input
        self.dag = sc_dag_in
        self.inst_sched_opt = inst_sched_opt
        assert self.inst_sched_opt == InstSchedOpt.ASAP

        self.rot_sched_opt = rot_sched_opt

        #
        self.s_trans_opt = s_trans_opt
        self.skip_h = skip_h
        # creg_names_cond is compilation metadata used by legacy/debug flows.
        # Keep the historical behavior by default; AE can disable the
        # propagation because its execution and analysis paths do not read it.
        self.propagate_creg_names_cond = propagate_creg_names_cond
        self.debug_print = debug_print

        ##
        self.can_overlap = True
        self.cannot_overlap = False
        ##
        self.free_cells_m = 0
        self.free_cells_y = 0
        self.free_cells_n = 0
        for i in range(len(plane_char_in)):
            self.free_cells_m += plane_char_in[i].count('M')
            self.free_cells_y += plane_char_in[i].count('Y')
            self.free_cells_n += plane_char_in[i].count('N')

        self.rot_list = self.keep_rotations()
        #
        self.req_queue = self.order_req_insts()
        self.init_ready_tracker()
        #
        self.curr_layer = 0
        #
        self.inst_schedule_trace = []
        self.req_schedule_trace = []

    def keep_rotations(self):
        def find_nearest_anc(dag, start, target_qn, inst_types):
            visited = set()
            queue = deque([start])
            #
            while queue:
                node = queue.popleft()
                for parent in dag.predecessor_indices(node):
                    if parent in visited:
                        continue

                    visited.add(parent)
                    cond_1 = dag[parent].inst_type in inst_types
                    try:
                        cond_2 = target_qn in dag[parent].qreg_names
                    except:
                        cond_2 = False
                    if cond_1 and cond_2:
                        return parent
                    else:
                        queue.append(parent)
            return None

        def find_nearest_desc(dag, start, target_qn, inst_types):
            visited = set()
            queue = deque([start])
            #
            while queue:
                node = queue.popleft()
                for child in dag.successor_indices(node):
                    if child in visited:
                        continue
                    visited.add(child)
                    cond_1 = dag[child].inst_type in inst_types
                    try:
                        cond_2 = target_qn in dag[child].qreg_names
                    except:
                        cond_2 = False
                    if cond_1 and cond_2:
                        return child
                    else:
                        queue.append(child)
            return

        # remove rotations from the dag and record its predecessor and succesor nodes
        rot_list = []
        for i in self.dag.node_indices():
            nd = self.dag[i]
            if nd.inst_type == InstType.ROTATION:
                assert len(self.dag.predecessors(i)) == 1
                [pred_i] = self.dag.predecessor_indices(i)
                pred_nd = self.dag[pred_i]
                nd.pred_nd = pred_nd
                #
                assert len(self.dag.successors(i)) == 1
                [succ_i] = self.dag.successor_indices(i)
                succ_nd = self.dag[succ_i]
                nd.succ_nd = succ_nd
                #
                rot_list.append(nd)
                self.dag.remove_node(i)
                #
                # try to find the origin_h
                if nd.origin_nd is None:
                    [qn] = nd.qreg_names
                    h_up = find_nearest_anc(self.dag, succ_i, qn, [InstType.TRANS_H])
                    h_down = find_nearest_desc(self.dag, pred_i, qn, [InstType.TRANS_H])
                    if h_up == h_down and h_up is not None:
                        origin_h = h_up
                        nd.origin_nd = self.dag[origin_h]
                    else:
                        nd.origin_nd = None
                else:
                    pass
                #
        return rot_list

    def order_req_insts(self):
        # To avoid deadlock at the resource state ports,
        # We need to carefully consider the port request order
        # Current strategy
        ## Earlier (i.e., deeper at the reversed DAG) requests -> Earlier allocation

        depths = {}
        for idx in reversed(rx.topological_sort(self.dag)):
            succ_depth = 0
            for child in self.dag.successor_indices(idx):
                succ_depth = max(succ_depth, depths[child] + 1)
            depths[idx] = succ_depth

        # ordering the req insts indices
        req_indices = [i for i in self.dag.node_indices() if self.dag[i].inst_type in [InstType.REQ_MY, InstType.REQ_Y, InstType.REQ_M]]
        depth_list = [depths[i] for i in req_indices]
        req_queue = deque((i, self.dag[i]) for _, i in sorted(zip(depth_list, req_indices), reverse=True))
        return req_queue

    def get_ready_insts(self):
        ready_indices = sorted(self.ready_indices)
        ready_nodes = [self.dag[i] for i in ready_indices]
        return list(zip(ready_indices, ready_nodes))

    def init_ready_tracker(self):
        node_indices = list(self.dag.node_indices())
        self.remaining_in_degrees = {
            i: self.dag.in_degree(i)
            for i in node_indices
        }
        self.ready_indices = {
            i
            for i, degree in self.remaining_in_degrees.items()
            if degree == 0
        }

    def remove_ready_node(self, node_id):
        successor_indices = list(self.dag.successor_indices(node_id))
        self.ready_indices.discard(node_id)
        del self.remaining_in_degrees[node_id]

        for successor_id in successor_indices:
            if successor_id not in self.remaining_in_degrees:
                continue
            self.remaining_in_degrees[successor_id] -= 1
            if self.remaining_in_degrees[successor_id] == 0:
                self.ready_indices.add(successor_id)

        self.dag.remove_node(node_id)

    #######################################

    def run(self):
        while len(self.dag) > 0:
            inst_schedule = []
            req_schedule = []

            ##### PREPROCESSING #####
            req_insts = [InstType.REQ_MY, InstType.REQ_Y, InstType.REQ_M]
            pauli_insts = [InstType.PAULI_X, InstType.PAULI_Y, InstType.PAULI_Z]
            skip_targets = ['qreg_in',
                            'qreg_out',
                            'creg_out',
                            InstType.INIT_Z
                            ] + pauli_insts
            has_changed = True

            while has_changed:
                has_changed = False
                for (id, nd) in self.get_ready_insts():
                    if nd.inst_type in (skip_targets+req_insts):
                        if nd.inst_type in req_insts:
                            req_id, _ = self.req_queue[0]
                            if id != req_id:
                                continue
                            if nd.inst_type == InstType.REQ_MY:
                                if not (self.free_cells_m and self.free_cells_y):
                                    continue
                                else:
                                    req_schedule.append(nd)
                                    self.free_cells_m -= 1
                                    self.free_cells_y -= 1
                                    self.req_queue.popleft()
                                    self.remove_ready_node(id)
                            elif nd.inst_type == InstType.REQ_Y:
                                if not (self.free_cells_y):
                                    continue
                                else:
                                    req_schedule.append(nd)
                                    self.free_cells_y -= 1
                                    self.req_queue.popleft()
                                    self.remove_ready_node(id)
                            elif nd.inst_type == InstType.REQ_M:
                                if not (self.free_cells_m):
                                    continue
                                else:
                                    req_schedule.append(nd)
                                    self.free_cells_m -= 1
                                    self.req_queue.popleft()
                                    self.remove_ready_node(id)
                            else:
                                raise Exception()
                        else: #skip targets
                            if (self.propagate_creg_names_cond
                                    and nd.inst_type in pauli_insts
                                    and nd.creg_names_cond):
                                children = self.dag.successors(id)
                                for child in children:
                                    child.creg_names_cond += nd.creg_names_cond
                                    child.creg_names_cond = list(set(child.creg_names_cond))
                            elif nd.inst_type == InstType.INIT_Z:
                                nd.asap_idx = self.curr_layer
                                self.free_cells_n -= 1
                            else:
                                pass
                            self.remove_ready_node(id)
                            has_changed = True


            if len(self.dag) == 0:
                if self.debug_print:
                    print("SCHEDULING SUCCESSFULLY FINISHED!")
                break

            #### LAYER SCHEDULING #####
            # ASAP scheduling
            # Schedule all instructions regardless of the latency
            # Valid logical instructions are:
            ## MEAS_Z, MEAS_XorZ, TRANS_H, TRANS_CX
            ## MEAS_RESET_Z added
            inst_to_be_scheduled = []
            for id, nd in self.get_ready_insts():
                if nd.inst_type in [InstType.MEAS_Z,
                                    InstType.MEAS_XorZ,
                                    InstType.MEAS_RESET_Z,
                                    InstType.TRANS_H,
                                    InstType.TRANS_CX,
                                    #
                                    InstType.TRANS_H_ROT,
                                    #
                                    InstType.TRANS_S,
                                    ]:
                    inst_to_be_scheduled.append((id, nd))
                else:
                    pass

            for id, nd in inst_to_be_scheduled:
                # free cell tracking
                ## when we can overlap two LQs
                if self.can_overlap:
                    if nd.inst_type == InstType.TRANS_CX:
                        [on_0, on_1] = nd.oreg_names
                        if self.s_trans_opt == STransOpt.GATE_TEL:
                            ## MY: M->Y & overlap, m is freed
                            if ("M" in on_0 and "Y" in on_1) or ("Y" in on_0 and "M" in on_1):
                                self.free_cells_m += 1

                            ## MQ: Y->Q & overlap, but Y is still at the Y port (no free)

                            ## YQ: Y->Q & overlap, y is freed
                            elif ("Y" in on_0 and "Q" in on_1) or ("Q" in on_0 and "Y" in on_1):
                                [qn_0, qn_1] = nd.qreg_names
                                self.free_cells_y += 1
                            else:
                                pass
                        elif self.s_trans_opt == STransOpt.TRANS_S:
                            ## MQ: M->Q m is freed
                            if ("M" in on_0 and "Q" in on_1):
                                self.free_cells_m += 1
                        else:
                            raise Exception()
                    elif nd.inst_type == InstType.MEAS_Z:
                        ## should occur at the normal cells
                        ## do not free any cell
                        ## instead the measrued cell is already occupied by the teleported M
                        pass
                    elif nd.inst_type == InstType.MEAS_RESET_Z:
                        ## do not free any cell
                        ## the same program qubit will be initialized at the place
                        pass
                    elif nd.inst_type == InstType.MEAS_XorZ:
                        # should occur at the Y port
                        # free the Y port
                        [qn] = nd.qreg_names
                        assert "Y" in qn
                        self.free_cells_y += 1
                    else:
                        assert nd.inst_type in [InstType.TRANS_H, InstType.TRANS_H_ROT, InstType.TRANS_S]
                        # do nothing

                ## When we cannot overlap two LQs
                elif self.cannot_overlap:
                    raise Exception()
                else:
                    raise Exception()

                # creg_cond propagation
                if (self.propagate_creg_names_cond
                        and nd.inst_type in [InstType.TRANS_H, InstType.TRANS_CX,
                                             InstType.TRANS_H_ROT, InstType.TRANS_S]):
                    if nd.creg_names_cond:
                        children = self.dag.successors(id)
                        for child in children:
                            child.creg_names_cond += nd.creg_names_cond
                            child.creg_names_cond = list(set(child.creg_names_cond))
                    else:
                        pass
                else:
                    pass
                self.remove_ready_node(id)
            #
            #
            inst_schedule = [nd for _, nd in inst_to_be_scheduled]

            # M/Y port clear request
            if self.can_overlap:
                pass
            else:
                raise Exception()
            #
            for req in req_schedule:
                req.asap_idx = self.curr_layer
            for inst in inst_schedule:
                inst.asap_idx = self.curr_layer
            #####
            self.inst_schedule_trace.append(inst_schedule)
            self.req_schedule_trace.append(req_schedule)
            self.curr_layer += 1

        ## rot schedule
        if self.inst_sched_opt == InstSchedOpt.ASAP:
            ##
            if self.rot_sched_opt == RotSchedOpt.FOLLOW_H:
                for rot in self.rot_list:
                    if rot.origin_nd is not None:
                        rot_idx = rot.origin_nd.asap_idx
                    else:
                        # can occur for others than ALL_ROT
                        if rot.pred_nd.inst_type == InstType.TRANS_CX:
                            rot_idx = rot.pred_nd.asap_idx+1
                        else:
                            rot_idx = rot.pred_nd.asap_idx
                    self.inst_schedule_trace[rot_idx].append(rot)
            ##
            elif self.rot_sched_opt == RotSchedOpt.ASAP:
                for rot in self.rot_list:
                    if rot.pred_nd.inst_type == InstType.TRANS_CX:
                        rot_idx = rot.pred_nd.asap_idx+1
                    else:
                        rot_idx = rot.pred_nd.asap_idx
                    self.inst_schedule_trace[rot_idx].append(rot)
            ##
            elif self.rot_sched_opt == RotSchedOpt.ALAP:
                for rot in self.rot_list:
                    rot_idx = rot.succ_nd.asap_idx
                    self.inst_schedule_trace[rot_idx].append(rot)
            ##
            elif self.rot_sched_opt == RotSchedOpt.DISTRIBUTE:
                ### check each rot's idx range to schedule
                for rot in self.rot_list:
                    if rot.pred_nd.inst_type in [InstType.TRANS_CX, InstType.MEAS_RESET_Z]:
                        earliest = rot.pred_nd.asap_idx+1
                    else:
                        earliest = rot.pred_nd.asap_idx
                    # latest
                    if rot.succ_nd.inst_type in [InstType.MEAS_Z, InstType.MEAS_XorZ, InstType.MEAS_RESET_Z]:
                        latest = rot.succ_nd.asap_idx-1
                    else:
                        latest = rot.succ_nd.asap_idx
                    #
                    rot.idx_interval = (earliest, latest)
                    #
                ### determine the idx to schedule each rot
                self.rot_list.sort(key=lambda x: x.idx_interval[1])
                ###
                used_idx_count = dict()
                for rot in self.rot_list:
                    (l, r) = rot.idx_interval
                    ## try to find smallest unused idx
                    test_idx = next((i for i in range(l, r+1) if not i in used_idx_count.keys()), None)
                    if test_idx is not None:
                        rot_idx = test_idx
                        if not rot_idx in used_idx_count.keys():
                            used_idx_count[rot_idx] = 1
                        else:
                            used_idx_count[rot_idx] += 1
                    # if a rot should be scheduled with others
                    else:
                        subdict = {k: v for k, v in used_idx_count.items() if k in range(l, r+1)}
                        # choose the min-used idx
                        rot_idx = min(subdict, key=subdict.get)
                        assert rot_idx in used_idx_count.keys()
                        used_idx_count[rot_idx] += 1
                    rot.rot_idx = rot_idx
                ### gather rots to the target indices
                rot_dict = dict()
                for rot in self.rot_list:
                    if not rot.rot_idx in rot_dict.keys():
                        rot_dict[rot.rot_idx] = [rot]
                    else:
                        rot_dict[rot.rot_idx].append(rot)
                ### schedule rots to the inst_schedule
                for scheduled_rot_idx, scheduled_rots in rot_dict.items():
                    # assert unique
                    assert len(scheduled_rots) == len(set(scheduled_rots))
                    #
                    inst_schedule = self.inst_schedule_trace[scheduled_rot_idx]
                    ##
                    scheduled_meas = [inst for inst in inst_schedule if inst.inst_type in [InstType.MEAS_Z, InstType.MEAS_XorZ]]
                    measured_qns = [next(qn for qn in inst.qreg_names) for inst in scheduled_meas]
                    ##
                    for rot in scheduled_rots:
                        [qn] = rot.qreg_names
                        if not qn in measured_qns:
                            inst_schedule.append(rot)
                        else:
                            pass
                    ##

            elif self.rot_sched_opt in [RotSchedOpt.AGGREGATE]:
                rotations = self.rot_list
                active = [True] * len(rotations)
                intervals = []
                candidates_by_idx = dict()
                candidate_counts = dict()

                # Rotation intervals never change during aggregation.  Build
                # their inverse index once and update only the counts when a
                # group is scheduled.
                for rot_i, rot in enumerate(rotations):
                    if rot.pred_nd.inst_type in [InstType.TRANS_CX, InstType.MEAS_RESET_Z]:
                        earliest = rot.pred_nd.asap_idx+1
                    else:
                        earliest = rot.pred_nd.asap_idx
                    if rot.succ_nd.inst_type in [InstType.MEAS_Z, InstType.MEAS_XorZ, InstType.MEAS_RESET_Z]:
                        latest = rot.succ_nd.asap_idx-1
                    else:
                        latest = rot.succ_nd.asap_idx

                    interval = range(earliest, latest+1)
                    intervals.append(interval)
                    for idx in interval:
                        candidates_by_idx.setdefault(idx, deque()).append(rot_i)
                        candidate_counts[idx] = candidate_counts.get(idx, 0) + 1

                remaining = len(rotations)
                while remaining:
                    # The old dict was rebuilt in remaining-rotation order on
                    # every iteration.  Its max() kept the first inserted key
                    # on ties.  (first active rotation, layer index) reproduces
                    # that ordering without rebuilding the dictionary.
                    for idx, candidates in candidates_by_idx.items():
                        while candidates and not active[candidates[0]]:
                            candidates.popleft()
                    max_rot_idx = min(
                        (idx for idx, count in candidate_counts.items() if count),
                        key=lambda idx: (
                            -candidate_counts[idx],
                            candidates_by_idx[idx][0],
                            idx,
                        ),
                    )
                    #
                    scheduled_rot_idx = max_rot_idx
                    scheduled_indices = [
                        rot_i
                        for rot_i in candidates_by_idx[scheduled_rot_idx]
                        if active[rot_i]
                    ]
                    scheduled_rots = [rotations[rot_i] for rot_i in scheduled_indices]
                    #####
                    # remove two duplicated rots
                    counts = dict()
                    for rot in scheduled_rots:
                        [qn] = rot.qreg_names
                        if not qn in counts.keys():
                            counts[qn] = 0
                        counts[qn] += 1
                    #
                    unique_rots = []
                    for rot in scheduled_rots:
                        [qn] = rot.qreg_names
                        if counts[qn] == 1:
                            unique_rots.append(rot)
                        else:
                            pass
                    #
                    inst_schedule = self.inst_schedule_trace[scheduled_rot_idx]
                    scheduled_meas = [inst for inst in inst_schedule if inst.inst_type in [InstType.MEAS_Z, InstType.MEAS_XorZ]]
                    measured_qns = [next(qn for qn in inst.qreg_names) for inst in scheduled_meas]
                    for rot in unique_rots:
                        [qn] = rot.qreg_names
                        if not qn in measured_qns:
                            inst_schedule.append(rot)
                        else:
                            pass
                    #
                    for rot_i in scheduled_indices:
                        active[rot_i] = False
                        remaining -= 1
                        for idx in intervals[rot_i]:
                            candidate_counts[idx] -= 1
                self.rot_list = []

            else:
                raise Exception()
        else:
            raise Exception()

        # Remove duplicated rotatations
        for idx, inst_schedule in enumerate(self.inst_schedule_trace):
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]
            if not rot_scheduled:
                continue
            #
            rot_qn_list = []
            for rot in rot_scheduled:
                rot_qn_list += rot.qreg_names
            rot_count_dict = Counter(rot_qn_list)
            rot_dups_dict = {qn: cnt for qn, cnt in rot_count_dict.items() if cnt > 1}
            #
            rots_to_remove = []
            for qn, cnt in rot_dups_dict.items():
                rot_dups = [rot for rot in rot_scheduled if qn in rot.qreg_names]
                if cnt % 2 == 1:
                    rot_dups = rot_dups[:-1]
                #
                rots_to_remove += rot_dups
            #
            for rot in rots_to_remove:
                inst_schedule.remove(rot)

        # Remove rotations with measurements
        for inst_schedule in self.inst_schedule_trace:
            rots_to_remove = []
            #
            rot_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.TRANS_H_ROT, InstType.ROTATION]]

            meas_scheduled = [inst for inst in inst_schedule if inst.inst_type in [InstType.MEAS_XorZ, InstType.MEAS_Z]]
            if not rot_scheduled or not meas_scheduled:
                continue
            #
            rot_qn_list = []
            for rot in rot_scheduled:
                rot_qn_list += rot.qreg_names
            meas_qn_list = []
            for meas in meas_scheduled:
                meas_qn_list += meas.qreg_names
            rot_meas_qn_list = list(set(rot_qn_list) & set(meas_qn_list))
            for qn in rot_meas_qn_list:
                rot = next(rot for rot in rot_scheduled if qn in rot.qreg_names)
                rots_to_remove.append(rot)
            #
            for rot in rots_to_remove:
                inst_schedule.remove(rot)

        self.insert_cx_boundary_alignment_rotations()
        self.max_rot_count = self.max_rot_per_layer()

        # debug
        if self.debug_print:
            for idx, (req_schedule, inst_schedule) in enumerate(zip(self.req_schedule_trace, self.inst_schedule_trace)):
                print(f"LAYER: {idx}")
                for req in req_schedule:
                    print(req.inst_name, req.qreg_names)
                for inst in inst_schedule:
                    if inst.inst_type == InstType.TRANS_CX:
                        print(inst.inst_name, inst.qreg_names, inst.oreg_names)
                    else:
                        print(inst.inst_name, inst.qreg_names)
                print()

    def insert_cx_boundary_alignment_rotations(self):
        if self.skip_h or self.s_trans_opt != STransOpt.TRANS_S:
            return

        req_types = [InstType.REQ_MY, InstType.REQ_Y, InstType.REQ_M]
        h_types = [InstType.TRANS_H, InstType.TRANS_H_ROT]
        rot_types = [InstType.ROTATION, InstType.TRANS_H_ROT]
        meas_types = [InstType.MEAS_Z, InstType.MEAS_XorZ, InstType.MEAS_RESET_Z]

        qreg_boundary = {}

        def ensure_qreg(qn):
            if qn not in qreg_boundary:
                qreg_boundary[qn] = False

        def toggle_qreg(qn):
            ensure_qreg(qn)
            qreg_boundary[qn] = not qreg_boundary[qn]

        def add_alignment_rotation(inst_schedule, qn, layer_idx):
            rot = InstRotation([qn])
            rot.asap_idx = layer_idx
            rot.is_boundary_alignment = True
            inst_schedule.append(rot)
            toggle_qreg(qn)

        for layer_idx, (req_schedule, inst_schedule) in enumerate(zip(self.req_schedule_trace, self.inst_schedule_trace)):
            for req in req_schedule:
                if req.inst_type in req_types:
                    for qn in req.qreg_names:
                        qreg_boundary[qn] = False

            for inst in inst_schedule:
                if inst.inst_type in h_types:
                    for qn in inst.qreg_names:
                        toggle_qreg(qn)

            for inst in inst_schedule:
                if inst.inst_type in meas_types:
                    for qn in inst.qreg_names:
                        if inst.inst_type == InstType.MEAS_RESET_Z:
                            qreg_boundary[qn] = False
                        else:
                            qreg_boundary.pop(qn, None)

            for inst in inst_schedule:
                if inst.inst_type in rot_types:
                    for qn in inst.qreg_names:
                        toggle_qreg(qn)

            for inst in list(inst_schedule):
                if inst.inst_type != InstType.TRANS_CX:
                    continue
                q0, q1 = inst.qreg_names
                ensure_qreg(q0)
                ensure_qreg(q1)
                if qreg_boundary[q0] == qreg_boundary[q1]:
                    continue
                if qreg_boundary[q0]:
                    add_alignment_rotation(inst_schedule, q0, layer_idx)
                else:
                    add_alignment_rotation(inst_schedule, q1, layer_idx)

    def count_rot_layers(self):
        rot_layer_count = 0
        for inst_schedule in self.inst_schedule_trace:
            rot_types = [InstType.ROTATION, InstType.TRANS_H_ROT]
            if any([inst.inst_type in rot_types for inst in inst_schedule]):
                rot_layer_count += 1
            else:
                pass

        # Debug
        for inst_schedule in self.inst_schedule_trace:
            if any([inst.inst_type == InstType.TRANS_H for inst in inst_schedule]):
                pass
        return rot_layer_count

    def count_rot_numbers(self):
        rot_num_count = 0
        for inst_schedule in self.inst_schedule_trace:
            rot_types = [InstType.ROTATION, InstType.TRANS_H_ROT]
            if any([inst.inst_type in rot_types for inst in inst_schedule]):
                rot_num_count += len([inst for inst in inst_schedule if inst.inst_type in rot_types])
            else:
                pass
        return rot_num_count


    def max_rot_per_layer(self):
        rot_count_trace = []
        for idx, inst_schedule in enumerate(self.inst_schedule_trace):
            rot_types = [InstType.ROTATION, InstType.TRANS_H_ROT]
            if any([inst.inst_type in rot_types for inst in inst_schedule]):
                rot_count = len([inst for inst in inst_schedule if inst.inst_type in rot_types])
            else:
                rot_count = 0
            rot_count_trace.append(rot_count)
        #
        return max(rot_count_trace)
