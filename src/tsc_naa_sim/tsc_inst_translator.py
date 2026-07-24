from tsc_instructions import *
from macro import *
#
import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit import Instruction
#
import rustworkx as rx
from types import SimpleNamespace
from collections import deque


## Static translation
### -> Separate "Resource state preparation" and "Logical operations" using them
### -> Or other necessary / fixed translation

class tsc_inst_translator:
    # input
    qc_qiskit_in = None ## {CX, H, S, T, X, Y, Z}

    # output
    sc_dag = rx.PyDiGraph()

    # intermediate
    m_id = 0 # virtual id for magic state patch
    y_id = 0 # virtual id for y state patch
    c_id = 0 # virtual id for classical registers

    leaf_nodes = dict()
    qubit_tracking = dict()
    qstat_tracking = dict()


    def __init__(
        self,
        qc_qiskit_in: qiskit.QuantumCircuit,
        rot_trans_opt: RotTransOpt,
        s_trans_opt: STransOpt,

    ) -> None:
        self.qc_qiskit_in = qc_qiskit_in
        #
        self.sc_dag = rx.PyDiGraph()
        #
        self.m_id = 0
        self.y_id = 0
        self.c_id = 0
        #
        self.leaf_nodes = dict()
        #
        self.qubit_tracking = dict()
        self.current_to_origin = dict()
        for q_id in range(qc_qiskit_in.num_qubits):
            q_name = f'Q{q_id}'
            self.qubit_tracking[q_name] = q_name
            self.current_to_origin[q_name] = q_name
        #
        self.qstat_tracking = dict()
        #
        self.inst_map = {
            'x': InstType.PAULI_X,
            'y': InstType.PAULI_Y,
            'z': InstType.PAULI_Z,
            'cx': InstType.TRANS_CX,
            'reset': InstType.INIT_Z,
            'measure': InstType.MEAS_Z
        }
        #
        self.rot_trans_opt = rot_trans_opt
        #
        self.s_trans_opt = s_trans_opt

    def run(self) -> None:
        qc_dag = circuit_to_dag(self.qc_qiskit_in)

        # tranalate & mark
        for qc_node in qc_dag.topological_op_nodes():
            self.translate_node(qc_node)


        ### ROT OPT ###
        nearest_cache = {}

        def find_nearest_anc(dag, start, target_qn, inst_types):
            cache_key = ("anc", start, target_qn, tuple(inst_types))
            if cache_key in nearest_cache:
                return nearest_cache[cache_key]
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
                        nearest_cache[cache_key] = parent
                        return parent
                    queue.append(parent)
            nearest_cache[cache_key] = None
            return None

        def find_nearest_desc(dag, start, target_qn, inst_types):
            cache_key = ("desc", start, target_qn, tuple(inst_types))
            if cache_key in nearest_cache:
                return nearest_cache[cache_key]
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
                        nearest_cache[cache_key] = child
                        return child
                    queue.append(child)
            nearest_cache[cache_key] = None
            return

        def insert_rot(cx_i, target_qn):

            sc_i = cx_to_sc[cx_i]
            ### predecessor
            pred_list = self.cx_dag.predecessor_indices(cx_i)
            ####
            cx_pred = None
            for pred in pred_list:
                qreg_names = self.cx_dag.get_all_edge_data(pred, cx_i)
                if target_qn in qreg_names:
                    cx_pred = pred
                    break

            ####
            sc_pred = None
            if cx_pred is None:
                sc_ancestors = rx.ancestors(self.sc_dag, cx_to_sc[cx_i])
                for n in alloc_nodes_by_qn.get(target_qn, []):
                    if n in sc_ancestors:
                        sc_pred = n
                        break
            else:
                sc_pred = cx_to_sc[cx_pred]
            assert sc_pred is not None

            ## insert
            rot_i = self.sc_dag.add_node(InstRotation([target_qn]))
            self.sc_dag[rot_i].origin_nd = None
            self.sc_dag.add_edge(sc_pred, rot_i, target_qn)
            self.sc_dag.add_edge(rot_i, sc_i, target_qn)

            return

        rot_prop_cache = {}

        def rot_prop_chain(cx_dag, start, target_qn):
            cache_key = (start, target_qn)
            if cache_key in rot_prop_cache:
                return rot_prop_cache[cache_key]
            desc_list = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for child in cx_dag.successor_indices(node):
                    if cx_dag.get_edge_data(node, child) == target_qn:
                        if child in desc_list: 
                            continue
                        desc_list.append(child)
                        queue.append(child)
            #
            rot_prop_cache[cache_key] = tuple(desc_list)
            return rot_prop_cache[cache_key]

        def inspect_rot_imapct(cx_i, target_qn):
            cx_nd = self.cx_dag[cx_i]
            desc_list = rot_prop_chain(self.cx_dag, cx_i, target_qn)
            target_nodes = [cx_nd]
            target_nodes += [self.cx_dag[desc] for desc in desc_list]
            #
            unmatch_reduction = 0
            for target_nd in target_nodes:
                for (q_name, q_stat) in zip(target_nd.qreg_names, target_nd.qreg_stats):
                    if q_name == target_qn:
                        old_stat = q_stat
                    else:
                        other_stat = q_stat
                ##
                if old_stat == QregStatus.ACTIVE_A:
                    new_stat = QregStatus.ACTIVE_B
                elif old_stat == QregStatus.ACTIVE_B:
                    new_stat = QregStatus.ACTIVE_A
                elif old_stat == QregStatus.ACTIVE_C:
                    new_stat = QregStatus.ACTIVE_D
                elif old_stat == QregStatus.ACTIVE_D:
                    new_stat = QregStatus.ACTIVE_C
                else:
                    raise Exception()
                ##
                if new_stat != other_stat:
                    unmatch_reduction -= 1
                else:
                    unmatch_reduction += 1
            return unmatch_reduction

            
        def propagate_rot_impact(cx_i, target_qn):
            cx_nd = self.cx_dag[cx_i]
            desc_list = rot_prop_chain(self.cx_dag, cx_i, target_qn)
            target_nodes = [cx_nd]
            target_nodes += [self.cx_dag[desc] for desc in desc_list]
            #
            for target_nd in target_nodes:
                for idx, (q_name, q_stat) in enumerate(zip(target_nd.qreg_names, target_nd.qreg_stats)):
                    if q_name == target_qn:
                        target_idx = idx
                        old_stat = q_stat
                ##
                if old_stat == QregStatus.ACTIVE_A:
                    new_stat = QregStatus.ACTIVE_B
                elif old_stat == QregStatus.ACTIVE_B:
                    new_stat = QregStatus.ACTIVE_A
                elif old_stat == QregStatus.ACTIVE_C:
                    new_stat = QregStatus.ACTIVE_D
                elif old_stat == QregStatus.ACTIVE_D:
                    new_stat = QregStatus.ACTIVE_C
                else:
                    raise Exception()
                ##
                target_nd.qreg_stats[target_idx] = new_stat
            return


        # build CX sub graph
        cx_nodes = [i for i, data in enumerate(self.sc_dag.nodes()) if data.inst_type == InstType.TRANS_CX]
        cx_node_set = set(cx_nodes)
        ##
        self.sc_order = list(rx.topological_sort(self.sc_dag))
        alloc_nodes_by_qn = dict()
        for n in reversed(self.sc_order):
            nd = self.sc_dag[n]
            if nd.inst_type in [InstType.INIT_Z, InstType.REQ_Y, InstType.REQ_MY]:
                for qn in nd.qreg_names:
                    if qn not in alloc_nodes_by_qn:
                        alloc_nodes_by_qn[qn] = []
                    alloc_nodes_by_qn[qn].append(n)
        self.cx_dag = rx.PyDiGraph()
        ## node
        sc_to_cx = {}
        cx_to_sc = {}

        for sc_i, data in enumerate(self.sc_dag.nodes()):
            if sc_i in cx_node_set:
                cx_i = self.cx_dag.add_node(data)
                sc_to_cx[sc_i] = cx_i
                cx_to_sc[cx_i] = sc_i
        ## edge
        for u, v in self.sc_dag.edge_list():
            q_name = self.sc_dag.get_edge_data(u,v)
            if u in cx_node_set and v in cx_node_set:
                self.cx_dag.add_edge(sc_to_cx[u], sc_to_cx[v], q_name)

            elif u in cx_node_set and v not in cx_node_set:
                if self.sc_dag[v].inst_type in [InstType.MEAS_XorZ, InstType.MEAS_Z]:
                    pass
                else:
                    try:
                        first_desc = find_nearest_desc(self.sc_dag, v, q_name, [InstType.TRANS_CX, InstType.MEAS_RESET_Z, InstType.MEAS_Z, InstType.MEAS_XorZ])
                        if self.sc_dag[first_desc].inst_type == InstType.TRANS_CX:
                            self.cx_dag.add_edge(sc_to_cx[u], sc_to_cx[first_desc], q_name)
                        else:
                            pass
                        ###
                    except:
                        pass
            else:
                pass
        #
        if self.rot_trans_opt == RotTransOpt.ALL_ROT:
            #
            h_indices = [n for n  in self.sc_dag.node_indices() if self.sc_dag[n].inst_type == InstType.TRANS_H]
            for h_i in h_indices:
                [target_qn] = self.sc_dag[h_i].qreg_names

                ## pred
                first_anc = find_nearest_anc(self.sc_dag, h_i, target_qn, [InstType.TRANS_CX, InstType.INIT_Z, InstType.REQ_Y, InstType.REQ_MY])
                ## succ
                first_desc = find_nearest_desc(self.sc_dag, h_i, target_qn, [InstType.TRANS_CX, InstType.MEAS_Z, InstType.MEAS_XorZ, InstType.MEAS_RESET_Z])

                # insert
                rot_i = self.sc_dag.add_node(InstRotation([target_qn]))
                self.sc_dag[rot_i].origin_nd = self.sc_dag[h_i]
                self.sc_dag.add_edge(first_anc, rot_i, target_qn)
                self.sc_dag.add_edge(rot_i, first_desc, target_qn)
                # Nearest-node searches depend on graph topology.  
                # Do not reuse results across an insertion.
                nearest_cache.clear()

                # propagate
                if self.sc_dag[first_desc].inst_type == InstType.TRANS_CX:
                    cx_i = sc_to_cx[first_desc]
                    propagate_rot_impact(cx_i, target_qn)

        elif self.rot_trans_opt in [RotTransOpt.NAIVE_SKIP, RotTransOpt.HEUR_ALLOC]:
            depths = {}
            for idx in reversed(rx.topological_sort(self.cx_dag)):
                succ_depth = 0
                for child in self.cx_dag.successor_indices(idx):
                    succ_depth = max(succ_depth, depths[child]+1)
                depths[idx] = succ_depth
            cx_indices = sorted(self.cx_dag.node_indices(), key=lambda x: depths[x], reverse=True)

            # Traverse CX nodes
            for cx_i in cx_indices:
                cx_nd = self.cx_dag[cx_i]
                # Check match
                unmatch = len(set(cx_nd.qreg_stats)) != 1
                if unmatch:
                    if self.rot_trans_opt == RotTransOpt.NAIVE_SKIP:
                        ## decide qreg to add rotation: NAIVE
                        for (q_name, q_stat) in zip(cx_nd.qreg_names, cx_nd.qreg_stats):
                            if q_stat == QregStatus.ACTIVE_B:
                                target_qn = q_name
                            else:
                                pass
                        ## insert rotation
                        insert_rot(cx_i, target_qn)
                        ### propagate stat change
                        propagate_rot_impact(cx_i, target_qn)

                    elif self.rot_trans_opt == RotTransOpt.HEUR_ALLOC:
                        ## decide qreg to add rotation:
                        [qn_1, qn_2] = cx_nd.qreg_names
                        # Avoid to insert rotation to M port or Y port
                        [on_1, on_2] = cx_nd.oreg_names
                        #####
                        if cx_nd.cx_type == CXType.YQ:
                            if "Y" in on_1:
                                target_qn = qn_2
                            elif "Y" in on_2:
                                target_qn = qn_1
                            else:
                                raise Exception()
                        #   
                        elif cx_nd.cx_type == CXType.MQ:
                            if "M" in on_1:
                                target_qn = qn_2
                            elif "M" in on_2:
                                target_qn = qn_1
                            else:
                                raise Exception()
                        else: 
                            assert cx_nd.cx_type != CXType.MY
                            assert cx_nd.cx_type == CXType.QQ
                            #
                            unmatch_reduction_1 = inspect_rot_imapct(cx_i, qn_1)
                            unmatch_reduction_2 = inspect_rot_imapct(cx_i, qn_2)
                            if unmatch_reduction_1 > unmatch_reduction_2:
                                target_qn = qn_1
                            else:
                                target_qn = qn_2
                        ##
                        insert_rot(cx_i, target_qn)
                        ##
                        propagate_rot_impact(cx_i, target_qn)
                    else:
                        raise Exception()
                    ##
                else:
                    pass

            # Check validity
            for cx_i in cx_indices:
                cx_nd = self.cx_dag[cx_i]
                if len(set(cx_nd.qreg_stats)) != 1:
                    raise Exception()
                    pass
        else:
            pass

        # To handle Qubit reuse scenarios
        ## Detect continuous MEAS & INIT for the same program qubit
        ## Merge them into the MEAS_RESET node
        ## Remove leaf INIT_Zs
        leaf_init_ids = set()
        for i in self.sc_dag.node_indices():
            nd = self.sc_dag[i]
            if nd.inst_type == InstType.INIT_Z and self.sc_dag.out_degree(i) == 0:
                leaf_init_ids.add(i)
        #
        for i in leaf_init_ids:
            self.sc_dag.remove_node(i)
        ## start
        merged_ids = set()
        for (u, v) in self.sc_dag.edge_list():
            parent = self.sc_dag[u]
            child = self.sc_dag[v]
            #
            if (parent.inst_type == InstType.MEAS_Z) and (child.inst_type == InstType.INIT_Z):
                if set(parent.qreg_names) == set(child.qreg_names):
                    # add
                    mr_inst = InstMeasResetZ(
                        qreg_names=parent.qreg_names, 
                        creg_name_out=parent.creg_name_out
                    )
                    mr_i = self.sc_dag.add_node(mr_inst)
                    # pred
                    pred_list = self.sc_dag.predecessor_indices(u)
                    for pred in pred_list:
                        ed = self.sc_dag.get_edge_data(pred, u) 
                        self.sc_dag.add_edge(pred, mr_i, ed)
                    # succ
                    succ_list = [nd for nd in self.sc_dag.successor_indices(u) if nd != v]
                    for succ in succ_list:
                        ed = self.sc_dag.get_edge_data(u, succ)
                        self.sc_dag.add_edge(mr_i, succ, ed)
                    ##
                    succ_list = self.sc_dag.successor_indices(v)
                    for succ in succ_list:
                        ed = self.sc_dag.get_edge_data(v, succ)
                        self.sc_dag.add_edge(mr_i, succ, ed)
                    # remove
                    merged_ids.add(u)
                    merged_ids.add(v)
                else:
                    pass
            else:
                pass
        for i in merged_ids:
            self.sc_dag.remove_node(i)


    def add_inst_node(self,
                      inst_type,
                      qreg_names,
                      creg_names_cond,
                      creg_name_out):

        if inst_type == InstType.INIT_Z:
            inst = InstInitZ(qreg_names)
            for q_name in qreg_names:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_A
        elif inst_type == InstType.REQ_MY:
            inst = InstReqMY(qreg_names)
            for q_name in qreg_names:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_A
        elif inst_type == InstType.REQ_Y:
            inst = InstReqY(qreg_names)
            for q_name in qreg_names:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_A
        elif inst_type == InstType.REQ_M:
            inst = InstReqM(qreg_names)
            for q_name in qreg_names:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_A
        elif inst_type == InstType.TRANS_S:
            inst = InstTransS(qreg_names, creg_names_cond)
        elif inst_type == InstType.MEAS_Z:
            inst = InstMeasZ(qreg_names, creg_name_out)
            for q_name in qreg_names:
                del self.qstat_tracking[q_name]
        elif inst_type == InstType.MEAS_XorZ:
            inst = InstMeasXorZ(qreg_names, creg_names_cond, creg_name_out)
            for q_name in qreg_names:
                del self.qstat_tracking[q_name]
        elif inst_type == InstType.PAULI_X:
            inst = InstPauliX(qreg_names, creg_names_cond)
        elif inst_type == InstType.PAULI_Y:
            inst = InstPauliY(qreg_names, creg_names_cond)
        elif inst_type == InstType.PAULI_Z:
            inst = InstPauliZ(qreg_names, creg_names_cond)
        elif inst_type == InstType.TRANS_H_ROT:
            inst = InstTransHRot(qreg_names)
        elif inst_type == InstType.TRANS_H:
            inst = InstTransH(qreg_names)
            #
            assert len(qreg_names) == 1
            [q_name] = qreg_names
            if self.qstat_tracking[q_name] == QregStatus.ACTIVE_A:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_B
            elif self.qstat_tracking[q_name] == QregStatus.ACTIVE_B:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_A
            elif self.qstat_tracking[q_name] == QregStatus.ACTIVE_C:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_D
            elif self.qstat_tracking[q_name] == QregStatus.ACTIVE_D:
                self.qstat_tracking[q_name] = QregStatus.ACTIVE_C
            else:
                raise Exception()

        elif inst_type == InstType.TRANS_CX:
            oreg_names = []
            qreg_stats = []
            for q_name in qreg_names:
                o_name = self.current_to_origin.get(q_name, q_name)
                oreg_names.append(o_name)
                #
                qreg_stat = self.qstat_tracking[q_name]
                qreg_stats.append(qreg_stat)
            if all(["Q" in on for on in oreg_names]):
                cx_type = CXType.QQ
            elif any(["M" in on for on in oreg_names]) and any(["Y" in on for on in oreg_names]):
                cx_type = CXType.MY
            elif any(["M" in on for on in oreg_names]):
                cx_type = CXType.MQ
            elif any(["Y" in on for on in oreg_names]):
                cx_type = CXType.YQ
            else:
                raise Exception()
            #
            inst = InstTransCX(qreg_names, oreg_names, cx_type)
            #
            inst.qreg_stats = qreg_stats
        else:
            raise Exception()

        # add node
        curr_nd = self.sc_dag.add_node(inst)

        #connect to predecessors
        ## qregs
        assert qreg_names
        for qreg_name in qreg_names:
            try:
                pred_node = self.leaf_nodes[qreg_name]
            except:
                nd = SimpleNamespace()
                nd.inst_name = qreg_name
                nd.inst_type = "qreg_in"
                pred_node = self.sc_dag.add_node(nd)
            self.sc_dag.add_edge(pred_node, curr_nd, qreg_name)
        
        ## cregs
        for creg_name_cond in creg_names_cond:
            try:
                pred_node = self.leaf_nodes[creg_name_cond]
            except:
                raise Exception()
            self.sc_dag.add_edge(pred_node, curr_nd, creg_name_cond)

        # connect to successor
        if creg_name_out: # indicates it's measurement
            nd = SimpleNamespace()
            nd.inst_name = creg_name_out
            nd.inst_type = "creg_out"
            cout_node = self.sc_dag.add_node(nd)
            self.sc_dag.add_edge(curr_nd, cout_node, creg_name_out)

        if inst_type in [InstType.MEAS_Z, InstType.MEAS_XorZ]:
            [qreg_name] = qreg_names
            nd = SimpleNamespace()
            nd.inst_name = qreg_name
            nd.inst_type = "qreg_out"
            qout_node = self.sc_dag.add_node(nd)
            self.sc_dag.add_edge(curr_nd, qout_node, qreg_name)


        # update leaf_nodes
        for qreg_name in qreg_names:
            self.leaf_nodes[qreg_name] = curr_nd
        if creg_name_out:
            self.leaf_nodes[creg_name_out] = cout_node

        return

    def translate_node(self, qc_node):
        direct_gates = ['cx', 'x', 'y', 'z', 'reset']
        rot_gates = ['h']
        teleport_gates = ['t', 's']
        custom_instruction_labels = [
            "plus_state_prepare",
            "Tdag_state_prepare",
            "X-meas",
            "decomposed_Tdag",
            "T_state_prepare"
        ]
        #
        qc_name = qc_node.name
        qc_label = qc_node.label
        qreg_names = [f'Q{self.qc_qiskit_in.find_bit(q).index}' for q in qc_node.qargs]
        qreg_names = [self.qubit_tracking[qreg_name] for qreg_name in qreg_names]

        if qc_name in direct_gates:
            self.add_inst_node(
                inst_type = self.inst_map[qc_name],
                qreg_names = qreg_names,
                creg_names_cond = [],
                creg_name_out = ""
            )
        elif qc_name == 'measure':
            c_name = f"C{self.c_id}"
            self.add_inst_node(
                inst_type = self.inst_map[qc_name],
                qreg_names = qreg_names,
                creg_names_cond = [],
                creg_name_out = c_name
            )
            self.c_id += 1

        elif qc_name in rot_gates and qc_name == 'h':
            if self.rot_trans_opt == RotTransOpt.BASE:
                self.add_inst_node(
                    inst_type = InstType.TRANS_H_ROT,
                    qreg_names = qreg_names,
                    creg_names_cond = [],
                    creg_name_out = ""
                )

            elif self.rot_trans_opt in [RotTransOpt.ALL_ROT, RotTransOpt.NAIVE_SKIP, RotTransOpt.HEUR_ALLOC]:
                self.add_inst_node(
                    inst_type = InstType.TRANS_H,
                    qreg_names = qreg_names,
                    creg_names_cond = [],
                    creg_name_out = ""
                )
            else:
                raise Exception()

        elif qc_name in teleport_gates and qc_name == 't':
            [q_name] = qreg_names
            if self.s_trans_opt == STransOpt.GATE_TEL:
                # request - M & Y together
                m_name = f"M{self.m_id}"
                y_name = f"Y{self.y_id}"
                self.add_inst_node(
                    inst_type = InstType.REQ_MY,
                    qreg_names = [m_name, y_name],
                    creg_names_cond = [],
                    creg_name_out = ""
                )
                self.m_id += 1
                self.y_id += 1

                # Body
                ## CX(M, Y)
                self.add_inst_node(
                    inst_type = InstType.TRANS_CX,
                    qreg_names = [m_name, y_name],
                    creg_names_cond = [],
                    creg_name_out = ""
                )
                ## CX(M, Q)
                self.add_inst_node(
                    inst_type = InstType.TRANS_CX,
                    qreg_names = [m_name, q_name],
                    creg_names_cond = [],
                    creg_name_out = ""
                )
                ## DMEAS_Z(Q) -> C1
                c_name_1 = f"C{self.c_id}"
                self.add_inst_node(
                    inst_type = InstType.MEAS_Z,
                    qreg_names = [q_name],
                    creg_names_cond = [],
                    creg_name_out = c_name_1
                )
                self.c_id += 1

                ## C1 -> COND_Y(M)
                self.add_inst_node(
                    inst_type = InstType.PAULI_Y,
                    qreg_names = [m_name],
                    creg_names_cond = [c_name_1],
                    creg_name_out = ""
                )

                ## C1 -> DMEAS_XorZ(Y) -> C2
                c_name_2 = f"C{self.c_id}"
                self.add_inst_node(
                    inst_type = InstType.MEAS_XorZ,
                    qreg_names = [y_name],
                    creg_names_cond = [c_name_1],
                    creg_name_out = c_name_2
                )
                self.c_id += 1

                ## C2 -> Z(M)
                self.add_inst_node(
                    inst_type = InstType.PAULI_Z,
                    qreg_names = [m_name],
                    creg_names_cond = [c_name_2],
                    creg_name_out = ""
                )

                # qubit name change with teleportation
                o_name = self.current_to_origin[q_name]
                self.qubit_tracking[o_name] = m_name
                del self.current_to_origin[q_name]
                self.current_to_origin[m_name] = o_name

            elif self.s_trans_opt == STransOpt.TRANS_S:
                # request - m only
                m_name = f"M{self.m_id}"
                self.add_inst_node(
                    inst_type = InstType.REQ_M,
                    qreg_names = [m_name],
                    creg_names_cond = [],
                    creg_name_out = ""
                )
                self.m_id += 1
                
                # Body
                ## CX (M, Q)
                self.add_inst_node(
                    inst_type = InstType.TRANS_CX,
                    qreg_names = [m_name, q_name],
                    creg_names_cond = [],
                    creg_name_out = ""
                )

                ## DMEAS_Z(Q) -> C
                c_name = f"C{self.c_id}"
                self.add_inst_node(
                    inst_type = InstType.MEAS_Z,
                    qreg_names = [q_name],
                    creg_names_cond = [],
                    creg_name_out = c_name
                )
                self.c_id += 1

                ## C -> COND_S
                self.add_inst_node(
                    inst_type = InstType.TRANS_S,
                    qreg_names = [m_name],
                    creg_names_cond = [c_name],
                    creg_name_out = ""
                )

                ## C -> COND_X
                self.add_inst_node(
                    inst_type = InstType.PAULI_X,
                    qreg_names = [m_name],
                    creg_names_cond = [c_name],
                    creg_name_out = ""
                )
                # 
                # qubit name change with teleportation
                o_name = self.current_to_origin[q_name]
                self.qubit_tracking[o_name] = m_name
                del self.current_to_origin[q_name]
                self.current_to_origin[m_name] = o_name
            else:
                raise Exception()

        elif qc_name in teleport_gates and qc_name == 's': 
            [q_name] = qreg_names
            if self.s_trans_opt == STransOpt.GATE_TEL:
                # request - Y
                y_name = f"Y{self.y_id}"
                self.add_inst_node(
                    inst_type = InstType.REQ_Y,
                    qreg_names = [y_name],
                    creg_names_cond = [],
                    creg_name_out = ""
                )
                self.y_id += 1

                # Body
                ## CX (Y, Q)
                self.add_inst_node(
                    inst_type = InstType.TRANS_CX,
                    qreg_names = [y_name, q_name],
                    creg_names_cond = [],
                    creg_name_out = ""
                )

                ## DMEAS_Z (Q) -> C
                c_name = f"C{self.c_id}"
                self.add_inst_node(
                    inst_type = InstType.MEAS_Z,
                    qreg_names = [q_name],
                    creg_names_cond = [],
                    creg_name_out = c_name
                )
                self.c_id += 1

                ## C -> Z(Y)
                self.add_inst_node(
                    inst_type = InstType.PAULI_Z,
                    qreg_names = [y_name],
                    creg_names_cond = [c_name],
                    creg_name_out = ""
                )

                # qubit name change with teleportation
                o_name = self.current_to_origin[q_name]
                self.qubit_tracking[o_name] = y_name
                del self.current_to_origin[q_name]
                self.current_to_origin[y_name] = o_name
            ##    
            elif self.s_trans_opt == STransOpt.TRANS_S:
                self.add_inst_node(
                    inst_type = InstType.TRANS_S,
                    qreg_names = qreg_names,
                    creg_names_cond = [],
                    creg_name_out = ""
                )
            else:
                raise Exception()


        elif qc_label in custom_instruction_labels or qc_name == 'tdg':
            if qc_label == "plus_state_prepare":
                decomposed_qc_names = ["reset", "h"]
            elif qc_label == "Tdag_state_prepare":
                decomposed_qc_names = ["reset", "h", "z", "s", "t"]
            elif qc_label == "X-meas":
                decomposed_qc_names = ["h", "measure"]
            elif qc_label == "T_state_prepare":
                decomposed_qc_names = ["reset", "h", "t"]
            elif qc_label == "decomposed_Tdag" or qc_name == "tdg":
                decomposed_qc_names = ["z", "s", "t"]
            else:
                raise Exception(qc_name, qc_label)

            for decomposed_qc_name in decomposed_qc_names:
                new_op = Instruction(name=decomposed_qc_name, num_qubits=qc_node.op.num_qubits, num_clbits=qc_node.op.num_clbits, params=qc_node.op.params)
                new_op.name = decomposed_qc_name
                new_node = DAGOpNode(op=new_op, qargs=qc_node.qargs, cargs=qc_node.cargs)

                self.translate_node(new_node)

        elif qc_name in ["barrier", "if_else"]:
            pass

        else:
            raise Exception(qc_name)


    def count_rot(self):
        assert self.sc_dag

        if self.rot_trans_opt == RotTransOpt.BASE:
            indices = [i for i in self.sc_dag.node_indices()]
            rot_nodes = [self.sc_dag[i] for i in indices if self.sc_dag[i].inst_type == InstType.TRANS_H_ROT]
            rot_count = len(rot_nodes)
        elif self.rot_trans_opt in [RotTransOpt.ALL_ROT, RotTransOpt.NAIVE_SKIP, RotTransOpt.HEUR_ALLOC]:
            indices = [i for i in self.sc_dag.node_indices()]
            rot_nodes = [self.sc_dag[i] for i in indices if self.sc_dag[i].inst_type == InstType.ROTATION]
            rot_count = len(rot_nodes)
        else:
            raise Exception()

        return rot_count
