import math
import random
import sys

import numpy as np

from macro import *
from tsc_instructions import *

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


TERM_Q = 0
TERM_M = 1
TERM_Y = 2

MOVE_STA = 0
MOVE_CJ = 1
MOVE_CV = 2


def _distance_value(r1, c1, r2, c2, move_type_code):
    r_diff = abs(r1 - r2)
    c_diff = abs(c1 - c2)
    if move_type_code == MOVE_STA:
        return r_diff ** (1 / 3) + c_diff ** (1 / 3)
    if move_type_code == MOVE_CJ:
        return r_diff ** (1 / 2) + c_diff ** (1 / 3)
    if move_type_code == MOVE_CV:
        return r_diff + c_diff
    raise ValueError("unknown move_type_code")


@njit(cache=True)
def _add_affected_terms(
    qubit,
    affected_offsets,
    affected_indices,
    term_seen,
    token,
    affected_buffer,
    count,
):
    for idx in range(affected_offsets[qubit], affected_offsets[qubit + 1]):
        term_id = affected_indices[idx]
        if term_seen[term_id] == token:
            continue
        term_seen[term_id] = token
        affected_buffer[count] = term_id
        count += 1
    return count


@njit(cache=True)
def _term_cost_sum(
    count,
    affected_buffer,
    current_mapping,
    term_kind,
    term_q0,
    term_q1,
    term_weight,
    distance_cache,
):
    cost = 0.0
    for idx in range(count):
        term_id = affected_buffer[idx]
        q0 = term_q0[term_id]
        r0 = current_mapping[q0, 0]
        c0 = current_mapping[q0, 1]
        kind = term_kind[term_id]
        if kind == TERM_Q:
            q1 = term_q1[term_id]
            r1 = current_mapping[q1, 0]
            c1 = current_mapping[q1, 1]
        elif kind == TERM_M:
            r1 = 0
            c1 = c0
        elif kind == TERM_Y:
            r1 = 1
            c1 = c0
        else:
            raise ValueError("unknown term kind")
        cost += term_weight[term_id] * distance_cache[r0, c0, r1, c1]
    return cost


@njit(cache=True)
def _apply_move(
    qubit_to_move,
    move_candidate_index,
    current_mapping,
    physical_to_program,
    pos_to_index,
    move_candidate_coords,
    affected_offsets,
    affected_indices,
    term_seen,
    token,
    affected_buffer,
    term_kind,
    term_q0,
    term_q1,
    term_weight,
    distance_cache,
):
    old_r = current_mapping[qubit_to_move, 0]
    old_c = current_mapping[qubit_to_move, 1]
    old_pos_idx = pos_to_index[old_r, old_c]
    new_r = move_candidate_coords[old_pos_idx, move_candidate_index, 0]
    new_c = move_candidate_coords[old_pos_idx, move_candidate_index, 1]
    qubit_be_affected = physical_to_program[new_r, new_c]

    count = 0
    count = _add_affected_terms(
        qubit_to_move,
        affected_offsets,
        affected_indices,
        term_seen,
        token,
        affected_buffer,
        count,
    )
    if qubit_be_affected > -1 and qubit_be_affected != qubit_to_move:
        count = _add_affected_terms(
            qubit_be_affected,
            affected_offsets,
            affected_indices,
            term_seen,
            token,
            affected_buffer,
            count,
        )

    ori_cost = _term_cost_sum(
        count,
        affected_buffer,
        current_mapping,
        term_kind,
        term_q0,
        term_q1,
        term_weight,
        distance_cache,
    )

    current_mapping[qubit_to_move, 0] = new_r
    current_mapping[qubit_to_move, 1] = new_c
    physical_to_program[new_r, new_c] = qubit_to_move
    physical_to_program[old_r, old_c] = qubit_be_affected
    if qubit_be_affected > -1:
        current_mapping[qubit_be_affected, 0] = old_r
        current_mapping[qubit_be_affected, 1] = old_c

    new_cost = _term_cost_sum(
        count,
        affected_buffer,
        current_mapping,
        term_kind,
        term_q0,
        term_q1,
        term_weight,
        distance_cache,
    )

    return new_cost - ori_cost, qubit_be_affected, old_r, old_c, new_r, new_c


@njit(cache=True)
def _recover_move(
    qubit_to_move,
    qubit_be_affected,
    old_r,
    old_c,
    new_r,
    new_c,
    current_mapping,
    physical_to_program,
):
    current_mapping[qubit_to_move, 0] = old_r
    current_mapping[qubit_to_move, 1] = old_c
    physical_to_program[old_r, old_c] = qubit_to_move
    physical_to_program[new_r, new_c] = qubit_be_affected
    if qubit_be_affected > -1:
        current_mapping[qubit_be_affected, 0] = new_r
        current_mapping[qubit_be_affected, 1] = new_c


def _sa_kernel_impl(
    initial_mapping,
    initial_cost,
    h,
    w,
    pos_to_index,
    move_candidate_coords,
    term_kind,
    term_q0,
    term_q1,
    term_weight,
    affected_offsets,
    affected_indices,
    distance_cache,
    move_qubits,
    move_candidate_indices,
    accept_uniforms,
    sa_l,
    sa_iter_limit,
    sa_init_perturb_num,
    sa_t1_initial,
    sa_t_frozen,
    sa_p,
    sa_k,
    sa_c,
):
    current_mapping = initial_mapping.copy()
    best_mapping = initial_mapping.copy()
    current_cost = initial_cost
    best_cost = initial_cost
    sa_t1 = sa_t1_initial

    physical_to_program = np.empty((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            physical_to_program[r, c] = -1
    for q in range(current_mapping.shape[0]):
        physical_to_program[current_mapping[q, 0], current_mapping[q, 1]] = q

    term_seen = np.zeros(term_kind.shape[0], dtype=np.int64)
    affected_buffer = np.empty(term_kind.shape[0], dtype=np.int64)
    token = 0
    status = 0

    uphill_sum = 0.0
    uphill_cnt = 0
    move_idx = 0

    for _ in range(sa_init_perturb_num):
        token += 1
        delta, _, _, _, _, _ = _apply_move(
            move_qubits[move_idx],
            move_candidate_indices[move_idx],
            current_mapping,
            physical_to_program,
            pos_to_index,
            move_candidate_coords,
            affected_offsets,
            affected_indices,
            term_seen,
            token,
            affected_buffer,
            term_kind,
            term_q0,
            term_q1,
            term_weight,
            distance_cache,
        )
        move_idx += 1
        current_cost += delta
        if current_cost < -1e-9:
            status = -1
            return (
                best_mapping,
                current_mapping,
                current_mapping.copy(),
                best_cost,
                current_cost,
                sa_t1,
                sa_t1,
                0,
                status,
            )
        if best_cost - current_cost > 1e-9:
            best_cost = current_cost
            best_mapping[:, :] = current_mapping[:, :]
        if delta > 0:
            uphill_sum += delta
            uphill_cnt += 1

    if uphill_cnt > 0:
        sa_t1 = (uphill_sum / uphill_cnt) / ((-1) * math.log(sa_p))
    sa_t = sa_t1
    random_mapping = current_mapping.copy()

    sa_n = 0
    while sa_t > sa_t_frozen:
        sa_n += 1
        sa_delta_cost_cnt = 0
        sa_delta_sum = 0.0

        for _ in range(sa_l):
            token += 1
            delta, qubit_be_affected, old_r, old_c, new_r, new_c = _apply_move(
                move_qubits[move_idx],
                move_candidate_indices[move_idx],
                current_mapping,
                physical_to_program,
                pos_to_index,
                move_candidate_coords,
                affected_offsets,
                affected_indices,
                term_seen,
                token,
                affected_buffer,
                term_kind,
                term_q0,
                term_q1,
                term_weight,
                distance_cache,
            )
            accept_value = accept_uniforms[move_idx - sa_init_perturb_num]
            move_idx += 1

            sa_delta_cost_cnt += 1
            sa_delta_sum += abs(delta)
            if delta <= 0:
                current_cost += delta
                if current_cost < -1e-9:
                    status = -1
                    return (
                        best_mapping,
                        current_mapping,
                        random_mapping,
                        best_cost,
                        current_cost,
                        sa_t,
                        sa_t1,
                        sa_n,
                        status,
                    )
                if best_cost - current_cost > 1e-9:
                    best_cost = current_cost
                    best_mapping[:, :] = current_mapping[:, :]
            else:
                accept = accept_value <= math.exp(-(delta) / sa_t)
                if accept:
                    current_cost += delta
                else:
                    _recover_move(
                        move_qubits[move_idx - 1],
                        qubit_be_affected,
                        old_r,
                        old_c,
                        new_r,
                        new_c,
                        current_mapping,
                        physical_to_program,
                    )

        if sa_n <= sa_k:
            sa_t = (sa_t1 * abs(sa_delta_sum) / sa_delta_cost_cnt) / sa_n / sa_c
        else:
            sa_t = (sa_t1 * abs(sa_delta_sum) / sa_delta_cost_cnt) / sa_n
        if sa_n > sa_iter_limit:
            break

    return (
        best_mapping,
        current_mapping,
        random_mapping,
        best_cost,
        current_cost,
        sa_t,
        sa_t1,
        sa_n,
        status,
    )


_sa_kernel_jit = njit(cache=True)(_sa_kernel_impl) if NUMBA_AVAILABLE else None


class sa_mapper:
    def __init__(
        self,
        num_lq,
        inst_schedule,
        plane_char,
        use_jit=True,
    ):
        self.move_type = MoveType.STA
        self.use_jit = use_jit
        self.initialize_param()
        self.n_qubit = num_lq
        self.inst_schedule = inst_schedule
        self.plane_char = plane_char
        self.can_overlap = True
        self.cannot_overlap = False
        self.list_gate = []
        self.list_qubit_dict_gate = dict()
        self.possible_positions = []
        self.possible_positions_np = None
        self.pos_to_index = None
        self.move_candidate_coords = None
        self.distance_cache = None
        self.term_kind = []
        self.term_q0 = []
        self.term_q1 = []
        self.term_weight = []
        self.affected_term_ids = []
        self.affected_offsets = None
        self.affected_indices = None
        self.sa_mapping = None
        self.random_mapping = None

    def initialize_param(self):
        self.sa_t = 100000.0
        self.sa_t1 = 4.0
        self.sa_t_frozen = 0.000001
        self.sa_p = 0.987
        self.sa_l = 400
        self.sa_n = 0
        self.sa_k = 7
        self.sa_c = 100
        self.sa_iter_limit = 1000

        self.sa_init_perturb_num = 100
        self.sa_uphill_avg_cnt = 0
        self.sa_uphill_sum = 0
        self.sa_delta_cost_cnt = 0
        self.sa_delta_sum = 0
        self.sa_delta = 0
        self.sa_n_trials = 1

        self.best_mapping = None
        self.best_cost = sys.maxsize
        self.current_mapping = None
        self.current_mapping_physical_to_program = None
        self.current_cost = sys.maxsize
        self.current_violation = 0
        self.tmp_violation = 0

    def preprocessing(self):
        self.list_gate = []
        self.list_qubit_dict_gate = dict()
        for insts in self.inst_schedule:
            list_cx = [
                inst.oreg_names
                for inst in insts
                if inst.inst_type == InstType.TRANS_CX
            ]
            if not list_cx:
                continue
            self.list_gate.append(list_cx)

        max_level = 10
        if self.can_overlap:
            self.list_weight = [1 - 0.1 * l for l in range(max_level)]
        elif self.cannot_overlap:
            self.list_weight = [1 for _ in range(max_level)]
        else:
            raise Exception()

        for i, gates in enumerate(self.list_gate):
            if i < max_level:
                weight = self.list_weight[i]
            else:
                weight = self.list_weight[-1]
            for gate in gates:
                if not any(["Q" in o_name for o_name in gate]):
                    continue
                if gate[0] in self.list_qubit_dict_gate:
                    if gate[1] in self.list_qubit_dict_gate[gate[0]]:
                        self.list_qubit_dict_gate[gate[0]][gate[1]] += weight
                    else:
                        self.list_qubit_dict_gate[gate[0]][gate[1]] = weight
                else:
                    self.list_qubit_dict_gate[gate[0]] = dict()
                    self.list_qubit_dict_gate[gate[0]][gate[1]] = weight

                if gate[1] not in self.list_qubit_dict_gate:
                    self.list_qubit_dict_gate[gate[1]] = dict()
                self.list_qubit_dict_gate[gate[1]][gate[0]] = self.list_qubit_dict_gate[gate[0]][gate[1]]

        self.prepare_plane_helpers()
        self.prepare_cost_terms()

    def parse_qubit_id(self, name):
        if isinstance(name, str) and name.startswith("Q"):
            digits = []
            for char in name[1:]:
                if not char.isdigit():
                    break
                digits.append(char)
            if digits:
                return int("".join(digits))
        return None

    def move_type_code(self):
        if self.move_type == MoveType.STA:
            return MOVE_STA
        if self.move_type == MoveType.CJ:
            return MOVE_CJ
        if self.move_type == MoveType.CV:
            return MOVE_CV
        raise Exception()

    def prepare_plane_helpers(self):
        h, w = len(self.plane_char), len(self.plane_char[0])
        self.possible_positions = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if self.plane_char[r][c] == "N"
        ]
        self.possible_positions_np = np.asarray(self.possible_positions, dtype=np.int64)

        self.pos_to_index = np.full((h, w), -1, dtype=np.int64)
        for idx, (r, c) in enumerate(self.possible_positions):
            self.pos_to_index[r, c] = idx

        num_candidates = 120
        self.move_candidate_coords = np.empty(
            (len(self.possible_positions), num_candidates, 2),
            dtype=np.int64,
        )
        for pos_idx, (old_r, old_c) in enumerate(self.possible_positions):
            cand_idx = 0
            for move_disr in range(-5, 6):
                for move_disc in range(-5, 6):
                    if move_disr == 0 and move_disc == 0:
                        continue
                    new_r = max(2, min(old_r + move_disr, h - 1))
                    new_c = max(0, min(old_c + move_disc, w - 1))
                    self.move_candidate_coords[pos_idx, cand_idx, 0] = new_r
                    self.move_candidate_coords[pos_idx, cand_idx, 1] = new_c
                    cand_idx += 1
            assert cand_idx == num_candidates

        self.distance_cache = np.empty((h, w, h, w), dtype=np.float64)
        move_type_code = self.move_type_code()
        for r1 in range(h):
            for c1 in range(w):
                for r2 in range(h):
                    for c2 in range(w):
                        self.distance_cache[r1, c1, r2, c2] = _distance_value(
                            r1,
                            c1,
                            r2,
                            c2,
                            move_type_code,
                        )

    def prepare_cost_terms(self):
        pair_weights = dict()
        port_m_weights = [0.0 for _ in range(self.n_qubit)]
        port_y_weights = [0.0 for _ in range(self.n_qubit)]

        for q_name, related_gates in self.list_qubit_dict_gate.items():
            q0 = self.parse_qubit_id(q_name)
            if q0 is None or q0 >= self.n_qubit:
                continue
            for target_name, weight in related_gates.items():
                q1 = self.parse_qubit_id(target_name)
                if q1 is not None:
                    if q1 >= self.n_qubit or q0 == q1:
                        continue
                    key = (q0, q1) if q0 < q1 else (q1, q0)
                    pair_weights[key] = weight
                elif "M" in target_name:
                    port_m_weights[q0] += weight
                elif "Y" in target_name:
                    port_y_weights[q0] += weight

        term_kind = []
        term_q0 = []
        term_q1 = []
        term_weight = []
        affected_term_ids = [[] for _ in range(self.n_qubit)]

        def add_term(kind, q0, q1, weight):
            if weight == 0:
                return
            term_id = len(term_kind)
            term_kind.append(kind)
            term_q0.append(q0)
            term_q1.append(-1 if q1 is None else q1)
            term_weight.append(weight)
            affected_term_ids[q0].append(term_id)
            if kind == TERM_Q:
                affected_term_ids[q1].append(term_id)

        for (q0, q1), weight in pair_weights.items():
            add_term(TERM_Q, q0, q1, weight)
        for q0, weight in enumerate(port_m_weights):
            add_term(TERM_M, q0, None, weight)
        for q0, weight in enumerate(port_y_weights):
            add_term(TERM_Y, q0, None, weight)

        self.term_kind = np.asarray(term_kind, dtype=np.int64)
        self.term_q0 = np.asarray(term_q0, dtype=np.int64)
        self.term_q1 = np.asarray(term_q1, dtype=np.int64)
        self.term_weight = np.asarray(term_weight, dtype=np.float64)
        self.affected_term_ids = affected_term_ids

        offsets = np.zeros(self.n_qubit + 1, dtype=np.int64)
        total = 0
        for q, term_ids in enumerate(affected_term_ids):
            offsets[q] = total
            total += len(term_ids)
        offsets[self.n_qubit] = total
        indices = np.empty(total, dtype=np.int64)
        cursor = 0
        for term_ids in affected_term_ids:
            for term_id in term_ids:
                indices[cursor] = term_id
                cursor += 1
        self.affected_offsets = offsets
        self.affected_indices = indices

    def run(self):
        print("[INFO] JIT SA-based placement")
        self.initialize_param()
        self.preprocessing()

        for trial in range(self.sa_n_trials):
            rng = random.Random(0)
            initial_mapping, initial_cost = self.init_sa_solution_arrays(rng)
            print(
                "[INFO] JIT SA-based placement: Iter {}, intial cost: {:4f}".format(
                    trial,
                    initial_cost,
                )
            )
            move_qubits, move_candidate_indices, accept_uniforms = self.precompute_rng(rng)

            kernel = _sa_kernel_jit if self.use_jit and NUMBA_AVAILABLE else _sa_kernel_impl
            (
                best_mapping,
                current_mapping,
                random_mapping,
                best_cost,
                current_cost,
                self.sa_t,
                self.sa_t1,
                self.sa_n,
                status,
            ) = kernel(
                initial_mapping,
                initial_cost,
                len(self.plane_char),
                len(self.plane_char[0]),
                self.pos_to_index,
                self.move_candidate_coords,
                self.term_kind,
                self.term_q0,
                self.term_q1,
                self.term_weight,
                self.affected_offsets,
                self.affected_indices,
                self.distance_cache,
                move_qubits,
                move_candidate_indices,
                accept_uniforms,
                self.sa_l,
                self.sa_iter_limit,
                self.sa_init_perturb_num,
                4.0,
                self.sa_t_frozen,
                self.sa_p,
                self.sa_k,
                self.sa_c,
            )
            if status != 0:
                raise RuntimeError("SA mapper produced a negative current_cost.")

            self.best_mapping = self.mapping_array_to_list(best_mapping)
            self.current_mapping = self.mapping_array_to_list(current_mapping)
            self.random_mapping = self.mapping_array_to_dict(random_mapping)
            self.best_cost = float(best_cost)
            self.current_cost = float(current_cost)
            print(
                "[INFO] JIT SA-based placement: Iter{}, cost: {:4f}".format(
                    trial,
                    self.best_cost,
                )
            )

        assert len(self.best_mapping) == self.n_qubit
        self.sa_mapping = self.mapping_list_to_dict(self.best_mapping)

    def init_sa_solution_arrays(self, rng):
        list_possible_position = list(self.possible_positions)
        list_shuffle_position = list(list_possible_position)
        rng.shuffle(list_shuffle_position)
        list_shuffle_position2 = list(list_shuffle_position)
        rng.shuffle(list_shuffle_position2)

        candidates = [
            (self.get_cost(list_possible_position), list_possible_position),
            (self.get_cost(list_shuffle_position), list_shuffle_position),
            (self.get_cost(list_shuffle_position2), list_shuffle_position2),
        ]
        current_cost, selected_positions = min(candidates, key=lambda item: item[0])

        mapping = np.empty((self.n_qubit, 2), dtype=np.int64)
        for i, pos in enumerate(selected_positions[: self.n_qubit]):
            mapping[i, 0] = pos[0]
            mapping[i, 1] = pos[1]

        return mapping, float(current_cost)

    def precompute_rng(self, rng):
        main_move_num = self.sa_l * (self.sa_iter_limit + 1)
        total_move_num = self.sa_init_perturb_num + main_move_num
        move_qubits = np.empty(total_move_num, dtype=np.int64)
        move_candidate_indices = np.empty(total_move_num, dtype=np.int64)
        accept_uniforms = np.empty(main_move_num, dtype=np.float64)

        for idx in range(self.sa_init_perturb_num):
            move_qubits[idx] = rng.randrange(self.n_qubit)
            move_candidate_indices[idx] = rng.randrange(self.move_candidate_coords.shape[1])

        for idx in range(main_move_num):
            move_idx = self.sa_init_perturb_num + idx
            move_qubits[move_idx] = rng.randrange(self.n_qubit)
            move_candidate_indices[move_idx] = rng.randrange(self.move_candidate_coords.shape[1])
            accept_uniforms[idx] = rng.uniform(0, 1)

        return move_qubits, move_candidate_indices, accept_uniforms

    def get_cost(self, mapping):
        mapping_array = self.mapping_to_array(mapping)
        cost = 0.0
        for term_id, kind in enumerate(self.term_kind):
            q0 = self.term_q0[term_id]
            r0, c0 = mapping_array[q0]
            if kind == TERM_Q:
                r1, c1 = mapping_array[self.term_q1[term_id]]
            elif kind == TERM_M:
                r1, c1 = 0, c0
            elif kind == TERM_Y:
                r1, c1 = 1, c0
            else:
                raise Exception()
            cost += self.term_weight[term_id] * self.distance_cache[r0, c0, r1, c1]
        return cost

    def distance_value(self, r1, c1, r2, c2):
        return _distance_value(r1, c1, r2, c2, self.move_type_code())

    def distance(self, pos1, pos2):
        (r1, c1), (r2, c2) = pos1, pos2
        if self.distance_cache is not None:
            return self.distance_cache[r1, c1, r2, c2]
        return self.distance_value(r1, c1, r2, c2)

    def mapping_to_array(self, mapping):
        if isinstance(mapping, np.ndarray):
            return mapping
        arr = np.empty((self.n_qubit, 2), dtype=np.int64)
        for i in range(self.n_qubit):
            r, c = mapping[i]
            arr[i, 0] = r
            arr[i, 1] = c
        return arr

    def mapping_array_to_list(self, mapping):
        return [(int(mapping[i, 0]), int(mapping[i, 1])) for i in range(mapping.shape[0])]

    def mapping_array_to_dict(self, mapping):
        return {
            f"Q{i}": (int(mapping[i, 0]), int(mapping[i, 1]))
            for i in range(mapping.shape[0])
        }

    def mapping_list_to_dict(self, mapping):
        return {f"Q{i}": pos for i, pos in enumerate(mapping)}


sa_mapper_jit = sa_mapper
