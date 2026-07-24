"""Count occupied cells per beat in a lattice surgery program.

Reads the same JSON format as visualize_computational_process.py.
For each beat (step), outputs the total number of unique cells occupied
by all instructions that are active at that beat.

Usage:
    python3 count_occupied_cells.py <json_file>
    python3 count_occupied_cells.py <json_file> --csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

Point = tuple[int, int]


class InstructionType(StrEnum):
    allocate = "ALLOCATE"
    allocate_2q = "ALLOCATE_2Q"
    allocate_magic_factory = "ALLOCATE_MAGIC_FACTORY"
    allocate_entanglement_factory = "ALLOCATE_ENTANGLEMENT_FACTORY"
    deallocate = "DEALLOCATE"
    init_zx = "INIT_ZX"
    meas_zx = "MEAS_ZX"
    meas_y = "MEAS_Y"
    twist = "TWIST"
    hadamard = "HADAMARD"
    rotate = "ROTATE"
    lattice_surgery = "LATTICE_SURGERY"
    lattice_surgery_magic = "LATTICE_SURGERY_MAGIC"
    lattice_surgery_multinode = "LATTICE_SURGERY_MULTINODE"
    move = "MOVE"
    move_magic = "MOVE_MAGIC"
    move_entanglement = "MOVE_ENTANGLEMENT"
    cnot = "CNOT"
    cnot_trans = "CNOT_TRANS"
    swap_trans = "SWAP_TRANS"
    move_trans = "MOVE_TRANS"
    xor = "XOR"
    and_ = "AND"
    or_ = "OR"
    probability_hint = "PROBABILITY_HINT"
    await_correction = "AWAIT_CORRECTION"


LATENCY_TBL: dict[InstructionType, int] = {
    InstructionType.allocate: 0,
    InstructionType.allocate_2q: 0,
    InstructionType.allocate_magic_factory: 0,
    InstructionType.allocate_entanglement_factory: 0,
    InstructionType.deallocate: 0,
    InstructionType.init_zx: 0,
    InstructionType.meas_zx: 0,
    InstructionType.meas_y: 2,
    InstructionType.twist: 2,
    InstructionType.hadamard: 0,
    InstructionType.rotate: 3,
    InstructionType.lattice_surgery: 1,
    InstructionType.lattice_surgery_magic: 1,
    InstructionType.lattice_surgery_multinode: 1,
    InstructionType.move: 1,
    InstructionType.move_magic: 1,
    InstructionType.move_entanglement: 1,
    InstructionType.cnot: 2,
    InstructionType.cnot_trans: 0,
    InstructionType.swap_trans: 0,
    InstructionType.move_trans: 0,
    InstructionType.xor: 0,
    InstructionType.and_: 0,
    InstructionType.or_: 0,
    InstructionType.probability_hint: 0,
    InstructionType.await_correction: 0,
}


@dataclass
class QubitInfo:
    id: int
    x: int
    y: int
    begin: int
    end: int


@dataclass
class FactoryInfo:
    id: int
    x: int
    y: int
    begin: int


class LightInstruction:
    """Minimal instruction representation for cell counting (no graphviz dependency)."""

    def __init__(self, index: int, inst: dict) -> None:
        self.index = index
        self._raw = inst
        try:
            self.type = InstructionType(inst["type"])
        except ValueError:
            self.type = None
        self.qtarget: list[int] = inst["qtarget"]
        self.mtarget: list[int] = inst["mtarget"]
        self.etarget: list[int] = inst.get("etarget", [])
        self.beat: int = inst["metadata"]["beat"]
        self.latency: int = LATENCY_TBL.get(self.type, 0) if self.type else 0
        self._ancilla_raw: list = inst.get("ancilla", [])
        self.aux: list[Point] = []

    def resolve_aux(self, qubits: dict[int, QubitInfo], factories: dict[int, FactoryInfo]) -> None:
        """Resolve aux cells from raw instruction data."""
        if self.type is None:
            return

        if self.type in {
            InstructionType.allocate,
            InstructionType.deallocate,
            InstructionType.init_zx,
            InstructionType.meas_zx,
            InstructionType.hadamard,
        }:
            q = qubits.get(self.qtarget[0])
            if q:
                self.aux = [(q.x, q.y)]

        elif self.type == InstructionType.allocate_2q:
            q0 = qubits.get(self.qtarget[0])
            q1 = qubits.get(self.qtarget[1])
            if q0 and q1:
                self.aux = [(q0.x, q0.y), (q1.x, q1.y)]

        elif self.type == InstructionType.allocate_magic_factory:
            f = factories.get(self.mtarget[0])
            if f:
                self.aux = [(f.x, f.y)]

        elif self.type == InstructionType.allocate_entanglement_factory:
            d1 = self._raw["dest1"]
            d2 = self._raw["dest2"]
            self.aux = [(d1[0], d1[1]), (d2[0], d2[1])]

        elif self.type == InstructionType.meas_y:
            q = qubits.get(self.qtarget[0])
            if not q:
                return
            d = self._raw["dir"]
            if d == 0:
                self.aux = [(q.x, q.y), (q.x + 1, q.y), (q.x + 1, q.y + 1), (q.x, q.y + 1)]
            elif d == 1:
                self.aux = [(q.x, q.y), (q.x, q.y + 1), (q.x - 1, q.y + 1), (q.x - 1, q.y)]
            elif d == 2:
                self.aux = [(q.x, q.y), (q.x + 1, q.y), (q.x + 1, q.y - 1), (q.x, q.y - 1)]
            elif d == 3:
                self.aux = [(q.x, q.y), (q.x, q.y - 1), (q.x - 1, q.y - 1), (q.x - 1, q.y)]

        elif self.type in {InstructionType.twist, InstructionType.rotate}:
            q = qubits.get(self.qtarget[0])
            if not q:
                return
            d = self._raw["dir"]
            if d == 0:
                self.aux = [(q.x, q.y), (q.x + 1, q.y)]
            elif d == 1:
                self.aux = [(q.x, q.y), (q.x, q.y + 1)]
            elif d == 2:
                self.aux = [(q.x, q.y), (q.x - 1, q.y)]
            elif d == 3:
                self.aux = [(q.x, q.y), (q.x, q.y - 1)]

        elif self.type in {InstructionType.lattice_surgery, InstructionType.cnot}:
            qs = [qubits[q] for q in self.qtarget if q in qubits]
            terminal = [(q.x, q.y) for q in qs]
            self.aux = terminal + [(p[0], p[1]) for p in self._ancilla_raw]

        elif self.type == InstructionType.lattice_surgery_magic:
            qs = [qubits[q] for q in self.qtarget if q in qubits]
            terminal = [(q.x, q.y) for q in qs]
            if self.mtarget:
                f = factories.get(self.mtarget[0])
                if f:
                    terminal.append((f.x, f.y))
            self.aux = terminal + [(p[0], p[1]) for p in self._ancilla_raw]

        elif self.type == InstructionType.lattice_surgery_multinode:
            qs = [qubits[q] for q in self.qtarget if q in qubits]
            terminal = [(q.x, q.y) for q in qs]
            if self.mtarget:
                f = factories.get(self.mtarget[0])
                if f:
                    terminal.append((f.x, f.y))
            self.aux = terminal + [(p[0], p[1]) for p in self._ancilla_raw]

        elif self.type == InstructionType.move:
            q = qubits.get(self.qtarget[0])
            if not q:
                return
            d = self._raw["dest"]
            terminal = [(q.x, q.y), (d[0], d[1])]
            self.aux = terminal + [(p[0], p[1]) for p in self._ancilla_raw]

        elif self.type == InstructionType.move_magic:
            q = qubits.get(self.qtarget[0])
            f = factories.get(self.mtarget[0]) if self.mtarget else None
            if q and f:
                terminal = [(q.x, q.y), (f.x, f.y)]
                self.aux = terminal + [(p[0], p[1]) for p in self._ancilla_raw]

        elif self.type == InstructionType.move_entanglement:
            q = qubits.get(self.qtarget[0])
            if q:
                terminal = [(q.x, q.y)]
                self.aux = terminal + [(p[0], p[1]) for p in self._ancilla_raw]

        elif self.type in {
            InstructionType.cnot_trans,
            InstructionType.swap_trans,
            InstructionType.move_trans,
        }:
            qs = [qubits[q] for q in self.qtarget if q in qubits]
            terminal = [(q.x, q.y) for q in qs]
            self.aux = terminal + [(p[0], p[1]) for p in self._ancilla_raw]


def parse_program(
    raw_insts: list[dict],
) -> tuple[list[LightInstruction], dict[int, QubitInfo], dict[int, FactoryInfo], int, int]:
    """Parse JSON program into instructions with resolved aux cells.

    Returns (instructions, qubits, factories, begin_beat, end_beat).
    """
    # Build instructions, skipping unknown types
    insts: list[LightInstruction] = []
    for i, raw in enumerate(raw_insts):
        inst = LightInstruction(i, raw)
        if inst.type is not None:
            insts.append(inst)

    if not insts:
        return [], {}, {}, 0, 0

    begin_beat = min(inst.beat for inst in insts)
    end_beat = max(inst.beat for inst in insts) + 1

    # Build qubit and factory maps (same logic as Circuit.__init__)
    qubits: dict[int, QubitInfo] = {}
    factories: dict[int, FactoryInfo] = {}
    for inst in insts:
        if inst.type == InstructionType.allocate:
            qid = inst.qtarget[0]
            x = inst._raw["dest"][0]
            y = inst._raw["dest"][1]
            qubits[qid] = QubitInfo(qid, x, y, inst.beat, end_beat)
        elif inst.type == InstructionType.allocate_magic_factory:
            fid = inst.mtarget[0]
            x = inst._raw["dest"][0]
            y = inst._raw["dest"][1]
            factories[fid] = FactoryInfo(fid, x, y, inst.beat)
        elif inst.type == InstructionType.deallocate:
            qid = inst.qtarget[0]
            if qid in qubits:
                qubits[qid].end = inst.beat

    # Resolve aux cells for each instruction
    for inst in insts:
        inst.resolve_aux(qubits, factories)

    return insts, qubits, factories, begin_beat, end_beat


@dataclass
class BeatStats:
    living_data_cells: int
    living_factories: int
    occupied_cells: int


def compute_stats_per_beat(
    insts: list[LightInstruction],
    qubits: dict[int, QubitInfo],
    factories: dict[int, FactoryInfo],
    begin_beat: int,
    end_beat: int,
) -> dict[int, BeatStats]:
    """Compute per-beat statistics.

    For each beat:
    - living_data_cells: number of qubits with begin <= beat < end
    - living_factories: number of factories with begin <= beat
    - occupied_cells: unique cells occupied by active instructions,
      EXCLUDING qtarget positions (qubit/factory cells referenced by qtarget)

    An instruction is "active" at a beat if:
        inst.beat <= beat < inst.beat + max(inst.latency, 1)
    """
    # Pre-compute occupied cells (excluding qtarget) per beat
    beat_occupied: dict[int, set[Point]] = defaultdict(set)
    for inst in insts:
        cells = inst.aux
        if not cells:
            continue
        start = inst.beat
        end = inst.beat + max(inst.latency, 1)
        # Compute qtarget positions to exclude
        qtarget_positions: set[Point] = set()
        for qid in inst.qtarget:
            q = qubits.get(qid)
            if q:
                qtarget_positions.add((q.x, q.y))
        # Cells excluding qtarget
        non_qtarget_cells = set(cells) - qtarget_positions
        for beat in range(start, end):
            beat_occupied[beat].update(non_qtarget_cells)

    # Compute all stats for each beat
    results: dict[int, BeatStats] = {}
    for beat in range(begin_beat, end_beat):
        living_data = sum(
            1 for q in qubits.values() if q.begin <= beat < q.end
        )
        living_fac = sum(
            1 for f in factories.values() if f.begin <= beat
        )
        occ = len(beat_occupied.get(beat, set()))
        results[beat] = BeatStats(living_data, living_fac, occ)

    return results


def parse_ml_json(json_path: str) -> dict:
    with open(json_path) as f:
        raw = json.load(f)

    insts, qubits, factories, begin_beat, end_beat = parse_program(raw["program"])

    [chip_width, chip_height, chip_depth] = raw["parameter"]["target"]["topology"][0]["coord"]

    if not insts:
        print("No instructions found.", file=sys.stderr)
        return

    stats = compute_stats_per_beat(insts, qubits, factories, begin_beat, end_beat)

    qubit_idle_counts = 0
    occupied_cell_counts = 0

    total_beats = end_beat - begin_beat
    for beat in range(begin_beat, end_beat):
        s = stats[beat]
        qubit_idle_counts += s.living_data_cells
        occupied_cell_counts += s.occupied_cells

    summary_dict = {}
    summary_dict['ham_name'] = None
    summary_dict['beats'] = total_beats
    summary_dict['idle_counts'] = qubit_idle_counts
    summary_dict['occupied_cell_counts'] = occupied_cell_counts
    summary_dict['space_time_volume'] = qubit_idle_counts + occupied_cell_counts
    summary_dict['chip_width'] = chip_width
    summary_dict['chip_height'] = chip_height
    summary_dict['chip_depth'] = chip_depth

    d = 15
    code_cycle_s = 0.00025
    print(f"Total Beats: {total_beats}")
    print(f"Execution time (sec): {total_beats*d*code_cycle_s}")
    print(f"Total idle counts: {qubit_idle_counts}")
    print(f"Occupied cell counts: {occupied_cell_counts}")
    print(f"Average idle cell: {qubit_idle_counts/total_beats}")
    print(f"Total space-time volume: {qubit_idle_counts + occupied_cell_counts}")
    print(f"Total physical_qubits: {(2*d*d-1) * chip_height* chip_width}")

    return summary_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count occupied cells per beat in a lattice surgery program."
    )
    parser.add_argument("json_file", help="Path to the JSON file (same format as visualizer).")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output in CSV format.",
    )
    args = parser.parse_args()

    with open(args.json_file) as f:
        raw = json.load(f)

    insts, qubits, factories, begin_beat, end_beat = parse_program(raw["program"])
    [chip_width, chip_height, chip_depth] = raw["parameter"]["target"]["topology"][0]["coord"]

    if not insts:
        print("No instructions found.", file=sys.stderr)
        return

    stats = compute_stats_per_beat(insts, qubits, factories, begin_beat, end_beat)

    qubit_idle_counts = 0
    occupied_cell_counts = 0

    total_beats = end_beat - begin_beat
    for beat in range(begin_beat, end_beat):
        s = stats[beat]
        qubit_idle_counts += s.living_data_cells
        occupied_cell_counts += s.occupied_cells

    d = 15
    code_cycle_s = 0.00025
    print(f"Total Beats: {total_beats}")
    print(f"Execution time (sec): {total_beats*d*code_cycle_s}")
    print(f"Total idle counts: {qubit_idle_counts}")
    print(f"Occupied cell counts: {occupied_cell_counts}")
    print(f"Average idle cell: {qubit_idle_counts/total_beats}")
    print(f"Total space-time volume: {qubit_idle_counts + occupied_cell_counts}")
    print(f"Physical_qubits per cell: {(2*d*d-1)}")
    print(f"Total physical_qubits: {(2*d*d-1) * chip_height* chip_width}")


if __name__ == "__main__":
    main()
