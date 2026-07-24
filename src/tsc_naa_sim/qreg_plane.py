from macro import *
from copy import deepcopy
from enum import Enum, auto
import random


_IMMUTABLE_TYPES = (int, float, str, bool, type(None), Enum)


def _is_fast_copy_value(value):
    if isinstance(value, _IMMUTABLE_TYPES):
        return True
    if isinstance(value, tuple):
        return all(_is_fast_copy_value(item) for item in value)
    return False


def _copy_value(value, memo):
    if _is_fast_copy_value(value):
        return value
    if isinstance(value, list):
        if all(_is_fast_copy_value(item) for item in value):
            return list(value)
        return deepcopy(value, memo)
    if isinstance(value, dict):
        if all(
            _is_fast_copy_value(k)
            and isinstance(v, list)
            and all(_is_fast_copy_value(item) for item in v)
            for k, v in value.items()
        ):
            return {k: list(v) for k, v in value.items()}
        return deepcopy(value, memo)
    return deepcopy(value, memo)


def _copy_attrs(src, dst, memo):
    for key, value in src.__dict__.items():
        setattr(dst, key, _copy_value(value, memo))


class Cell:
    def __init__(self, pos, type, status_q, status_op):
        self.pos = pos
        self.type = type
        self.status_q = status_q
        self.status_op = status_op
        #
        self.occupants = []
        self.lock_dict = dict()

        self.route_dir = []

    def init_tnorm(self):
        self.enter_tnorm_c = None
        self.exit_tnorm_c = None
        self.enter_tnorm_e = None
        self.exit_tnorm_e = None
        self.enter_tnorm_s = None
        self.exit_tnorm_s = None
        self.enter_tnorm_se = None
        self.exit_tnorm_se = None
        return

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        _copy_attrs(self, copied, memo)
        return copied

class QubitPlane:
    def __init__(self,
                 width,
                 height,
                 field,
                 lattice_bound,
                 cell_size,
                 plane_type
                 ):
        self.w = width
        self.h = height
        self.field = field
        self.lattice_bound = lattice_bound
        self.cell_size = cell_size
        self.plane_type = plane_type

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            if key == "field":
                copied.field = [[deepcopy(cell, memo) for cell in row] for row in value]
            else:
                setattr(copied, key, _copy_value(value, memo))
        return copied

    def get_free_pos(self, cell_type):
        ret = []
        for r in range(self.h):
            for c in range(self.w):
                cell = self.field[r][c]
                if cell.type == cell_type and cell.status_q == CellStatusQ.FREE:
                    ret.append(cell.pos)
        return ret

    def get_occ_pos(self, cell_type):
        ret = []
        for r in range(self.h):
            for c in range(self.w):
                cell = self.field[r][c]
                if cell.type == cell_type and cell.status_q == CellStatusQ.OCCUPIED_Q:
                    ret.append(cell.pos)
        return ret

    def reset_status_op(self):
        for r in range(self.h):
            for c in range(self.w):
                cell = self.field[r][c]
                cell.status_op = CellStatusOp.IDLE
                cell.route_dir = []
                cell.is_dst = []
                cell.is_src = []
                cell.pass_type = []
        return

# Quantum regsiter
class QregStatus(Enum):
    INACTIVE = auto()
    ACTIVE_A = auto()
    ACTIVE_B = auto()
    ACTIVE_C = auto()
    ACTIVE_D = auto()

class Qreg:
    def __init__(self,
                 name,
                 status,
                 pos,
                 ):
        self.name = name
        self.status = status
        self.pos = pos

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        _copy_attrs(self, copied, memo)
        return copied

# Classical register
class Creg:
    def __init__(self,
                 name,
                 status,
                 value):
        self.name = name
        self.status = status
        self.value = value
        self.dep_insts = []

    def __deepcopy__(self, memo):
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        _copy_attrs(self, copied, memo)
        return copied