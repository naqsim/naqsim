from enum import Enum, auto

################
class RunOpt(Enum):
    IGNORE_NONE = auto()
    IGNORE_PC = auto()
    IGNORE_ROT = auto()
    IGNORE_PC_ROT = auto()
################

class IntervalType(Enum):
    QOP = auto()
    PICK = auto()
    SHUTTLE = auto()
    DROP = auto()
    DELAY = auto()

class PhyOpType(Enum):
    ZONE = auto()
    SELECT = auto()

class MoveType(Enum):
    CV = auto()
    CJ = auto()
    STA = auto()


class CXType(Enum):
    QQ = auto()
    MQ = auto()
    YQ = auto()
    MY = auto()

class MoveDir(Enum):
    UP_Y = auto()
    DOWN_Y = auto()
    LEFT_X = auto()
    RIGHT_X = auto()
    UP_Q = auto()
    DOWN_Q = auto()
    LEFT_D = auto()
    RIGHT_D = auto()


class CellType(Enum):
    INVALID = auto()
    NORMAL = auto()
    PORT_M = auto()
    PORT_Y = auto()   

class CellStatusQ(Enum):
    FREE = auto()
    OCCUPIED_Q = auto()
    OVERLAP_Q = auto()
    TEMPOVER_Q = auto()

class CellStatusOp(Enum):
    IDLE = auto()
    MOV = auto()
    RD = auto()
    RH = auto()
    RD_RH = auto()
    RH_RD = auto()

class AodType(Enum):
    AODH = auto()
    AODD = auto()
    AODR = auto() # for direct rotation 

class LatticeBound(Enum):
    SHUTTLE_PATH = auto()

class CellSize(Enum):
    SMALLEST= auto()
    DOUBLE_TE = auto()
    DOUBLE_DIR = auto()

class PlaneType(Enum):
    ALLQ = auto()

####
class LockType(Enum):
    # center
    SQR = auto()
    QR = auto()
    QL = auto()
    DR = auto()
    DL = auto()
    #
    SOL = auto() # ensure sole Q i.e., prohibit next drop, but allow passing moves
    #
    QRR = auto()
    QLL = auto()
    DRR = auto()
    DLL = auto()

    # east
    E = auto()
    # south
    S = auto()
    # south-east
    SE = auto()

LOCK_TYPES_C = [
    LockType.SQR,
    # 
    LockType.QR,
    LockType.QL,
    LockType.DR,
    LockType.DL,
    #
    LockType.QRR,
    LockType.QLL,
    LockType.DRR,
    LockType.DLL
]

LOCK_TYPES_E = [
    LockType.E
]

LOCK_TYPES_S = [
    LockType.S
]

LOCK_TYPES_SE = [
    LockType.SE
]
    
class CellPart(Enum):
    C = auto()
    E = auto()
    S = auto()
    SE = auto()

class MoveStep(Enum):
    ONE_STEP_XY = auto()
    ONE_STEP_DQ = auto()
    TWO_STEP = auto()

class RotType(Enum):
    REFL = auto()
    DIR_IDEAL = auto()
    DIR_CHANGE = auto()
    DIR_TOGL = auto()

class ReflType(Enum):
    STATIC_SE = auto()
    STATIC_TE = auto()

### Rotations
class RotTransOpt(Enum):
    BASE = auto()
    ALL_ROT = auto()
    NAIVE_SKIP = auto()
    HEUR_ALLOC = auto()

class InstSchedOpt(Enum):
    ASAP = auto()

class RotSchedOpt(Enum):
    FOLLOW_H = auto()
    ASAP = auto()
    ALAP = auto()
    AGGREGATE = auto()
    DISTRIBUTE = auto()

class RotPlaneOpt(Enum):
    ALL_ROT = auto()
    DEDICATED_ROT = auto()

## AOD 
class AodSchedOpt(Enum):
    NAIVE_DRAIN = auto()
    LAST_FINISH = auto()
    FIRST_FINISH = auto()

##
class MoveTwostepOpt(Enum):
    DQ_XY = auto()
    XY_XY = auto()

##
class STransOpt(Enum):
    GATE_TEL = auto()
    TRANS_S = auto()