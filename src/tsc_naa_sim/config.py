from macro import *
from misc import *

class experiment_config:
    # baseline for case study
    def __init__(self):
        # plane: Baseline
        self.lattice_bound = LatticeBound.SHUTTLE_PATH
        self.cell_size = CellSize.SMALLEST
        self.plane_type = PlaneType.ALLQ

        # pysical hardware parameters/regimes
        ## latency / error rates
        self.hw_cfg = getJsonData("case_study_hwcfg.json")
        ## regimes
        self.move_type = MoveType.STA
        self.cx_type = PhyOpType.SELECT
        self.meas_type = PhyOpType.SELECT

        # qec
        self.code_dist = 27
        self.rounds = 1

        # software options
        ## instruction translator
        ### rotation translation options
        #### BASE / ALL_ROT / NAIVE_SKIP / HEUR_ALLOC
        self.rot_trans_opt = RotTransOpt.BASE

        ## instruction scheduler
        ### scheduling policy
        #### ASAP
        self.inst_sched_opt = InstSchedOpt.ASAP
        ### rotation scheduling policy
        #### FOLLOW_H / ASAP / ALAP / AGGREGATE
        self.rot_sched_opt = RotSchedOpt.FOLLOW_H

        ## uop scheduler
        ## rot_type
        self.rot_type = RotType.REFL
        self.refl_type_h = ReflType.STATIC_SE
        self.refl_type_d = ReflType.STATIC_SE
        # rot_plane
        self.rot_plane_opt = RotPlaneOpt.ALL_ROT
        self.num_rot_cell = None

        # mov routing
        self.mov_twostep_opt = MoveTwostepOpt.DQ_XY

        ## aod scheduler
        self.aod_sched_opt = AodSchedOpt.NAIVE_DRAIN
        ## aod grouping

        ### aod
        self.num_aodh_max = 4
        self.num_aodd_max = 4
        self.num_aodr_max = 4
        self.is_aod_infinite = True
        ### grouping
        self.skip_uop_grouping = True

        ###
        self.s_trans_opt = STransOpt.GATE_TEL
        self.skip_h = False