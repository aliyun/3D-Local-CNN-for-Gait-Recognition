#! /usr/bin/env python

from .solver import Solver
from .baseline import BaselineSolver
from .pretreatment import PretreatmentC, PretreatmentO

from .vis_analysis import VisAnalysis
from .data_analysis import DataAnalysis
from .test_sampling_module import TestSamplingModule

from .localcnn import LocalCNNSolver
from .c2d_local import C2D_Local_Solver
from .c2d_load import C2D_Load_Solver
from .c2d_concat import C2D_Local_Concat_Solver

# C3D
from .c3d import C3D_Solver
from .c3d_local import Local3dSolver
from .c3d_local_fix import Local3dFixSolver
# from .c3d_local_v1 import Local3dSolverV1
from .c3d_local_fusion_6branch import Local3dFusion6BranchSolver
