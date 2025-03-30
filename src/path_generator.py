from pathlib import Path
import numpy as np
from fk_solver import FKSolver
import transform_utils as T
import utils
from numba import njit
from pydrake.solvers import MathematicalProgram
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    Parser,
    RigidTransform,
    RotationMatrix
)
from drake_utils import(
    mapidx_og2drake
)
import logging
from pydrake.multibody.tree import JacobianWrtVariable

class PathGenerator():

    def __init__(
        self,
        robot_urdf_path,
        eef_name,
        initial_joint_pos,
        dof_idx,
        trans_world2robot,
        og_joint_name,
        pos_step_size,
        rot_step_size
    ):
        self.path_costs = []
        self.dof_idx = dof_idx
        self.ndof = len(dof_idx)
        # Transformation from world to robot base (robot base pose in world frame)
        self.trans_world2robot = trans_world2robot
        self.pos_step_size = pos_step_size
        self.rot_step_size = rot_step_size
        logging.getLogger("drake").setLevel(logging.ERROR)
        # set drake plant for fk and jacobian
        self.set_plant(robot_urdf_path, eef_name, dof_idx, og_joint_name, initial_joint_pos)




    def set_start(self,eef_pose, eef_twist, keypoints, keypoint_movable_mask):
        """
        Initialize Start Condition of Decision variable

        Args:
            - eef_pose: start eef pose in world frame [x,y,z,qx,qy,qz,qw]
            - eef_twist: start eef twist in world frame [wx,wy,wz,vx,vy,vz]
            - keypoints: keypoint position in world frame [num_keypoints, 3]
            - keypoint_movable_mask(bool): Whether the keypoints are on the object being grasped
        """
        # transform of eef in world
        trans_world2eef = T.pose2mat([eef_pose[:3], eef_pose[3:]])
        # transform of world in eef
        trans_eef2world = T.pose_inv(trans_world2eef)

        # setup keypoint (moveable keypoint relative to eef)
        self.keypoints_eef = self.transform_keypoints(trans_eef2world, keypoints, keypoint_movable_mask)

        # set zero
        eef_twist = np.zeros(6) # HACK: Remeber to remove this after testing
        eef_pose_euler = np.concatenate([eef_pose[:3], T.quat2euler(eef_pose[3:])])

        self.start_var = np.concatenate([eef_pose_euler, eef_twist])
        self.keypoint_movable_mask = keypoint_movable_mask



    def set_target(self, target_eef_pose, target_eef_twist):
        """
        Set target pose and twist

        Args:
            - target_eef_pose: target eef pose for optimization in world frame [x,y,z,qx,qy,qz,qw]
            - target_eef_twist: target eef twist for optimization in world frame [vx,vy,vz,qx,qy,qz]
        """
        eef_pose_euler = np.concatenate([target_eef_pose[:3], T.quat2euler(target_eef_pose[3:])])
        self.target_var = np.concatenate([eef_pose_euler, target_eef_twist])


    def set_steps(self):
        """
        Calculate an appropriate number of control points, including start and goal
        """
        start_pose = self.start_var[:6]
        end_pose = self.target_var[:6]
        # FIXME: Might need consider twist as well to calculate number of control point
        self.num_steps = self.get_linear_interpolation_steps(start_pose, end_pose, self.pos_step_size, self.rot_step_size)
        self.num_steps = np.clip(self.num_steps, 3 ,6)


    def set_costs(self):
        """
        Set up cost functions, cost functions are computed in task space(world frame)
        """
        # TODO: Now hardcoded cost function, after testing remeber to rewrite this
        def stage3_path_constraint1(eef_pose, keypoints, eef_velocity):
            """The pen should keep aligned with the intended throwing direction during execution."""
            dx = keypoints[5][0] - eef_pose[0]
            dy = keypoints[5][1] - eef_pose[1]
            cost = (eef_velocity[0] * dy - eef_velocity[1] * dx)**2 - 1e-4  # Small tolerance delta
            return cost
        
        self.path_costs.append(stage3_path_constraint1)

    def set_plant(self,urdf_path, eef_name, dof_idx, og_joint_name, initial_joint_pose):
        # set up urdf string
        urdf_path = Path(urdf_path)
        with open(urdf_path, "r") as f:
            urdf_content = f.read()
        urdf_content = urdf_content.replace(".STL", ".obj")

        # build plant
        self.plant = MultibodyPlant(time_step=0.0)
        self.parser = Parser(self.plant)
        model_indices = self.parser.AddModelsFromString(urdf_content, "urdf")
        self.model_index = model_indices[0]
        self.plant.Finalize()
        self.eef_name = eef_name

        # build context
        self.context = self.plant.CreateDefaultContext()
        self.drake_ndof = self.plant.num_positions(self.model_index)

        # map og dof index to drake dof index
        drake_joint_name = self.plant.GetPositionNames(self.model_index)
        self.drake_dof_idx = mapidx_og2drake(dof_idx, og_joint_name, drake_joint_name)
        self.drake_twist_idx = np.array(self.drake_dof_idx) - 1
        
        # set initial joint pos
        q_all = np.zeros(self.drake_ndof)
        q_all[self.drake_dof_idx] = initial_joint_pose
        self.plant.SetPositions(self.context, self.model_index, q_all)

        # set base pose in world frame
        self.base_body = self.plant.GetBodyByName("base_link", self.model_index)
        rot = RotationMatrix(self.trans_world2robot[:3, :3])
        p = self.trans_world2robot[:3, 3]
        T_world_base = RigidTransform(rot, p)
        self.plant.SetFreeBodyPose(self.context, self.base_body, T_world_base)
        self.eef_body = self.plant.GetBodyByName(self.eef_name, self.model_index)


    def compute_fk(self, q):
        # set joint angle
        q_all = np.zeros(self.drake_ndof)
        q_all[self.drake_dof_idx] = q
        self.plant.SetPositions(self.context, self.model_index, q_all)
        # eef pose in world frame
        T_world_eef = self.plant.EvalBodyPoseInWorld(self.context, self.eef_body)
        pos,ori = T.mat2pose(T_world_eef.GetAsMatrix4())
        pose = np.concatenate([pos, T.quat2euler(ori)])
        return pose, T_world_eef
    
    def compute_twist(self, dq):
        Jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            with_respect_to=JacobianWrtVariable.kV,  
            frame_B=self.eef_body.body_frame(),
            p_BoBp_B= np.zeros(3, dtype=np.float64),
            frame_A=self.base_body.body_frame(),             
            frame_E=self.base_body.body_frame()              
        )
        Jacobian = np.vstack([Jacobian[3:], Jacobian[:3]])[:, self.drake_twist_idx]
        # twist in base frame
        twist_base = np.dot(Jacobian, dq)
        # twist in world frame
        twist_world = T.vel_in_A_to_vel_in_B(twist_base[:3], twist_base[3:], self.trans_world2robot)
        return twist_world


    def joint_path_cost(self, q, dq):
        """
        Below are the steps and formulation
        1. Decision variables:
            - Since we are using fetch robot, dof has 1 base , 7 joint
            - trajectory of joint angles: [q1,q2...qn], each q [1x8]
            - trajectory of joint velocity: [dq1, dq2.....dqn], each dq [1x8]
        1. Based on current q's, use FK(q) to compute eef pose in world frame: trans_world2eef
        2. use trans_world2eef to update movable keypoints pose(keypoint_ee and keypoint_movable_mask) in world frame to ensure correct cost
        3. Based on current dq's, use Jacobian * dq to compute eef twist in world frame
        4. Compute Cost as sum of path_costs by passing
            - updated keypoint in world frame
            - eef_pose in world
            - eef_twist in world

        I dont know how to do:
            1. We need to determine how many control number we should have
            2. We need B-spline interpolation during optimization to make sure smoothness
        """
        if self.path_costs is None or len(self.path_costs) == 0:
            return 0
        
        # FK to get eef pose in world for given q
        eef_pos_world, trans_world2eef= self.compute_fk(q)
        # Update Keypoint based on eef pose in world frame
        keypoint_world = self.transform_keypoints(trans_world2eef, self.keypoints_eef, self.keypoint_movable_mask)
        # Jacobian to get eef twist in world
        eef_twist_world = self.compute_twist(dq)

        path_cost = 0
        for cost_fn in self.path_costs:
            cost = cost_fn(eef_pos_world, keypoint_world, eef_twist_world[:3])
            path_cost += np.clip(cost, 0, np.inf)
        return 200.0 * path_cost
    

    def joint_path_total_cost(self,x):
        n = self.num_steps
        d = self.ndof
        q_reshaped = x[0 : n*d].reshape(n, d)
        dq_reshaped = x[n*d : 2*n*d].reshape(n, d)
        total_cost = 0.0
        # sum up cost for each step
        for i in range(n):
            cost_i = self.joint_path_cost(q_reshaped[i], dq_reshaped[i])
            total_cost += cost_i
        return total_cost


    def set_optimizer(self):
        self.prog = MathematicalProgram()
        # decision variables
        self.q = self.prog.NewContinuousVariables(self.num_steps, self.ndof, name='joint angle')
        self.dq = self.prog.NewContinuousVariables(self.num_steps, self.ndof, name='joint velocity')
        all_vars = np.concatenate((self.q.reshape(-1), self.dq.reshape(-1)))
        # add cost
        def path_cost(x):
            return [self.joint_path_total_cost(x)]
        self.prog.AddCost(path_cost, vars=all_vars)
        
    @staticmethod
    @njit(cache=True, fastmath=True)
    def angle_between_rotmat(P, Q):
        R = np.dot(P, Q.T)
        cos_theta = (np.trace(R)-1)/2
        if cos_theta > 1:
            cos_theta = 1
        elif cos_theta < -1:
            cos_theta = -1
        return np.arccos(cos_theta)

    def get_linear_interpolation_steps(self, start_pose, end_pose, pos_step_size, rot_step_size):
        """
        Given start and end pose, calculate the number of steps to interpolate between them.
        Args:
            start_pose: [6] position + euler or [4, 4] pose or [7] position + quat
            end_pose: [6] position + euler or [4, 4] pose or [7] position + quat
            pos_step_size: position step size
            rot_step_size: rotation step size
        Returns:
            num_path_poses: number of poses to interpolate
        """
        if start_pose.shape == (6,) and end_pose.shape == (6,):
            start_pos, start_euler = start_pose[:3], start_pose[3:]
            end_pos, end_euler = end_pose[:3], end_pose[3:]
            start_rotmat = T.euler2mat(start_euler)
            end_rotmat = T.euler2mat(end_euler)
        elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
            start_pos = start_pose[:3, 3]
            start_rotmat = start_pose[:3, :3]
            end_pos = end_pose[:3, 3]
            end_rotmat = end_pose[:3, :3]
        elif start_pose.shape == (7,) and end_pose.shape == (7,):
            start_pos, start_quat = start_pose[:3], start_pose[3:]
            start_rotmat = T.quat2mat(start_quat)
            end_pos, end_quat = end_pose[:3], end_pose[3:]
            end_rotmat = T.quat2mat(end_quat)
        else:
            raise ValueError('start_pose and end_pose not recognized')
        pos_diff = np.linalg.norm(start_pos - end_pos)
        rot_diff = self.angle_between_rotmat(start_rotmat, end_rotmat)
        pos_num_steps = np.ceil(pos_diff / pos_step_size)
        rot_num_steps = np.ceil(rot_diff / rot_step_size)
        num_path_poses = int(max(pos_num_steps, rot_num_steps))
        num_path_poses = max(num_path_poses, 2)  # at least start and end poses
        return num_path_poses

    @staticmethod
    def transform_keypoints(transform, keypoints, movable_mask):
        assert transform.shape == (4, 4)
        transformed_keypoints = keypoints.copy()
        if movable_mask.sum() > 0:
            transformed_keypoints[movable_mask] = np.dot(keypoints[movable_mask], transform[:3, :3].T) + transform[:3, 3]
        return transformed_keypoints