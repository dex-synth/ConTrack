"""Compile the common MJCF and resolve its actual physical indexing."""

import mujoco
import numpy as np


class Scene:
    def __init__(self, path, layout):
        self.model = m = mujoco.MjModel.from_xml_path(str(path))
        self.layout = layout
        joints = np.array([m.joint(n).id for n in layout['joint_names']])
        if not np.isin(m.jnt_type[joints], [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]).all():
            raise ValueError('Controlled joints must be scalar')
        self.qids, self.vids = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        self.aids = []
        for j in joints:
            ids = np.flatnonzero((m.actuator_trnid[:, 0] == j) & (m.actuator_trntype == 0))
            if len(ids) != 1:
                raise ValueError('Expected one joint actuator per controlled joint')
            a = ids[0]
            if (m.actuator_gaintype[a] != 0 or m.actuator_biastype[a] != 1 or m.actuator_dyntype[a] != 0
                    or m.actuator_gainprm[a, 0] <= 0
                    or not np.allclose(m.actuator_biasprm[a, :2], [0, -m.actuator_gainprm[a, 0]])
                    or not np.array_equal(m.actuator_gear[a], [1, 0, 0, 0, 0, 0])):
                raise ValueError('Expected common unit-gear position actuators')
            self.aids.append(int(a))
        if m.nu != len(joints) or m.na or m.nflex:
            raise ValueError('Uncontrolled actuators, activation state and flex are unsupported')
        self.part_ids = [m.body(n).id for n in layout['part_bodies']]
        self.limits = np.where(m.jnt_limited[joints, None], m.jnt_range[joints], [-np.inf, np.inf])
        self.body_slots = np.full((m.nbody, 2), -1, dtype=np.int32)
        for name, slot in layout['contact_bodies'].items():
            self.body_slots[m.body(name).id] = slot
        self.body_parts = np.full(m.nbody, -1, dtype=np.int32)
        self.body_parts[self.part_ids] = np.arange(len(self.part_ids))

    def object_qpos(self, demo):
        """Recover consistent generalized coordinates, never independent child poses."""
        m = self.model
        q = np.tile(m.qpos0, (len(demo.time_s), 1))
        for obj in demo.objects.values():
            if obj.root_joint:  # A fixed root has no coordinates; the FK check below still covers it.
                joint = m.joint(obj.root_joint)
                if joint.type[0] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError('Expected free root')
                address = joint.qposadr[0]
                q[:, address:address+3], q[:, address+3:address+7] = demo.part_pose(obj.root_part)
            for hinge in obj.joints:
                joint = m.joint(hinge.mjcf_name)
                if joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('Only passive hinges are currently supported')
                q[:, joint.qposadr[0]] = hinge.position_rad
        arrays = demo.arrays
        d = mujoco.MjData(m)
        max_error = 0.0
        for i in range(len(q)):
            d.qpos[:] = q[i]
            mujoco.mj_kinematics(m, d)
            error = np.max(np.abs(d.xpos[self.part_ids] - arrays['part_pos_m'][i]))
            dot = np.abs(np.sum(d.xquat[self.part_ids] * arrays['part_quat_wxyz'][i], -1))
            max_error = max(max_error, float(error))
            if error > 1e-5 or np.any(1-dot > 1e-5):
                raise ValueError(f'Object reference violates the common joint tree at frame {i}')
        return q, max_error

    def audit(self):
        m = self.model
        fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_gravcomp', 'body_pos', 'body_quat',
                  'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited', 'jnt_stiffness', 'qpos_spring',
                  'dof_damping', 'dof_armature',
                  'dof_frictionloss', 'geom_bodyid', 'geom_type', 'geom_size', 'geom_friction',
                  'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity',
                  'geom_condim', 'geom_margin', 'geom_gap', 'geom_pos', 'geom_quat',
                  'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange',
                  'actuator_ctrllimited', 'actuator_forcelimited')
        return {k: getattr(m, k).tolist() for k in fields}
