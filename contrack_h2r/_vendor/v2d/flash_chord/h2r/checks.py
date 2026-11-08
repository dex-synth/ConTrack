# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.spatial.transform import Rotation
from .inputs import DIGITS, frame_names


def check_ik_model(solver, model, state, side, fingers, wrist_ids, base):
    """Compare derived Pinocchio FK to common MuJoCo FK at nonzero poses."""
    import mujoco

    from .wrist import wrist_coordinates

    rng = np.random.default_rng(17)
    finger_ids = [model.joint(name).qposadr[0] for name in fingers]
    ranges = np.array([model.joint(name).range for name in fingers])
    np.testing.assert_allclose(solver.robot.model.lowerPositionLimit[7:], ranges[:, 0], atol=1e-8)
    np.testing.assert_allclose(solver.robot.model.upperPositionLimit[7:], ranges[:, 1], atol=1e-8)
    max_error = 0.0
    for _ in range(5):
        q = solver.robot.q0.copy()
        q[:3] = rng.uniform(-0.3, 0.3, 3)
        q[3:7] = Rotation.from_euler("XYZ", rng.uniform(-2, 2, 3)).as_quat()
        q[7:] = rng.uniform(ranges[:, 0], ranges[:, 1])
        solver.configuration.update(q)
        state.qpos[:] = 0
        state.qpos[wrist_ids] = wrist_coordinates(q[:3], q[[6, 3, 4, 5]], base)
        state.qpos[finger_ids] = q[7:]
        mujoco.mj_forward(model, state)
        for name in frame_names(side)[:6]:
            pose = solver.configuration.get_transform_frame_to_world(name)
            common = state.body(name)
            error = np.max(np.abs(pose.translation - common.xpos))
            max_error = max(max_error, float(error))
            np.testing.assert_allclose(pose.translation, common.xpos, atol=2e-6, rtol=0)
            np.testing.assert_allclose(pose.rotation, common.xmat.reshape(3, 3), atol=2e-6, rtol=0)
        for digit in DIGITS:
            pose = solver.configuration.get_transform_frame_to_world(f"{side}_{digit}_tip_site")
            np.testing.assert_allclose(
                pose.translation, state.body(f"{side}_{digit}_fingertip").xpos, atol=2e-6, rtol=0
            )
            # Native task-site orientation is a half turn around distal Z.
            expected = state.body(f"{side}_{digit}_DP").xmat.reshape(3, 3) @ np.diag([-1, -1, 1])
            np.testing.assert_allclose(pose.rotation, expected, atol=2e-6, rtol=0)
    return max_error
