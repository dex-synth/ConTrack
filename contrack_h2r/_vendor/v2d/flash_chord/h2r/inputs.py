# SPDX-License-Identifier: Apache-2.0
DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def frame_names(side):
    return [f"{side}_hand_C_MC", *[f"{side}_{d}_DP" for d in DIGITS], *[f"{side}_{d}_fingertip" for d in DIGITS]]
