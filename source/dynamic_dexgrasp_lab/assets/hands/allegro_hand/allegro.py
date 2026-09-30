"""Runtime asset constants for Allegro Hand."""

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG


ALLEGRO_JOINT_NAMES = (
    "index_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_0",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_0",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_0",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
)
ALLEGRO_ACTUATED_JOINT_NAMES = ALLEGRO_JOINT_NAMES
ALLEGRO_HOME_JOINT_POSITIONS = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
)
ALLEGRO_HOME_JOINT_POS_MAP = {
    name: value for name, value in zip(ALLEGRO_JOINT_NAMES, ALLEGRO_HOME_JOINT_POSITIONS, strict=True)
}
ALLEGRO_POSITION_ACTION_OFFSET_MAP = {
    name: value for name, value in zip(ALLEGRO_JOINT_NAMES, ALLEGRO_HOME_JOINT_POSITIONS, strict=True)
}
ALLEGRO_JOINT_LIMIT_LOWER_MAP = {
    "index_joint_0": -0.47,
    "index_joint_1": -0.196,
    "index_joint_2": -0.174,
    "index_joint_3": -0.227,
    "middle_joint_0": -0.47,
    "middle_joint_1": -0.196,
    "middle_joint_2": -0.174,
    "middle_joint_3": -0.227,
    "ring_joint_0": -0.47,
    "ring_joint_1": -0.196,
    "ring_joint_2": -0.174,
    "ring_joint_3": -0.227,
    "thumb_joint_0": 0.263,
    "thumb_joint_1": -0.105,
    "thumb_joint_2": -0.189,
    "thumb_joint_3": -0.162,
}
ALLEGRO_JOINT_LIMIT_UPPER_MAP = {
    "index_joint_0": 0.47,
    "index_joint_1": 1.61,
    "index_joint_2": 1.709,
    "index_joint_3": 1.618,
    "middle_joint_0": 0.47,
    "middle_joint_1": 1.61,
    "middle_joint_2": 1.709,
    "middle_joint_3": 1.618,
    "ring_joint_0": 0.47,
    "ring_joint_1": 1.61,
    "ring_joint_2": 1.709,
    "ring_joint_3": 1.618,
    "thumb_joint_0": 1.396,
    "thumb_joint_1": 1.163,
    "thumb_joint_2": 1.644,
    "thumb_joint_3": 1.719,
}
ALLEGRO_POSITION_ACTION_SCALE_MAP = {
    joint_name: max(
        ALLEGRO_POSITION_ACTION_OFFSET_MAP[joint_name] - ALLEGRO_JOINT_LIMIT_LOWER_MAP[joint_name],
        ALLEGRO_JOINT_LIMIT_UPPER_MAP[joint_name] - ALLEGRO_POSITION_ACTION_OFFSET_MAP[joint_name],
    )
    for joint_name in ALLEGRO_ACTUATED_JOINT_NAMES
}
ALLEGRO_HAND_TRANSFORM_POS = (0.0, 0.0, 0.0)
ALLEGRO_HAND_TRANSFORM_QUAT = (0.92387953, 0.0, 0.38268343, 0.0)
ALLEGRO_HAND_OBJECT_RELATIVE_POS = (0.0, -0.30641778, -0.25711504)
ALLEGRO_HAND_OBJECT_RELATIVE_QUAT = (0.27059805, 0.27059805, 0.65328148, 0.65328148)
ALLEGRO_CAM_OBJECT_RELATIVE_POS = (-0.272372, 0.040000, 0.335367)
ALLEGRO_CAM_OBJECT_RELATIVE_QUAT = (0.360753, -0.608159, 0.608157, -0.360756)
ALLEGRO_PALM_BODY_NAME = "palm_link"
ALLEGRO_GEO_BODY_NAMES = (
    "index_link_0",
    "index_link_1",
    "index_link_2",
    "index_link_3",
    "middle_link_0",
    "middle_link_1",
    "middle_link_2",
    "middle_link_3",
    "ring_link_0",
    "ring_link_1",
    "ring_link_2",
    "ring_link_3",
    "thumb_link_0",
    "thumb_link_1",
    "thumb_link_2",
    "thumb_link_3",
)
ALLEGRO_GEO_BODY_WEIGHTS = (
    1.0,
    1.0,
    1.0,
    3.0,
    1.0,
    1.0,
    1.0,
    3.0,
    1.0,
    1.0,
    1.0,
    3.0,
    1.0,
    1.0,
    1.0,
    3.0,
)
ALLEGRO_ARTICULATION_ROOT_PRIM_PATH = ""
ALLEGRO_CONTACT_BODY_NAMES = ALLEGRO_GEO_BODY_NAMES
ALLEGRO_CONTACT_FINGER_BODY_NAMES = (
    ("index_link_0", "index_link_1", "index_link_2", "index_link_3"),
    ("middle_link_0", "middle_link_1", "middle_link_2", "middle_link_3"),
    ("ring_link_0", "ring_link_1", "ring_link_2", "ring_link_3"),
    ("thumb_link_0", "thumb_link_1", "thumb_link_2", "thumb_link_3"),
)
ALLEGRO_CONTACT_FINGERTIP_BODY_NAMES = (
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
    "thumb_link_3",
)


__all__ = [
    "ALLEGRO_HAND_CFG",
    "ALLEGRO_JOINT_NAMES",
    "ALLEGRO_ACTUATED_JOINT_NAMES",
    "ALLEGRO_HOME_JOINT_POSITIONS",
    "ALLEGRO_HOME_JOINT_POS_MAP",
    "ALLEGRO_POSITION_ACTION_OFFSET_MAP",
    "ALLEGRO_POSITION_ACTION_SCALE_MAP",
    "ALLEGRO_HAND_TRANSFORM_POS",
    "ALLEGRO_HAND_TRANSFORM_QUAT",
    "ALLEGRO_HAND_OBJECT_RELATIVE_POS",
    "ALLEGRO_HAND_OBJECT_RELATIVE_QUAT",
    "ALLEGRO_CAM_OBJECT_RELATIVE_POS",
    "ALLEGRO_CAM_OBJECT_RELATIVE_QUAT",
    "ALLEGRO_PALM_BODY_NAME",
    "ALLEGRO_GEO_BODY_NAMES",
    "ALLEGRO_GEO_BODY_WEIGHTS",
    "ALLEGRO_ARTICULATION_ROOT_PRIM_PATH",
    "ALLEGRO_CONTACT_BODY_NAMES",
    "ALLEGRO_CONTACT_FINGER_BODY_NAMES",
    "ALLEGRO_CONTACT_FINGERTIP_BODY_NAMES",
]
