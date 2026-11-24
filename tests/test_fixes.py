"""题目所述六类接触阴影缺陷的行为回归测试。"""

import math

import pytest

from graphics import contact_shadow_reproject as cs


def camera(origin=(0.0, 0.0, 0.0), frame=1, fov_deg=60.0, resolution=(128, 72)):
    return {
        "frame": frame,
        "origin": list(origin),
        "target": [origin[0], origin[1], origin[2] - 1.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y": math.radians(fov_deg),
        "aspect": 16.0 / 9.0,
        "near": 0.1,
        "far": 100.0,
        "resolution": list(resolution),
    }


# 1. 重投影必须用当帧矩阵，并校验历史 ---------------------------------------

def test_reproject_uses_current_matrix_not_previous():
    previous = camera([0.0, 0.0, 0.0], frame=11)
    current = camera([0.4, 0.0, 0.0], frame=12)
    positions = [[0.0, 0.0, -5.0], [1.0, -0.5, -6.0]]

    report = cs.reproject_matrix(positions, current, previous)
    assert report["matrix_source"] == "current_frame"
    assert report["valid_count"] == 2

    first = report["samples"][0]
    # 当帧矩阵下，世界点相对相机偏左；上一帧矩阵会把它投到屏幕正中。
    assert first["uv"][0] < 0.5
    assert first["naive_prev_matrix_uv"][0] == pytest.approx(0.5)
    # 屏幕运动量非零，证明历史确实被重投影而不是原地复用
    assert abs(first["motion_px"][0]) > 1.0


def test_reproject_invalidates_gaps_and_marks_invalid_samples():
    previous = camera([0.0, 0.0, 0.0], frame=1)
    current = camera([0.0, 0.0, 0.0], frame=9)
    report = cs.reproject_matrix([[0.0, 0.0, -5.0]], current, previous)
    assert "frame_gap" in report["samples"][0]["reasons"]
    assert report["samples"][0]["valid"] is False

    behind = cs.reproject_matrix([[0.0, 0.0, 5.0]], current, current)
    assert "behind_camera" in behind["samples"][0]["reasons"]

    outside = cs.reproject_matrix([[50.0, 0.0, -5.0]], current, current)
    assert "out_of_bounds" in outside["samples"][0]["reasons"]


# 2. 边缘按厚度继续步进，而不是一穿透就停 -----------------------------------

def test_thickness_lets_grazing_steps_continue():
    grazing = cs.thickness_step(0.03, 0.1)
    assert grazing["decision"] == "grazing"
    assert grazing["continue"] is True
    assert grazing["occluded"] is False

    occluded = cs.thickness_step(0.3, 0.1)
    assert occluded["decision"] == "occluded"
    assert occluded["continue"] is False
    assert occluded["occluded"] is True


def test_ray_march_survives_contact_edge_without_splitting():
    # 轮廓处三个掠射样本（旧逻辑第一步就停），之后才是真实遮挡。
    result = cs.ray_march([5.03, 5.06, 5.09, 5.30], [5.0, 5.0, 5.0, 5.0],
                          thickness=0.1, bias=0.01)
    assert result["grazing_steps"] == 3
    assert result["steps"] == 4
    assert result["occluded"] is True
    # 旧实现对照：第一处穿透就终止，接地阴影在轮廓处断成两截。
    assert result["naive_steps"] == 1
    assert result["naive_occluded"] is True

    # 只有掠射、没有真实遮挡时，阴影不应被误判为接触阴影。
    graze_only = cs.ray_march([5.01, 5.02, 5.03], [5.0, 5.0, 5.0],
                              thickness=0.1, bias=0.01)
    assert graze_only["occluded"] is False
    assert graze_only["visible"] == 1.0


# 3. compute / graphics 队列之间插栅栏 ---------------------------------------

def test_queue_fence_requires_signal_and_wait():
    ok = cs.queue_fence(1)
    assert ok["satisfied"] is True
    assert ok["hazard"] is False
    assert ok["signal_value"] == ok["wait_value"]
    # 演示对照：图形队列忘记 wait 时会被识别为整块缺失的 hazard。
    assert ok["demo_comparison"]["missing_wait_hazard"] is True

    compute = cs.Queue("compute")
    graphics = cs.Queue("graphics")
    fence = cs.Fence()
    hazard = cs.queue_fence(compute, graphics, fence,
                            signal_value=1, wait_value=2)
    assert hazard["hazard"] is True
    assert "wait_exceeds_signal" in hazard["reasons"]


# 4. 步长按世界空间换算，半分辨率保持世界步长 --------------------------------

def test_world_space_step_is_constant_across_depths():
    world_step = 0.02
    for depth in (0.5, 2.0, 10.0, 50.0):
        info = cs.step_world_space(world_step, depth, 60.0, 1080)
        assert info["pixel_step"] * info["world_per_pixel"] == \
            pytest.approx(world_step)
    near = cs.step_world_space(world_step, 1.0, 60.0, 1080)
    far = cs.step_world_space(world_step, 20.0, 60.0, 1080)
    assert near["pixel_step"] > far["pixel_step"]


def test_half_res_halves_pixel_step_but_keeps_world_step():
    shadow = [[0.9, 0.1], [0.9, 0.1]]
    depth = [[1.0, 1.0, 5.0, 5.0] for _ in range(4)]
    result = cs.half_res(shadow, depth, (4, 4), world_step=0.02)
    assert result["world_step"] == pytest.approx(0.02)
    assert result["pixel_step_ratio"] == pytest.approx(0.5)
    assert result["half_pixel_step"] * 2 == pytest.approx(
        result["full_pixel_step"])
    assert len(result["upsampled"]) == 4
    assert len(result["upsampled"][0]) == 4


# 5. 相机瞬移清空历史 --------------------------------------------------------

def test_teleport_invalidates_history_on_cut():
    base = camera([0.0, 0.0, 0.0], frame=3)
    small = camera([0.2, 0.0, 0.0], frame=4)
    jump = camera([12.0, 0.0, 0.0], frame=4)
    fov_cut = camera([0.0, 0.0, 0.0], frame=4, fov_deg=75.0)

    assert cs.teleport_invalidate(base, small)["invalidate_history"] is False
    assert cs.teleport_invalidate(base, jump)["reasons"] == ["position_jump"]
    assert cs.teleport_invalidate(base, jump)["invalidate_history"] is True
    assert "fov_jump" in cs.teleport_invalidate(base, fov_cut)["reasons"]
    gap = camera([0.0, 0.0, 0.0], frame=10)
    assert cs.teleport_invalidate(base, gap)["invalidate_history"] is True


# 6. 历史累积按运动量裁剪 ----------------------------------------------------

def test_history_clip_weights_by_motion_and_clamps():
    history = [[0.2] * 3 for _ in range(3)]
    current = [[0.8] * 3 for _ in range(3)]
    motion = [[0.0, 2.0, 100.0] for _ in range(3)]

    result = cs.history_clip(history, current, motion)
    # 静止像素材历史权重最高，运动越大权重越低，超过阈值直接丢弃。
    assert result["weights"][0][0] > result["weights"][0][1] > 0.0
    assert result["weights"][0][2] == 0.0
    assert result["discarded_pixels"] == 3
    # 邻域 AABB 裁剪：历史 0.2 被拉回当前帧邻域 0.8 附近。
    assert result["clipped"][0][0] == pytest.approx(0.8)
    # 丢弃历史的像素完全回退当前帧。
    assert result["blended"][0][2] == pytest.approx(0.8)


# 配套契约 ---------------------------------------------------------------

def test_support_contracts_are_stable():
    assert cs.validation(1)["ok"] is True
    assert cs.validation(("shadow", [[0.5, 1.5]]))["ok"] is False
    assert cs.audit(1)["counts"]["events"] == 3
    assert cs.fallback(1)["mode"] == "neutral_full_res_recompute"
    assert cs.determinism(1)["deterministic"] is True
    assert cs.metrics(1)["resolution"] == (4, 4)
    assert cs.depth_bias()["world_space_constant"] is True
    assert cs.depth_bias(10.0, 0.5)["world_space"] is True
    assert cs.softness(1)["world_space_consistent"] is True
    assert cs.sample_sequence(3, 8) == cs.sample_sequence(3, 8)
    assert len(cs.sample_sequence(0, 8)) == 8
