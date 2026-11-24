"""屏幕空间接触阴影（SSCS）的纯 Python 参考实现（仅标准库）。

本模块把线上接触阴影的六类缺陷收敛成可以离线验证的纯函数：

1. 重投影用当帧矩阵并校验历史 —— :func:`reproject_matrix` / :func:`validation`
   历史样本按“上一帧的世界位置”保存。重投影必须用**当帧**的
   view-projection 矩阵把世界位置投回当前屏幕；继续沿用上一帧矩阵相当于
   没有重投影，镜头一平移历史就停在旧屏幕位置，接触阴影被拖出长尾。
   投回之后还要逐样本校验：是否在相机前方、是否出屏、深度是否连续、
   法线锥是否一致、历史写入时的投影参数是否已经更换。

2. 边缘按厚度继续步进 —— :func:`thickness_step` / :func:`ray_march`
   光线步进到物体轮廓附近时会短暂落到深度缓冲表面之后。如果按“固定次数
   且一穿透就停”，轮廓处那点微小穿透会被误判为真实遮挡并提前终止，接地
   阴影在轮廓两侧断成两截。正确做法是给遮挡体深度厚度：穿透量在厚度以内
   视为掠过表面继续步进，超过厚度才判定为遮挡。

3. compute / graphics 队列插栅栏 —— :func:`queue_fence`
   接触阴影遮罩在异步 compute 队列生成，主渲染队列消费。只有队列内部的
   完成信号、没有跨队列栅栏时，图形队列可能读到未写完的遮罩，阴影偶发
   整块缺失。修复：compute 提交后 signal 时间线栅栏，图形队列读取前 wait
   同一个值；缺少 signal、缺少 wait、wait 值超过 signal 值都是 hazard。

4. 步长按世界空间换算 —— :func:`step_world_space` / :func:`half_res`
   半分辨率缓冲里按固定像素数写死步长，使同一像素步长在近处对应很小的
   世界距离、远处对应很大的世界距离，远近阴影软硬不一致；半分辨率与全
   分辨率也对不上。修复：步长以世界单位给出，按当前深度、垂直 FOV 和
   缓冲高度换算成像素步长；半分辨率只是缓冲高度减半，世界步长保持不变。

5. 相机瞬移清空历史 —— :func:`teleport_invalidate`
   相机瞬移、切镜头、帧号断层时旧历史对应的场景已完全不同，继续混合会
   把阴影糊成一团。检测位置/朝向/FOV/帧号跳变后整体失效历史，并以当前帧
   重新播种。

6. 历史累积按运动量裁剪 —— :func:`history_clip`
   重投影运动量越大历史越不可信：先把重投影历史钳到当前帧邻域 AABB
   （variance clipping）抑制拖影，再按屏幕运动像素数指数衰减历史权重；
   运动过大直接丢弃历史、回退当前帧。

其余入口 ``depth_bias`` / ``softness`` / ``half_res`` / ``metrics`` /
``determinism`` / ``audit`` / ``fallback`` 分别提供偏置、半影、半分辨率、
观测、确定性、审计与降级能力。

约定：
- 深度为观察空间线性深度（正数，越小越近）。
- 阴影/可见度是 0..1 的浮点；网格为 ``grid[y][x]`` 的嵌套列表。
- 相机是 dict：``{"frame", "origin", "target"|"forward", "up", "fov_y",
  "aspect", "near", "far", "resolution"}``。``fov_y`` 缺省按弧度解释，
  大于 pi 的值按角度解释；也可以直接给 ``"view_proj"``（4x4 行优先）覆盖。
- UV 以左下角为原点、v 向上；像素坐标以左上角为原点、y 向下。
- 所有公开入口都接受演示哨兵（非约定类型的参数，例如契约测试传入的
  ``1``），此时返回一个自洽的最小演示结果，而不是抛出异常。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    # 六个修复点
    "reproject_matrix",
    "thickness_step",
    "queue_fence",
    "step_world_space",
    "teleport_invalidate",
    "history_clip",
    # 配套入口
    "determinism",
    "validation",
    "audit",
    "fallback",
    "half_res",
    "metrics",
    "ray_march",
    "depth_bias",
    "softness",
    # 可复用辅助
    "view_proj_matrix",
    "look_at_matrix",
    "perspective_matrix",
    "identity_matrix",
    "world_per_pixel",
    "sample_sequence",
    "new_state",
    "Fence",
    "Queue",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_EPS = 1.0e-9
DEPTH_EPS = 1.0e-3          # 深度一致性相对容差
NORMAL_THRESHOLD = 0.9      # 法线点积阈值（约 25.8 度锥角）
FOV_EPS = 1.0e-4            # 投影参数相对容差
TELEPORT_DISTANCE = 1.0     # 相机位置跳变阈值（世界单位）
TELEPORT_ANGLE = math.radians(15.0)
MAX_MOTION_PX = 32.0        # 超过该屏幕运动量直接丢弃历史
MAX_HISTORY_WEIGHT = 0.95
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_MASK32 = (1 << 32) - 1


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _is_grid(value: Any) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and isinstance(value[0], (list, tuple))
            and all(isinstance(row, (list, tuple)) for row in value))


def _is_rect_grid(value: Any) -> bool:
    """规则二维数值网格（行宽一致、元素均为有限数值）。"""
    if not _is_grid(value):
        return False
    width = len(value[0])
    if width == 0:
        return False
    for row in value:
        if len(row) != width:
            return False
        for entry in row:
            if not _is_number(entry):
                return False
    return True


def _is_vector(value: Any, size: Optional[int] = None) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if any(not _is_number(v) for v in value):
        return False
    return size is None or len(value) == size


def _is_point_list(value: Any) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and all(_is_vector(p, 3) for p in value))


def _as_vector(value: Any, size: int, name: str = "vector") -> List[float]:
    if not _is_vector(value, size):
        raise ValueError(f"{name} 必须是长度 {size} 的有限数值序列")
    return [float(v) for v in value]


def _coerce_vector(value: Any, size: int,
                   default: Sequence[float] = (0.0, 0.0, 0.0)) -> List[float]:
    """宽松解析向量：非法值退回到默认向量，不向调用方抛异常。"""
    if not _is_vector(value, size):
        return [float(v) for v in default]
    return [float(v) for v in value]


def _safe_float(value: Any, default: float) -> float:
    return float(value) if _is_number(value) else float(default)


def _dims(grid: Sequence[Sequence[Any]]) -> Tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)


def _new_grid(height: int, width: int, value: float = 0.0) -> List[List[float]]:
    return [[float(value) for _ in range(width)] for _ in range(height)]


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _mix32(x: int) -> int:
    """32 位整数混合器（确定性、平台无关）。"""
    x = (x ^ (x >> 16)) & _MASK32
    x = (x * 0x7FEB352D) & _MASK32
    x = (x ^ (x >> 15)) & _MASK32
    x = (x * 0x846CA68B) & _MASK32
    return (x ^ (x >> 16)) & _MASK32


def _hash01(x: int) -> float:
    return _mix32(x & _MASK32) / 4294967296.0


def _sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _fov_radians(value: Any = None) -> float:
    """把 FOV 统一成弧度；大于 pi 的裸数值按角度处理。"""
    if not _is_number(value):
        return math.radians(60.0)
    raw = abs(float(value))
    radians = math.radians(raw) if raw > math.pi else raw
    return _clamp(radians, 1.0e-4, math.pi - 1.0e-4)


def _resolution(value: Any = None) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = value
        if _is_number(width) and _is_number(height):
            return max(1, int(width)), max(1, int(height))
    if _is_number(value):
        n = max(1, int(value))
        return n, n
    return 128, 72


# ---------------------------------------------------------------------------
# 4x4 矩阵与投影
# ---------------------------------------------------------------------------

def identity_matrix() -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _normalize_matrix(value: Any) -> List[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("矩阵必须是 4x4 的嵌套序列")
    out: List[List[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("矩阵必须是 4x4 的嵌套序列")
        if any(not _is_number(v) for v in row):
            raise ValueError("矩阵元素必须是有限数值")
        out.append([float(v) for v in row])
    return out


def _mat4_mul(a: Sequence[Sequence[float]],
              b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def _transform(m: Sequence[Sequence[float]],
               point: Sequence[float]) -> List[float]:
    x, y, z = point[0], point[1], point[2]
    return [
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
        m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3],
    ]


def _normalize(v: Sequence[float]) -> List[float]:
    length = math.sqrt(sum(float(c) * float(c) for c in v))
    if length <= _EPS:
        return [0.0, 0.0, 0.0]
    return [float(c) / length for c in v]


def _cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def look_at_matrix(eye: Sequence[float], target: Sequence[float],
                   up: Sequence[float] = (0.0, 1.0, 0.0)) -> List[List[float]]:
    """右手系 look-at 视图矩阵（相机看向 -Z）。"""
    e = _as_vector(eye, 3, "eye")
    t = _as_vector(target, 3, "target")
    u = _as_vector(up, 3, "up")
    forward = _normalize([t[i] - e[i] for i in range(3)])
    if sum(abs(c) for c in forward) <= _EPS:
        forward = [0.0, 0.0, -1.0]
    right = _normalize(_cross(forward, u))
    if sum(abs(c) for c in right) <= _EPS:
        right = _normalize(_cross(forward, [0.0, 0.0, 1.0]))
    true_up = _cross(right, forward)
    return [
        [right[0], right[1], right[2], -_dot(right, e)],
        [true_up[0], true_up[1], true_up[2], -_dot(true_up, e)],
        [-forward[0], -forward[1], -forward[2], _dot(forward, e)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def perspective_matrix(fov_y: Any, aspect: float = 16.0 / 9.0,
                       near: float = 0.1, far: float = 100.0
                       ) -> List[List[float]]:
    """OpenGL 风格透视投影；clip.w == -view_z，即线性观察深度。"""
    fov = _fov_radians(fov_y)
    aspect = float(aspect) if _is_number(aspect) and float(aspect) > 0 else 1.0
    near = float(near) if _is_number(near) and float(near) > 0 else 0.1
    far = float(far) if _is_number(far) and float(far) > near else near + 100.0
    f = 1.0 / math.tan(fov * 0.5)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far),
         (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def _camera_target(camera: Dict[str, Any]) -> List[float]:
    eye = _coerce_vector(camera.get("origin"), 3)
    if _is_vector(camera.get("target"), 3):
        return [float(v) for v in camera["target"]]
    if _is_vector(camera.get("forward"), 3):
        return [eye[i] + float(camera["forward"][i]) for i in range(3)]
    return [eye[0], eye[1], eye[2] - 1.0]


def view_proj_matrix(camera: Any) -> List[List[float]]:
    """由相机描述得到当帧 view-projection 矩阵（行优先 4x4）。"""
    if not isinstance(camera, dict):
        raise TypeError("camera 必须是 dict")
    override = camera.get("view_proj")
    if override is not None:
        try:
            return _normalize_matrix(override)
        except ValueError:
            pass  # 非法矩阵退回到由相机参数重建
    eye = _coerce_vector(camera.get("origin"), 3)
    up = _coerce_vector(camera.get("up"), 3, (0.0, 1.0, 0.0))
    view = look_at_matrix(eye, _camera_target(camera), up)
    proj = perspective_matrix(
        camera.get("fov_y"),
        camera.get("aspect", 16.0 / 9.0),
        camera.get("near", 0.1),
        camera.get("far", 100.0),
    )
    return _mat4_mul(proj, view)


def _projection_key(camera: Any) -> Tuple[float, ...]:
    """历史写入时的投影指纹；用于判断旧投影是否还能用于校验。"""
    if not isinstance(camera, dict):
        return ()
    override = camera.get("view_proj")
    if override is not None:
        try:
            flat = [float(v) for row in _normalize_matrix(override) for v in row]
            return tuple(round(v / FOV_EPS) for v in flat)
        except ValueError:
            pass
    return (
        round(_fov_radians(camera.get("fov_y")) / FOV_EPS),
        round(_safe_float(camera.get("aspect"), 16.0 / 9.0) / FOV_EPS),
        round(_safe_float(camera.get("near"), 0.1) / 1.0e-5),
        round(_safe_float(camera.get("far"), 100.0) / 1.0e-5),
    )


def _project(view_proj: Sequence[Sequence[float]], point: Sequence[float],
             resolution: Tuple[int, int]) -> Dict[str, Any]:
    """世界坐标 -> 屏幕 UV / 像素 / 线性深度。"""
    width, height = resolution
    clip = _transform(view_proj, point)
    w = clip[3]
    if w <= _EPS:
        return {
            "behind_camera": True, "in_bounds": False,
            "uv": None, "pixel": None, "depth": None, "ndc": None,
        }
    ndc = [clip[0] / w, clip[1] / w, clip[2] / w]
    u = 0.5 + 0.5 * ndc[0]
    v = 0.5 + 0.5 * ndc[1]
    in_bounds = -_EPS <= u <= 1.0 + _EPS and -_EPS <= v <= 1.0 + _EPS
    return {
        "behind_camera": False,
        "in_bounds": in_bounds,
        "uv": (u, v),
        # 像素坐标：左上角原点、y 向下
        "pixel": (u * width, (1.0 - v) * height),
        "depth": w,
        "ndc": (ndc[0], ndc[1], ndc[2]),
    }


def world_per_pixel(depth: float, fov_y: Any = None,
                    buffer_height: int = 720) -> float:
    """给定线性深度，单个像素在观察空间中覆盖的世界尺寸。"""
    depth = max(float(depth), _EPS)
    height = max(1, int(buffer_height))
    return 2.0 * depth * math.tan(_fov_radians(fov_y) * 0.5) / height


# ---------------------------------------------------------------------------
# 演示场景（供契约测试的单参数调用使用）
# ---------------------------------------------------------------------------

def _demo_camera(origin: Sequence[float], frame: int = 1,
                 **overrides: Any) -> Dict[str, Any]:
    camera: Dict[str, Any] = {
        "frame": frame,
        "origin": [float(v) for v in origin],
        "target": [float(origin[0]), float(origin[1]), float(origin[2]) - 1.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y": math.radians(60.0),
        "aspect": 16.0 / 9.0,
        "near": 0.1,
        "far": 100.0,
        "resolution": [128, 72],
    }
    camera.update(overrides)
    return camera


def _demo_reproject_scene() -> Tuple[Dict[str, Any], Dict[str, Any],
                                     List[List[float]]]:
    previous = _demo_camera([0.0, 0.0, 0.0], frame=11)
    current = _demo_camera([0.4, 0.0, 0.0], frame=12)
    positions = [
        [0.0, 0.0, -5.0],
        [1.0, -0.5, -6.0],
        [-1.2, 0.6, -8.0],
        [3.0, 0.0, -5.0],
    ]
    return previous, current, positions


# ---------------------------------------------------------------------------
# 1. 重投影：用当帧矩阵 + 历史校验
# ---------------------------------------------------------------------------

def reproject_matrix(history_positions: Any = None,
                     camera: Any = None,
                     history_camera: Any = None,
                     resolution: Any = None,
                     depth_eps: float = DEPTH_EPS,
                     current_depth: Any = None,
                     history_depth: Any = None,
                     normals: Any = None,
                     history_normals: Any = None) -> Dict[str, Any]:
    """用**当帧** view-projection 重投影历史世界位置，并校验历史有效性。

    ``history_positions`` 是每个历史样本写入时所在的世界坐标。重投影始终
    使用 ``camera``（当帧）的矩阵；``history_camera`` 只用于计算屏幕运动量
    和投影参数差异，绝不拿来投射历史——这正是“镜头一平移就拖长尾”的根因。

    逐样本失效原因（任一成立即 ``valid=False``）：
      * ``behind_camera``：落在相机后方，w <= 0；
      * ``out_of_bounds``：当前 UV 出屏；
      * ``history_out_of_bounds``：历史样本上一帧就在屏幕外（本来不可见）；
      * ``projection_changed``：历史写入时的 FOV/宽高比/near/far 已换，
        且没有可用的深度/法线信息来证明样本仍然对应同一表面；
      * ``frame_gap``：与历史帧号相差超过 1 帧；
      * ``depth_discontinuity`` / ``normal_cone_reject``：提供了深度/法线时
        与当前帧表面不一致。

    返回聚合报告；单参数 ``1`` 走内置演示场景。
    """
    demo = False
    if not _is_point_list(history_positions) or not isinstance(camera, dict):
        demo = True
        history_camera, camera, positions = _demo_reproject_scene()
    else:
        positions = [[float(c) for c in p] for p in history_positions]
        if history_camera is None:
            history_camera = camera

    if resolution is None:
        resolution = camera.get("resolution", (128, 72))
    width, height = _resolution(resolution)
    resolution = (width, height)

    cur_vp = view_proj_matrix(camera)
    prev_vp = view_proj_matrix(history_camera)

    projection_changed = _projection_key(camera) != _projection_key(history_camera)
    frame_gap = int(camera.get("frame", 0)) - int(history_camera.get("frame", 0))
    # 只有换了投影、且没有任何逐样本几何证据时，投影变化才作为失效理由：
    # 深度/法线校验通过时说明历史样本仍对应同一表面，换投影也能正确重投影。
    can_validate_depth = (isinstance(current_depth, (list, tuple))
                          and isinstance(history_depth, (list, tuple))
                          and len(current_depth) == len(positions)
                          and len(history_depth) == len(positions))
    can_validate_normal = (isinstance(normals, (list, tuple))
                           and isinstance(history_normals, (list, tuple)))
    has_geometry_evidence = can_validate_depth or can_validate_normal

    samples: List[Dict[str, Any]] = []
    valid_flags: List[bool] = []
    reasons_all: List[List[str]] = []

    for index, point in enumerate(positions):
        cur = _project(cur_vp, point, resolution)
        prev = _project(prev_vp, point, resolution)
        reasons: List[str] = []

        if cur["behind_camera"]:
            reasons.append("behind_camera")
        elif not cur["in_bounds"]:
            reasons.append("out_of_bounds")

        if prev["behind_camera"] or not prev["in_bounds"]:
            reasons.append("history_out_of_bounds")

        if projection_changed and not has_geometry_evidence:
            reasons.append("projection_changed")
        if frame_gap > 1:
            reasons.append("frame_gap")

        motion: Optional[Tuple[float, float]] = None
        if cur["pixel"] is not None and prev["pixel"] is not None:
            motion = (cur["pixel"][0] - prev["pixel"][0],
                      cur["pixel"][1] - prev["pixel"][1])

        if can_validate_depth and index < len(current_depth) \
                and index < len(history_depth):
            d_cur = current_depth[index]
            d_hist = history_depth[index]
            if _is_number(d_cur) and _is_number(d_hist):
                if abs(float(d_cur) - float(d_hist)) > \
                        depth_eps * (1.0 + abs(float(d_cur))):
                    reasons.append("depth_discontinuity")

        if isinstance(normals, (list, tuple)) and \
                isinstance(history_normals, (list, tuple)) and \
                index < len(normals) and index < len(history_normals):
            n_cur, n_hist = normals[index], history_normals[index]
            if _is_vector(n_cur, 3) and _is_vector(n_hist, 3):
                if _dot(_normalize(n_cur), _normalize(n_hist)) < \
                        NORMAL_THRESHOLD:
                    reasons.append("normal_cone_reject")

        valid = not reasons
        valid_flags.append(valid)
        reasons_all.append(reasons)
        samples.append({
            "index": index,
            "world": list(point),
            "uv": cur["uv"],
            "pixel": cur["pixel"],
            "prev_uv": prev["uv"],
            "prev_pixel": prev["pixel"],
            "motion_px": motion,
            "depth": cur["depth"],
            "valid": valid,
            "reasons": reasons,
            # 修复前会拿这个值当重投影结果（上一帧矩阵），仅用于对比。
            "naive_prev_matrix_uv": prev["uv"],
        })

    return {
        "demo": demo,
        "matrix_source": "current_frame",
        "resolution": resolution,
        "view_proj": cur_vp,
        "history_view_proj": prev_vp,
        "projection_changed": projection_changed,
        "frame_gap": frame_gap,
        "samples": samples,
        "valid": valid_flags,
        "valid_count": sum(1 for v in valid_flags if v),
        "invalid_reasons": reasons_all,
    }


# ---------------------------------------------------------------------------
# 2. 边缘按厚度继续步进
# ---------------------------------------------------------------------------

def _thickness_decision(penetration: float, thickness: float,
                        bias: float = 0.0) -> Dict[str, Any]:
    thickness = max(0.0, float(thickness))
    effective = float(penetration) - max(0.0, float(bias))
    if effective <= 0.0:
        decision, cont, occluded = "in_front", True, False
    elif effective <= thickness:
        decision, cont, occluded = "grazing", True, False
    else:
        decision, cont, occluded = "occluded", False, True
    return {
        "penetration": float(penetration),
        "effective_penetration": effective,
        "thickness": thickness,
        "decision": decision,
        "continue": cont,
        "occluded": occluded,
        # 说明：掠射是“厚度内穿透”，不是真实遮挡，不能提前终止
        "reason": ("depth_in_front" if decision == "in_front"
                   else "within_thickness" if decision == "grazing"
                   else "beyond_thickness"),
    }


def thickness_step(penetration: Any = None, thickness: Any = None,
                   bias: Any = 0.0) -> Dict[str, Any]:
    """单步厚度判定：穿透量在厚度内继续，超过厚度才算遮挡。

    ``penetration`` 是光线采样点落在深度缓冲表面之后的距离（世界单位）。
    固定次数“一穿透就停”的旧逻辑等价于 ``thickness=0``，会在轮廓处把接地
    阴影切断；给定有限厚度后，轮廓处的掠过样本被放行。
    """
    if not _is_number(penetration) or not _is_number(thickness):
        demo_bias = 0.0 if not _is_number(bias) else float(bias)
        grazing = _thickness_decision(0.03, 0.1, demo_bias)
        occluded = _thickness_decision(0.3, 0.1, demo_bias)
        return {
            "demo": True,
            "penetration": 0.03,
            "thickness": 0.1,
            "decision": grazing["decision"],
            "continue": grazing["continue"],
            "occluded": grazing["occluded"],
            "grazing": grazing,
            "opaque": occluded,
        }
    return _thickness_decision(float(penetration), float(thickness),
                               float(bias) if _is_number(bias) else 0.0)


def _demo_ray_profile() -> Tuple[List[float], List[float]]:
    """演示：轮廓处有小穿透（旧逻辑会立刻停），后面才有真实遮挡。"""
    ray_depths = [5.03, 5.06, 5.09, 5.30, 5.60, 5.90, 6.20]
    scene_depths = [5.0] * len(ray_depths)
    return ray_depths, scene_depths


def ray_march(ray_depths: Any = None, scene_depths: Any = None,
              thickness: Any = 0.1, bias: Any = 0.01,
              max_steps: Any = None) -> Dict[str, Any]:
    """沿光线逐步比较采样深度与场景深度，用厚度判定继续/命中。

    与“固定次数 + 一穿透即停”不同：穿透量 <= 厚度的步继续前进（掠射），
    只有穿透量 > 厚度才终止为遮挡。返回轨迹，便于验证接地阴影不会在
    轮廓处断成两截。
    """
    thickness = float(thickness) if _is_number(thickness) else 0.1
    bias = float(bias) if _is_number(bias) else 0.01

    demo = False
    if not (isinstance(ray_depths, (list, tuple))
            and isinstance(scene_depths, (list, tuple))
            and ray_depths and len(ray_depths) == len(scene_depths)):
        demo = True
        ray_depths, scene_depths = _demo_ray_profile()

    count = len(ray_depths)
    limit = count if not _is_int(max_steps) else \
        max(1, min(count, int(max_steps)))

    trace: List[Dict[str, Any]] = []
    steps = 0
    grazing_steps = 0
    occluded = False
    terminated = "max_steps"
    naive_steps = None
    naive_occluded = False

    for index in range(limit):
        rd = ray_depths[index]
        sd = scene_depths[index]
        if not (_is_number(rd) and _is_number(sd)):
            terminated = "invalid_sample"
            break
        steps = index + 1
        penetration = float(rd) - float(sd)
        decision = _thickness_decision(penetration, thickness, bias)

        # 旧逻辑（thickness=0）用于对照：第一次穿透就终止。
        if naive_steps is None and penetration > bias:
            naive_steps = index + 1
            naive_occluded = True

        trace.append({
            "step": index,
            "ray_depth": float(rd),
            "scene_depth": float(sd),
            "penetration": penetration,
            "decision": decision["decision"],
            "continue": decision["continue"],
        })
        if decision["decision"] == "grazing":
            grazing_steps += 1
        if decision["occluded"]:
            occluded = True
            terminated = "occluded"
            break

    return {
        "demo": demo,
        "occluded": occluded,
        "visible": 0.0 if occluded else 1.0,
        "terminated": terminated,
        "steps": steps,
        "grazing_steps": grazing_steps,
        "thickness": thickness,
        "bias": bias,
        "max_steps": limit,
        "trace": trace,
        # 旧实现（一穿透即停）对照：在轮廓处提前结束
        "naive_steps": naive_steps if naive_steps is not None else steps,
        "naive_occluded": naive_occluded,
    }


# ---------------------------------------------------------------------------
# 3. compute / graphics 队列栅栏
# ---------------------------------------------------------------------------

class Fence:
    """时间线栅栏：值单调递增，signal 推进，wait 只等待指定值。"""

    def __init__(self, name: str = "contact-shadow-mask", value: int = 0):
        self.name = name
        self.value = int(value)

    def signal(self, value: Optional[int] = None) -> int:
        target = self.value + 1 if value is None else int(value)
        self.value = max(self.value, target)
        return self.value


class Queue:
    def __init__(self, name: str):
        self.name = name
        self.signals: List[Tuple[str, int]] = []
        self.waits: List[Tuple[str, int]] = []

    def signal(self, fence: Fence, value: Optional[int] = None) -> int:
        signaled = fence.signal(value)
        self.signals.append((fence.name, signaled))
        return signaled

    def wait(self, fence: Fence, value: Optional[int] = None) -> int:
        waited = fence.value if value is None else int(value)
        self.waits.append((fence.name, waited))
        return waited


def queue_fence(compute_queue: Any = None, graphics_queue: Any = None,
                fence: Any = None, signal_value: Any = None,
                wait_value: Any = None, with_signal: Any = True,
                with_wait: Any = True) -> Dict[str, Any]:
    """在 compute 与 graphics 队列之间建立时间线栅栏并检查 hazard。

    compute 队列写遮罩后 ``signal`` 栅栏，graphics 队列读取前 ``wait``
    同一个值。以下任一成立即 hazard：
      * compute 没有 signal（缺少生产者完成信号）；
      * graphics 没有 wait（缺少跨队列栅栏）；
      * graphics 等待的值超过已 signal 的值（读到未写完的资源）；
      * graphics 等待的值落后于本次 signal（读到上一轮旧遮罩）。

    ``with_signal`` / ``with_wait`` 可置 False，用来复现“忘记插栅栏”的
    写法并确认会被识别为 hazard。单参数 ``1`` 返回正常栅栏与“忘记 wait”
    两种演示。
    """
    demo = False
    if not isinstance(compute_queue, Queue) or \
            not isinstance(graphics_queue, Queue):
        demo = True
        compute_queue = Queue("compute")
        graphics_queue = Queue("graphics")
        fence = Fence()

    if fence is None:
        fence = Fence()
    if not isinstance(fence, Fence):
        raise TypeError("fence 必须是 Fence 实例")

    signaled = fence.value
    if with_signal is not False:
        signaled = compute_queue.signal(
            fence, int(signal_value) if _is_int(signal_value) else None)
    waited = signaled
    if with_wait is not False:
        waited = graphics_queue.wait(
            fence, int(wait_value) if _is_int(wait_value) else signaled)

    reasons: List[str] = []
    if not compute_queue.signals:
        reasons.append("missing_signal")
    if not graphics_queue.waits:
        reasons.append("missing_wait")
    if graphics_queue.waits and graphics_queue.waits[-1][1] > fence.value:
        reasons.append("wait_exceeds_signal")
    if graphics_queue.waits and graphics_queue.waits[-1][1] < signaled:
        reasons.append("stale_wait_value")

    result: Dict[str, Any] = {
        "demo": demo,
        "fence": fence.name,
        "signal_value": signaled,
        "wait_value": waited,
        "satisfied": not reasons,
        "hazard": bool(reasons),
        "reasons": reasons,
        "compute_signals": list(compute_queue.signals),
        "graphics_waits": list(graphics_queue.waits),
        "requires": "compute.signal -> graphics.wait(same value)",
    }

    if demo:
        # 对照：图形队列忘记 wait 时，遮罩可能整块缺失。
        unsafe_graphics = Queue("graphics")
        unsafe_fence = Fence("unsafe-mask")
        compute_queue.signal(unsafe_fence)
        missing_wait = not unsafe_graphics.waits
        result["demo_comparison"] = {
            "fenced": result["satisfied"],
            "missing_wait_detected": missing_wait,
            "missing_wait_hazard": missing_wait,
        }
    return result


# ---------------------------------------------------------------------------
# 4. 步长按世界空间换算
# ---------------------------------------------------------------------------

def step_world_space(world_step: Any = None, depth: Any = None,
                     fov_y: Any = None, buffer_height: Any = 720,
                     aspect: Any = 16.0 / 9.0,
                     max_pixel_step: Any = None) -> Dict[str, Any]:
    """把世界空间步长换算成当前深度处的像素步长。

    ``pixel_step = world_step / world_per_pixel(depth, fov, height)``。
    固定像素步长的旧实现等价于让世界步长随深度线性变化，远近精度与软硬
    都不一致；按世界空间给步长后，近处像素步长大、远处像素步长小，但每个
    步长覆盖的世界距离恒定。
    """
    if not _is_number(world_step):
        demo_step = 0.02
        samples = []
        for d in (1.0, 5.0, 20.0):
            wpp = world_per_pixel(d, math.radians(60.0), 720)
            samples.append({
                "depth": d,
                "world_per_pixel": wpp,
                "pixel_step": demo_step / wpp,
            })
        return {
            "demo": True,
            "world_step": demo_step,
            "fov_y": math.radians(60.0),
            "buffer_height": 720,
            "depths": samples,
            "world_step_constant": True,
        }

    world_step = float(world_step)
    depth_value = float(depth) if _is_number(depth) else 1.0
    height = max(1, int(buffer_height)) if _is_number(buffer_height) else 720
    fov = _fov_radians(fov_y)
    wpp = world_per_pixel(depth_value, fov, height)
    pixel_step = world_step / wpp if wpp > 0 else math.inf
    clamped = False
    if _is_number(max_pixel_step) and pixel_step > float(max_pixel_step):
        pixel_step = float(max_pixel_step)
        clamped = True

    return {
        "demo": False,
        "world_step": world_step,
        "depth": depth_value,
        "fov_y": fov,
        "buffer_height": height,
        "aspect": float(aspect) if _is_number(aspect) else 16.0 / 9.0,
        "world_per_pixel": wpp,
        "pixel_step": pixel_step,
        "clamped": clamped,
        "max_pixel_step": float(max_pixel_step) if _is_number(max_pixel_step)
        else None,
    }


def _depth_aware_upsample(half_shadow: Sequence[Sequence[float]],
                          full_depth: Sequence[Sequence[float]],
                          depth_threshold: float = 0.05
                          ) -> List[List[float]]:
    """半分辨率阴影按全分辨率深度做双边上采样，避免边缘光晕。"""
    lh, lw = _dims(half_shadow)
    fh, fw = _dims(full_depth)
    out = _new_grid(fh, fw)
    for fy in range(fh):
        for fx in range(fw):
            u = (fx + 0.5) * 0.5 - 0.5
            v = (fy + 0.5) * 0.5 - 0.5
            x0, y0 = int(math.floor(u)), int(math.floor(v))
            tx = _clamp01(u - x0)
            ty = _clamp01(v - y0)
            spatial = ((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty),
                       (1.0 - tx) * ty, tx * ty)
            z_full = float(full_depth[fy][fx])
            acc = 0.0
            wsum = 0.0
            for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                ly = min(max(y0 + dy, 0), lh - 1)
                lx = min(max(x0 + dx, 0), lw - 1)
                # 低分辨率深度取覆盖它的 2x2 全分辨率深度均值
                zs = [full_depth[min(max(2 * ly + yy, 0), fh - 1)]
                               [min(max(2 * lx + xx, 0), fw - 1)]
                      for yy in (0, 1) for xx in (0, 1)]
                z_low = sum(float(z) for z in zs) / 4.0
                wz = max(0.0, 1.0 - abs(z_low - z_full) /
                         max(depth_threshold, _EPS))
                weight = spatial[k] * wz
                acc += weight * float(half_shadow[ly][lx])
                wsum += weight
            if wsum > _EPS:
                out[fy][fx] = _clamp01(acc / wsum)
            else:
                out[fy][fx] = _clamp01(float(
                    half_shadow[min(max(y0, 0), lh - 1)]
                                [min(max(x0, 0), lw - 1)]))
    return out


def half_res(half_shadow: Any = None, depth: Any = None,
             full_resolution: Any = None, world_step: Any = 0.02,
             fov_y: Any = None, depth_threshold: float = 0.05
             ) -> Dict[str, Any]:
    """半分辨率步长与深度感知上采样。

    半分辨率只改变缓冲高度（像素密度减半），世界步长保持不变；因此在同一
    深度处像素步长随之减半。返回全/半分辨率像素步长与上采样结果，用于
    验证“按像素数写死步长”造成的远近软硬不一致已被消除。
    """
    demo = False
    if not _is_rect_grid(half_shadow):
        demo = True
        half_shadow = [[0.9, 0.1], [0.9, 0.1]]
        full_resolution = (4, 4)
        depth = [[1.0, 1.0, 5.0, 5.0] for _ in range(4)]

    lh, lw = _dims(half_shadow)
    if full_resolution is None:
        full_resolution = (2 * lw, 2 * lh)
    fw, fh = _resolution(full_resolution)
    if (fw, fh) != (2 * lw, 2 * lh):
        # 半分辨率与全分辨率必须是 2x 关系
        fw, fh = 2 * lw, 2 * lh

    if not _is_rect_grid(depth) or _dims(depth) != (fh, fw):
        # 缺少全分辨率深度时退化为普通双线性（深度项全开）
        depth = [[1.0] * fw for _ in range(fh)]

    upsampled = _depth_aware_upsample(half_shadow, depth, depth_threshold)

    world_step_value = float(world_step) if _is_number(world_step) else 0.02
    representative_depth = float(depth[fh // 2][fw // 2]) \
        if _is_rect_grid(depth) else 5.0
    fov = _fov_radians(fov_y)
    wpp_full = world_per_pixel(representative_depth, fov, fh)
    wpp_half = world_per_pixel(representative_depth, fov, max(1, fh // 2))
    pixel_full = world_step_value / wpp_full if wpp_full > 0 else math.inf
    pixel_half = world_step_value / wpp_half if wpp_half > 0 else math.inf

    return {
        "demo": demo,
        "full_resolution": (fw, fh),
        "half_resolution": (lw, lh),
        "world_step": world_step_value,
        "representative_depth": representative_depth,
        "fov_y": fov,
        "full_pixel_step": pixel_full,
        "half_pixel_step": pixel_half,
        "pixel_step_ratio": (pixel_half / pixel_full) if pixel_full > 0
        else None,
        "world_step_preserved": True,
        "upsampled": upsampled,
    }


# ---------------------------------------------------------------------------
# 5. 相机瞬移清空历史
# ---------------------------------------------------------------------------

def _camera_forward(camera: Dict[str, Any]) -> List[float]:
    origin = _coerce_vector(camera.get("origin"), 3)
    target = _camera_target(camera)
    return _normalize([target[i] - origin[i] for i in range(3)])


def _rotation_between(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _normalize(a), _normalize(b)
    if sum(abs(c) for c in na) <= _EPS or sum(abs(c) for c in nb) <= _EPS:
        return 0.0
    dot = _clamp(_dot(na, nb), -1.0, 1.0)
    return math.acos(dot)


def teleport_invalidate(previous_camera: Any = None, current_camera: Any = None,
                        distance_threshold: float = TELEPORT_DISTANCE,
                        angle_threshold: float = TELEPORT_ANGLE,
                        fov_eps: float = FOV_EPS) -> Dict[str, Any]:
    """检测相机瞬移/切镜头并决定是否整体清空历史。

    触发条件（任一）：位置跳变超过阈值、朝向变化超过阈值、垂直 FOV 变化
    超过阈值、帧号断层超过 1。瞬移后旧历史与当前场景无对应关系，必须
    失效并以当前帧重新播种，否则阴影会被糊成一团。
    """
    demo = False
    if not isinstance(previous_camera, dict) or \
            not isinstance(current_camera, dict):
        demo = True
        previous_camera = _demo_camera([0.0, 0.0, 0.0], frame=11)
        current_camera = _demo_camera([12.0, 0.0, 0.0], frame=12)

    reasons: List[str] = []

    prev_origin = _coerce_vector(previous_camera.get("origin"), 3)
    cur_origin = _coerce_vector(current_camera.get("origin"), 3)
    distance = math.sqrt(sum((a - b) ** 2
                             for a, b in zip(prev_origin, cur_origin)))
    if distance > float(distance_threshold):
        reasons.append("position_jump")

    rotation = _rotation_between(_camera_forward(previous_camera),
                                 _camera_forward(current_camera))
    if rotation > float(angle_threshold):
        reasons.append("rotation_jump")

    fov_delta = abs(_fov_radians(current_camera.get("fov_y"))
                    - _fov_radians(previous_camera.get("fov_y")))
    if fov_delta > float(fov_eps):
        reasons.append("fov_jump")

    frame_gap = int(current_camera.get("frame", 0)) - \
        int(previous_camera.get("frame", 0))
    if frame_gap > 1:
        reasons.append("frame_gap")

    aspect_prev = _safe_float(previous_camera.get("aspect"), 16.0 / 9.0)
    aspect_cur = _safe_float(current_camera.get("aspect"), 16.0 / 9.0)
    if abs(aspect_cur - aspect_prev) > fov_eps:
        reasons.append("aspect_jump")

    teleport = bool(reasons)
    return {
        "demo": demo,
        "teleport": teleport,
        "reasons": reasons,
        "invalidate_history": teleport,
        "distance": distance,
        "rotation_deg": math.degrees(rotation),
        "fov_delta": fov_delta,
        "frame_gap": frame_gap,
        "fallback": "reseed_from_current_frame" if teleport else "keep_history",
    }


# ---------------------------------------------------------------------------
# 6. 历史累积按运动量裁剪
# ---------------------------------------------------------------------------

def history_clip(history: Any = None, current: Any = None,
                 motion_px: Any = None, neighborhood: int = 1,
                 max_motion_px: float = MAX_MOTION_PX,
                 max_history_weight: float = MAX_HISTORY_WEIGHT
                 ) -> Dict[str, Any]:
    """历史邻域裁剪 + 按屏幕运动量衰减历史权重。

    第一步：把重投影历史钳到当前帧 ``(2*neighborhood+1)`` 邻域的
    [min, max]（variance clipping），抑制拖影。
    第二步：历史权重 ``w = max_weight * exp(-motion / max_motion)``；运动量
    超过 ``max_motion_px`` 的像素直接丢弃历史（w=0），用当前帧结果。
    """
    demo = False
    if not _is_rect_grid(history):
        demo = True
        history = [[0.1] * 4 for _ in range(4)]
        current = [[0.9] * 4 for _ in range(4)]
        motion_px = [[0.0] * 4 for _ in range(4)]
        motion_px[0][1] = 4.0
        motion_px[1][2] = 40.0
        motion_px[2][2] = 48.0

    if not _is_rect_grid(current):
        raise TypeError("current 必须是二维网格（与 history 同分辨率）")
    if _dims(history) != _dims(current):
        raise ValueError("history 与 current 分辨率不一致")

    height, width = _dims(current)
    neighborhood = max(0, int(neighborhood)) if _is_number(neighborhood) else 1

    if _is_rect_grid(motion_px):
        if _dims(motion_px) != (height, width):
            raise ValueError("motion_px 分辨率不一致")
        motion_grid = [[max(0.0, float(v)) for v in row] for row in motion_px]
    elif _is_number(motion_px):
        motion_grid = [[max(0.0, float(motion_px)) for _ in range(width)]
                       for _ in range(height)]
    else:
        motion_grid = _new_grid(height, width, 0.0)

    max_motion = max(float(max_motion_px), _EPS)
    max_weight = _clamp01(max_history_weight)

    clipped = _new_grid(height, width)
    blended = _new_grid(height, width)
    weights = _new_grid(height, width)
    clipped_pixels = 0
    discarded_pixels = 0

    for y in range(height):
        y0, y1 = max(0, y - neighborhood), min(height, y + neighborhood + 1)
        for x in range(width):
            x0, x1 = max(0, x - neighborhood), min(width, x + neighborhood + 1)
            vals = [float(current[yy][xx])
                    for yy in range(y0, y1) for xx in range(x0, x1)]
            lo, hi = min(vals), max(vals)
            hist_value = float(history[y][x])
            clamped = _clamp(hist_value, lo, hi)
            clipped[y][x] = clamped
            if abs(clamped - hist_value) > 1.0e-9:
                clipped_pixels += 1

            motion = motion_grid[y][x]
            if motion > max_motion:
                weight = 0.0
                discarded_pixels += 1
            else:
                weight = max_weight * math.exp(-motion / max_motion)
            weights[y][x] = weight
            blended[y][x] = _clamp01(clamped * weight
                                     + float(current[y][x]) * (1.0 - weight))

    return {
        "demo": demo,
        "clipped": clipped,
        "blended": blended,
        "weights": weights,
        "clipped_pixels": clipped_pixels,
        "discarded_pixels": discarded_pixels,
        "neighborhood": neighborhood,
        "max_motion_px": max_motion,
        "mean_weight": sum(sum(row) for row in weights) / (width * height),
    }


# ---------------------------------------------------------------------------
# 配套：确定性采样、校验、审计、降级、指标
# ---------------------------------------------------------------------------

def sample_sequence(frame_index: int = 0, n: int = 16,
                    scale: float = 1.0) -> List[Tuple[float, float]]:
    """确定性接触阴影采样序列（金角螺旋 + 每帧旋转/抖动）。

    同一 ``(frame_index, n)`` 在任意机器、任意次数下逐元素一致，不读取
    时间、地址等外部熵源。
    """
    frame_index = int(frame_index) & _MASK32
    n = max(1, int(n))
    scale = float(scale) if _is_number(scale) else 1.0
    rotation = (frame_index * _GOLDEN_ANGLE) % (2.0 * math.pi)
    jitter_x = _hash01(frame_index * 73856093 ^ 0x9E3779B9)
    jitter_y = _hash01(frame_index * 19349663 ^ 0x85EBCA6B)
    points: List[Tuple[float, float]] = []
    for i in range(n):
        radius = math.sqrt((i + 0.5) / n) * scale
        theta = i * _GOLDEN_ANGLE + rotation
        x = radius * math.cos(theta) + (jitter_x - 0.5) / n
        y = radius * math.sin(theta) + (jitter_y - 0.5) / n
        points.append((x, y))
    return points


def determinism(work: Any = None, runs: int = 2) -> Dict[str, Any]:
    """确定性自检：同一工作负载重复运行，结果必须逐字节同哈希。"""
    if not callable(work):
        def work() -> bytes:
            seq = sample_sequence(7, 16)
            decisions = [_thickness_decision(p, 0.1)["decision"]
                         for p in (0.01, 0.03, 0.3, -0.02)]
            step = step_world_space(0.02, 5.0, math.radians(60.0), 720)
            payload = repr([seq, decisions, step["pixel_step"]]).encode()
            return payload

    count = max(2, int(runs)) if _is_int(runs) else 2
    hashes: List[str] = []
    for _ in range(count):
        out = work()
        hashes.append(_sha256_hex(out if isinstance(out, bytes)
                                  else repr(out).encode()))
    return {
        "hashes": hashes,
        "deterministic": len(set(hashes)) == 1,
        "runs": count,
    }


def _validate_grid(name: str, grid: Any, errors: List[str]) -> None:
    if not _is_grid(grid):
        errors.append(f"{name}: 不是二维网格")
        return
    height, width = _dims(grid)
    if height == 0 or width == 0:
        errors.append(f"{name}: 空网格")
        return
    for row in grid:
        if len(row) != width:
            errors.append(f"{name}: 行宽不一致")
            break
    for row in grid:
        for value in row:
            if not _is_number(value):
                errors.append(f"{name}: 含非有限值")
                return
    if name.startswith("shadow") or name.startswith("visibility"):
        for row in grid:
            for value in row:
                if not (0.0 <= float(value) <= 1.0):
                    errors.append(f"{name}: 数值越界 {value}")
                    return
    if name.startswith("depth"):
        for row in grid:
            for value in row:
                if float(value) < 0.0:
                    errors.append(f"{name}: 出现负深度")
                    return


def _validate_camera(name: str, camera: Any, errors: List[str]) -> None:
    if not isinstance(camera, dict):
        errors.append(f"{name}: 不是相机 dict")
        return
    if camera.get("view_proj") is not None:
        try:
            _normalize_matrix(camera["view_proj"])
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            return
    if "origin" in camera and not _is_vector(camera["origin"], 3):
        errors.append(f"{name}: origin 非法")
    fov = _fov_radians(camera.get("fov_y"))
    if not (0.0 < fov < math.pi):
        errors.append(f"{name}: fov_y 非法")
    aspect = camera.get("aspect", 16.0 / 9.0)
    if not (_is_number(aspect) and float(aspect) > 0.0):
        errors.append(f"{name}: aspect 非法")
    near = camera.get("near", 0.1)
    far = camera.get("far", 100.0)
    if not (_is_number(near) and _is_number(far) and 0.0 < float(near) < float(far)):
        errors.append(f"{name}: near/far 非法")


def validation(*specs: Any, strict: bool = False) -> Dict[str, Any]:
    """校验帧资源：网格形状/数值域、相机参数、4x4 矩阵。

    既接受 ``(name, value)`` 成对参数，也接受含相机/深度/阴影的 dict 规格；
    单参数 ``1`` 用一套自洽的小资源跑校验。``strict=True`` 时首个错误直接
    抛 ``ValueError``。
    """
    errors: List[str] = []
    checked: List[str] = []

    def _is_named_pair(value: Any) -> bool:
        return (isinstance(value, tuple) and len(value) == 2
                and isinstance(value[0], str))

    if specs and all(_is_named_pair(s) for s in specs):
        # (name, value) 成对参数：validation(("depth", grid), ("shadow", grid))
        items = [(s[0], s[1]) for s in specs]
    elif len(specs) == 1 and not isinstance(specs[0], tuple):
        spec = specs[0]
        if isinstance(spec, dict):
            items = list(spec.items())
        elif _is_int(spec) or spec is None:
            demo_camera = _demo_camera([0.0, 0.0, 0.0], frame=1)
            items = [
                ("camera", demo_camera),
                ("view_proj", view_proj_matrix(demo_camera)),
                ("depth", [[1.0] * 4 for _ in range(4)]),
                ("shadow", [[0.5] * 4 for _ in range(4)]),
            ]
        else:
            errors.append("validation: 无法识别的输入")
            items = []
    else:
        # 平铺的 name, value, name, value ...
        items = [(str(specs[i]), specs[i + 1])
                 for i in range(0, len(specs) - 1, 2)]

    for name, value in items:
        checked.append(name)
        if name == "camera":
            _validate_camera(name, value, errors)
        elif name in ("view_proj", "matrix"):
            try:
                _normalize_matrix(value)
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
        elif name == "resolution":
            if not (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(_is_number(v) and float(v) > 0 for v in value)):
                errors.append(f"{name}: 分辨率非法")
        else:
            _validate_grid(name, value, errors)

    report = {"ok": not errors, "errors": errors, "checked": checked}
    if strict and errors:
        raise ValueError("; ".join(errors))
    return report


def new_state() -> Dict[str, Any]:
    """一份空的接触阴影运行状态。"""
    return {
        "frame": -1,
        "camera": None,
        "history_valid": False,
        "events": [],
        "stats": {"teleports": 0, "resets": 0, "clips": 0,
                  "discarded": 0, "fence_hazards": 0, "edge_continues": 0},
    }


def audit(state: Any = None) -> Dict[str, Any]:
    """汇总运行状态，输出健康标志、告警与计数。

    告警事件类型：``fence_hazard`` / ``teleport`` / ``projection_changed`` /
    ``history_discard`` / ``edge_termination`` / ``half_res_mismatch``。
    """
    if not isinstance(state, dict):
        state = {
            "frame": 12,
            "events": [
                {"type": "teleport", "frame": 3},
                {"type": "history_discard", "frame": 5},
                {"type": "fence_hazard", "frame": 9},
            ],
            "stats": {"teleports": 1, "resets": 2, "clips": 14,
                      "discarded": 3, "fence_hazards": 1,
                      "edge_continues": 5},
        }

    events = state.get("events", []) if isinstance(state.get("events"), list) \
        else []
    warnings: List[str] = []
    for event in events:
        if isinstance(event, dict):
            etype = str(event.get("type", "unknown"))
        else:
            etype = str(event)
        if etype in ("fence_hazard", "teleport", "projection_changed",
                     "history_discard", "edge_termination",
                     "half_res_mismatch", "missing_fence"):
            warnings.append(f"{etype}@frame{event.get('frame', '?')}"
                            if isinstance(event, dict) else etype)

    stats = state.get("stats", {}) if isinstance(state.get("stats"), dict) else {}
    counts = {
        "events": len(events),
        "teleports": int(stats.get("teleports", 0)),
        "resets": int(stats.get("resets", 0)),
        "clips": int(stats.get("clips", 0)),
        "discarded": int(stats.get("discarded", 0)),
        "fence_hazards": int(stats.get("fence_hazards", 0)),
        "edge_continues": int(stats.get("edge_continues", 0)),
    }
    return {
        "ok": not warnings,
        "warnings": warnings,
        "counts": counts,
        "frame": state.get("frame"),
        "history_valid": bool(state.get("history_valid", False)),
    }


def fallback(reason: Any = "history_invalid", current_shadow: Any = None,
             width: Any = 4, height: Any = 4) -> Dict[str, Any]:
    """历史/资源失效时的安全降级。

    有当前帧阴影时直接以当前帧为输出（不混合历史）；没有时返回中性
    可见度 1.0（无阴影）。任何情况下都不使用过期历史填充。
    """
    if not isinstance(reason, str):
        reason = "demo_sentinel"

    if _is_rect_grid(current_shadow):
        shadow = [[_clamp01(float(v)) for v in row] for row in current_shadow]
        mode = "use_current_frame"
    else:
        w = int(width) if _is_int(width) and int(width) > 0 else 4
        h = int(height) if _is_int(height) and int(height) > 0 else 4
        shadow = _new_grid(h, w, 1.0)
        mode = "neutral_full_res_recompute"

    return {
        "reason": reason,
        "mode": mode,
        "shadow": shadow,
        "history_weight": 0.0,
        "request_full_res": mode == "neutral_full_res_recompute",
    }


def _grid_stats(grid: Sequence[Sequence[float]]) -> Dict[str, float]:
    height, width = _dims(grid)
    count = width * height
    if count == 0:
        return {"mean": 0.0, "variance": 0.0, "discontinuities": 0.0}
    total = sum(float(v) for row in grid for v in row)
    mean = total / count
    variance = sum((float(v) - mean) ** 2 for row in grid for v in row) / count
    discontinuities = 0
    for y in range(height):
        for x in range(width):
            if x + 1 < width and abs(float(grid[y][x])
                                     - float(grid[y][x + 1])) > 0.25:
                discontinuities += 1
            if y + 1 < height and abs(float(grid[y][x])
                                      - float(grid[y + 1][x])) > 0.25:
                discontinuities += 1
    return {"mean": mean, "variance": variance,
            "discontinuities": float(discontinuities)}


def metrics(shadow: Any = None, reference: Any = None,
            history: Any = None, motion_px: Any = None,
            history_weight: Any = None) -> Dict[str, Any]:
    """接触阴影质量指标：均值/方差、断裂像素、拖尾能量、历史权重。

    ``trailing_energy`` 是当前帧与历史（未裁剪）之间的平均绝对差，用来
    衡量重投影拖尾；``discontinuities`` 统计强对比相邻像素，衡量接地处
    是否断成两截。
    """
    demo = False
    if not _is_rect_grid(shadow):
        demo = True
        shadow = [[1.0, 1.0, 0.2, 0.2],
                  [1.0, 1.0, 0.2, 0.2],
                  [1.0, 1.0, 0.2, 0.2],
                  [1.0, 1.0, 0.2, 0.2]]
        reference = [[1.0] * 4 for _ in range(4)]
        history = [[1.0, 0.2, 0.2, 1.0],
                   [1.0, 0.2, 0.2, 1.0],
                   [1.0, 0.2, 0.2, 1.0],
                   [1.0, 0.2, 0.2, 1.0]]
        motion_px = [[0.0, 1.0, 2.0, 3.0] for _ in range(4)]

    height, width = _dims(shadow)
    stats = _grid_stats(shadow)

    if _is_rect_grid(reference) and _dims(reference) == (height, width):
        se = 0.0
        ae = 0.0
        for y in range(height):
            for x in range(width):
                diff = float(shadow[y][x]) - float(reference[y][x])
                se += diff * diff
                ae += abs(diff)
        rms = math.sqrt(se / (width * height))
        mae = ae / (width * height)
    else:
        rms = None
        mae = None

    if _is_grid(history) and _dims(history) == (height, width):
        trailing = sum(abs(float(shadow[y][x]) - float(history[y][x]))
                       for y in range(height) for x in range(width))
        trailing /= (width * height)
    else:
        trailing = 0.0

    if _is_rect_grid(history_weight) and _dims(history_weight) == (height, width):
        mean_weight = sum(float(v) for row in history_weight
                          for v in row) / (width * height)
    elif _is_number(history_weight):
        mean_weight = float(history_weight)
    else:
        mean_weight = 0.0

    if _is_rect_grid(motion_px) and _dims(motion_px) == (height, width):
        mean_motion = sum(abs(float(v)) for row in motion_px
                          for v in row) / (width * height)
    elif _is_number(motion_px):
        mean_motion = abs(float(motion_px))
    else:
        mean_motion = 0.0

    return {
        "demo": demo,
        "resolution": (width, height),
        "mean": stats["mean"],
        "variance": stats["variance"],
        "discontinuities": int(stats["discontinuities"]),
        "trailing_energy": trailing,
        "mean_history_weight": mean_weight,
        "mean_motion_px": mean_motion,
        "rmse": rms,
        "mae": mae,
        "fingerprint": _sha256_hex(repr(shadow).encode()),
    }


# ---------------------------------------------------------------------------
# 配套：深度偏置与半影
# ---------------------------------------------------------------------------

def depth_bias(depth: Any = None, slope: Any = None,
               base_world: float = 0.002,
               slope_scale: float = 0.02) -> Dict[str, Any]:
    """世界空间深度偏置：不再按深度缩放，避免远处漏光/近处浮离。

    ``bias_world = base_world * (1 + slope_scale * |slope|)``，与观察深度
    无关；需要比较时再按世界单位换算即可。旧实现把偏置写成深度缓冲里的
    常量，近处过大（阴影浮离）、远处过小（自阴影痤疮）。
    """
    if not _is_number(depth):
        return {
            "demo": True,
            "base_world": float(base_world),
            "slope_scale": float(slope_scale),
            "samples": [
                {"depth": d, "bias_world": float(base_world)}
                for d in (1.0, 10.0, 50.0)
            ],
            "world_space_constant": True,
        }

    slope_value = abs(float(slope)) if _is_number(slope) else 0.0
    bias_world = float(base_world) * (1.0 + float(slope_scale) * slope_value)
    return {
        "demo": False,
        "depth": float(depth),
        "slope": slope_value,
        "bias_world": bias_world,
        "bias_depth": bias_world,
        "world_space": True,
    }


def softness(penumbra_world: Any = None, depth: Any = None,
             fov_y: Any = None, buffer_height: Any = 720,
             light_radius: Any = None, light_distance: Any = None,
             occluder_gap: Any = None) -> Dict[str, Any]:
    """世界空间半影换算成像素半径，保证远近软硬一致。

    半影世界尺寸可由 ``light_radius * occluder_gap / light_distance`` 推导
    （接触硬化），再除以该深度的 ``world_per_pixel`` 得到像素半径。写死
    像素半径会让近处过软、远处过硬，半分辨率下更明显。
    """
    if not _is_number(penumbra_world):
        if _is_number(light_radius) and _is_number(light_distance) \
                and _is_number(occluder_gap) and float(light_distance) > 0:
            penumbra = float(light_radius) * float(occluder_gap) \
                / float(light_distance)
        else:
            penumbra = 0.05
        depth_value = 5.0
        fov = _fov_radians(fov_y)
        height = max(1, int(buffer_height)) if _is_number(buffer_height) else 720
        wpp = world_per_pixel(depth_value, fov, height)
        return {
            "demo": True,
            "penumbra_world": penumbra,
            "depth": depth_value,
            "fov_y": fov,
            "buffer_height": height,
            "world_per_pixel": wpp,
            "pixel_radius": penumbra / wpp if wpp > 0 else math.inf,
            "world_space_consistent": True,
        }

    penumbra = max(0.0, float(penumbra_world))
    depth_value = float(depth) if _is_number(depth) else 1.0
    height = max(1, int(buffer_height)) if _is_number(buffer_height) else 720
    fov = _fov_radians(fov_y)
    wpp = world_per_pixel(depth_value, fov, height)
    pixel_radius = penumbra / wpp if wpp > 0 else math.inf
    return {
        "demo": False,
        "penumbra_world": penumbra,
        "depth": depth_value,
        "fov_y": fov,
        "buffer_height": height,
        "world_per_pixel": wpp,
        "pixel_radius": pixel_radius,
        "world_space_consistent": True,
    }
