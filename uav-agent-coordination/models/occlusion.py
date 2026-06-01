import numpy as np
import config
from models.kinematics import calculate_3d_direction_vector


def generate_target_samples() -> np.ndarray:
    """生成圆柱形真目标的采样点"""
    samples = []
    center_x, center_y, center_z = config.TARGET_CENTER
    radius = config.TARGET_RADIUS
    height = config.TARGET_HEIGHT

    # 底面圆周采样点
    bottom_z = center_z - height / 2
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        samples.append(np.array([x, y, bottom_z]))

    # 顶面圆周采样点
    top_z = center_z + height / 2
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        samples.append(np.array([x, y, top_z]))

    # 侧面中线圆周采样点
    mid_z = center_z
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        samples.append(np.array([x, y, mid_z]))

    # 轴线采样点
    for z in np.linspace(bottom_z, top_z, 5):
        samples.append(np.array([center_x, center_y, z]))

    return np.array(samples)


def line_sphere_intersection(line_start: np.ndarray, line_end: np.ndarray,
                            sphere_center: np.ndarray, sphere_radius: float) -> bool:
    """判断三维空间中线段与球面是否相交"""
    line_vector = line_end - line_start
    line_length_squared = np.dot(line_vector, line_vector)

    if line_length_squared == 0:
        return np.linalg.norm(line_start - sphere_center) <= sphere_radius

    # 计算球心到线段起点的向量在直线上的投影比例
    t = np.dot(sphere_center - line_start, line_vector) / line_length_squared

    # 找到线段上距离球心最近的点
    if t < 0.0:
        closest_point = line_start
    elif t > 1.0:
        closest_point = line_end
    else:
        closest_point = line_start + t * line_vector

    # 计算最近点到球心的距离
    distance = np.linalg.norm(closest_point - sphere_center)

    return distance <= sphere_radius


def is_effectively_occluded(missile_position: np.ndarray, smoke_position: np.ndarray,
                           target_samples: np.ndarray) -> bool:
    """判断烟幕是否对导弹形成有效遮蔽（所有采样点都被遮挡）"""
    for sample in target_samples:
        if not line_sphere_intersection(missile_position, sample,
                                       smoke_position, config.SMOKE_EFFECTIVE_RADIUS):
            return False
    return True


def calculate_occlusion_rate(missile_position: np.ndarray, smoke_positions: list,
                            target_samples: np.ndarray) -> float:
    """计算遮蔽率（被遮挡的采样点比例）"""
    if not smoke_positions:
        return 0.0

    occluded_count = 0
    for sample in target_samples:
        for smoke_pos in smoke_positions:
            if line_sphere_intersection(missile_position, sample,
                                       smoke_pos, config.SMOKE_EFFECTIVE_RADIUS):
                occluded_count += 1
                break

    return occluded_count / len(target_samples)
