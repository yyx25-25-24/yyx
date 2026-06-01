"""快速验证项目能否正常导入并计算问题一结果（不弹窗、不跑优化）。"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def check_imports():
    import numpy
    import scipy
    import pandas
    import matplotlib
    from config import SMOKE_EFFECTIVE_TIME
    from agents.uav_agent import UAVAgent
    from agents.missile_agent import MissileAgent
    from models.occlusion import generate_target_samples, is_effectively_occluded
    from algorithms.differential_evolution import DifferentialEvolutionOptimizer
    from algorithms.genetic_algorithm import EnhancedGeneticAlgorithm
    print("[OK] 所有模块导入成功")
    print(f"     numpy={numpy.__version__}, scipy={scipy.__version__}")


def check_problem1():
    from config import SMOKE_EFFECTIVE_TIME
    from agents.uav_agent import UAVAgent
    from agents.missile_agent import MissileAgent
    from models.occlusion import generate_target_samples, is_effectively_occluded

    uav = UAVAgent("FY1")
    uav.set_flight_parameters(120.0, 180.0)
    uav.add_smoke_plan(1.5, 3.6)
    missile = MissileAgent("M1")
    target_samples = generate_target_samples()

    total = 0.0
    current_time = 1.5 + 3.6
    end_time = current_time + SMOKE_EFFECTIVE_TIME
    is_occluded = False
    start = None

    while current_time < end_time:
        uav.update(current_time)
        missile.update(current_time)
        if not missile.is_active:
            break
        smoke_positions = uav.get_active_smoke_positions(current_time)
        occluded = any(
            is_effectively_occluded(missile.current_position, pos, target_samples)
            for pos in smoke_positions
        )
        if occluded and not is_occluded:
            is_occluded = True
            start = current_time
        elif not occluded and is_occluded and start is not None:
            total += current_time - start
            is_occluded = False
        current_time += 0.01

    if is_occluded and start is not None:
        total += min(end_time, current_time) - start

    assert len(uav.smoke_clouds) > 0, "烟幕云团未初始化"
    assert total > 0, f"遮蔽时长为 0，计算逻辑异常"
    print(f"[OK] 问题一遮蔽时长 = {total:.6f} 秒")


def check_line_sphere_edge_case():
    from models.occlusion import line_sphere_intersection
    import numpy as np

    line_start = np.array([0.0, 0.0, 0.0])
    line_end = np.array([2.0, 0.0, 0.0])
    sphere_center = np.array([1.0, 1.0, 0.0])
    sphere_radius = 1.0

    assert line_sphere_intersection(line_start, line_end, sphere_center, sphere_radius), \
        "线段与球面相交的边界条件未通过"
    print("[OK] 线段-球相交边界测试通过")


def check_differential_evolution_quick():
    from algorithms.differential_evolution import DifferentialEvolutionOptimizer

    optimizer = DifferentialEvolutionOptimizer("FY1", "M1", popsize=5, maxiter=5, seed=123)
    result = optimizer.optimize()
    assert isinstance(result, dict), "差分进化返回类型错误"
    assert "success" in result, "差分进化结果缺少 success 字段"
    print("[OK] 差分进化 quick 测试通过")


if __name__ == "__main__":
    check_imports()
    check_problem1()
    check_line_sphere_edge_case()
    check_differential_evolution_quick()
    print("\n验证通过。完整优化请运行: python main.py --scenario 1")
