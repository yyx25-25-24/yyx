import argparse
import json
import os
import random
import numpy as np

# 确保相对路径 result/、docs/ 始终指向项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

import config
from agents.coordinator import MultiAgentEnvironment
from agents.uav_agent import UAVAgent
from agents.missile_agent import MissileAgent
from algorithms.differential_evolution import DifferentialEvolutionOptimizer
from algorithms.genetic_algorithm import EnhancedGeneticAlgorithm
from models.occlusion import generate_target_samples, is_effectively_occluded
from models.simulator import is_occluded_by_smokes
from utils.visualization import (
    plot_3d_trajectory,
    plot_optimization_convergence,
    plot_occlusion_timeline
)
from utils.io_utils import (
    save_problem3_result,
    save_problem4_result,
    save_problem5_result,
    generate_text_report,
    save_result_json
)


def _ensure_docs_dir():
    os.makedirs("docs", exist_ok=True)


def _ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def _configure_random_seed(seed: int = None):
    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    return seed


def _get_runtime_optimizer_settings(problem_type: str, demo: bool = False, quick: bool = False) -> dict:
    if problem_type == "problem2":
        settings = {
            "de_population": config.DE_POPULATION_SIZE,
            "de_maxiter": config.DE_MAX_ITERATIONS,
        }
        if demo:
            settings.update({"de_population": 20, "de_maxiter": 30})
        if quick:
            settings.update({"de_population": 10, "de_maxiter": 10})
        return settings

    settings = {
        "ga_population": config.GA_POPULATION_SIZE[problem_type],
        "ga_max_iterations": config.GA_MAX_ITERATIONS[problem_type],
    }
    if demo and problem_type in config.DEMO_MODE_SETTINGS:
        settings.update({
            "ga_population": config.DEMO_MODE_SETTINGS[problem_type].get("ga_population", settings["ga_population"]),
            "ga_max_iterations": config.DEMO_MODE_SETTINGS[problem_type].get("ga_maxiter", settings["ga_max_iterations"]),
        })
    if quick:
        settings["ga_population"] = max(5, int(settings["ga_population"] * config.QUICK_MODE_FACTOR))
        settings["ga_max_iterations"] = max(5, int(settings["ga_max_iterations"] * config.QUICK_MODE_FACTOR))
    return settings


def _save_result_files(result: dict, problem_type: str, output_dir: str):
    _ensure_output_dir(output_dir)
    json_path = os.path.join(output_dir, f"scenario{problem_type}_result.json")
    save_result_json(result, json_path)


RUN_CONFIG = {
    "demo": False,
    "quick": False,
    "seed": None,
    "output_dir": config.OUTPUT_DIR,
}


def solve_problem1():
    """求解问题一：固定参数下的有效遮蔽时长计算"""
    print("=" * 80)
    print("问题一：固定参数烟幕干扰有效遮蔽时长计算")
    print("=" * 80)

    uav_speed = 120.0
    drop_delay = 1.5
    explosion_delay = 3.6
    direction_angle = 180.0

    uav = UAVAgent("FY1")
    uav.set_flight_parameters(uav_speed, direction_angle)
    uav.add_smoke_plan(drop_delay, explosion_delay)

    missile = MissileAgent("M1")
    environment = MultiAgentEnvironment([uav], [missile])

    start_time = drop_delay + explosion_delay
    end_time = start_time + config.SMOKE_EFFECTIVE_TIME
    time_step = 0.01

    print(f"\n开始计算遮蔽过程...")
    print(f"起爆时刻: {start_time}s")
    print(f"烟幕有效时间窗口: [{start_time}, {end_time}]s")
    print("-" * 40)

    occlusion_times, occlusion_details, total_occlusion_time = environment.evaluate_occlusion(
        start_time, end_time, time_step=time_step
    )

    print("\n" + "=" * 40)
    print(f"最终结果: 有效遮蔽时长 = {total_occlusion_time:.6f} 秒")
    print("=" * 40)

    print("\n生成可视化图表...")
    _ensure_docs_dir()
    plot_3d_trajectory([uav], [missile], "问题一：固定参数烟幕干扰3D轨迹",
                      "docs/problem1_trajectory.png")

    result = {
        "success": True,
        "problem": 1,
        "uav_speed": uav_speed,
        "drop_delay": drop_delay,
        "explosion_delay": explosion_delay,
        "direction_angle": direction_angle,
        "total_occlusion_time": total_occlusion_time,
        "occlusion_times": occlusion_times,
        "occlusion_details": {k: sorted(list(v)) for k, v in occlusion_details.items()},
    }
    _save_result_files(result, 1, RUN_CONFIG["output_dir"])

    with open("docs/problem1_result.txt", "w", encoding="utf-8") as f:
        f.write("问题一：固定参数烟幕干扰有效遮蔽时长计算结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"无人机速度: {uav_speed} m/s\n")
        f.write(f"投放延迟: {drop_delay} s\n")
        f.write(f"起爆延迟: {explosion_delay} s\n")
        f.write(f"飞行方向: {direction_angle}°\n\n")
        f.write(f"有效遮蔽时长: {total_occlusion_time:.6f} 秒\n")

    print("结果已保存到: docs/problem1_result.txt")


def solve_problem2():
    """求解问题二：单无人机单弹最优投放策略"""
    print("=" * 80)
    print("问题二：单无人机单弹最优投放策略优化")
    print("=" * 80)

    settings = _get_runtime_optimizer_settings("problem2", demo=RUN_CONFIG["demo"], quick=RUN_CONFIG["quick"])
    optimizer = DifferentialEvolutionOptimizer(
        "FY1", "M1",
        popsize=settings["de_population"],
        maxiter=settings["de_maxiter"],
        seed=RUN_CONFIG["seed"]
    )
    result = optimizer.optimize()

    if not result["success"]:
        print("优化失败，未找到有效解")
        return

    print("\n" + "=" * 60)
    print("最优解找到!")
    print(f"最大遮蔽时长: {result['occlusion_time']:.4f} 秒")
    print("\n最优参数:")
    print(f"无人机速度: {result['uav_speed']:.2f} m/s")
    print(f"投放延迟: {result['drop_delay']:.4f} s")
    print(f"起爆延迟: {result['explosion_delay']:.4f} s")
    print(f"飞行方向: {result['direction_angle']:.2f}°")
    print("\n关键位置:")
    print(f"投放位置: ({result['drop_position'][0]:.1f}, {result['drop_position'][1]:.1f}, {result['drop_position'][2]:.1f}) m")
    print(f"起爆位置: ({result['explosion_position'][0]:.1f}, {result['explosion_position'][1]:.1f}, {result['explosion_position'][2]:.1f}) m")
    print("=" * 60)

    uav = UAVAgent("FY1")
    uav.set_flight_parameters(result["uav_speed"], result["direction_angle"])
    uav.add_smoke_plan(result["drop_delay"], result["explosion_delay"])

    missile = MissileAgent("M1")

    print("\n生成可视化图表...")
    _ensure_docs_dir()
    plot_3d_trajectory([uav], [missile], "问题二：最优烟幕干扰3D轨迹",
                      "docs/problem2_trajectory.png")

    _save_result_files(result, 2, RUN_CONFIG["output_dir"])
    generate_text_report(result, "problem2", "docs/problem2_report.txt")


def solve_problem3():
    """求解问题三：单无人机三弹最优投放策略"""
    print("=" * 80)
    print("问题三：单无人机三弹最优投放策略优化")
    print("=" * 80)

    settings = _get_runtime_optimizer_settings("problem3", demo=RUN_CONFIG["demo"], quick=RUN_CONFIG["quick"])
    optimizer = EnhancedGeneticAlgorithm(
        "problem3",
        uav_id="FY1",
        missile_id="M1",
        population_size=settings["ga_population"],
        max_iterations=settings["ga_max_iterations"],
        random_seed=RUN_CONFIG["seed"]
    )
    result = optimizer.optimize()

    if not result["success"]:
        print("优化失败，未找到有效解")
        return

    print("\n" + "=" * 60)
    print("最优解找到!")
    print(f"总有效遮蔽时间: {result['total_occlusion_time']:.3f} 秒")
    print("\n无人机参数:")
    print(f"飞行方向: {result['uav_direction']:.1f}°")
    print(f"飞行速度: {result['uav_speed']:.1f} m/s")
    print("\n烟幕弹详情:")
    for smoke in result["smoke_details"]:
        print(f"烟幕弹{smoke['smoke_id']}:")
        print(f"  投放时间: {smoke['drop_time']:.2f}s")
        print(f"  起爆延迟: {smoke['explosion_delay']:.2f}s")
        print(f"  投放点: ({smoke['drop_position'][0]:.1f}, {smoke['drop_position'][1]:.1f}, {smoke['drop_position'][2]:.1f}) m")
        print(f"  起爆点: ({smoke['explosion_position'][0]:.1f}, {smoke['explosion_position'][1]:.1f}, {smoke['explosion_position'][2]:.1f}) m")
    print("=" * 60)

    uav = UAVAgent("FY1")
    uav.set_flight_parameters(result["uav_speed"], result["uav_direction"])
    for smoke in result["smoke_details"]:
        uav.add_smoke_plan(smoke["drop_time"], smoke["explosion_delay"])

    missile = MissileAgent("M1")

    print("\n生成可视化图表...")
    _ensure_docs_dir()
    plot_3d_trajectory([uav], [missile], "问题三：三弹协同烟幕干扰3D轨迹",
                      "docs/problem3_trajectory.png")

    plot_optimization_convergence(result["fitness_history"], result["best_fitness_history"],
                                 "问题三：优化收敛曲线", "docs/problem3_convergence.png")

    save_problem3_result(result)
    _save_result_files(result, 3, RUN_CONFIG["output_dir"])
    generate_text_report(result, "problem3", "docs/problem3_report.txt")


def solve_problem4():
    """求解问题四：三无人机单弹协同最优投放策略"""
    print("=" * 80)
    print("问题四：三无人机单弹协同最优投放策略优化")
    print("=" * 80)

    settings = _get_runtime_optimizer_settings("problem4", demo=RUN_CONFIG["demo"], quick=RUN_CONFIG["quick"])
    optimizer = EnhancedGeneticAlgorithm(
        "problem4",
        uav_ids=["FY1", "FY2", "FY3"],
        missile_id="M1",
        population_size=settings["ga_population"],
        max_iterations=settings["ga_max_iterations"],
        random_seed=RUN_CONFIG["seed"]
    )
    result = optimizer.optimize()

    if not result["success"]:
        print("优化失败，未找到有效解")
        return

    print("\n" + "=" * 60)
    print("最优解找到!")
    print(f"总有效遮蔽时间: {result['total_occlusion_time']:.2f} 秒")
    print("\n各无人机策略:")
    for uav in result["uav_details"]:
        print(f"{uav['uav_id']}:")
        print(f"  飞行方向: {uav['direction']:.1f}°")
        print(f"  飞行速度: {uav['speed']:.1f} m/s")
        print(f"  投放时间: {uav['drop_time']:.2f}s")
        print(f"  起爆延迟: {uav['explosion_delay']:.2f}s")
        print(f"  投放点: ({uav['drop_position'][0]:.1f}, {uav['drop_position'][1]:.1f}, {uav['drop_position'][2]:.1f}) m")
        print(f"  起爆点: ({uav['explosion_position'][0]:.1f}, {uav['explosion_position'][1]:.1f}, {uav['explosion_position'][2]:.1f}) m")
    print("=" * 60)

    uavs = []
    for uav_detail in result["uav_details"]:
        uav = UAVAgent(uav_detail["uav_id"])
        uav.set_flight_parameters(uav_detail["speed"], uav_detail["direction"])
        uav.add_smoke_plan(uav_detail["drop_time"], uav_detail["explosion_delay"])
        uavs.append(uav)

    missile = MissileAgent("M1")

    print("\n生成可视化图表...")
    _ensure_docs_dir()
    plot_3d_trajectory(uavs, [missile], "问题四：三机协同烟幕干扰3D轨迹",
                      "docs/problem4_trajectory.png")

    plot_optimization_convergence(result["fitness_history"], result["best_fitness_history"],
                                 "问题四：优化收敛曲线", "docs/problem4_convergence.png")

    save_problem4_result(result)
    _save_result_files(result, 4, RUN_CONFIG["output_dir"])
    generate_text_report(result, "problem4", "docs/problem4_report.txt")


def solve_problem5():
    """求解问题五：五无人机多弹对三导弹协同最优投放策略"""
    print("=" * 80)
    print("问题五：五无人机多弹对三导弹协同最优投放策略优化")
    print("=" * 80)

    settings = _get_runtime_optimizer_settings("problem5", demo=RUN_CONFIG["demo"], quick=RUN_CONFIG["quick"])
    optimizer = EnhancedGeneticAlgorithm(
        "problem5",
        population_size=settings["ga_population"],
        max_iterations=settings["ga_max_iterations"],
        random_seed=RUN_CONFIG["seed"]
    )
    result = optimizer.optimize()

    if not result["success"]:
        print("优化失败，未找到有效解")
        return

    print("\n" + "=" * 60)
    print("最优解找到!")
    print(f"总有效遮蔽时间: {result['total_occlusion_time']:.2f} 秒")
    print(f"总计使用烟幕弹: {result['total_smokes_used']}枚")
    print("\n各导弹遮蔽时间:")
    for missile_id, time in result["missile_occlusion_times"].items():
        print(f"  {missile_id}: {time:.2f}秒")
    print("\n各无人机策略:")
    for uav in result["uav_details"]:
        print(f"{uav['uav_id']}:")
        print(f"  飞行方向: {uav['direction']:.1f}°")
        print(f"  飞行速度: {uav['speed']:.1f} m/s")
        print(f"  活跃烟幕弹: {uav['active_smokes']}/3枚")
    print("=" * 60)

    uavs = []
    for uav_detail in result["uav_details"]:
        uav = UAVAgent(uav_detail["uav_id"])
        uav.set_flight_parameters(uav_detail["speed"], uav_detail["direction"])
        for smoke in uav_detail["smoke_details"]:
            uav.add_smoke_plan(smoke["drop_time"], smoke["explosion_delay"])
        uavs.append(uav)

    missiles = [MissileAgent(missile_id) for missile_id in ["M1", "M2", "M3"]]

    print("\n生成可视化图表...")
    _ensure_docs_dir()
    plot_3d_trajectory(uavs, missiles, "问题五：多机多弹协同烟幕干扰3D轨迹",
                      "docs/problem5_trajectory.png")

    plot_optimization_convergence(result["fitness_history"], result["best_fitness_history"],
                                 "问题五：优化收敛曲线", "docs/problem5_convergence.png")

    save_problem5_result(result)
    if "occlusion_details" in result:
        plot_occlusion_timeline(result["occlusion_details"],
                                "问题五：导弹遮蔽时间线", "docs/problem5_occlusion_timeline.png")
    _save_result_files(result, 5, RUN_CONFIG["output_dir"])
    generate_text_report(result, "problem5", "docs/problem5_report.txt")


SCENARIO_HANDLERS = {
    1: solve_problem1,
    2: solve_problem2,
    3: solve_problem3,
    4: solve_problem4,
    5: solve_problem5,
}


def main():
    parser = argparse.ArgumentParser(description="基于AI Agent的无人机多机协同干扰任务调度系统")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5],
                       help="作战场景编号 (1-5)")
    parser.add_argument("--problem", type=int, choices=[1, 2, 3, 4, 5],
                       help="要解决的问题编号 (1-5)，与 --scenario 等价")
    parser.add_argument("--all", action="store_true",
                       help="求解所有场景")
    parser.add_argument("--demo", action="store_true",
                       help="快速演示模式，使用更少迭代和更小种群")
    parser.add_argument("--quick", action="store_true",
                       help="快速自检模式，仅运行轻量版本")
    parser.add_argument("--seed", type=int,
                       help="指定随机数种子，以便结果复现")
    parser.add_argument("--output", type=str, default=config.OUTPUT_DIR,
                       help="结果导出目录，默认 output")

    args = parser.parse_args()
    scenario = args.scenario or args.problem
    RUN_CONFIG["demo"] = args.demo
    RUN_CONFIG["quick"] = args.quick
    RUN_CONFIG["seed"] = args.seed
    RUN_CONFIG["output_dir"] = args.output
    _configure_random_seed(RUN_CONFIG["seed"])

    if args.all:
        for handler in SCENARIO_HANDLERS.values():
            handler()
    elif scenario in SCENARIO_HANDLERS:
        SCENARIO_HANDLERS[scenario]()
    else:
        parser.print_help()
        print("\n示例:")
        print("  python main.py --scenario 1  # 单无人机单弹场景")
        print("  python main.py --scenario 5  # 多无人机多导弹场景")
        print("  python main.py --all         # 求解所有场景")


if __name__ == "__main__":
    main()
