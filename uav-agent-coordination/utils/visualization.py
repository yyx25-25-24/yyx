import numpy as np
import matplotlib.pyplot as plt
import config
from models.kinematics import calculate_uav_position, calculate_smoke_position, calculate_missile_position
from models.occlusion import generate_target_samples

# 设置中文字体
plt.rcParams['font.sans-serif'] = config.PLOT_FONT
plt.rcParams['axes.unicode_minus'] = False


def plot_3d_trajectory(uav_agents: list, missile_agents: list,
                       title: str, save_path: str = None):
    """绘制3D轨迹图"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制真目标
    _plot_cylinder(ax, np.array(config.TARGET_BOTTOM_CENTER), config.TARGET_RADIUS, config.TARGET_HEIGHT, 'g', '真目标')

    # 绘制假目标
    ax.scatter(*config.DECOY_TARGET, color='orange', s=150, marker='*', label='假目标')

    # 绘制导弹轨迹
    for missile in missile_agents:
        times = np.linspace(0, missile.total_flight_time, 200)
        positions = np.array([
            calculate_missile_position(missile.initial_position, missile.direction,
                                      missile.speed, t)
            for t in times
        ])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'r-', linewidth=2, label=f'{missile.missile_id} 轨迹')
        ax.scatter(*missile.initial_position, color='red', s=100, marker='o',
                  label=f'{missile.missile_id} 初始位置')

    # 绘制无人机轨迹和烟幕弹
    colors = ['b', 'c', 'm', 'y', 'k']
    for i, uav in enumerate(uav_agents):
        if uav.speed == 0:
            continue

        max_time = max([drop_time for drop_time, _ in uav.smoke_plan] + [0]) + 10
        times = np.linspace(0, max_time, 100)
        positions = np.array([
            calculate_uav_position(uav.initial_position, uav.speed, uav.direction_angle, t)
            for t in times
        ])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                f'{colors[i % len(colors)]}--', linewidth=2, label=f'{uav.uav_id} 轨迹')
        ax.scatter(*uav.initial_position, color=colors[i % len(colors)], s=100, marker='^',
                  label=f'{uav.uav_id} 初始位置')

        for j, (drop_time, explosion_delay) in enumerate(uav.smoke_plan):
            drop_pos = calculate_uav_position(
                uav.initial_position, uav.speed, uav.direction_angle, drop_time
            )
            explosion_time = drop_time + explosion_delay
            explosion_pos = calculate_smoke_position(
                drop_pos, uav.speed_vector, explosion_delay, 0
            )

            ax.scatter(*drop_pos, color=colors[i % len(colors)], s=80, marker='o',
                      label=f'{uav.uav_id} 烟幕{j+1} 投放点')

            ax.scatter(*explosion_pos, color=colors[i % len(colors)], s=120, marker='*',
                      label=f'{uav.uav_id} 烟幕{j+1} 起爆点')

            drop_times = np.linspace(0, explosion_delay, 50)
            drop_positions = np.array([
                calculate_smoke_position(drop_pos, uav.speed_vector, t, t - explosion_delay)
                for t in drop_times
            ])
            ax.plot(drop_positions[:, 0], drop_positions[:, 1], drop_positions[:, 2],
                    f'{colors[i % len(colors)]}:', linewidth=2, alpha=0.7)

            sink_times = np.linspace(0, config.SMOKE_EFFECTIVE_TIME, 100)
            sink_positions = np.array([
                calculate_smoke_position(drop_pos, uav.speed_vector, explosion_delay + t, t)
                for t in sink_times
            ])
            ax.plot(sink_positions[:, 0], sink_positions[:, 1], sink_positions[:, 2],
                    f'{colors[i % len(colors)]}-', linewidth=3, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"3D轨迹图已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_occlusion_timeline(occlusion_times: dict, title: str, save_path: str = None):
    """绘制遮蔽时间线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    all_times = set()
    for times in occlusion_times.values():
        all_times.update(times)

    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        time_range = np.arange(min_time - 1, max_time + 1, config.TIME_STEP_VISUAL)

        total_occlusion = np.zeros_like(time_range)
        for i, t in enumerate(time_range):
            for times in occlusion_times.values():
                if t in times:
                    total_occlusion[i] = 1
                    break

        ax1.fill_between(time_range, 0, total_occlusion, alpha=0.7, color='red', label='总体遮蔽')

    ax1.set_ylabel('遮蔽状态')
    ax1.set_title('总体遮蔽时间线')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    colors = ['orange', 'purple', 'brown', 'green', 'blue']
    for i, (name, times) in enumerate(occlusion_times.items()):
        if not times:
            continue

        time_array = np.array(sorted(times))
        ax2.scatter(time_array, np.full_like(time_array, i),
                   color=colors[i % len(colors)], s=10, label=name)

    ax2.set_yticks(range(len(occlusion_times)))
    ax2.set_yticklabels(list(occlusion_times.keys()))
    ax2.set_ylabel('对象')
    ax2.set_xlabel('时间 (s)')
    ax2.set_title('各对象单独遮蔽效果')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"遮蔽时间线图已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_optimization_convergence(fitness_history: list, best_fitness_history: list,
                                 title: str, save_path: str = None):
    """绘制优化收敛曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    generations = range(len(fitness_history))

    ax1.plot(generations, fitness_history, 'b-', label='平均适应度', alpha=0.7)
    ax1.plot(generations, best_fitness_history, 'r-', label='最佳适应度', linewidth=2)
    ax1.set_xlabel('迭代代数')
    ax1.set_ylabel('适应度 (秒)')
    ax1.set_title('适应度演化过程')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    if len(best_fitness_history) > 10:
        window = 10
        moving_avg = np.convolve(best_fitness_history, np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(best_fitness_history)+1), moving_avg, 'g-', linewidth=2)
        ax2.set_xlabel('迭代代数')
        ax2.set_ylabel('滑动平均适应度 (秒)')
        ax2.set_title(f'收敛性分析 ({window}代滑动平均)')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"优化收敛曲线已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_target_samples(save_path: str = None):
    """绘制目标采样点"""
    samples = generate_target_samples()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    _plot_cylinder(ax, np.array(config.TARGET_BOTTOM_CENTER), config.TARGET_RADIUS, config.TARGET_HEIGHT, 'b', alpha=0.3)

    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
              c='r', marker='o', s=50, label='采样点')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'圆柱形目标及采样点 (共{len(samples)}个)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"目标采样点图已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _plot_cylinder(ax, bottom_center: np.ndarray, radius: float, height: float,
                  color: str, label: str = None, alpha: float = 0.6):
    """绘制圆柱体"""
    theta = np.linspace(0, 2 * np.pi, 50)
    z = np.linspace(bottom_center[2], bottom_center[2] + height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = bottom_center[0] + radius * np.cos(theta_grid)
    y_grid = bottom_center[1] + radius * np.sin(theta_grid)

    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, label=label)

    ax.plot(bottom_center[0] + radius * np.cos(theta),
            bottom_center[1] + radius * np.sin(theta),
            np.full_like(theta, bottom_center[2]),
            color=color, alpha=alpha)
    ax.plot(bottom_center[0] + radius * np.cos(theta),
            bottom_center[1] + radius * np.sin(theta),
            np.full_like(theta, bottom_center[2] + height),
            color=color, alpha=alpha)
