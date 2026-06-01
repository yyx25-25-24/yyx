# 物理常数
GRAVITY = 9.8  # 重力加速度 m/s²

# 导弹参数
MISSILE_SPEED = 300.0  # 导弹飞行速度 m/s

# 烟幕参数
SMOKE_SINK_SPEED = 3.0  # 烟幕云团下沉速度 m/s
SMOKE_EFFECTIVE_RADIUS = 10.0  # 烟幕有效遮蔽半径 m
SMOKE_EFFECTIVE_TIME = 20.0  # 烟幕有效持续时间 s

# 目标参数
TARGET_RADIUS = 7.0  # 真目标半径 m
TARGET_HEIGHT = 10.0  # 真目标高度 m
TARGET_CENTER = (0.0, 200.0, 5.0)  # 真目标中心坐标
TARGET_BOTTOM_CENTER = (0.0, 200.0, 0.0)  # 真目标底面中心坐标

# 假目标参数
DECOY_TARGET = (0.0, 0.0, 0.0)  # 假目标坐标

# 无人机参数
UAV_SPEED_MIN = 70.0  # 无人机最小速度 m/s
UAV_SPEED_MAX = 140.0  # 无人机最大速度 m/s
MIN_DROP_INTERVAL = 1.0  # 单架无人机最小投弹间隔 s

# 初始位置
MISSILE_INITIAL_POSITIONS = {
    "M1": (20000.0, 0.0, 2000.0),
    "M2": (19000.0, 600.0, 2100.0),
    "M3": (18000.0, -600.0, 1900.0)
}

UAV_INITIAL_POSITIONS = {
    "FY1": (17800.0, 0.0, 1800.0),
    "FY2": (12000.0, 1400.0, 1400.0),
    "FY3": (6000.0, -3000.0, 700.0),
    "FY4": (11000.0, 2000.0, 1800.0),
    "FY5": (13000.0, -2000.0, 1300.0)
}

# 算法参数
# 差分进化算法
DE_POPULATION_SIZE = 30
DE_MAX_ITERATIONS = 100
DE_MUTATION_FACTOR = 0.8
DE_CROSSOVER_PROBABILITY = 0.9
DE_TOLERANCE = 1e-6

# 增强遗传算法
GA_POPULATION_SIZE = {
    "problem3": 20,
    "problem4": 150,
    "problem5": 200
}
GA_MAX_ITERATIONS = {
    "problem3": 100,
    "problem4": 200,
    "problem5": 400
}
GA_MUTATION_RATE = {
    "problem3": 0.15,
    "problem4": 0.1,
    "problem5": 0.2
}
GA_CROSSOVER_PROBABILITY = 0.9
GA_ELITISM_SIZE = {
    "problem3": 20,
    "problem4": 30,
    "problem5": 50
}
GA_EARLY_STOP = {
    "problem3": 50,
    "problem4": 50,
    "problem5": 60
}

# 时间步长
TIME_STEP_FINE = 0.005  # 精细时间步长 s
TIME_STEP_COARSE = 0.02  # 粗时间步长 s
TIME_STEP_SCAN = 0.05  # 扫描时间步长 s
TIME_STEP_VISUAL = 0.1  # 可视化时间步长 s

# 可视化设置
PLOT_FONT = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
PLOT_DPI = 300

# 运行输出目录
OUTPUT_DIR = 'output'

# CLI 演示/快速模式默认参数
DEMO_MODE_SETTINGS = {
    'problem2': {'de_population': 20, 'de_maxiter': 30, 'ga_population': 20, 'ga_maxiter': 30},
    'problem3': {'ga_population': 20, 'ga_maxiter': 30},
    'problem4': {'ga_population': 50, 'ga_maxiter': 60},
    'problem5': {'ga_population': 80, 'ga_maxiter': 80},
}
QUICK_MODE_FACTOR = 0.25
DEFAULT_RANDOM_SEED = 42
