import pandas as pd
import numpy as np
import config


def save_problem3_result(result: dict, file_path: str = "result/result1.xlsx"):
    """保存问题三结果到Excel"""
    if not result["success"]:
        print("优化失败，无法保存结果")
        return

    data = []
    for smoke in result["smoke_details"]:
        data.append({
            "烟幕弹编号": smoke["smoke_id"],
            "无人机编号": "FY1",
            "飞行方向(度)": round(result["uav_direction"], 1),
            "飞行速度(m/s)": round(result["uav_speed"], 1),
            "投放时间(s)": round(smoke["drop_time"], 2),
            "起爆延迟(s)": round(smoke["explosion_delay"], 2),
            "投放点X(m)": round(smoke["drop_position"][0], 1),
            "投放点Y(m)": round(smoke["drop_position"][1], 1),
            "投放点Z(m)": round(smoke["drop_position"][2], 1),
            "起爆点X(m)": round(smoke["explosion_position"][0], 1),
            "起爆点Y(m)": round(smoke["explosion_position"][1], 1),
            "起爆点Z(m)": round(smoke["explosion_position"][2], 1)
        })

    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"问题三结果已保存到: {file_path}")
    print(f"总有效遮蔽时间: {result['total_occlusion_time']:.3f}秒")


def save_problem4_result(result: dict, file_path: str = "result/result2.xlsx"):
    """保存问题四结果到Excel"""
    if not result["success"]:
        print("优化失败，无法保存结果")
        return

    data = []
    for uav in result["uav_details"]:
        data.append({
            "无人机编号": uav["uav_id"],
            "飞行方向(度)": round(uav["direction"], 1),
            "飞行速度(m/s)": round(uav["speed"], 1),
            "投放时间(s)": round(uav["drop_time"], 2),
            "起爆延迟(s)": round(uav["explosion_delay"], 2),
            "投放点X(m)": round(uav["drop_position"][0], 1),
            "投放点Y(m)": round(uav["drop_position"][1], 1),
            "投放点Z(m)": round(uav["drop_position"][2], 1),
            "起爆点X(m)": round(uav["explosion_position"][0], 1),
            "起爆点Y(m)": round(uav["explosion_position"][1], 1),
            "起爆点Z(m)": round(uav["explosion_position"][2], 1)
        })

    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"问题四结果已保存到: {file_path}")
    print(f"总有效遮蔽时间: {result['total_occlusion_time']:.2f}秒")


def save_problem5_result(result: dict, file_path: str = "result/result3.xlsx"):
    """保存问题五结果到Excel"""
    if not result["success"]:
        print("优化失败，无法保存结果")
        return

    data = []
    smoke_id = 1

    for uav in result["uav_details"]:
        for smoke in uav["smoke_details"]:
            data.append({
                "烟幕弹编号": smoke_id,
                "无人机编号": uav["uav_id"],
                "飞行方向(度)": round(uav["direction"], 1),
                "飞行速度(m/s)": round(uav["speed"], 1),
                "投放时间(s)": round(smoke["drop_time"], 2),
                "起爆延迟(s)": round(smoke["explosion_delay"], 2),
                "投放点X(m)": round(smoke["drop_position"][0], 1),
                "投放点Y(m)": round(smoke["drop_position"][1], 1),
                "投放点Z(m)": round(smoke["drop_position"][2], 1),
                "起爆点X(m)": round(smoke["explosion_position"][0], 1),
                "起爆点Y(m)": round(smoke["explosion_position"][1], 1),
                "起爆点Z(m)": round(smoke["explosion_position"][2], 1)
            })
            smoke_id += 1

    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"问题五结果已保存到: {file_path}")
    print(f"总有效遮蔽时间: {result['total_occlusion_time']:.2f}秒")
    print("各导弹遮蔽时间:")
    for missile_id, time in result["missile_occlusion_times"].items():
        print(f"  {missile_id}: {time:.2f}秒")
    print(f"总计使用烟幕弹: {result['total_smokes_used']}枚")


def save_result_json(result: dict, file_path: str):
    """保存结果为 JSON 文件"""
    import json

    def _json_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(file_path, 'w', encoding='utf-8') as out_file:
        json.dump(result, out_file, default=_json_encoder, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存到: {file_path}")


def generate_text_report(result: dict, problem_type: str, file_path: str):
    """生成文本报告"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"问题{problem_type[-1]} 烟幕干扰策略优化结果报告\n")
        f.write("=" * 80 + "\n\n")

        if not result["success"]:
            f.write("优化失败，未找到有效解\n")
            return

        f.write(f"优化方法: {'差分进化算法' if problem_type == 'problem2' else '增强遗传算法'}\n")
        if "optimization_time" in result:
            f.write(f"优化耗时: {result['optimization_time']:.2f}秒\n\n")
        else:
            f.write("\n")

        if problem_type == "problem2":
            f.write("【最优参数】\n")
            f.write(f"无人机速度: {result['uav_speed']:.2f} m/s\n")
            f.write(f"投放延迟: {result['drop_delay']:.4f} s\n")
            f.write(f"起爆延迟: {result['explosion_delay']:.4f} s\n")
            f.write(f"飞行方向: {result['direction_angle']:.2f}°\n\n")

            f.write("【关键位置】\n")
            f.write(f"投放位置: ({result['drop_position'][0]:.1f}, {result['drop_position'][1]:.1f}, {result['drop_position'][2]:.1f}) m\n")
            f.write(f"起爆位置: ({result['explosion_position'][0]:.1f}, {result['explosion_position'][1]:.1f}, {result['explosion_position'][2]:.1f}) m\n\n")

            f.write(f"最大有效遮蔽时间: {result['occlusion_time']:.6f}秒\n")

        elif problem_type == "problem3":
            f.write("【无人机参数】\n")
            f.write(f"飞行方向: {result['uav_direction']:.1f}°\n")
            f.write(f"飞行速度: {result['uav_speed']:.1f} m/s\n\n")

            f.write("【烟幕弹详情】\n")
            for smoke in result["smoke_details"]:
                f.write(f"烟幕弹{smoke['smoke_id']}:\n")
                f.write(f"  投放时间: {smoke['drop_time']:.2f}s\n")
                f.write(f"  起爆延迟: {smoke['explosion_delay']:.2f}s\n")
                f.write(f"  投放点: ({smoke['drop_position'][0]:.1f}, {smoke['drop_position'][1]:.1f}, {smoke['drop_position'][2]:.1f}) m\n")
                f.write(f"  起爆点: ({smoke['explosion_position'][0]:.1f}, {smoke['explosion_position'][1]:.1f}, {smoke['explosion_position'][2]:.1f}) m\n\n")

            f.write(f"总有效遮蔽时间(去重后): {result['total_occlusion_time']:.3f}秒\n")

        elif problem_type == "problem4":
            f.write("【各无人机策略】\n")
            for uav in result["uav_details"]:
                f.write(f"{uav['uav_id']}:\n")
                f.write(f"  飞行方向: {uav['direction']:.1f}°\n")
                f.write(f"  飞行速度: {uav['speed']:.1f} m/s\n")
                f.write(f"  投放时间: {uav['drop_time']:.2f}s\n")
                f.write(f"  起爆延迟: {uav['explosion_delay']:.2f}s\n")
                f.write(f"  投放点: ({uav['drop_position'][0]:.1f}, {uav['drop_position'][1]:.1f}, {uav['drop_position'][2]:.1f}) m\n")
                f.write(f"  起爆点: ({uav['explosion_position'][0]:.1f}, {uav['explosion_position'][1]:.1f}, {uav['explosion_position'][2]:.1f}) m\n\n")

            f.write(f"总有效遮蔽时间: {result['total_occlusion_time']:.2f}秒\n")

        elif problem_type == "problem5":
            f.write("【总体性能】\n")
            f.write(f"总有效遮蔽时间: {result['total_occlusion_time']:.2f}秒\n")
            f.write(f"平均每导弹遮蔽时间: {result['total_occlusion_time']/3:.2f}秒\n")
            f.write(f"总计使用烟幕弹: {result['total_smokes_used']}枚\n\n")

            f.write("【各导弹遮蔽详情】\n")
            for missile_id, time in result["missile_occlusion_times"].items():
                f.write(f"{missile_id}: {time:.2f}秒\n")

            f.write("\n【各无人机策略】\n")
            for uav in result["uav_details"]:
                f.write(f"{uav['uav_id']}:\n")
                f.write(f"  飞行方向: {uav['direction']:.1f}°\n")
                f.write(f"  飞行速度: {uav['speed']:.1f} m/s\n")
                f.write(f"  活跃烟幕弹: {uav['active_smokes']}/3枚\n")

                for smoke in uav["smoke_details"]:
                    f.write(f"  烟幕弹{smoke['smoke_id']}:\n")
                    f.write(f"    投放时间: {smoke['drop_time']:.2f}s\n")
                    f.write(f"    起爆延迟: {smoke['explosion_delay']:.2f}s\n")
                    f.write(f"    投放点: ({smoke['drop_position'][0]:.1f}, {smoke['drop_position'][1]:.1f}, {smoke['drop_position'][2]:.1f}) m\n")
                    f.write(f"    起爆点: ({smoke['explosion_position'][0]:.1f}, {smoke['explosion_position'][1]:.1f}, {smoke['explosion_position'][2]:.1f}) m\n")

                f.write("\n")

    print(f"文本报告已保存到: {file_path}")
