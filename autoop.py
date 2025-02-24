import sys
import csv
import random
import re  # 用于解析字符串中的 Infeasible block IDs
import logging
from pathlib import Path
import datetime
import pandas as pd

# beb_chargers 相关 import
from beb_chargers.scripts.script_helpers import build_trips_df, build_charger_location_inputs
from beb_chargers.opt.charger_location import ChargerLocationModel
from beb_chargers.gtfs_beb import GTFSData

logging.basicConfig(level=logging.INFO)

# ============== 全局不变参数 ==============
alpha = 190 * 365 * 12 / 60
depot_coords = (40.819197, -73.957060)
site_fname = Path.cwd().parent / 'GTFS' / 'whole_station.csv'
gtfs_dir = Path.cwd().parent / 'GTFS'
routes_filename = 'routes.txt'
battery_cap = 300  # 可按需求调整
ocl_date = datetime.datetime(2025, 3, 14)

# 需要测试的线路前缀
ALL_PREFIXES = ['X','Q','B','Bx']

# 需要测试的 3×3×3 组合
N_MAX_LIST = [1, 4, 8]
S_COST_LIST = [200000, 500000, 800000]
C_COST_LIST = [600000, 800000, 1000000]

# 输出目录
OUTPUT_DIR = Path(r"E:/UW-Seattle/UW/25WI/CET 593/Project1/CET593/result/auto")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_ITERATIONS = 10

def read_routes_file(prefix: str, fname: str) -> list:
    """
    读取 routes.txt 文件，返回所有以 prefix 开头的 route_id。
    """
    route_ids = []
    with open(fname, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['route_id'].startswith(prefix):
                route_ids.append(row['route_id'])
    return route_ids

def random_select_routes(route_ids, percentage=1.0) -> list:
    num_to_select = round(len(route_ids) * percentage)
    return random.sample(route_ids, num_to_select) if route_ids else []

def main_select_routes(prefix: str, fname: str, num_iterations=10) -> list:
    route_ids = read_routes_file(prefix, fname)
    print(f"\n[{prefix}] Total number of routes matching prefix '{prefix}': {len(route_ids)}")

    all_results = []
    for i in range(num_iterations):
        selected = random_select_routes(route_ids)
        all_results.append(selected)
        print(f"Iteration {i + 1} - selected {len(selected)} routes: {selected}")

    return all_results

def run_two_stage_optimization(prefix: str,
                               n_max: int,
                               s_cost: float,
                               c_cost: float,
                               output_file: Path):
    """
    针对单个 prefix + (n_max, s_cost, c_cost) 的组合执行“两步求解”并将
    全部日志写入 output_file。
    """

    # ============== 给 logging 添加一个 FileHandler，用于记录详细日志 ==============
    logger = logging.getLogger()  # 根 logger
    logger.setLevel(logging.INFO)

    # 创建 FileHandler，写入本轮输出文件 (mode='w' 覆盖写)
    fh = logging.FileHandler(output_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)

    # 也可添加一个 StreamHandler 输出到控制台 (若你还想在控制台也看见同样信息)
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)

    # 将 handler 加到 logger
    logger.addHandler(fh)
    # logger.addHandler(console_handler)

    # 在日志文件开头简单标识一下本次测试
    logging.info(f"\n======== Prefix: {prefix}, n_max={n_max}, s_cost={s_cost}, c_cost={c_cost} ========\n")

    # ============== 开始具体求解流程 ==============
    results = main_select_routes(prefix, routes_filename, num_iterations=NUM_ITERATIONS)
    beb_routes = results[0] if len(results) > 0 else []

    gtfs = GTFSData.from_dir(gtfs_dir)
    beb_trips = build_trips_df(
        gtfs=gtfs,
        date=ocl_date,
        routes=beb_routes,
        depot_coords=depot_coords,
        add_depot_dh=True,
        add_kwh_per_mi=False,
        add_durations=False,
        routes_60=[]
    )
    beb_trips['kwh_per_mi'] = 2.0

    logging.info(
        '{}: There are {} total trips to be served by {} BEB blocks.'.format(
            ocl_date.strftime('%m/%d/%y'),
            len(beb_trips),
            beb_trips['block_id'].nunique()
        )
    )

    loc_df = pd.read_csv(site_fname)
    loc_df['max_chargers'] = n_max
    chg_pwrs = 450/60
    loc_df['kw'] = chg_pwrs * 60
    loc_df['fixed_cost'] = s_cost
    loc_df['charger_cost'] = c_cost

    # -------- 第一次求解 --------
    opt_kwargs = build_charger_location_inputs(
        gtfs=gtfs,
        trips_df=beb_trips,
        chargers_df=loc_df,
        depot_coords=depot_coords,
        battery_cap=battery_cap
    )
    clm = ChargerLocationModel(**opt_kwargs)
    clm.solve(alpha=alpha, opt_gap=0, bu_kwh=battery_cap)

    # 如果 log_results() 返回 None，需要替换为空字符串
    results_1 = clm.log_results() or ""

    logging.info("\n===== Optimization Results (First Run) =====\n" + results_1)

    # 匹配不可行 block
    infeasible_blocks = []
    match = re.search(r"Infeasible block IDs:\s*\[(.*)\]", results_1)
    if match:
        blocks_str = match.group(1)
        infeasible_blocks = [b.strip().strip("'").strip('"') for b in blocks_str.split(',') if b.strip()]

    logging.info(f"Detected infeasible blocks: {len(infeasible_blocks)}")

    # -------- 剔除不可行 block 并再次求解 --------
    if infeasible_blocks:
        beb_trips_filtered = beb_trips[~beb_trips["block_id"].isin(infeasible_blocks)]
        logging.info(
            f"Filter out infeasible blocks => remain {len(beb_trips_filtered)} trips, "
            f"{beb_trips_filtered['block_id'].nunique()} blocks."
        )
    else:
        beb_trips_filtered = beb_trips

    opt_kwargs_filtered = build_charger_location_inputs(
        gtfs=gtfs,
        trips_df=beb_trips_filtered,
        chargers_df=loc_df,
        depot_coords=depot_coords,
        battery_cap=battery_cap
    )
    clm_filtered = ChargerLocationModel(**opt_kwargs_filtered)
    clm_filtered.solve(alpha=alpha, opt_gap=0, bu_kwh=battery_cap)

    results_2 = clm_filtered.log_results() or ""
    logging.info("\n===== Optimization Results (After Filtering) =====\n" + results_2)

    # ============== 移除文件 handler，结束本次输出 ==============
    logger.removeHandler(fh)
    fh.close()

    print(f"[DONE] {prefix} - n_max={n_max}, s_cost={s_cost}, c_cost={c_cost} => Logs at: {output_file}")

def main():
    random.seed(42)

    # 对每个前缀 + 3×3×3 参数组合都执行一轮优化，并将日志输出到单独文件
    for prefix in ALL_PREFIXES:
        combination_index = 1
        for nm in N_MAX_LIST:
            for sc in S_COST_LIST:
                for cc in C_COST_LIST:
                    file_name = f"{prefix}_{combination_index}.txt"
                    out_path = OUTPUT_DIR / file_name
                    run_two_stage_optimization(prefix, nm, sc, cc, out_path)
                    combination_index += 1

if __name__ == "__main__":
    main()
