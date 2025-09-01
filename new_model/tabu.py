import pandas as pd
import random
import time
from collections import deque
import copy

# --- 1. 資料載入與準備 ---
def load_data():
    """
    載入所有必要的 CSV 檔案，並將資料準備成以列表為主的結構。
    返回一個包含所有模型參數的字典。
    """
    try:
        df_w = pd.read_csv("./data/Wij.csv")
        df_c = pd.read_csv("./data/Cij.csv")
        df_flights = pd.read_csv("./data/flight.csv")
        df_G = pd.read_csv("./data/Gij.csv")
    except FileNotFoundError as e:
        print(f"錯誤：找不到資料檔案 {e.filename}。請確認 'data' 資料夾與所有 CSV 檔存在。")
        return None

    params = {}
    
    # 基本集合參數
    flight_ids = sorted(df_flights['flight_id'].astype(int).tolist())
    num_flights = len(flight_ids)
    params['gates'] = [1, 2, 3, 4, 5, 6]
    params['apron'] = [0]
    params['locations'] = params['apron'] + params['gates']
    num_locations = len(params['locations'])
    params['num_flights'] = num_flights

    # 初始化所有參數為列表結構
    params['A'] = [0] * num_flights
    params['S'] = [0] * num_flights
    params['U'] = [0] * num_flights
    params['Y_orig'] = [0] * num_flights
    params['G'] = [[0] * num_locations for _ in range(num_flights)]
    params['C'] = [[] for _ in range(num_flights)]
    params['W'] = [[0] * num_locations for _ in range(num_locations)]
    
    # 填充航班相關資料 (A, S, U, Y_orig)
    for _, row in df_flights.iterrows():
        i = int(row["flight_id"])
        params['A'][i] = int(row["A_i"])
        params['S'][i] = int(row["S_i"])
        params['U'][i] = int(row["U_i"])
        for k_idx, k in enumerate(params['gates']):
            if row[f'Y_k{k}'] == 1:
                params['Y_orig'][i] = k
                break

    # 填充相容性資料 G[i][k]
    for _, row in df_G.iterrows():
        i = int(row['flight_id'])
        for k in params['gates']:
            params['G'][i][k] = int(row[f'gate_{k}'])
        params['G'][i][0] = 1 # 所有飛機都可停在停機坪

    # 填充相連航班資料 C[i] = [j1, j2, ...]
    for _, row in df_c.iterrows():
        i = int(row['flight_id_i'])
        for j_str in df_c.columns[1:]:
             j = int(j_str)
             if row[j_str] == 1:
                 params['C'][i].append(j)

    # 填充更換成本資料 W[k1][k2]
    for _, row in df_w.iterrows():
        from_k = int(row['from_gate'])
        for to_k in params['gates']:
            params['W'][from_k][to_k] = int(row[f'to_gate_{to_k}'])

    params['B'] = 30  # 停機坪懲罰
    return params

# --- 2. 成本計算 (目標函數) ---
def calculate_cost(assignments, params):
    """
    計算給定指派方案(assignments)的總成本。
    assignments: 一個列表，索引為 flight_id，值為 location_id
    """
    total_delay = 0
    total_change_penalty = 0
    total_apron_penalty = 0

    # 計算停機坪與更換登機門的成本
    for i, new_gate in enumerate(assignments):
        if new_gate == 0:
            total_apron_penalty += params['B']

        original_gate = params['Y_orig'][i]
        if new_gate != original_gate and new_gate != 0:
            total_change_penalty += params['W'][original_gate][new_gate]

    # 計算總延誤時間
    gate_schedule = [[] for _ in params['locations']]
    for i, gate in enumerate(assignments):
        gate_schedule[gate].append(i)

    flight_start_times = [0] * params['num_flights']

    for k in params['locations']:
        sorted_flights = sorted(gate_schedule[k], key=lambda f: params['U'][f])
        last_finish_time = -1
        for flight in sorted_flights:
            start_time = max(params['U'][flight], last_finish_time)
            flight_start_times[flight] = start_time
            last_finish_time = start_time + params['S'][flight]
    
    for i in range(params['num_flights']):
        delay = flight_start_times[i] - params['A'][i]
        total_delay += delay

    return total_delay + total_change_penalty + total_apron_penalty

# --- 3. 初始解生成 ---
def generate_initial_solution(params):
    """生成一個合法的初始解"""
    num_flights = params['num_flights']
    assignments = [-1] * num_flights
    assigned_flights = set()

    # 優先處理相連航班
    for i in range(num_flights):
        if i in assigned_flights:
            continue
        
        connected_group = {i}.union(set(params['C'][i]))
        for f in connected_group: # 確保雙向連接
            connected_group.update(params['C'][f])
            
        best_gate = None
        for k in params['locations']:
            if all(params['G'][f][k] == 1 for f in connected_group):
                best_gate = k
                break
        
        best_gate = best_gate if best_gate is not None else 0
        
        for f in connected_group:
            assignments[f] = best_gate
            assigned_flights.add(f)
            
    return assignments

# --- 4. 鄰域與移動操作 ---
def get_neighborhood(solution, params):
    """生成鄰域解及對應的移動記錄"""
    neighborhood = []
    num_flights = params['num_flights']
    
    flight_to_move = random.randrange(num_flights)
    
    # 找到其所在的完整相連群組
    connected_group = {flight_to_move}
    group_queue = [flight_to_move]
    visited = {flight_to_move}
    while group_queue:
        f1 = group_queue.pop(0)
        for f2 in params['C'][f1]:
            if f2 not in visited:
                visited.add(f2)
                connected_group.add(f2)
                group_queue.append(f2)

    current_gate = solution[flight_to_move]

    for new_gate in params['locations']:
        if new_gate == current_gate:
            continue

        if all(params['G'][f][new_gate] == 1 for f in connected_group):
            new_solution = solution[:] # 複製列表
            move = []
            for f in connected_group:
                new_solution[f] = new_gate
                move.append((f, current_gate)) # 禁忌對象:(航班, 原登機門)
            neighborhood.append((new_solution, move))
            
    return neighborhood

# --- 5. 禁忌搜尋主算法 ---
def tabu_search(params):
    """執行禁忌搜尋演算法"""
    MAX_ITERATIONS = 1000
    TABU_TENURE = 20
    
    current_solution = generate_initial_solution(params)
    current_cost = calculate_cost(current_solution, params)
    
    best_solution = current_solution[:]
    best_cost = current_cost
    
    tabu_list = deque(maxlen=TABU_TENURE)
    
    print(f"初始解成本: {best_cost:.2f}")

    start_time = time.time()
    for i in range(MAX_ITERATIONS):
        neighborhood = get_neighborhood(current_solution, params)
        if not neighborhood: continue

        best_neighbor, best_neighbor_cost, best_move = None, float('inf'), None

        for neighbor, move in neighborhood:
            neighbor_cost = calculate_cost(neighbor, params)
            is_tabu = any(m in tabu_list for m in move)
            aspiration_criterion_met = neighbor_cost < best_cost

            if (not is_tabu or aspiration_criterion_met):
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor, best_neighbor_cost, best_move = neighbor, neighbor_cost, move
        
        if best_neighbor is None: continue

        current_solution, current_cost = best_neighbor, best_neighbor_cost
        if best_move: 
            tabu_list.extend(best_move)
        
        if current_cost < best_cost:
            best_solution, best_cost = current_solution[:], current_cost
            print(f"迭代 {i+1}: 找到新的最佳解！成本: {best_cost:.2f}")

    end_time = time.time()
    print("-" * 50)
    print(f"搜尋完成！總耗時: {end_time - start_time:.2f} 秒")
    return best_solution, best_cost

# --- 6. 主執行區 ---
if __name__ == "__main__":
    params = load_data()
    
    if params:
        best_solution_ts, best_cost_ts = tabu_search(params)
        print(f"禁忌搜尋找到的最佳解成本: {best_cost_ts:.2f}")
        print("-" * 75)
        print(f"{'航班':<8} | {'原指派':<10} | {'新指派':<10} | {'抵達時間':<12} | {'更換成本':<10}")
        print("-" * 75)
        
        for i in range(params['num_flights']):
            orig_gate = params['Y_orig'][i]
            new_gate = best_solution_ts[i]
            
            cost = 0
            if new_gate != orig_gate and new_gate != 0:
                cost = params['W'][orig_gate][new_gate]

            orig_gate_str = f"登機門 {orig_gate}"
            new_gate_str = f"停機坪 {new_gate}" if new_gate == 0 else f"登機門 {new_gate}"

            print(f"Flight {i:<3} | {orig_gate_str:<10} | {new_gate_str:<10} | {params['U'][i]:<12.2f} | {cost:<10.2f}")