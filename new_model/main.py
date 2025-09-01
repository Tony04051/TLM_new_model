import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import traceback

def solve_gate_assignment():
    try:
        # --- Sets ---

        df_w = pd.read_csv("./data/Wij.csv")
        df_c = pd.read_csv("./data/cij.csv")
        df_flights = pd.read_csv("./data/flight.csv")
        df_G = pd.read_csv("./data/Gij.csv")
        df_N = pd.read_csv("./data/Nij.csv")

        gates = [1, 2, 3, 4, 5, 6]   # 登機門 K 的集合
        apron = [0]               # 停機坪 (以 0 代表)
        locations = apron + gates # 登機門(包含停機坪) K^0 的集合
        
        K = gates
        K_0 = locations

        flights = [] # 航班 I 的集合
        flights_S = []
        flights_L = []
        flights_XL = []
        A = {}
        U = {}
        S = {}
        Y = {}
        B = {}  # Bi: 航班i指派至停機坪的懲罰成本    

        for _, row in df_flights.iterrows():
            flights.append(int(row["flight_id"]))
            A[int(row["flight_id"])] = int(row["A_i"])
            S[int(row["flight_id"])] = int(row["S_i"])
            U[int(row["flight_id"])] = int(row["U_i"])
            B[int(row["flight_id"])] = int(row["B"])
            for gate in range(1, len(gates) + 1):
                if row[f'Y_k{gate}'] == 1:
                    Y[(int(row["flight_id"]), gate)] = 1
                else:
                    Y[(int(row["flight_id"]), gate)] = 0
            if row["S"] == 1:
                flights_S.append(int(row["flight_id"])) 
            elif row["L"] == 1:
                flights_L.append(int(row["flight_id"]))
            elif row["XL"] == 1:
                flights_XL.append(int(row["flight_id"]))
            
        Y_orig = {}
        for key, value in Y.items():
            flight_id, gate = key
            if value == 1:
               Y_orig[flight_id] = gate 

        I = flights

        # G_ik: 航班 i 能否被指派去登機門 k
        G = {}
        for _, row in df_G.iterrows():
            flight_id = int(row['flight_id'])
            for gate in range (1, len(gates) + 1):
                if (row[f'gate_{gate}'] == 1):
                    G[(flight_id, gate)] = 1
                else:
                    G[(flight_id, gate)] = 0
                G[(flight_id, 0)] = 1

        # C_ij: 航班 j 是否為 i 的下一個航段
        C = {}
        for _, row in df_c.iterrows():
            for j in flights:
                if row[f'{j}'] == 1:
                    C[(int(row['flight_id_i']), j)] = 1

        # W_kj: 從登機門 k 更換至 j 的懲罰成本
        W = {}
        for _, row in df_w.iterrows():
            for to_gate in K:
                W[(int(row['from_gate']), to_gate)] = int(row[f'to_gate_{to_gate}'])
        
        # Nij: =1代表登機門i和登機門j相鄰，否則為0
        N = set()
        for _, row in df_N.iterrows():
            for j in gates:
                if row[f'{j}'] == 1:
                    N.add((int(row['gate']), j))
                    N.add((j, int(row['gate'])))
        
        M = 100000

        m = gp.Model("GAP")

        # --- Variables ---
        y = m.addVars(I, K_0, vtype=GRB.BINARY, name="y")
        t = m.addVars(I, vtype=GRB.CONTINUOUS, lb=0.0, name="t")
        z = m.addVars(I, vtype=GRB.CONTINUOUS, lb=0.0, name="z")
        delta = m.addVars(flights_XL, flights_L, vtype=GRB.BINARY, name="delta")

        # --- Objective ---
        objective = (
            gp.quicksum(t[i] - A[i] for i in I) +
            gp.quicksum(z[i] for i in I) +
            gp.quicksum(B[i] * y[i, 0] for i in I)
        )
        m.setObjective(objective, GRB.MINIMIZE)

        # --- Constraints ---
        
        # 1. 每個航班必須被指派到一個位置
        m.addConstrs((y.sum(i, '*') == 1 for i in I), name="assignment")

        # 2. 航班 i 只能被指派到相容的登機門 k
        m.addConstrs((y[i, k] <= G[i, k] for i in I for k in K_0), name="compatibility")
        
        # 合併第一和第三條限制式
        # for i in I:
        #     m.addConstr(
        #         gp.quicksum(y[i, k] * G[i, k] for k in K_0) == 1,
        #         name=f"sum_yG_eq1_i{i}"
        #     )

        # 3. 登機門間隔限制
        m.addConstrs(
            (t[i] + S[i] - t[j] <= M * (2 - y[i, k] - y[j, k])
             for i in I for j in I if U[i] <= U[j] for k in K_0 if i != j),
            name="separation"
        )
        
        # 4. 作業開始時間不能早於航班更新後的抵達時間
        m.addConstrs((t[i] >= U[i] for i in I), name="arrival_time")

        # 5. 計算更換登機門的懲罰成本 z_i
        m.addConstrs(
            (z[i] >= W[k, j] * y[i, j] * Y[i, k]
             for i in I for k in K for j in K),
            name="gate_change_penalty"
        )
        
        # 6. 相連航班必須使用同一個登機門
        m.addConstrs(
            (C.get((i, j), 0) * (y[i, k] - y[j, k]) == 0
             for i in I for j in I if C.get((i, j), 0) == 1 for k in K_0),
            name="continuous_flights"
        )
        
        # 7. 巨大型飛機旁邊不能有廣體飛機
        # a. delta[i, j] = 1 如果航班i的服務結束時間比航班j的開始服務時間還要早：
        m.addConstrs(
            t[j] >= t[i] + S[i] - M * (1 - delta[i, j]) - M * (2 - y[i, k] - y[j, l])
            for i in flights_XL for j in flights_L for k, l in N
        )
        # b. delta[i, j] = 0 如果航班i的服務結束時間比航班j的開始服務時間還要晚：
        m.addConstrs(
            t[i] >= t[j] + S[j] - M * delta[i, j] - M * (2 - y[i, k] - y[j, l])
            for i in flights_XL for j in flights_L for k, l in N
        )
        m.addConstr

        m.optimize()
        
        # --- Output ---
        print("-" * 100)
        if m.Status == GRB.OPTIMAL:
            print(f"找到最佳解！目標值 (總成本): {m.ObjVal:.2f}")
            print("-" * 100)
            print(f"{'航班':<8} | {'原指派':<10} | {'新指派':<10} | {'更新後抵達':<12} | {'作業開始':<12} | {'作業結束':<12} | {'更換成本':<10}")
            print("-" * 100)
            
            results_list = []

            for i in I:
                new_gate_str = "N/A"
                for k in K_0:
                    if y[i, k].X > 0.5:
                        new_gate_str = f"停機坪" if k == 0 else f"登機門 {k}"
                        break
                
                orig_gate = Y_orig.get(i, 0)
                orig_gate_str = f"停機坪" if orig_gate == 0 else f"登機門 {orig_gate}"

                start_time = t[i].X
                end_time = start_time + S[i]

                print(f"Flight {i:<3} | {orig_gate_str:<10} | {new_gate_str:<10} | {U[i]:<12.2f} | {start_time:<12.2f} | {end_time:<12.2f} | {z[i].X:<10.2f}")

                result_row = {
                    '航班ID': i,
                    '原始指派': orig_gate_str,
                    '最新指派': new_gate_str,
                    '更新後抵達時間': U[i],
                    '作業開始時間': round(start_time, 2),
                    '作業結束時間': round(end_time, 2),
                    '更換登機門成本': round(z[i].X, 2)
                }
                results_list.append(result_row)

            # --- save result as a csv file ---
            try:
                results_df = pd.DataFrame(results_list)
                output_filename = 'results.csv'
                results_df.to_csv(output_filename, index=False, encoding='utf-8')
                print("-" * 100)
                print(f"result saved to: {output_filename}")
            except Exception as e:
                print(e) 

        elif m.Status == GRB.INFEASIBLE:
            m.computeIIS()
            m.write("model_iis.ilp")
            print("已將導致無解的最小衝突限制子系統寫入 model_iis.ilp")
            
        else:
            print(m.Status)

    except gp.GurobiError as e:
        print(f"Gurobi Error code {e.errno}: {e}")
    except Exception as e:
        print(f"An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    solve_gate_assignment()