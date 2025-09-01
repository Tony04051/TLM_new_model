import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def create_gantt_chart():
    """
    讀取登機門指派結果的 CSV 檔案，並產生甘特圖。
    """
    try:
        # 讀取 Gurobi 最佳化後的結果
        df = pd.read_csv('results.csv')
    except FileNotFoundError:
        print("錯誤：找不到 'gate_assignment_results.csv' 檔案。")
        print("請先執行主要的 Gurobi 腳本來產生結果檔案。")
        return

    # --- 1. 資料前處理 ---
    # 從 '最新指派' 欄位中提取數字，以便排序 (停機坪設為 0)
    df['gate_numeric'] = df['最新指派'].apply(lambda x: int(x.split(' ')[-1]) if '登機門' in x else 0)
    # 根據登機門號碼排序，確保 Y 軸順序正確
    df = df.sort_values(by='gate_numeric', ascending=True)
    
    # --- 2. 設定中文字體 ---
    # 為了在圖中正確顯示中文，需要設定字體。
    # 請根據您的作業系統選擇一個可用的中文字體。
    # Windows: 'Microsoft JhengHei' (微軟正黑體), 'SimHei'
    # macOS: 'Arial Unicode MS', 'PingFang TC'
    # Linux: 'WenQuanYi Zen Hei', 'Noto Sans CJK TC'

    plt.rcParams['font.sans-serif'] = ['Heiti TC']  # MacOS 內建中文字型
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號


    # --- 3. 繪製甘特圖 ---
    fig, ax = plt.subplots(figsize=(20, 10))

    # Y 軸標籤 (使用數字 0, 1, 2...)
    # 取得所有使用到的登機門/停機坪的數字編號並排序
    gate_numbers = sorted(df['gate_numeric'].unique())
    # 建立一個從登機門編號到 Y 軸位置的映射
    gate_to_y_pos = {gate: i for i, gate in enumerate(gate_numbers)}
    y_pos = np.arange(len(gate_numbers))

    unique_flights = df['航班ID'].unique()
    num_flights = len(unique_flights)

    # 先獲取連續的 colormap 物件
    cmap = matplotlib.colormaps.get_cmap('viridis') 

    # 使用 np.linspace 產生 N 個介於 0 和 1 之間的均勻間隔值，
    # 再傳入 cmap 來取得一個包含 N 個 RGBA 顏色的 NumPy 陣列
    colors_list = cmap(np.linspace(0, 1, num_flights))

    # 建立航班 ID 到顏色的映射字典 (現在是從陣列中索引，而不是呼叫函式)
    flight_colors = {flight_id: colors_list[i] for i, flight_id in enumerate(unique_flights)}

    # 逐一繪製每個航班的長條
    for index, task in df.iterrows():
        start_time = task['作業開始時間']
        end_time = task['作業結束時間']
        duration = end_time - start_time
        
        # 直接使用數字編號找到 Y 軸位置
        gate_num = task['gate_numeric']
        y = gate_to_y_pos[gate_num]
        
        # 繪製水平長條
        ax.barh(
            y, 
            duration, 
            left=start_time, 
            height=0.6, 
            align='center',
            color=flight_colors[task['航班ID']],
            edgecolor='black'
        )
        
        # 在長條中間加上航班 ID 文字
        text_color = 'white' if sum(flight_colors[task['航班ID']][:3]) < 1.5 else 'black'
        ax.text(
            start_time + duration / 2, 
            y, 
            f"{task['航班ID']}", 
            ha='center', 
            va='center',
            color=text_color,
            fontweight='bold'
        )

    # --- 4. 圖表格式化 ---
    ax.set_yticks(y_pos)
    # 使用數字作為 Y 軸的座標標籤
    ax.set_yticklabels(gate_numbers, fontsize=12)
    ax.invert_yaxis()  # 讓 y=0 (停機坪) 在最下面

    ax.set_xlabel('Time(mins)', fontsize=14)
    # 將 Y 軸標題改為英文
    ax.set_ylabel('Gate', fontsize=14)
    ax.set_title('GAP problem', fontsize=18, fontweight='bold')
    
    # 設定 X 軸的刻度，使其更易讀
    max_time = df['作業結束時間'].max()
    ax.set_xticks(np.arange(0, max_time + 60, 60))
    ax.tick_params(axis='x', rotation=45)
    
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # --- 5. 儲存圖檔 ---
    output_filename = 'result.png'
    plt.savefig(output_filename, dpi=300)
    print(f"甘特圖已成功儲存至檔案: {output_filename}")
    
    # 顯示圖表 (可選)
    # plt.show()


if __name__ == "__main__":
    create_gantt_chart()
