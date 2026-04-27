import numpy as np
import os
from openpyxl import load_workbook, Workbook
from openpyxl.styles import PatternFill
import matplotlib.pyplot as plt

# ==========================================
# 1. FILE & ANGLE CONFIGURATION
# ==========================================
FILE_CONFIGS = [
    {'filename': 'Roll0.xlsx',   'expected_roll': 0.0},
    {'filename': 'Roll30.xlsx',  'expected_roll': 30.0},
    {'filename': 'Roll45.xlsx',  'expected_roll': 45.0},
    {'filename': 'Roll60.xlsx',  'expected_roll': 60.0},
    {'filename': 'Roll90.xlsx',  'expected_roll': 90.0},
    {'filename': 'Roll-30.xlsx', 'expected_roll': -30.0},
    {'filename': 'Roll-45.xlsx', 'expected_roll': -45.0},
    {'filename': 'Roll-60.xlsx', 'expected_roll': -60.0},
    {'filename': 'Roll-90.xlsx', 'expected_roll': -90.0},
]

OUTPUT_FILE = 'Batch_Validation_Output.xlsx'

# Highlight Configuration
HIGHLIGHT_TARGET = 'Roll'  
HIGHLIGHT_COLOR = 'FFFFE0' # Light Yellow

ORDERED_SENSORS = ["21", "22", "23"]

# ==========================================
# 2. STATISTICAL ENGINE
# ==========================================
def calculate_roll_stats(target_df, expected_val):
    """Calculates all scientific metrics for Roll axis"""
    if target_df.empty:
        return {"Expected": expected_val, "Mean": "", "MAE": "", "RMSE": "", "Bias": "", "STD": "", "Lower_LOA": "", "Upper_LOA": ""}
    
    # 1. Raw Measured Average
    measured_mean = np.mean(target_df['roll'])
    
    # 2. Error Calculations
    raw_err = target_df['roll'] - expected_val
    abs_err = np.abs(raw_err)
    
    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(raw_err**2))
    bias = np.mean(raw_err)
    std = np.std(raw_err)
    
    # 3. 95% Limits of Agreement
    lower_loa = bias - (1.96 * std)
    upper_loa = bias + (1.96 * std)
    
    return {
        "Expected": expected_val,
        "Mean": round(measured_mean, 3),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Bias": round(bias, 3),
        "STD": round(std, 3),
        "Lower_LOA": round(lower_loa, 3),
        "Upper_LOA": round(upper_loa, 3)
    }

# ==========================================
# 3. MAIN PROCESSING FUNCTION
# ==========================================
def process_batch():
    print("🚀 Starting 3-Axis Batch Processing...")
    
    # Dictionaries to hold the row data for each sheet
    results = {
        '21': [],
        '22': [],
        '23': [],
        'Combined_Average': [] 
    }
    
    for config in FILE_CONFIGS:
        fname = config['filename']
        print(f"📂 Processing: {fname}...")
        
        # --- Handle Missing Files ---
        if not os.path.exists(fname):
            print(f"   ⚠️ File not found. Creating blank row.")
            empty_row = {'File_Name': fname, 'Sample_Count': 0}
            empty_row.update({
                'Expected_Roll': config['expected_roll'],
                'Roll_Mean': "",
                'Roll_MAE': "", 'Roll_RMSE': "", 'Roll_Bias': "", 
                'Roll_STD': "", 'Roll_Lower_LOA': "", 'Roll_Upper_LOA': ""
            })
            for key in results.keys():
                results[key].append(empty_row)
            continue
            
        # --- Read and Filter Data ---
        wb = load_workbook(fname)
        ws = wb.active
        
        # Get headers
        headers = [cell.value for cell in ws[1]]
        
        # Find column indices
        col_indices = {header: idx for idx, header in enumerate(headers)}
        
        if 'roll' not in col_indices:
            print(f"   ❌ No 'roll' column found in {fname}.")
            continue
            
        # Read and filter data
        paused_data = {header: [] for header in headers}
        for row in ws.iter_rows(min_row=2, values_only=True):
            is_paused = row[col_indices.get('is_paused', -1)] if 'is_paused' in col_indices else True
            if is_paused == True:
                for header, value in zip(headers, row):
                    paused_data[header].append(value)
        
        # Create simple data structure
        class SimpleDataFrame:
            def __init__(self, data_dict):
                self.data = data_dict
                first_key = list(data_dict.keys())[0] if data_dict else None
                self._len = len(data_dict[first_key]) if first_key else 0
            
            def __len__(self):
                return self._len
            
            @property
            def empty(self):
                return self._len == 0
            
            def __getitem__(self, key):
                if isinstance(key, str):
                    return np.array(self.data[key])
                else:
                    raise NotImplementedError
        
        paused_df = SimpleDataFrame(paused_data)
        
        if paused_df.empty:
            print(f"   ❌ No paused data found in {fname}.")
            continue
            
        # --- Helper Function to Build Row Data ---
        def build_row(sensor_data):
            row_data = {'File_Name': fname, 'Sample_Count': len(sensor_data)}
            
            # Process Roll only
            expected = config['expected_roll']
            stats = calculate_roll_stats(sensor_data, expected)
            
            row_data.update({
                'Expected_Roll': stats['Expected'],
                'Roll_Mean': stats['Mean'],
                'Roll_MAE': stats['MAE'],
                'Roll_RMSE': stats['RMSE'],
                'Roll_Bias': stats['Bias'],
                'Roll_STD': stats['STD'],
                'Roll_Lower_LOA': stats['Lower_LOA'],
                'Roll_Upper_LOA': stats['Upper_LOA']
            })
            return row_data

        # --- Process Individual Sensors ---
        for sensor in ORDERED_SENSORS:
            sensor_mask = np.array(paused_df.data['sensor_id']) == int(sensor)
            sensor_roll_data = np.array(paused_df.data['roll'])[sensor_mask]
            
            # Create sensor-specific DataFrame
            sensor_df = SimpleDataFrame({'roll': sensor_roll_data})
            results[sensor].append(build_row(sensor_df))
            
        # --- Process Combined Average (All Sensors Pooled) ---
        results['Combined_Average'].append(build_row(paused_df))

    # ==========================================
    # 4. EXCEL EXPORT & HIGHLIGHTING
    # ==========================================
    print(f"\n💾 Saving all tables to: {OUTPUT_FILE}")
    wb = Workbook()
    
    # Write each dictionary to a separate sheet
    for sheet_name, data_list in results.items():
        ws = wb.create_sheet(sheet_name)
        
        if not data_list:
            continue
            
        # Write headers
        headers = list(data_list[0].keys())
        for col, header in enumerate(headers, 1):
            ws.cell(row=1, column=col, value=header)
        
        # Write data
        for row, data_row in enumerate(data_list, 2):
            for col, header in enumerate(headers, 1):
                ws.cell(row=row, column=col, value=data_row.get(header, ''))
        
        # Apply Highlighting if configured
        if HIGHLIGHT_TARGET:
            target_str = str(HIGHLIGHT_TARGET).lower()
            fill_style = PatternFill(start_color=HIGHLIGHT_COLOR, end_color=HIGHLIGHT_COLOR, fill_type='solid')
            
            # Check header row for the target string
            for col_idx, cell in enumerate(ws[1], 1):
                if cell.value and target_str in str(cell.value).lower():
                    # Highlight entire column
                    for row_idx in range(1, ws.max_row + 1):
                        ws.cell(row=row_idx, column=col_idx).fill = fill_style
    
    # Remove default sheet if it exists
    if 'Sheet' in wb.sheetnames:
        wb.remove(wb['Sheet'])
    
    wb.save(OUTPUT_FILE)

    print(f"✨ Batch Processing Complete! Highlight target was set to '{HIGHLIGHT_TARGET}'.")
    
    # Generate validation plot
    generate_validation_plot(results)

# ==========================================
# 5. PLOT GENERATION
# ==========================================
def generate_validation_plot(results):
    """Generate IMU Roll Accuracy Validation chart"""
    print("\n📊 Generating Roll validation plot...")
    
    # Extract roll angles from file names
    Roll_angles = [0, 30, 45, 60, 90, -30, -45, -60, -90]
    
    # Prepare data for plotting
    sensor_labels = ['21', '22', '23', 'Combined_Average']
    sensor_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']  # Blue, Green, Orange, Purple
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(Roll_angles))
    width = 0.2
    
    for idx, sensor in enumerate(sensor_labels):
        mae_values = []
        std_values = []
        
        for row_data in results[sensor]:
            # Extract Roll_MAE and Roll_STD from the row
            mae = row_data.get('Roll_MAE', 0)
            std = row_data.get('Roll_STD', 0)
            mae_values.append(mae if isinstance(mae, (int, float)) and not np.isnan(mae) else 0)
            std_values.append(std if isinstance(std, (int, float)) and not np.isnan(std) else 0)
        
        # Ensure we have the right number of values
        while len(mae_values) < len(Roll_angles):
            mae_values.append(0)
            std_values.append(0)
        mae_values = mae_values[:len(Roll_angles)]
        std_values = std_values[:len(Roll_angles)]
        
        offset = (idx - 1.5) * width
        ax.bar(x + offset, mae_values, width, label=f'Sensor {sensor}' if sensor != 'Combined_Average' else 'Overall System',
               color=sensor_colors[idx], yerr=std_values, capsize=5, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Expected Roll Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('IMU Roll Angle Accuracy Validation\nError bars show ±1 standard deviation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{angle}°' for angle in Roll_angles])
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Roll_Validation_Chart.png', dpi=300, bbox_inches='tight')
    print("✅ Roll validation chart saved as 'Roll_Validation_Chart.png'")

if __name__ == "__main__":
    process_batch()