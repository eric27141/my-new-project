import numpy as np
import os
from openpyxl import load_workbook, Workbook
from openpyxl.styles import PatternFill
import matplotlib.pyplot as plt

# ==========================================
# 1. FILE & ANGLE CONFIGURATION
# ==========================================
FILE_CONFIGS = [
    {'filename': 'Yaw0.xlsx',   'expected_pitch': 0.0},
    {'filename': 'Pitch30.xlsx',  'expected_pitch': 30.0},
    {'filename': 'Pitch45.xlsx',  'expected_pitch': 45.0},
    {'filename': 'Pitch60.xlsx',  'expected_pitch': 60.0},
    {'filename': 'Pitch90.xlsx',  'expected_pitch': 90.0},
    {'filename': 'Pitch-30.xlsx', 'expected_pitch': -30.0},
    {'filename': 'Pitch-45.xlsx', 'expected_pitch': -45.0},
    {'filename': 'Pitch-60.xlsx', 'expected_pitch': -60.0},
    {'filename': 'Pitch-90.xlsx', 'expected_pitch': -90.0},
]

OUTPUT_FILE = 'Batch_Validation_Output.xlsx'

# Highlight Configuration
HIGHLIGHT_TARGET = 'Pitch'  
HIGHLIGHT_COLOR = 'FFFFE0' # Light Yellow

ORDERED_SENSORS = ["21", "22", "23"]

# ==========================================
# 2. STATISTICAL ENGINE
# ==========================================
def calculate_pitch_stats(target_df, expected_val):
    """Calculates all scientific metrics for Pitch axis"""
    if target_df.empty:
        return {"Expected": expected_val, "Mean": "", "MAE": "", "RMSE": "", "Bias": "", "STD": "", "Lower_LOA": "", "Upper_LOA": ""}
    
    # 1. Raw Measured Average
    measured_mean = np.mean(target_df['pitch'])
    
    # 2. Error Calculations
    raw_err = target_df['pitch'] - expected_val
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
        '23': []
    }
    
    for config in FILE_CONFIGS:
        fname = config['filename']
        print(f"📂 Processing: {fname}...")
        
        # --- Handle Missing Files ---
        if not os.path.exists(fname):
            print(f"   ⚠️ File not found. Creating blank row.")
            empty_row = {'File_Name': fname, 'Sample_Count': 0}
            empty_row.update({
                'Expected_Pitch': config['expected_pitch'],
                'Pitch_Mean': "",
                'Pitch_MAE': "", 'Pitch_RMSE': "", 'Pitch_Bias': "", 
                'Pitch_STD': "", 'Pitch_Lower_LOA': "", 'Pitch_Upper_LOA': ""
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
        
        if 'pitch' not in col_indices:
            print(f"   ❌ No 'pitch' column found in {fname}.")
            # Create empty data for missing files
            empty_df = SimpleDataFrame({'pitch': np.array([])})
            for sensor in ORDERED_SENSORS:
                sensor_df = SimpleDataFrame({'pitch': np.array([])})
                results[sensor].append(build_row(sensor_df))
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
        
        # --- Helper Function to Build Row Data ---
        def build_row(sensor_data):
            row_data = {'File_Name': fname, 'Sample_Count': len(sensor_data)}
            
            # Process Pitch only
            expected = config['expected_pitch']
            stats = calculate_pitch_stats(sensor_data, expected)
            
            row_data.update({
                'Expected_Pitch': stats['Expected'],
                'Pitch_Mean': stats['Mean'],
                'Pitch_MAE': stats['MAE'],
                'Pitch_RMSE': stats['RMSE'],
                'Pitch_Bias': stats['Bias'],
                'Pitch_STD': stats['STD'],
                'Pitch_Lower_LOA': stats['Lower_LOA'],
                'Pitch_Upper_LOA': stats['Upper_LOA']
            })
            return row_data
        
        if paused_df.empty:
            print(f"   ❌ No paused data found in {fname}.")
            # Still add empty rows for this file
            for sensor in ORDERED_SENSORS:
                sensor_df = SimpleDataFrame({'yaw': np.array([])})
                results[sensor].append(build_row(sensor_df))
            continue

        # --- Process Individual Sensors ---
        for sensor in ORDERED_SENSORS:
            sensor_mask = np.array(paused_df.data['sensor_id']) == int(sensor)
            sensor_pitch_data = np.array(paused_df.data['pitch'])[sensor_mask]
            
            # Create sensor-specific DataFrame
            sensor_df = SimpleDataFrame({'pitch': sensor_pitch_data})
            results[sensor].append(build_row(sensor_df))

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
    """Generate IMU Pitch Accuracy Validation chart"""
    print("\n📊 Generating Pitch validation plot...")
    
    # Extract pitch angles from file names
    Pitch_angles = [0, 30, 45, 60, 90, -30, -45, -60, -90]
    
    # Prepare data for plotting - only individual sensors
    sensor_labels = ['21', '22', '23']
    sensor_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(Pitch_angles))
    width = 0.25  # Wider bars since we have fewer sensors
    
    for idx, sensor in enumerate(sensor_labels):
        mae_values = []
        std_values = []
        
        for row_data in results[sensor]:
            # Extract Pitch_MAE and Pitch_STD from the row
            mae = row_data.get('Pitch_MAE', 0)
            std = row_data.get('Pitch_STD', 0)
            mae_values.append(mae if isinstance(mae, (int, float)) and not np.isnan(mae) else 0)
            std_values.append(std if isinstance(std, (int, float)) and not np.isnan(std) else 0)
        
        # Ensure we have the right number of values
        while len(mae_values) < len(Pitch_angles):
            mae_values.append(0)
            std_values.append(0)
        mae_values = mae_values[:len(Pitch_angles)]
        std_values = std_values[:len(Pitch_angles)]
        
        offset = (idx - 1) * width  # Center the three bars
        ax.bar(x + offset, mae_values, width, label=f'Sensor {sensor}',
               color=sensor_colors[idx], yerr=std_values, capsize=5, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Expected Pitch Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('IMU Pitch Angle Accuracy Validation\nError bars show ±1 standard deviation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{angle}°' for angle in Pitch_angles])
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Pitch_Validation_Chart.png', dpi=300, bbox_inches='tight')
    print("✅ Pitch validation chart saved as 'Pitch_Validation_Chart.png'")

if __name__ == "__main__":
    process_batch()