import subprocess
import os
import pandas as pd
import numpy as np
import tensorflow as tf

def generate_xfoil_data(naca_code, alpha_start, alpha_end, alpha_step, reynolds):
    polar_filename = f"polar_{naca_code}.txt"
    
    # Delete old polar file if it exists
    if os.path.exists(polar_filename):
        os.remove(polar_filename)

    commands = [
        f"NACA {naca_code}",   # Generate the airfoil shape
        "PPAR",        # Enter Paneling Parameters
        "N 200",       # Increase number of nodes to 200 (standard is ~160)
        "P",           # Set node distribution 
        "1",           # Use a distribution that clusters more at the LE/TE
        "", "",        # Enter twice to return to main menu
        "PANE",                # Smooth the paneling (critical for convergence)
        "OPER",                # Enter operation mode
        f"Visc {reynolds}",    # Turn on viscous mode and set Re
        "ITER 200",            # Increase iterations for better convergence
        "PACC",                # Toggle Polar Accumulation on
        polar_filename,        # Name of the save file
        "",                    # Hit enter to skip dump file
        f"ASEQ {alpha_start} {alpha_end} {alpha_step}", # Run the alpha sweep
        "PACC",                # Toggle Polar Accumulation off
        "QUIT"                 # Exit XFOIL
    ]
    
    command_string = "\n".join(commands) + "\n"

    # 2. Run XFOIL silently using subprocess
    process = subprocess.Popen(
        ['xvfb-run', 'xfoil'], 
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send the commands
    stdout, stderr = process.communicate(input=command_string)

    # 3. polar file into pandas
    try:
        with open(polar_filename) as f:
            lines = f.readlines()
        data_start = next(i for i, l in enumerate(lines) if l.strip().startswith('---')) + 1 # data_start skips useless lines in polar_xxxx.txt
        df = pd.read_csv(polar_filename, skiprows=data_start, sep=r'\s+',                    # sep=r'\s+' to handle irregular spacing.
                         names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'Top_Itr', 'Bot_Itr'])
        
        # Add inputs to the dataframe for the NN
        df['NACA'] = naca_code
        df['Re'] = reynolds

        os.remove(polar_filename)  # remove polar_####.txt

        return df[['NACA', 'Re', 'alpha', 'CL', 'CD']] # Return only what we need
        
    except FileNotFoundError:
        print(f"XFOIL failed to converge for NACA {naca_code} at Re={reynolds}")
        return pd.DataFrame() # Return empty df if it failed


def generate_naca():
    naca_list = []
    
    # 1. Symmetrical Airfoils (00XX) - Teaches thickness vs drag
    # Range: 6% (thin) to 24% (thick)
    symmetrical = [f"00{t:02d}" for t in [6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 22, 24]]
    naca_list.extend(symmetrical)
    
    # 2. Low Camber (24XX) - Standard general aviation shapes
    # Varying thicknesses for the same camber
    low_camber = [f"24{t:02d}" for t in [8, 10, 12, 14, 15, 18, 20, 24]]
    naca_list.extend(low_camber)
    
    # 3. High Camber (44XX & 64XX) - High lift/STOL shapes
    high_camber = [f"{c}4{t:02d}" for c in [4, 6] for t in [10, 12, 15, 18, 21]]
    naca_list.extend(high_camber)
    
    # 4. Camber Position Sweep (X2XX, X4XX, X6XX)
    # Teaches the NN how moving the "peak" of the curve affects pitching moment
    pos_sweep = [f"4{p}{t:02d}" for p in [2, 3, 5, 6] for t in [10, 12, 15]]
    naca_list.extend(pos_sweep)
    
    # 5. Fine-tuning the "Common" shapes
    # Adding more variations around the most popular NACA airfoils (like 2412)
    fine_tune = ["1408", "1410", "1412", "3412", "5412", "2312", "2512", "2612"]
    naca_list.extend(fine_tune)

    # Remove duplicates and cap at 60 (though this logic gives ~60 unique ones)
    return list(dict.fromkeys(naca_list))[:60]


# ========================= RUN XFOIL AND SAVE AS CSV ===============================
print("Running XFOIL simulation...")
naca_pack = generate_naca()
reynolds_list = [1e5, 5*1e5, 1e6, 5*1e6, 1e7]
all_data_frames = []

total = len(naca_pack) * len(reynolds_list)
count = 0

for code in naca_pack:
    for reynold in reynolds_list:
        count += 1
        # The flush=True ensures you see the progress even if the cloud is lagging
        print(f"[{count}/{total}] NACA {code} | Re {reynold:,.0f}...", end=" ", flush=True)
        
        try:
            results_df = generate_xfoil_data(naca_code=code, alpha_start=-5, alpha_end=15, alpha_step=1, reynolds=reynold)
            
            if not results_df.empty:
                # Double check that these columns exist for the NN
                results_df['NACA'] = code
                results_df['Re'] = reynold
                all_data_frames.append(results_df)
                print("Done.")
            else:
                print("Failed (No Convergence).")
        except Exception as e:
            print(f"Error: {e}")

if all_data_frames:
    master_df = pd.concat(all_data_frames, ignore_index=True)
    master_df.to_csv("airfoil_dataset.csv", index=False)
    print(f"\nSuccess! Saved {len(master_df)} rows to airfoil_dataset.csv")

# delete .bl file
import glob
for f in glob.glob("*.bl"):
    os.remove(f)
# ===================================================================================

def create_aerospace_dataset(csv_file, batch_size=32):
    df = pd.read_csv(csv_file)
    
    # Extract physics from NACA 
    naca_str = df['NACA'].astype(str).str.zfill(4)  # ensure NACA is 4 digits
    df['camber'] = naca_str.str[0].astype(float) / 100.0
    df['camber_pos'] = naca_str.str[1].astype(float) / 10.0
    df['thickness'] = naca_str.str[2:4].astype(float) / 100.0
    

    inputs = df[['camber', 'camber_pos', 'thickness', 'Re', 'alpha']].values.astype(np.float32)
    outputs = df[['CL', 'CD']].values.astype(np.float32)
    
    # NORMALIZATION (Crucial for Neural Networks)
    # Neural networks struggle when Re is 1,000,000 but thickness is 0.12. 
    # We scale all inputs to have a mean of 0 and std of 1.
    x_mean = inputs.mean(axis=0)
    x_std = inputs.std(axis=0) + 1e-8
    inputs_scaled = (inputs - x_mean) / x_std
    
    # Create the tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((inputs_scaled, outputs))
    
    # Shuffle and batch for training
    dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, x_mean, x_std

# --- Build the Neural Network ---
def build_surrogate_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(5,)),   # 5 neurons in input (camber, camber_pos, thickness, Re, alpha)
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),  
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)                        # 2 neurons in output (CL, CD)
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse', 
        metrics=['mae']
    )
    
    return model
"""
# ========================= MODEL BUILD AND TRAIN ===============================
# 1. Prepare the data
train_dataset, x_mean, x_std = create_aerospace_dataset("airfoil_dataset.csv", batch_size=32)
np.save("x_mean.npy", x_mean)
np.save("x_std.npy", x_std)     # save these for later

# 2. Build the model
model = build_surrogate_model()

# 3. Train it!
print("Training the Aerodynamic Surrogate...")
history = model.fit(train_dataset, epochs=200)
# ================================================================================



# --- 1. SETUP YOUR INPUT ---
naca = "2412"
alpha = 5.0
reynolds = 1000000

# Feature Engineering (NACA 4-digit logic)
camber = float(naca[0]) / 100.0
camber_pos = float(naca[1]) / 10.0
thickness = float(naca[2:4]) / 100.0

# Create the raw vector [camber, pos, thickness, Re, alpha]
raw_input = np.array([[camber, camber_pos, thickness, reynolds, alpha]])

# --- 2. NORMALIZE (Must use the SAME mean/std from your training data) ---
# scaled_input = (raw_input - x_mean) / x_std
scaled_input = (raw_input - x_mean) / x_std

# --- 3. GET PREDICTION ---
# model.predict returns a 2D array: [[CL, CD]]
predictions = model.predict(scaled_input)

cl_final = predictions[0][0]
cd_final = predictions[0][1]

print(f"For NACA {naca} at alpha {alpha}:")
print(f"-> Predicted CL: {cl_final:.4f}")
print(f"-> Predicted CD: {cd_final:.4f}")
"""