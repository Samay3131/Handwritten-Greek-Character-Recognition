#%%
import time

# List of script file paths
script_files = [
    "C:/My_Project/Preprocessing.py",
    "C:/My_Project/EDA Greek Classification.py",
    "C:/My_Project/EDA GCDB Dataset.py",
    "C:/My_Project/PCA.py",
    "C:/My_Project/KNN.py",
    "C:/My_Project/descision trees.py",
    "C:/My_Project/Tensorflow GCDB Dataset.py"
]

# Execute each script using exec()
total_execution_time = 0

for script_file in script_files:
    print(f"Executing {script_file}...")
    start_time = time.time()
    with open(script_file, "r") as file:
        script_content = file.read()
        exec(script_content)
    end_time = time.time()
    
    execution_time = end_time - start_time
    total_execution_time += execution_time
    print(f"{script_file} executed in {execution_time:.2f} seconds")

total_execution_time_minutes = total_execution_time / 60
print("All scripts executed successfully")
print(f"Total execution time: {total_execution_time_minutes:.2f} minutes")

 # %%
