# run_all.py

import sys
# Assuming your modified main function is in a file named 'your_script.py'
from train_mlp_club import main # <--- IMPORT THE MAIN FUNCTION

def run_all_experiments():
    optimizers = ['SGD', 'AdamW', 'GDM', 'DMGD']
    epochs_list = [5, 15, 50, 200, 500, 3999]

    total_runs = len(optimizers) * len(epochs_list)
    current_run = 0

    for optim in optimizers:
        for epochs in epochs_list:
            current_run += 1
            print(f"\n========================================================")
            print(f"  STARTING RUN {current_run}/{total_runs}: {optim} for {epochs} epochs")
            print(f"========================================================\n")
            
            # Call the main function with the specific optimizer and epochs
            try:
                main(chosen_optim=optim, epochs=epochs)
            except Exception as e:
                print(f"ERROR in run {current_run} ({optim}, {epochs} epochs): {e}")
                # You might want to log this error and continue to the next run
                # or re-raise the exception depending on your needs.
                # sys.exit(1) # Exit immediately on error
                
    print("\nAll experiments finished.")

if __name__ == '__main__':
    run_all_experiments()