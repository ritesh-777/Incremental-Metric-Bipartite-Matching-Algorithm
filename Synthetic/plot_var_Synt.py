import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 16})  # default font size for everything

def generate_data_avg_with_std(start_n, end_n, gap, instances, delta, batchSize):
    prpr_cost_all = []  # Store all individual costs for std calculation
    prpr_time_all = []  # Store all individual times for std calculation
    prpr_cost_avg = []
    prpr_time_avg = []
    prpr_cost_std = []
    prpr_time_std = []
    list_of_n = []
    
    for i in range(start_n, end_n+1, gap):
        prpr_cost_iavg = []
        prpr_time_iavg = []
        
        for j in range(instances):
            prpr_csv = f"PlotData/10000/{j+1}_PRPR_{delta}000_Synt_Eu_2dim.csv"
            prpr_df = pd.read_csv(prpr_csv).head(int(i/batchSize))
            
            prpr_df['cost'] = prpr_df['cost'].astype(float)
            prpr_df['execution_time'] = prpr_df['execution_time'].astype(float)
            
            # Calculate cost per request and time per request for this instance
            cost_per_request = prpr_df.loc[:, 'cost'].iloc[-1] / i
            time_per_request = prpr_df['execution_time'].sum() / i
            
            prpr_cost_iavg.append(cost_per_request)
            prpr_time_iavg.append(time_per_request)
        
        # Store all values for this n
        prpr_cost_all.append(prpr_cost_iavg)
        prpr_time_all.append(prpr_time_iavg)
        
        # Calculate mean and std for this n
        prpr_cost_avg.append(np.mean(prpr_cost_iavg))
        prpr_time_avg.append(np.mean(prpr_time_iavg))
        prpr_cost_std.append(np.std(prpr_cost_iavg))
        prpr_time_std.append(np.std(prpr_time_iavg))
        
        list_of_n.append(i)
    
    # Convert to numpy arrays for easier manipulation
    list_of_n = np.array(list_of_n)
    prpr_cost_avg = np.array(prpr_cost_avg)
    prpr_time_avg = np.array(prpr_time_avg)
    prpr_cost_std = np.array(prpr_cost_std)
    prpr_time_std = np.array(prpr_time_std)
    
    # Plot 1: Cost with std deviation shading
    plt.figure(figsize=(10, 6))
    
    # Plot mean line
    plt.plot(list_of_n, prpr_cost_avg, color='#1f77b4', linewidth=2.5, label='PR (mean)')
    
    # Plot std deviation shading
    plt.fill_between(list_of_n, 
                     prpr_cost_avg - prpr_cost_std, 
                     prpr_cost_avg + prpr_cost_std, 
                     color='#1f77b4', alpha=0.3, label='±1 Std')
    
    plt.title(f"Mean ± 1 Std across instances\nBatch Incremental PR: Average Cost vs. Number of Requests")
    plt.xlabel("Number of requests")
    plt.ylabel("Average Cost per Request")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out1 = f"synt_cost_vs_n_{delta}_with_std.png"
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved PRPR cost plot with std to {out1}")
    
    # Plot 2: Runtime with std deviation shading
    plt.figure(figsize=(10, 6))
    
    # Plot mean line
    plt.plot(list_of_n, prpr_time_avg, color='#ff7f0e', linewidth=2.5, label='PR (mean)')
    
    # Plot std deviation shading
    plt.fill_between(list_of_n, 
                     prpr_time_avg - prpr_time_std, 
                     prpr_time_avg + prpr_time_std, 
                     color='#ff7f0e', alpha=0.3, label='±1 Std')
    
    plt.title(f"Mean ± 1 Std across instances\nBatch Incremental PR: Average Runtime vs. Number of Requests")
    plt.xlabel("Number of requests")
    plt.ylabel("Average Time per Request (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out2 = f"synt_runtime_vs_n_{delta}_with_std.png"
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved PRPR runtime plot with std to {out2}")
    
    # Optional: Print statistics
    print(f"\nStatistics for PRPR Algorithm:")
    print(f"Cost - Mean across all n: {np.mean(prpr_cost_avg):.4f}")
    print(f"Cost - Average std deviation: {np.mean(prpr_cost_std):.4f}")
    print(f"Runtime - Mean across all n: {np.mean(prpr_time_avg):.4f}")
    print(f"Runtime - Average std deviation: {np.mean(prpr_time_std):.4f}")

# Call the function
generate_data_avg_with_std(1000, 10000, 1000, 10, "0.001", 200)