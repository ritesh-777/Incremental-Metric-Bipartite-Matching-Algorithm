import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})  # default font size for everything

def generate_data_avg(start_n, end_n, gap, instances, delta, batchSize):
    pr_cost_avg = []
    pr_time_avg = []
    prpr_cost_avg = []
    prpr_time_avg = []
    greedy_cost_avg = []
    greedy_time_avg = []
    qt_cost_avg = []
    qt_time_avg = []
    dyneuc_cost_avg = []
    dyneuc_time_avg = []
    opt_cost_avg = []
    opt_time_avg = []
    list_of_n = []
    for i in range (start_n, end_n+1, gap):
        pr_cost_iavg = []
        pr_time_iavg = []
        prpr_cost_iavg = []
        prpr_time_iavg = []
        greedy_cost_iavg = []
        greedy_time_iavg = []
        qt_cost_iavg = []
        qt_time_iavg = []
        dyneuc_cost_iavg = []
        dyneuc_time_iavg = []
        opt_cost_iavg = []
        opt_time_iavg = []
        for j in range (instances):

            skip_n = batchSize - 1   # how many rows to skip
            take_every = skip_n + 1  # take the row after skipping
            number_of_rows_to_select = int((end_n-1000) / batchSize)
            #print(number_of_columns_to_select)
            '''
            pr_csv = f"PlotData/{i}/{j+1}_PR_{delta}000_Real_Eu_2dim.csv"
            pr_df = pd.read_csv(pr_csv)
            pr_df.sort_values(by='n', inplace=True)
            pr_df['cost'] = pr_df['cost'].astype(float)
            pr_df['run_time'] = pr_df['run_time'].astype(float)
            pr_cost_iavg.append(pr_df.loc[:, 'cost'].mean())
            pr_time_iavg.append(pr_df.loc[:, 'run_time'].mean())
            '''
            
            prpr_csv = f"PlotData/10000/{j+1}_PRPR_{delta}000_Synt_Eu_2dim.csv"
            prpr_df = pd.read_csv(prpr_csv).head(int(i/batchSize))
            #prpr_df.sort_values(by='n', inplace=True)
            #print(prpr_df)
            prpr_df['cost'] = prpr_df['cost'].astype(float)
            selected_prpr_cost_df = prpr_df['cost']/((prpr_df.index+1)*batchSize)
            print(selected_prpr_cost_df.mean(), i , [int(i/batchSize)-1])
            prpr_df['execution_time'] = prpr_df['execution_time'].astype(float)
            selected_prpr_time_df = prpr_df['execution_time']#/((prpr_df.index+1)*batchSize)
            #prpr_cost_iavg.append(selected_prpr_cost_df[int(i/batchSize)-1])
            #prpr_time_iavg.append(selected_prpr_cost_df.mean())
            #print(selected_prpr_time_df.mean())
            prpr_time_iavg.append(prpr_df['execution_time'].sum()/i)
            prpr_cost_iavg.append(prpr_df['cost'].iloc[-1]/i)
            #prpr_time_iavg.append(prpr_df.loc[:, 'run_time'].mean())
            print(prpr_cost_iavg)
            
            
            greedy_csv = f"PlotData/10000/{j+1}_Greedy_{delta}000_Synt_Eu_2dim.csv"
            greedy_df = pd.read_csv(greedy_csv).head(i)
            #greedy_df.sort_values(by='n', inplace=True)
            greedy_df['cost'] = greedy_df['cost'].astype(float)
            selected_greedy_cost_df = greedy_df['cost'].iloc[take_every-1::take_every]
            #print(selected_greedy_cost_df)
            #selected_greedy_cost_df = selected_greedy_cost_df/(selected_greedy_cost_df.index+1)
            #print(selected_greedy_cost_df)
            greedy_df['execution_time'] = greedy_df['execution_time'].astype(float)
            # Partition into k groups using integer division of index
            greedy_df["Group"] = greedy_df.index // batchSize
            # Compute average per group
            selected_greedy_time_df = greedy_df.groupby("Group")["execution_time"].mean().reset_index(drop=True)
            #selected_greedy_time_df = greedy_df['run_time'].iloc[take_every-1::take_every]
            #print(selected_greedy_time_df.mean())
            greedy_cost_iavg.append(selected_greedy_cost_df.iloc[-1]/i)
            #greedy_cost_iavg.append(selected_greedy_cost_df[i-1])
            #greedy_time_iavg.append(selected_greedy_time_df.mean())
            #greedy_cost_iavg.append(greedy_df.loc[:, 'cost'].mean())
            #greedy_cost_iavg.append(greedy_df['cost'].iloc[-1])
            greedy_time_iavg.append(greedy_df.loc[:, 'execution_time'].mean())
            
            qt_csv = f"PlotData/10000/{j+1}_QT_{delta}000_Synt_Eu_2dim.csv"
            qt_df = pd.read_csv(qt_csv).head(i)
            #qt_df.sort_values(by='n', inplace=True)
            qt_df['cost'] = qt_df['cost'].astype(float)
            selected_qt_cost_df = qt_df['cost'].iloc[take_every-1::take_every]
            #selected_qt_cost_df = selected_qt_cost_df/(selected_qt_cost_df.index+1)
            qt_df['run_time'] = qt_df['run_time'].astype(float)
            # Partition into k groups using integer division of index
            qt_df["Group"] = qt_df.index // batchSize
            # Compute average per group
            selected_qt_time_df = qt_df.groupby("Group")["run_time"].mean().reset_index(drop=True)
            #selected_qt_time_df = qt_df['run_time'].iloc[take_every-1::take_every]
            qt_cost_iavg.append(selected_qt_cost_df.iloc[-1]/i)
            #qt_time_iavg.append(selected_qt_time_df.mean())
            #qt_cost_iavg.append(qt_df.loc[:, 'cost'].mean())
            qt_time_iavg.append(qt_df.loc[:, 'run_time'].mean())

            dyneuc_csv = f"PlotData/10000/{j+1}_DynEuc_B16_{delta}000_Synt_Eu_2dim.csv"
            dyneuc_df = pd.read_csv(dyneuc_csv).head(i)
            #dyneuc_df.sort_values(by='n', inplace=True)
            dyneuc_df['cost'] = dyneuc_df['cost'].astype(float)
            selected_dyneuc_cost_df = dyneuc_df['cost'].iloc[take_every-1::take_every]
            #selected_dyneuc_cost_df = selected_dyneuc_cost_df/(selected_dyneuc_cost_df.index+1)
            dyneuc_df['execution_time'] = dyneuc_df['execution_time'].astype(float)
            # Partition into k groups using integer division of index
            dyneuc_df["Group"] = dyneuc_df.index // batchSize
            # Compute average per group
            selected_dyneuc_time_df = dyneuc_df.groupby("Group")["execution_time"].mean().reset_index(drop=True)
            #selected_dyneuc_time_df = dyneuc_df['execution_time'].iloc[take_every-1::take_every]
            dyneuc_cost_iavg.append(selected_dyneuc_cost_df.iloc[-1]/i)
            #dyneuc_time_iavg.append(selected_dyneuc_time_df.mean())
            #dyneuc_cost_iavg.append(dyneuc_df.loc[:, 'cost'].mean())
            dyneuc_time_iavg.append(dyneuc_df.loc[:, 'execution_time'].mean())
            
            '''
            if i <= 1000 :
                opt_csv = f"PlotData/10000/{j+1}_Opt_{i}_{delta}000_Synt_Eu_2dim.csv"
                opt_df = pd.read_csv(opt_csv).head(i)
                #opt_df.sort_values(by='n', inplace=True)
                opt_df['cost'] = opt_df['cost'].astype(float)
                selected_opt_cost_df = opt_df['cost'].iloc[take_every-1::take_every]
                #selected_opt_cost_df = selected_opt_cost_df/(selected_opt_cost_df.index+1)
                opt_df['run_time'] = opt_df['run_time'].astype(float)
                # Partition into k groups using integer division of index
                opt_df["Group"] = opt_df.index // batchSize
                # Compute average per group
                selected_opt_time_df = opt_df.groupby("Group")["run_time"].mean().reset_index(drop=True)
                #selected_opt_time_df = opt_df['run_time'].iloc[take_every-1::take_every]
                opt_cost_iavg.append(selected_opt_cost_df.iloc[-1]/i)
                #opt_time_iavg.append(selected_opt_time_df.mean())
                #opt_cost_iavg.append(opt_df.loc[:, 'cost'].mean())
                opt_time_iavg.append(opt_df.loc[:, 'run_time'].mean())
            '''
        '''
        pr_cost_avg.append(sum(pr_cost_iavg)/float(len(pr_cost_iavg)))
        pr_time_avg.append(sum(pr_time_iavg)/float(len(pr_time_iavg)))
        '''
        prpr_cost_avg.append(sum(prpr_cost_iavg)/float(len(prpr_cost_iavg)))
        prpr_time_avg.append(sum(prpr_time_iavg)/float(len(prpr_time_iavg)))

        greedy_cost_avg.append(sum(greedy_cost_iavg)/float(len(greedy_cost_iavg)))
        greedy_time_avg.append(sum(greedy_time_iavg)/float(len(greedy_time_iavg)))
        
        qt_cost_avg.append(sum(qt_cost_iavg)/float(len(qt_cost_iavg)))
        qt_time_avg.append(sum(qt_time_iavg)/float(len(qt_time_iavg)))

        dyneuc_cost_avg.append(sum(dyneuc_cost_iavg)/float(len(dyneuc_cost_iavg)))
        dyneuc_time_avg.append(sum(dyneuc_time_iavg)/float(len(dyneuc_time_iavg)))

        '''
        if i <= 1000 :
            opt_cost_avg.append(sum(opt_cost_iavg)/float(len(opt_cost_iavg)))
            opt_time_avg.append(sum(opt_time_iavg)/float(len(opt_time_iavg)))
        '''
        
        list_of_n.append(i)
    plt.figure(figsize=(8, 5))
    #plt.plot(list_of_n, pr_cost_avg, marker='o', label="Push–Relabel")
    plt.plot(list_of_n, prpr_cost_avg, marker='o', label="Batch Incremental PR")
    plt.plot(list_of_n, greedy_cost_avg, marker='s', label="Greedy")
    plt.plot(list_of_n, qt_cost_avg, marker='^', label="Quad Tree")
    plt.plot(list_of_n, dyneuc_cost_avg, marker='d', label="Dynamic Euclidean")
    #plt.plot([1000], opt_cost_avg, marker='*', label="Hungarian(Optimal)")
    #plt.plot(cr_df["n"], cr_df["max_cr_qt"], marker='^', label="Quadtree")
    plt.title(f"Average Cost vs. Number of Requests (n)")
    plt.xlabel("n (number of requests)")
    plt.ylabel("Average Cost (PR vs Greedy vs QT vs DyEu)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out1 = f"cost_vs_n_{delta}_synt.png"
    plt.savefig(out1)
    plt.show()
    print(f"Saved competitive-ratio plot to {out1}")


    plt.figure(figsize=(8, 5))
    #plt.plot(list_of_n, pr_time_avg, marker='o', label="Push–Relabel")
    plt.plot(list_of_n, prpr_time_avg, marker='o', label="Batch Incremental PR")
    plt.plot(list_of_n, greedy_time_avg, marker='s', label="Greedy")
    plt.plot(list_of_n, qt_time_avg, marker='^', label="Quad Tree")
    plt.plot(list_of_n, dyneuc_time_avg, marker='d', label="Dynamic Euclidean")
    #plt.plot([1000], opt_time_avg, marker='*', label="Optimal")
    #plt.plot(cr_df["n"], cr_df["max_cr_qt"], marker='^', label="Quadtree")
    plt.title(f"Average Match Time per Request vs. Number of Requests (n)")
    plt.xlabel("n (number of requests)")
    plt.ylabel("Average Time per Request (PR vs Greedy vs QT vs DyEu)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out2 = f"match_time_vs_n_{delta}_synt.png"
    plt.savefig(out2)
    plt.show()
    print(f"Saved match-time plot to {out2}")



generate_data_avg(1000, 10000, 1000, 10, "0.001", 200)


'''
# --- Configuration (match your CSV naming) ---
delta = 0.125
min_bound = 0.0
max_bound = 100.0
dimensions = 2

# Construct file name prefixes based on the experimental settings
#prefix = f"{delta}_{min_bound}_{max_bound}_Synthetic_{dimensions}dim"
prefix = f"0.001_Real_Eu_2dim"
cr_csv = f"CR_{prefix}.csv"
time_csv = f"Time_{prefix}.csv"

# --- Load data ---
# Competitive ratios CSV contains columns: n, max_cr_pr, max_cr_greedy, max_cr_qt
cr_df = pd.read_csv(cr_csv)
cr_df.sort_values(by='n', inplace=True)
# Time CSV contains columns: n, pr_avg_time, greedy_avg_time, qt_time, opt_avg_time
time_df = pd.read_csv(time_csv)
time_df.sort_values(by='n', inplace=True)

# --- Plot 1: Competitive Ratio vs n for PR, Greedy, and Quadtree ---
plt.figure(figsize=(8, 5))
plt.plot(cr_df["n"], cr_df["max_cr_pr"], marker='o', label="Push–Relabel")
plt.plot(cr_df["n"], cr_df["max_cr_greedy"], marker='s', label="Greedy")
plt.plot(cr_df["n"], cr_df["max_cr_qt"], marker='^', label="Quadtree")
plt.title(f"Max Competitive Ratio vs. Number of Servers (n) [{prefix}]")
plt.xlabel("n (number of servers/requests)")
plt.ylabel("Max Competitive Ratio (Algorithm / OPT)")
plt.legend()
plt.grid(True)
plt.tight_layout()
out1 = f"competitive_ratio_vs_n_{prefix}.png"
plt.savefig(out1)
plt.show()
print(f"Saved competitive-ratio plot to {out1}")

# --- Plot 2: Average Match Time vs n for PR, Greedy, Quadtree, and Optimal ---
plt.figure(figsize=(8, 5))
plt.plot(time_df["n"], time_df["pr_avg_time"], marker='o', label="Push–Relabel")
plt.plot(time_df["n"], time_df["greedy_avg_time"], marker='s', label="Greedy")
plt.plot(time_df["n"], time_df["qt_time"], marker='d', label="Quadtree")
plt.plot(time_df["n"], time_df["opt_avg_time"], marker='^', label="Optimal Algo")
plt.title(f"Average Match Time per Request vs. Number of Servers (n) [{prefix}]")
plt.xlabel("n (number of servers/requests)")
plt.ylabel("Average Time per Request (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
out2 = f"match_time_vs_n_{prefix}.png"
plt.savefig(out2)
plt.show()
print(f"Saved match-time plot to {out2}")
'''