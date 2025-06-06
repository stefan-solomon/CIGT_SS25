#Show Q-table as image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_q_table(Q, env):
    cols = []
    for col in env.action_map.values():
        if col == "commission_solar":
            cols.append("Com_Solar")
        elif col == "commission_nuclear":
            cols.append("Com_Nuclear")
        elif col == "decommission_solar":
            cols.append("Decom_Solar")
        elif col == "decommission_nuclear":
            cols.append("Decom_Nuclear")
        elif col == "decommission_coal":
            cols.append("Decom_Coal")
        else:
            cols.append(col)
    q_table = pd.DataFrame.from_dict(Q, orient='index', columns=cols)
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_table, annot=True, cmap='coolwarm', cbar_kws={'label': 'Q-value'})
    plt.title("Q-table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States (n_solar, n_nuclear, n_coal)")
    plt.show()