import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("RecallVsHashBits.csv")

for dataset in ["siftsmall", "sift"]:
    filtered_df = df[df["dataset"] == "sift"]

    plt.figure(figsize=(8, 4))
    plt.xlabel("Hash bits")
    plt.ylabel("Recall")
    plt.title(f"Recall vs. hash bits for CudaLshAnnIndex ({dataset})")
    plt.grid()
    plt.xlim(0, 4096)
    plt.ylim(0, 1)

    for column in ["at1", "at5", "at10", "at100", "at1000"]:
        plt.plot(filtered_df["hash_bits"], filtered_df[column], label=f"recall{column.replace('at', '@')}")
    
    plt.legend()

    plt.savefig(f"RecallVsHashBits_{dataset}.png")