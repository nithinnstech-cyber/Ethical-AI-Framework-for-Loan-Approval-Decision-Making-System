import pandas as pd

# Example metrics (replace with your actual model's accuracy if you have it)
accuracy = 0.8961

# Create a DataFrame
metrics = {
    'Metric': ['Accuracy'],
    'Value': [accuracy]
}

df = pd.DataFrame(metrics)

# Save the file to outputs
df.to_csv("../outputs/performance_metrics.csv", index=False)

print("✅ performance_metrics.csv generated at ../outputs/performance_metrics.csv")
