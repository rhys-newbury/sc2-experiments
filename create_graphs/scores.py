import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sc2_replay_reader import Result

# Your list of scores
# data = [
#     [0, 1, 2, 3, 4],
#     [0, 1, 2],
#     [0, 1, 2, 3, 4, 54, 5, 5, 5, 5, 5, 4, 56, 454, 45],
#     [0, 1, 2, 4, 5, 6, 76]
# ]

file_name = "output"
print(int(Result.Win))
input()
data0 = []
file_path = f"{file_name}0.scores"

with open(file_path, "r") as file:
    for line in tqdm(file, desc="Reading lines", unit=" lines"):
        data0.append(
            [
                [
                    float(y.strip())
                    for y in x.split(";")
                    if x.strip() != "" and len(x.split(";")) != 0
                ]
                for x in line.split(",")
                if x.strip() != ""
            ]
        )

# Assuming f"{file_name}0.scores" is in the current working directory
file_path = f"{file_name}1.scores"
data1 = []
with open(file_path, "r") as file:
    for line in tqdm(file, desc="Reading lines", unit=" lines"):
        data1.append(
            [
                [
                    float(y.strip())
                    for y in x.split(";")
                    if x.strip() != "" and len(x.split(";")) != 0
                ]
                for x in line.split(",")
                if x.strip() != ""
            ]
        )

x_values0 = [point[1] for game_data in data0 for point in game_data]
y_values0 = [point[0] for game_data in data0 for point in game_data]

x_values1 = [point[1] for game_data in data1 for point in game_data]
y_values1 = [point[0] for game_data in data1 for point in game_data]

common_x = np.arange(min(min(x_values0), min(x_values1)), 44800, 100)
print("find commonx")
final0 = []
for line in data0:
    if len(line) == 0:
        continue
    xs = [point[1] for point in line]
    ys = [point[0] for point in line]

    interpolated_y = np.interp(common_x, xs, ys)
    interpolated_y[common_x > max(xs)] = np.nan

    final0.append(list(interpolated_y))
print("interp0 done")
final0 = np.array(final0)
print("np arrayed it")
final1 = []
for line in data1:
    if len(line) == 0:
        continue
    xs = [point[1] for point in line]
    ys = [point[0] for point in line]

    interpolated_y = np.interp(common_x, xs, ys)
    interpolated_y[common_x > max(xs)] = np.nan

    final1.append(interpolated_y)
print("interp1 done")

final1 = np.array(final1)

common_x = common_x / 22.4 / 60

# Calculate means and standard deviations for each time point
means0 = np.nanmean(final0, axis=0)
std_devs0 = np.nanstd(final0, axis=0)
print("means0 done")

# Calculate means and standard deviations for each time point
means1 = np.nanmean(final1, axis=0)
std_devs1 = np.nanstd(final1, axis=0)
print("means1 done")

# Plotting with fill_between
plt.plot(common_x, means0, label="Scores (Win)")
plt.fill_between(common_x, means0 - std_devs0, means0 + std_devs0, alpha=0.2)

plt.plot(common_x, means1, label="Scores (Loss)")
plt.fill_between(common_x, means1 - std_devs1, means1 + std_devs1, alpha=0.2)

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Score")
plt.title("Scores Over Time with Standard Deviation")
plt.legend()
plt.grid(True)
plt.savefig("scores.png")
