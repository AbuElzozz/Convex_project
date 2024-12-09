import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# Load the dataset
data = pd.read_csv("StudentsPerformance.csv")

# Extract scores
scores = data[['math score', 'reading score', 'writing score']]
scores_array = scores.to_numpy()

# Calculate the variance of the original scores
original_variance = {
    'math': np.var(scores['math score']),
    'reading': np.var(scores['reading score']),
    'writing': np.var(scores['writing score'])
}

print(original_variance)

# Convex Optimization
adjusted_scores = cp.Variable(scores_array.shape)
alpha = 0.1
objective = cp.Minimize(cp.sum_squares(adjusted_scores - cp.mean(adjusted_scores, axis=0)) + alpha * cp.sum_squares(adjusted_scores - scores_array))
constraints = [adjusted_scores >= 0, adjusted_scores <= 100]
problem = cp.Problem(objective, constraints)
problem.solve()
balanced_scores = adjusted_scores.value

# Put balanced scores to DataFrame
scores['math_adjusted'] = balanced_scores[:, 0]
scores['reading_adjusted'] = balanced_scores[:, 1]
scores['writing_adjusted'] = balanced_scores[:, 2]

# Calculate the variance of the optimized scores
optimized_variance = {
    'math': np.var(scores['math_adjusted']),
    'reading': np.var(scores['reading_adjusted']),
    'writing': np.var(scores['writing_adjusted'])
}
print(optimized_variance)

# Visual Compare Between Original Varience and after optimization 
labels = ['Math', 'Reading', 'Writing']
original_vals = [original_variance['math'], original_variance['reading'], original_variance['writing']]
optimized_vals = [optimized_variance['math'], optimized_variance['reading'], optimized_variance['writing']]

x = np.arange(len(labels))  
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, original_vals, width, label='Original Variance')
rects2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized Variance')

for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

ax.set_ylabel('Variance')
ax.set_title('Variance Before and After Optimization')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

# Check convexity
is_convex = problem.is_dcp()
print("Is the variance problem convex?", is_convex)

# Add Outlier to violate the convexity 
np.random.seed(42)
scores['math_non_convex'] = scores['math score'] + np.random.randint(-30, 30, size=scores.shape[0])
scores['reading_non_convex'] = scores['reading score'] + np.random.randint(-30, 30, size=scores.shape[0])
scores['writing_non_convex'] = scores['writing score'] + np.random.randint(-30, 30, size=scores.shape[0])

# # Clip scores to valid range [0, 100]
# scores[['math_non_convex', 'reading_non_convex', 'writing_non_convex']] = scores[[
#     'math_non_convex', 'reading_non_convex', 'writing_non_convex']].clip(0, 100)

# Restore Convexity
outlier_scores_array = scores[['math_non_convex', 'reading_non_convex', 'writing_non_convex']].to_numpy()
restored_scores = cp.Variable(outlier_scores_array.shape)
alpha = 0.1
objective_restore = cp.Minimize(cp.sum_squares(restored_scores - cp.mean(restored_scores, axis=0)) + alpha * cp.sum_squares(restored_scores - outlier_scores_array))
constraints_restore = [restored_scores >= 0, restored_scores <= 100]
problem_restore = cp.Problem(objective_restore, constraints_restore)
problem_restore.solve()
restored_scores_value = restored_scores.value

# Add results to DataFrame
scores['math_restored'] = restored_scores_value[:, 0]
scores['reading_restored'] = restored_scores_value[:, 1]
scores['writing_restored'] = restored_scores_value[:, 2]

# Visualize Convexity Check (Straight Line Increasing improve that is convex)
iterations = np.arange(1, 21)
variance_values = [np.var(scores_array[:i, :]) for i in iterations]

plt.figure(figsize=(10, 5))
plt.plot(iterations, variance_values, marker='o', label='Variance of Scores')
plt.title("Variance Across Subsets of Scores (Convexity Check)")

convexity_text = 'Convex' if is_convex else 'Non-Convex'
plt.text(15, 0.15, f'Optimization is {convexity_text}', fontsize=12, color='red', ha='center')
plt.xlabel("Subset Size")
plt.ylabel("Variance")
plt.legend()
plt.grid(True)
plt.show()

# Visualize Original, Non-Convex, and Restored Scores
plt.figure(figsize=(15, 5))

for i, subject in enumerate(['math', 'reading', 'writing']):
    plt.subplot(1, 3, i + 1)
    plt.hist(scores[f'{subject} score'], bins=20, alpha=0.7, label='Original')
    plt.hist(scores[f'{subject}_non_convex'], bins=20, alpha=0.7, label='Non-Convex')
    plt.hist(scores[f'{subject}_restored'], bins=20, alpha=0.7, label='Restored')
    plt.title(f"{subject.capitalize()} Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

for i, subject in enumerate(['math', 'reading', 'writing']):
    plt.subplot(1, 3, i + 1)
    plt.hist(scores[f'{subject}_restored'], bins=20, alpha=0.7, label='Restored')
    plt.title(f"{subject.capitalize()} Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout()
plt.show()


from scipy.spatial import ConvexHull

# Function to check and visualize convex hull
def plot_convex_hull(scores_data, title, ax):
    # Ensure the data is in 2D form
    hull = ConvexHull(scores_data)
    ax.scatter(scores_data[:, 0], scores_data[:, 1], c='blue', label='Data Points')
    for simplex in hull.simplices:
        ax.plot(scores_data[simplex, 0], scores_data[simplex, 1], 'r-', label='Convex Hull' if simplex[0] == 0 else "")
    ax.set_title(title)
    ax.legend()

# Prepare data for convex hull visualization
scores_2d = scores[['math score', 'reading score']].to_numpy()  # Original scores (2D)
non_convex_2d = scores[['math_non_convex', 'reading_non_convex']].to_numpy()  # Non-convex scores (2D)
restored_2d = scores[['math_restored', 'reading_restored']].to_numpy()  # Restored scores (2D)

# Plotting convex hulls
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

plot_convex_hull(scores_2d, "Convex Hull - Original Scores", axs[0])
plot_convex_hull(non_convex_2d, "Convex Hull - Non-Convex Scores", axs[1])
plot_convex_hull(restored_2d, "Convex Hull - Restored Scores", axs[2])

plt.tight_layout()
plt.show()
