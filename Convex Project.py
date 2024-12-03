import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Load the dataset
data = pd.read_csv("StudentsPerformance.csv")

# Extract scores
scores = data[['math score', 'reading score', 'writing score']]
scores_array = scores.to_numpy()

# 1. Calculate the variance of the original scores
original_variance = {
    'math': np.var(scores['math score']),
    'reading': np.var(scores['reading score']),
    'writing': np.var(scores['writing score'])
}
print(original_variance)

# 2. Convex Optimization - Balancing Scores
adjusted_scores = cp.Variable(scores_array.shape)
objective = cp.Minimize(cp.sum_squares(adjusted_scores - cp.mean(adjusted_scores, axis=0)))
constraints = [adjusted_scores >= 0, adjusted_scores <= 100]
problem = cp.Problem(objective, constraints)
problem.solve()
balanced_scores = adjusted_scores.value

# Add balanced scores to DataFrame
scores['math_adjusted'] = balanced_scores[:, 0]
scores['reading_adjusted'] = balanced_scores[:, 1]
scores['writing_adjusted'] = balanced_scores[:, 2]

# 3. Calculate the variance of the optimized scores
optimized_variance = {
    'math': np.var(scores['math_adjusted']),
    'reading': np.var(scores['reading_adjusted']),
    'writing': np.var(scores['writing_adjusted'])
}
print(optimized_variance)

# 4. Plotting the comparison of variance before and after optimization in a single chart
labels = ['Math', 'Reading', 'Writing']
original_vals = [original_variance['math'], original_variance['reading'], original_variance['writing']]
optimized_vals = [optimized_variance['math'], optimized_variance['reading'], optimized_variance['writing']]

x = np.arange(len(labels))  # label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, original_vals, width, label='Original Variance')
rects2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized Variance')

# Annotating variance values on top of the bars
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
print("Is the variance minimization problem convex?", is_convex)

# 2. Modify Scores (Introduce Non-Convexity)
np.random.seed(42)
scores['math_non_convex'] = scores['math score'] + np.random.randint(-30, 30, size=scores.shape[0])
scores['reading_non_convex'] = scores['reading score'] + np.random.randint(-30, 30, size=scores.shape[0])
scores['writing_non_convex'] = scores['writing score'] + np.random.randint(-30, 30, size=scores.shape[0])

# Clip scores to valid range [0, 100]
scores[['math_non_convex', 'reading_non_convex', 'writing_non_convex']] = scores[[
    'math_non_convex', 'reading_non_convex', 'writing_non_convex']].clip(0, 100)

# 3. Restore Convexity
restored_scores = cp.Variable(scores_array.shape)
objective_restore = cp.Minimize(cp.sum_squares(restored_scores - cp.mean(restored_scores, axis=0)))
constraints_restore = [restored_scores >= 0, restored_scores <= 100]
problem_restore = cp.Problem(objective_restore, constraints_restore)
problem_restore.solve()
restored_scores_value = restored_scores.value

# Add results to DataFrame
scores['math_restored'] = restored_scores_value[:, 0]
scores['reading_restored'] = restored_scores_value[:, 1]
scores['writing_restored'] = restored_scores_value[:, 2]

# 4. Visualize Convexity Check
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

# 5. Visualize Original, Non-Convex, and Restored Scores
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
