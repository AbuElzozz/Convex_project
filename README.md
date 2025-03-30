# Convex_project
# Convex Optimization for Score Balancing

This project demonstrates the application of convex optimization to balance student scores across different subjects while maintaining fairness and validity constraints.

## Project Overview

The goal is to adjust student scores in math, reading, and writing to:
1. Reduce variance between subjects
2. Maintain scores within valid ranges (0-100)
3. Preserve the original score distribution as much as possible

## Key Features

- **Convex Optimization**: Uses CVXPY to formulate and solve the score balancing problem
- **Variance Reduction**: Minimizes differences in variance across subjects
- **Outlier Handling**: Demonstrates how to restore convexity when outliers are present
- **Visualizations**: Includes multiple plots to compare original and optimized scores

## Methodology

1. Load student performance data
2. Calculate original score variances
3. Formulate convex optimization problem:
   - Objective: Minimize variance + penalty for deviation from original scores
   - Constraints: Scores must be between 0 and 100
4. Solve optimization problem
5. Analyze results and visualize comparisons

## Results

The optimization successfully:
- Reduces variance between subjects while keeping scores valid
- Handles outliers through convex restoration
- Maintains the overall score distribution characteristics

![](D:\University\Level 3\First Semester\Convex\Convex_project\Presentation\Figure_1.png)
![](D:\University\Level 3\First Semester\Convex\Convex_project\Presentation\Figure_2.png)
![](D:\University\Level 3\First Semester\Convex\Convex_project\Presentation\Figure_3.png)
![](D:\University\Level 3\First Semester\Convex\Convex_project\Presentation\Figure_4.png)
![](D:\University\Level 3\First Semester\Convex\Convex_project\Presentation\Figure_5.png)
![](D:\University\Level 3\First Semester\Convex\Convex_project\Presentation\Figure_6.png)


## Visualization Examples

1. Variance comparison before and after optimization
2. Convexity check through subset variance analysis
3. Score distribution histograms (original vs. optimized)
4. Convex hull visualizations of score distributions

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - cvxpy
  - scipy

## Usage

1. Clone the repository
2. Install required packages
3. Run `Convex Project.py`
4. View generated visualizations

## Dataset

Uses the "StudentsPerformance.csv" dataset containing student scores in three subjects.

## License

This project is open-source and available under the MIT License.