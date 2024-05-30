import numpy as np
from scipy.stats import chi2

# Define the measurement and track parameters
state_dim = 3  # 3D state (e.g., x, y, z)

# Predefined tracks and reports in 3D with time
tracks = np.array([
    [6, 6, 10, 0],
    [15, 15, 10, 10],
    [7, 7, 10, 20],
    [8, 8, 10, 30],
    [9, 9, 10, 40],
    [10, 10, 10, 45],
    [11, 11, 10, 50],
    [12, 12, 10, 0],
    [13, 13, 10, 10],
    [14, 14, 10, 20]
])

reports = np.array([
    [7, 7, 10, 0],
    [16, 16, 10, 10],
    [7, 7, 10, 30],
    [80, 80, 80, 30],
    [90, 90, 90, 40],
    [100, 100, 100, 50],
    [7, 7, 10, 45],
    [120, 120, 10, 0],
    [130, 130, 10, 10],
    [140, 140, 100, 20]
])

# Chi-squared gating threshold for 95% confidence interval
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

# Covariance matrix of the measurement errors (assumed to be identity for simplicity)
cov_matrix = np.eye(state_dim)
cov_inv = np.linalg.inv(cov_matrix)

print("Covariance Matrix:\n", cov_matrix)
print("Chi-squared Threshold:", chi2_threshold)

# Perform residual error check using Chi-squared gating
association_list = []
for i, track in enumerate(tracks):
    for j, report in enumerate(reports):
        distance = mahalanobis_distance(track, report, cov_inv)
        time_diff = abs(track[3] - report[3])
        if distance < chi2_threshold and time_diff <= 50:
            association_list.append((i, j))
            print(f"Track {i} associated with Report {j}, Mahalanobis distance: {distance:.4f}, Time difference: {time_diff}")

# Clustering reports and tracks based on associations and time
clusters = []
while association_list:
    cluster_tracks = set()
    cluster_reports = set()
    stack = [association_list.pop(0)]
    while stack:
        track_idx, report_idx = stack.pop()
        cluster_tracks.add(track_idx)
        cluster_reports.add(report_idx)
        new_assoc = [(t, r) for t, r in association_list if t in cluster_tracks or r in cluster_reports]
        for assoc in new_assoc:
            if assoc not in stack:
                stack.append(assoc)
        association_list = [assoc for assoc in association_list if assoc not in new_assoc]
    
    # Check if the cluster satisfies the time constraint
    cluster_times = set([tracks[t][3] for t in cluster_tracks] + [reports[r][3] for r in cluster_reports])
    if len(cluster_times) <= 50:
        clusters.append((list(cluster_tracks), list(cluster_reports)))

print("Clusters:", clusters)

# Define a function to generate hypotheses for each cluster
def generate_hypotheses(tracks, reports):
    num_tracks = len(tracks)
    num_reports = len(reports)
    base = num_reports + 1
    
    hypotheses = []
    for count in range(base**num_tracks):
        hypothesis = []
        for track_idx in range(num_tracks):
            report_idx = (count // (base**track_idx)) % base
            hypothesis.append((track_idx, report_idx - 1))
        
        # Check if the hypothesis is valid (each report and track is associated with at most one entity)
        if is_valid_hypothesis(hypothesis):
            hypotheses.append(hypothesis)
    
    return hypotheses

def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

# Define a function to calculate probabilities for each hypothesis
def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance**2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

# Define a function to get association weights
def get_association_weights(hypotheses, probabilities):
    num_tracks = len(hypotheses[0])
    association_weights = [[] for _ in range(num_tracks)]
    
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                association_weights[track_idx].append((report_idx, prob))
    
    for track_weights in association_weights:
        track_weights.sort(key=lambda x: x[0])  # Sort by report index
        report_probs = {}
        for report_idx, prob in track_weights:
            if report_idx not in report_probs:
                report_probs[report_idx] = prob
            else:
                report_probs[report_idx] += prob
        track_weights[:] = [(report_idx, prob) for report_idx, prob in report_probs.items()]
    
    return association_weights

# Find the most likely association for each report
def find_max_associations(hypotheses, probabilities):
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs

# Process each cluster and generate hypotheses
for track_idxs, report_idxs in clusters:
    print("Cluster Tracks:", track_idxs)
    print("Cluster Reports:", report_idxs)
    
    cluster_tracks = tracks[track_idxs]
    cluster_reports = reports[report_idxs]
    hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
    probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, cov_inv)
    association_weights = get_association_weights(hypotheses, probabilities)
    max_associations, max_probs = find_max_associations(hypotheses, probabilities)
    
    print("Hypotheses:")
    print("Tracks/Reports:", ["t" + str(i+1) for i in track_idxs])
    for hypothesis, prob  in zip(hypotheses, probabilities):
        formatted_hypothesis = ["r" + str(report_idxs[r]+1) if r != -1 else "0" for _, r in hypothesis]
        print(f"Hypothesis: {formatted_hypothesis}, Probability: {prob:.4f}")
    
    for track_idx, weights in enumerate(association_weights):
        for report_idx, weight in weights:
            print(f"Track t{track_idxs[track_idx]+1}, Report r{report_idxs[report_idx]+1}: {weight:.4f}")
    
    for report_idx, association in enumerate(max_associations):
        if association != -1:
            print(f"Most likely association for Report r{report_idxs[report_idx]+1}: Track t{track_idxs[association]+1}, Probability: {max_probs[report_idx]:.4f}")
    
    # Print which tracks have which reports in the group with time less than 50 milliseconds
    for track_idx, report_idx in zip(track_idxs, report_idxs):
        if abs(tracks[track_idx][3] - reports[report_idx][3]) <= 50:
            print(f"Track t{track_idx+1} has Report r{report_idx+1} in the group with time less than 50 milliseconds.")