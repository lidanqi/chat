#include <iostream>

#include <vector>

#include <cmath>

using namespace std;

// Define the TreeNode structure
struct TreeNode {
  int feature;
  double threshold;
  int left_child;
  int right_child;
  double output;
};

// Define the gini_impurity function
double gini_impurity(vector < double > & labels) {
  int n_samples = labels.size();
  if (n_samples == 0) {
    return 0.0;
  }
  int count1 = 0, count2 = 0;
  for (int i = 0; i < n_samples; i++) {
    if (labels[i] > 0) {
      count1++;
    } else {
      count2++;
    }
  }
  double p1 = (double) count1 / n_samples;
  double p2 = (double) count2 / n_samples;
  return 1.0 - p1 * p1 - p2 * p2;
}

// Define the split_data function
pair < vector < int > , vector < int >> split_data(vector < vector < double >> & data, vector < double > & labels, int feature, double threshold) {
  vector < int > left_indices, right_indices;
  int n_samples = data.size();
  for (int i = 0; i < n_samples; i++) {
    if (data[i][feature] <= threshold) {
      left_indices.push_back(i);
    } else {
      right_indices.push_back(i);
    }
  }
  vector < int > left_labels(left_indices.size()), right_labels(right_indices.size());
  for (int i = 0; i < left_indices.size(); i++) {
    left_labels[i] = labels[left_indices[i]];
  }
  for (int i = 0; i < right_indices.size(); i++) {
    right_labels[i] = labels[right_indices[i]];
  }
  return make_pair(left_labels, right_labels);
}

// Define the find_best_split function
pair < int, double > find_best_split(vector < vector < double >> & data, vector < double > & labels) {
  int n_features = data[0].size();
  int n_samples = data.size();
  double best_impurity = 1.0;
  int best_feature = 0;
  double best_threshold = 0.0;
  for (int i = 0; i < n_features; i++) {
    vector < double > feature_values(n_samples);
    for (int j = 0; j < n_samples; j++) {
      feature_values[j] = data[j][i];
    }
    sort(feature_values.begin(), feature_values.end());
    for (int j = 0; j < n_samples - 1; j++) {
      double threshold = (feature_values[j] + feature_values[j + 1]) / 2.0;
      auto[left_labels, right_labels] = split_data(data, labels, i, threshold);
      double impurity = (double) left_labels.size() / n_samples * gini_impurity(left_labels) +
        (double) right_labels.size() / n_samples * gini_impurity(right_labels);
      if (impurity < best_impurity) {
        best_impurity = impurity;
        best_feature = i;
        best_threshold = threshold;
      }
    }
  }
  return make_pair(best_feature, best_threshold);
}

// Define the build_tree function
int build_tree(vector < vector < double >> & data, vector < double > & labels, vector < TreeNode > & nodes) {
  int n_samples = data.size();
  int n_features = data[0].size();
  double output = 0.0;
  for (int i = 0; i < n_samples; i++) {
    output += labels[i];
  }
  output /= n_samples;
  if (output == 0.0) {
    nodes.push_back({
      -1,
      -1.0,
      -1,
      -1,
      -1.0
    });
    return nodes.size() - 1;
  }
  if (output == 1.0) {
    nodes.push_back({
      -1,
      -1.0,
      -1,
      -1,
      1.0
    });
    return nodes.size() - 1;
  }
  if (n_samples == 0) {
    nodes.push_back({
      -1,
      -1.0,
      -1,
      -1,
      output
    });
    return nodes.size() - 1;
  }
  auto[best_feature, best_threshold] = find_best_split(data, labels);
  auto[left_labels, right_labels] = split_data(data, labels, best_feature, best_threshold);
  int left_child = build_tree(data, left_labels, nodes);
  int right_child = build_tree(data, right_labels, nodes);
  nodes.push_back({
    best_feature,
    best_threshold,
    left_child,
    right_child,
    -1.0
  });
  return nodes.size() - 1;
}

int main() {
  // Load the data
  int n_samples = 100;
  int n_features = 10;
  vector < vector < double >> data(n_samples, vector < double > (n_features));
  vector < double > labels(n_samples);
  for (int i = 0; i < n_samples; i++) {
    for (int j = 0; j < n_features; j++) {
      data[i][j] = (double) rand() / RAND_MAX;
    }
    labels[i] = (double) rand() / RAND_MAX;
  }

  // Build the decision tree
  vector < TreeNode > nodes;
  int root = build_tree(data, labels, nodes);

  // Make predictions
  vector < double > predictions(n_samples);
  for (int i = 0; i < n_samples; i++) {
    int index = root;
    while (nodes[index].output == -1.0) {
      if (data[i][nodes[index].feature] <= nodes[index].threshold) {
        index = nodes[index].left_child;
      } else {
        index = nodes[index].right_child;
      }
    }
    predictions[i] = nodes[index].output;
  }

  // Evaluate the model
  double mse = 0.0;
  for (int i = 0; i < n_samples; i++) {
    mse += pow(predictions[i] - labels[i], 2);
  }
  mse /= n_samples;
  cout << "Mean squared error: " << mse << endl;

  return 0;
}