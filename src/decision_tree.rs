use crate::parse::{Diagnosis, Sample, DIMENSIONS};

pub struct DecisionTree {
    max_depth: usize,
    min_samples_split: usize,
    min_gain: f64,
    root: Option<Node>,
}

#[derive(Debug)]
pub enum Node {
    Leaf(Diagnosis),
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize, min_gain: f64) -> Self {
        Self {
            max_depth,
            min_samples_split,
            min_gain,
            root: None,
        }
    }

    pub fn fit(&mut self, samples: &[Sample]) {
        self.root = Some(self.build_tree(samples, 0));
    }

    pub fn predict(&self, sample: &Sample) -> Diagnosis {
        match &self.root {
            Some(node) => Self::traverse_tree(node, sample),
            None => panic!("cannot predict without training"),
        }
    }

    fn build_tree(&self, samples: &[Sample], depth: usize) -> Node {
        if depth >= self.max_depth || samples.len() < self.min_samples_split {
            return Node::Leaf(Self::get_majority_class(samples));
        }

        let (split_feature, split_threshold, split_gain) = Self::get_best_split(samples);
        if split_gain < self.min_gain {
            return Node::Leaf(Self::get_majority_class(samples));
        }

        let (left_samples, right_samples): (Vec<_>, Vec<_>) = samples
            .iter()
            .partition(|sample| sample.features[split_feature] < split_threshold);

        Node::Split {
            feature_index: split_feature,
            threshold: split_threshold,
            left: Box::new(self.build_tree(&left_samples, depth + 1)),
            right: Box::new(self.build_tree(&right_samples, depth + 1)),
        }
    }

    fn get_best_split(samples: &[Sample]) -> (usize, f64, f64) {
        let mut best_feature_index = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        for feature_index in 0..DIMENSIONS {
            let thresholds: Vec<f64> = samples.iter().map(|s| s.features[feature_index]).collect();

            for &threshold in &thresholds {
                let (gain, _) = Self::get_information_gain(samples, feature_index, threshold);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature_index = feature_index;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature_index, best_threshold, best_gain)
    }

    fn get_information_gain(
        samples: &[Sample],
        feature_index: usize,
        threshold: f64,
    ) -> (f64, Vec<Sample>) {
        let (left_samples, right_samples): (Vec<_>, Vec<_>) = samples
            .iter()
            .partition(|sample| sample.features[feature_index] < threshold);

        let total_entropy = Self::get_entropy(samples);
        let left_entropy = Self::get_entropy(&left_samples);
        let right_entropy = Self::get_entropy(&right_samples);

        let gain = total_entropy
            - (left_samples.len() as f64 / samples.len() as f64) * left_entropy
            - (right_samples.len() as f64 / samples.len() as f64) * right_entropy;

        (gain, right_samples)
    }

    fn get_entropy(samples: &[Sample]) -> f64 {
        let total = samples.len() as f64;
        let benign_count = samples
            .iter()
            .filter(|sample| sample.label == Diagnosis::Benign)
            .count() as f64;
        let malignant_count = samples
            .iter()
            .filter(|sample| sample.label == Diagnosis::Malignant)
            .count() as f64;

        let benign_probability = benign_count / total;
        let malignant_probability = malignant_count / total;

        -benign_probability * benign_probability.ln()
            - malignant_probability * malignant_probability.ln()
    }

    fn get_majority_class(samples: &[Sample]) -> Diagnosis {
        let benign_count = samples
            .iter()
            .filter(|sample| sample.label == Diagnosis::Benign)
            .count();
        let malignant_count = samples
            .iter()
            .filter(|sample| sample.label == Diagnosis::Malignant)
            .count();

        if benign_count > malignant_count {
            Diagnosis::Benign
        } else {
            Diagnosis::Malignant
        }
    }

    fn traverse_tree(node: &Node, sample: &Sample) -> Diagnosis {
        match node {
            Node::Leaf(label) => *label,
            Node::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if sample.features[*feature_index] < *threshold {
                    Self::traverse_tree(left, sample)
                } else {
                    Self::traverse_tree(right, sample)
                }
            }
        }
    }
}
