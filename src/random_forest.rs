use crate::decision_tree::DecisionTree;
use crate::parse::{Diagnosis, Sample};
use rand::seq::SliceRandom;

pub struct RandomForest {
    num_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_gain: f64,
    trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn new(
        num_trees: usize,
        max_depth: usize,
        min_samples_split: usize,
        min_gain: f64,
    ) -> Self {
        Self {
            num_trees,
            max_depth,
            min_samples_split,
            min_gain,
            trees: Vec::new(),
        }
    }

    pub fn train(&mut self, samples: &[Sample]) {
        let mut rng = rand::thread_rng();

        for _ in 0..self.num_trees {
            let bootstrap_samples: Vec<Sample> = (0..samples.len())
                .map(|_| *samples.choose(&mut rng).unwrap())
                .collect();

            let mut tree = DecisionTree::new(self.max_depth, self.min_samples_split, self.min_gain);
            tree.fit(&bootstrap_samples);

            self.trees.push(tree);
        }
    }

    pub fn predict(&self, sample: &Sample) -> Diagnosis {
        let mut benign_vote_count = 0;
        let mut malignant_vote_count = 0;

        for tree in &self.trees {
            match tree.predict(sample) {
                Diagnosis::Benign => benign_vote_count += 1,
                Diagnosis::Malignant => malignant_vote_count += 1,
            }
        }

        if benign_vote_count > malignant_vote_count {
            Diagnosis::Benign
        } else {
            Diagnosis::Malignant
        }
    }

    pub fn calculate_accuracy(&self, samples: &[Sample]) -> f64 {
        let correct = samples
            .iter()
            .filter(|sample| self.predict(sample) == sample.label)
            .count();
        correct as f64 / samples.len() as f64 * 100.0
    }
}
