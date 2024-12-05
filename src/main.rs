use dt::{
    decision_tree::DecisionTree,
    parse::{csv_entries_to_samples, Sample},
    random_forest::RandomForest,
};

fn split(samples: &[Sample], train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (samples.len() as f64 * train_ratio) as usize;

    let (first, second) = samples.split_at(train_size);
    (first.to_vec(), second.to_vec())
}

fn main() {
    const DATA_FILEPATH: &str = "data/breast-cancer.csv";

    let entries = dt::parse::parse(DATA_FILEPATH).unwrap();
    assert!(!entries.is_empty());

    let samples = csv_entries_to_samples(entries);

    const TRAIN_RATIO: f64 = 0.6;
    let (train_samples, test_samples) = split(&samples, TRAIN_RATIO);
    assert!(!train_samples.is_empty());
    assert!(!test_samples.is_empty());

    run_decision_tree(&train_samples, &test_samples);
    run_random_forest(&train_samples, &test_samples);
}

fn run_decision_tree(train_samples: &[Sample], test_samples: &[Sample]) {
    const MAX_DEPTH: usize = 3;
    const MIN_SAMPLES_SPLIT: usize = 10;
    const MIN_GAIN: f64 = 0.1;

    let mut tree = DecisionTree::new(MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_GAIN);
    tree.fit(train_samples);

    let tree_accuracy = tree.calculate_accuracy(test_samples);
    println!("Decision tree accuracy: {tree_accuracy:.3}%");
}

fn run_random_forest(train_samples: &[Sample], test_samples: &[Sample]) {
    const NUM_TREES: usize = 100;
    const MAX_DEPTH: usize = 10;
    const MIN_SAMPLES_SPLIT: usize = 5;
    const MIN_GAIN: f64 = 0.01;

    let mut forest = RandomForest::new(NUM_TREES, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_GAIN);
    forest.train(train_samples);

    let forest_accuracy = forest.calculate_accuracy(test_samples);
    println!("Random forest accuracy: {forest_accuracy:.3}%");
}
