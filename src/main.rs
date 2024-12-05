use dt::{
    decision_tree::DecisionTree,
    parse::{csv_entries_to_samples, Sample},
};

fn split(samples: &[Sample], train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (samples.len() as f64 * train_ratio) as usize;

    let (first, second) = samples.split_at(train_size);
    (first.to_vec(), second.to_vec())
}

fn calculate_accuracy(tree: &DecisionTree, test_samples: &[Sample]) -> f64 {
    let correct = test_samples
        .iter()
        .filter(|sample| tree.predict(sample) == sample.label)
        .count();
    correct as f64 / test_samples.len() as f64 * 100.0
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

    const MAX_DEPTH: usize = 3;
    const MIN_SAMPLES_SPLIT: usize = 10;
    const MIN_GAIN: f64 = 0.1;

    let mut tree = DecisionTree::new(MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_GAIN);
    tree.fit(&train_samples);

    let accuracy = calculate_accuracy(&tree, &test_samples);
    println!("Decision tree accuracy: {accuracy:.3}%");
}
