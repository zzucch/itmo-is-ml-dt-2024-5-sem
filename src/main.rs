use dt::parse::{csv_entries_to_samples, Sample};

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
}
