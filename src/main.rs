use std::{error::Error, path::Path};

use dt::{
    decision_tree::DecisionTree,
    parse::{csv_entries_to_samples, Sample},
    random_forest::RandomForest,
};
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{Color, BLUE, RED, WHITE},
};
use smartcore::{
    linalg::basic::matrix::DenseMatrix,
    tree::decision_tree_classifier::{
        DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion,
    },
};

fn split(samples: &[Sample], train_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (samples.len() as f64 * train_ratio) as usize;

    let (first, second) = samples.split_at(train_size);
    (first.to_vec(), second.to_vec())
}

fn plot_curve(
    values: &[(i32, f64)],
    title: &str,
    label: &str,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(Path::new(filename), (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_value = values.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);
    let max_value = values
        .iter()
        .map(|&(_, y)| y)
        .fold(f64::NEG_INFINITY, f64::max);
    let margin = 0.1 * (max_value - min_value);

    #[allow(clippy::range_plus_one)]
    let x_range = values[0].0..values.last().unwrap().0 + 1;
    let y_range = (min_value - margin)..(max_value + margin);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(values.iter().copied(), &RED))?
        .label(label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .draw()?;

    println!("plot saved to {filename}");

    Ok(())
}

fn plot_train_test_accuracy(
    train_values: &[(i32, f64)],
    test_values: &[(i32, f64)],
    title: &str,
    train_label: &str,
    test_label: &str,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(Path::new(filename), (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_train_value = train_values
        .iter()
        .map(|&(_, y)| y)
        .fold(f64::INFINITY, f64::min);
    let max_train_value = train_values
        .iter()
        .map(|&(_, y)| y)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_test_value = test_values
        .iter()
        .map(|&(_, y)| y)
        .fold(f64::INFINITY, f64::min);
    let max_test_value = test_values
        .iter()
        .map(|&(_, y)| y)
        .fold(f64::NEG_INFINITY, f64::max);

    let min_value = min_train_value.min(min_test_value);
    let max_value = max_train_value.max(max_test_value);
    let margin = 0.1 * (max_value - min_value);

    #[allow(clippy::range_plus_one)]
    let x_range = train_values[0].0..train_values.last().unwrap().0 + 1;
    let y_range = (min_value - margin)..(max_value + margin);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(train_values.iter().copied(), &RED))?
        .label(train_label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

    chart
        .draw_series(LineSeries::new(test_values.iter().copied(), &BLUE))?
        .label(test_label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .draw()?;

    println!("plot saved to {filename}");

    Ok(())
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

    explore_my_decision_tree(&train_samples);
    explore_smartcore_decision_tree(&train_samples);

    explore_my_decision_tree_accuracy(&train_samples, &test_samples);
    explore_smartcore_decision_tree_accuracy(&train_samples, &test_samples);

    explore_my_random_forest_accuracy(&train_samples, &test_samples);
    explore_smartcore_random_forest_accuracy(&train_samples, &test_samples);

    explore_gbdt_accuracy(&train_samples, &test_samples);
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
    const TREE_COUNT: usize = 100;
    const MAX_DEPTH: usize = 100;
    const MIN_SAMPLES_SPLIT: usize = 5;
    const MIN_GAIN: f64 = 0.01;

    let mut forest = RandomForest::new(TREE_COUNT, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_GAIN);
    forest.train(train_samples);

    let forest_accuracy = forest.calculate_accuracy(test_samples);
    println!("Random forest accuracy: {forest_accuracy:.3}%");
}

const MIN_SAMPLES_SPLITS: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
];

fn explore_my_decision_tree(train_samples: &[Sample]) {
    const MAX_DEPTH: usize = 1000;
    const MIN_GAIN: f64 = 0.00001;

    let mut height_data = Vec::new();

    for &min_samples_split in &MIN_SAMPLES_SPLITS {
        let mut tree = DecisionTree::new(MAX_DEPTH, min_samples_split, MIN_GAIN);
        tree.fit(train_samples);

        let tree_height = tree.get_height();
        height_data.push((
            i32::try_from(min_samples_split).unwrap(),
            tree_height as f64,
        ));
    }

    plot_curve(
        &height_data,
        "Tree height vs min_samples_split (my decision tree)",
        "Tree height",
        "tree_height_vs_min_samples_split_my.png",
    )
    .unwrap();
}

fn explore_smartcore_decision_tree(train_samples: &[Sample]) {
    let x = DenseMatrix::from_2d_array(
        &train_samples
            .iter()
            .map(|sample| sample.features.as_slice())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let y = train_samples
        .iter()
        .map(|sample| sample.label as u32)
        .collect::<Vec<u32>>();

    let mut height_data = Vec::new();

    for &min_samples_split in &MIN_SAMPLES_SPLITS {
        let params = DecisionTreeClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_leaf: 0,
            min_samples_split,
            seed: Some(1234),
        };

        let tree = DecisionTreeClassifier::fit(&x, &y, params).unwrap();

        let tree_height = tree.depth();

        height_data.push((
            i32::try_from(min_samples_split).unwrap(),
            tree_height as f64,
        ));
    }

    plot_curve(
        &height_data,
        "Tree height vs min_samples_split (smartcore decision tree)",
        "Tree height",
        "tree_height_vs_min_samples_split_smartcore.png",
    )
    .unwrap();
}

const MAX_DEPTH_VALUES: [usize; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

fn explore_my_decision_tree_accuracy(train_samples: &[Sample], test_samples: &[Sample]) {
    const MIN_SAMPLES_SPLIT: usize = 0;
    const MIN_GAIN: f64 = 0.000_000_000_1;

    let mut train_accuracy_data = Vec::new();
    let mut test_accuracy_data = Vec::new();

    for &max_depth in &MAX_DEPTH_VALUES {
        let mut tree = DecisionTree::new(max_depth, MIN_SAMPLES_SPLIT, MIN_GAIN);
        tree.fit(train_samples);

        let train_accuracy = tree.calculate_accuracy(train_samples);
        let test_accuracy = tree.calculate_accuracy(test_samples);

        train_accuracy_data.push((i32::try_from(max_depth).unwrap(), train_accuracy));
        test_accuracy_data.push((i32::try_from(max_depth).unwrap(), test_accuracy));
    }

    plot_train_test_accuracy(
        &train_accuracy_data,
        &test_accuracy_data,
        "Accuracy vs Tree depth (my decision tree)",
        "Train accuracy",
        "Test accuracy",
        "decision_tree_accuracy_vs_depth_my.png",
    )
    .unwrap();
}

fn explore_smartcore_decision_tree_accuracy(train_samples: &[Sample], test_samples: &[Sample]) {
    let x = DenseMatrix::from_2d_array(
        &train_samples
            .iter()
            .map(|sample| sample.features.as_slice())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let y = train_samples
        .iter()
        .map(|sample| sample.label as u32)
        .collect::<Vec<u32>>();

    let mut train_accuracy_data = Vec::new();
    let mut test_accuracy_data = Vec::new();

    for &max_depth in &MAX_DEPTH_VALUES {
        let params = DecisionTreeClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Some(u16::try_from(max_depth).unwrap()),
            min_samples_leaf: 0,
            min_samples_split: 10,
            seed: Some(1234),
        };

        let tree = DecisionTreeClassifier::fit(&x, &y, params).unwrap();

        let train_predictions = tree.predict(&x).unwrap();
        let train_accuracy = calculate_accuracy_from_predictions(&train_predictions, train_samples);

        let test_x = DenseMatrix::from_2d_array(
            &test_samples
                .iter()
                .map(|sample| sample.features.as_slice())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let test_predictions = tree.predict(&test_x).unwrap();
        let test_accuracy = calculate_accuracy_from_predictions(&test_predictions, test_samples);

        train_accuracy_data.push((i32::try_from(max_depth).unwrap(), train_accuracy));
        test_accuracy_data.push((i32::try_from(max_depth).unwrap(), test_accuracy));
    }

    plot_train_test_accuracy(
        &train_accuracy_data,
        &test_accuracy_data,
        "Accuracy vs Tree depth (smartcore decision tree)",
        "Train accuracy",
        "Test accuracy",
        "accuracy_vs_depth_smartcore.png",
    )
    .unwrap();
}

fn calculate_accuracy_from_predictions(predictions: &[u32], true_labels: &[Sample]) -> f64 {
    let correct = predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(pred, sample)| **pred == sample.label as u32)
        .count();
    correct as f64 / predictions.len() as f64 * 100.0
}

const TREE_COUNTS: [usize; 14] = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150];

fn explore_my_random_forest_accuracy(train_samples: &[Sample], test_samples: &[Sample]) {
    const MAX_DEPTH: usize = 10;
    const MIN_SAMPLES_SPLIT: usize = 5;
    const MIN_GAIN: f64 = 0.01;

    let mut train_accuracy_data = Vec::new();
    let mut test_accuracy_data = Vec::new();

    for &num_trees in &TREE_COUNTS {
        let mut forest = RandomForest::new(num_trees, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_GAIN);
        forest.train(train_samples);

        let train_accuracy = forest.calculate_accuracy(train_samples);
        let test_accuracy = forest.calculate_accuracy(test_samples);

        train_accuracy_data.push((i32::try_from(num_trees).unwrap(), train_accuracy));
        test_accuracy_data.push((i32::try_from(num_trees).unwrap(), test_accuracy));
    }

    plot_train_test_accuracy(
        &train_accuracy_data,
        &test_accuracy_data,
        "Accuracy vs Number of trees (my random forest)",
        "Train Accuracy",
        "Test Accuracy",
        "random_forest_accuracy_vs_tree_count_my.png",
    )
    .unwrap();
}

fn explore_smartcore_random_forest_accuracy(train_samples: &[Sample], test_samples: &[Sample]) {
    let x = DenseMatrix::from_2d_array(
        &train_samples
            .iter()
            .map(|sample| sample.features.as_slice())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let y = train_samples
        .iter()
        .map(|sample| sample.label as u32)
        .collect::<Vec<u32>>();

    let mut train_accuracy_data = Vec::new();
    let mut test_accuracy_data = Vec::new();

    for &tree_count in &TREE_COUNTS {
        let params =
            smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters {
                max_depth: Some(10),
                min_samples_split: 5,
                min_samples_leaf: 2,
                n_trees: u16::try_from(tree_count).unwrap(),
                ..Default::default()
            };

        let forest = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            &x, &y, params,
        )
        .unwrap();

        let train_predictions = forest.predict(&x).unwrap();
        let train_accuracy = calculate_accuracy_from_predictions(&train_predictions, train_samples);

        let test_x = DenseMatrix::from_2d_array(
            &test_samples
                .iter()
                .map(|sample| sample.features.as_slice())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let test_predictions = forest.predict(&test_x).unwrap();
        let test_accuracy = calculate_accuracy_from_predictions(&test_predictions, test_samples);

        train_accuracy_data.push((i32::try_from(tree_count).unwrap(), train_accuracy));
        test_accuracy_data.push((i32::try_from(tree_count).unwrap(), test_accuracy));
    }

    plot_train_test_accuracy(
        &train_accuracy_data,
        &test_accuracy_data,
        "Accuracy vs Number of trees (smartcore random forest)",
        "Train Accuracy",
        "Test Accuracy",
        "random_forest_accuracy_vs_tree_count_smartcore.png",
    )
    .unwrap();
}

#[allow(clippy::cast_possible_truncation)]
fn explore_gbdt_accuracy(train_samples: &[Sample], test_samples: &[Sample]) {
    use gbdt::config::Config;
    use gbdt::decision_tree::{Data, DataVec};
    use gbdt::gradient_boost::GBDT;

    let mut train_accuracy_data = Vec::new();
    let mut test_accuracy_data = Vec::new();

    for &num_trees in &TREE_COUNTS {
        let mut config = Config::new();
        config.set_feature_size(train_samples[0].features.len());
        config.set_iterations(num_trees);

        let mut gbdt = GBDT::new(&config);

        let mut training_data: DataVec = train_samples
            .iter()
            .map(|sample| {
                Data::new_training_data(
                    sample.features.map(|feature| feature as f32).to_vec(),
                    1.0,
                    sample.label as i32 as f32,
                    None,
                )
            })
            .collect();

        gbdt.fit(&mut training_data);

        let train_data: DataVec = train_samples
            .iter()
            .map(|sample| {
                Data::new_test_data(sample.features.map(|feature| feature as f32).to_vec(), None)
            })
            .collect();

        let test_data: DataVec = test_samples
            .iter()
            .map(|sample| {
                Data::new_test_data(sample.features.map(|feature| feature as f32).to_vec(), None)
            })
            .collect();

        let train_predictions = gbdt.predict(&train_data);
        let test_predictions = gbdt.predict(&test_data);

        let train_accuracy =
            calculate_accuracy_from_gbdt_predictions(&train_predictions, train_samples);
        let test_accuracy =
            calculate_accuracy_from_gbdt_predictions(&test_predictions, test_samples);

        train_accuracy_data.push((i32::try_from(num_trees).unwrap(), train_accuracy));
        test_accuracy_data.push((i32::try_from(num_trees).unwrap(), test_accuracy));
    }

    plot_train_test_accuracy(
        &train_accuracy_data,
        &test_accuracy_data,
        "Accuracy vs Number of trees (GBDT)",
        "Train accuracy",
        "Test accuracy",
        "accuracy_vs_tree_count_gbdt.png",
    )
    .unwrap();
}

#[allow(clippy::cast_possible_truncation)]
fn calculate_accuracy_from_gbdt_predictions(predictions: &[f32], true_labels: &[Sample]) -> f64 {
    let correct = predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(prediction, sample)| prediction.round() as i32 == sample.label as i32)
        .count();
    correct as f64 / predictions.len() as f64 * 100.0
}
