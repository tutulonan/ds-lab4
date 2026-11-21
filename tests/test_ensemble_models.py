"""Тесты контрактной функции, метрик и отсутствия утечки данных."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult
from ensemble_models import SELECTED_MODEL, _make_pipeline, _metrics, run_ensemble_experiment


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-3, 3, 80).reshape(-1, 1)
        self.y = (self.x[:, 0] > 0).astype(int)
        self.data = EnsembleInput(self.x, self.y, np.array([[-2.], [-1.], [1.], [2.]]), [0, 0, 1, 1])

    def test_contract_and_result_shapes(self):
        result = run_ensemble_experiment(self.data)
        self.assertIsInstance(result, EnsembleResult)
        self.assertIsInstance(result.model_name, str)
        self.assertTrue(result.model_name)
        self.assertEqual(result.probabilities.shape, (4,))
        self.assertEqual(result.predictions.shape, (4,))
        self.assertTrue(np.isfinite(result.probabilities).all())
        self.assertTrue(np.all((result.probabilities >= 0) & (result.probabilities <= 1)))
        self.assertTrue(np.isin(result.predictions, [0, 1]).all())

    def test_learns_simple_signal(self):
        result = run_ensemble_experiment(self.data)
        assert_array_equal(result.predictions, [0, 0, 1, 1])
        self.assertEqual(result.roc_auc, 1)
        self.assertEqual(result.accuracy, 1)

    def test_metrics_match_independent_calls(self):
        data = replace(self.data, test_target=[0, 1, 0, 1])
        result = run_ensemble_experiment(data)
        self.assertEqual(result.roc_auc, roc_auc_score(data.test_target, result.probabilities))
        self.assertEqual(result.accuracy, accuracy_score(data.test_target, result.predictions))
        self.assertEqual(result.precision, precision_score(data.test_target, result.predictions, zero_division=0))
        self.assertEqual(result.recall, recall_score(data.test_target, result.predictions, zero_division=0))
        self.assertEqual(result.f1, f1_score(data.test_target, result.predictions, zero_division=0))

    def test_auc_uses_probabilities(self):
        values = _metrics([0, 0, 1, 1], [.1, .4, .35, .8], [0, 0, 0, 1])
        self.assertEqual(values['roc_auc'], .75)
        self.assertEqual(values['accuracy'], .75)
        self.assertEqual(values['precision'], 1)
        self.assertEqual(values['recall'], .5)
        self.assertAlmostEqual(values['f1'], 2/3)

    def test_threshold(self):
        result = run_ensemble_experiment(self.data)
        assert_array_equal(result.predictions, (result.probabilities >= .5).astype(int))

    def test_test_target_does_not_affect_model(self):
        a = run_ensemble_experiment(self.data)
        b = run_ensemble_experiment(replace(self.data, test_target=[1, 1, 0, 0]))
        assert_array_equal(a.probabilities, b.probabilities)
        assert_array_equal(a.predictions, b.predictions)
        self.assertNotEqual(a.accuracy, b.accuracy)

    def test_repeated_seed_is_reproducible(self):
        a, b = run_ensemble_experiment(self.data), run_ensemble_experiment(self.data)
        assert_array_equal(a.probabilities, b.probabilities)
        self.assertEqual(a.roc_auc, b.roc_auc)

    def test_missing_numeric_values(self):
        x = self.x.copy()
        x[0, 0] = np.nan
        result = run_ensemble_experiment(replace(self.data, train_features=x, test_features=[[np.nan], [2]], test_target=[0, 1]))
        self.assertTrue(np.isfinite(result.probabilities).all())

    def test_preprocessing_fits_only_training_features(self):
        x = self.x.copy()
        x[0, 0] = np.nan
        pipeline = _make_pipeline(x, SELECTED_MODEL, 42)
        with patch('ensemble_models._make_pipeline', return_value=pipeline):
            run_ensemble_experiment(replace(self.data, train_features=x, test_features=[[1e6], [2e6]], test_target=[0, 1]))
        assert_allclose(pipeline.named_steps['preprocess'].statistics_, np.nanmedian(x, axis=0))

    def test_categorical_dataframe_and_unseen_category(self):
        train = pd.DataFrame({'number': self.x[:, 0], 'category': ['a']*40 + ['b']*40})
        test = pd.DataFrame({'number': [-1, 1], 'category': ['new', 'b']})
        result = run_ensemble_experiment(EnsembleInput(train, self.y, test, [0, 1]))
        self.assertEqual(result.probabilities.shape, (2,))
        self.assertTrue(np.isfinite(result.probabilities).all())

    def test_inputs_unchanged_and_column_target(self):
        x, y = self.x.copy(), self.y.reshape(-1, 1).copy()
        before_x, before_y = x.copy(), y.copy()
        run_ensemble_experiment(replace(self.data, train_features=x, train_target=y))
        assert_array_equal(x, before_x)
        assert_array_equal(y, before_y)

    def test_single_test_class_has_undefined_auc(self):
        result = run_ensemble_experiment(replace(self.data, test_target=[1, 1, 1, 1]))
        self.assertTrue(np.isnan(result.roc_auc))
        self.assertTrue(np.isfinite(result.accuracy))

    def test_zero_positive_predictions(self):
        values = _metrics([0, 1], [.1, .2], [0, 0])
        self.assertEqual(values['precision'], 0)
        self.assertEqual(values['recall'], 0)
        self.assertEqual(values['f1'], 0)

    def test_invalid_target_labels(self):
        for y in [np.full(80, 2), np.full(80, .5), np.zeros(80), [0, 1]]:
            with self.subTest(target=str(y[:3])), self.assertRaises(ValueError):
                run_ensemble_experiment(replace(self.data, train_target=y))

    def test_invalid_shapes(self):
        cases = [replace(self.data, test_features=[[1, 2]]),
                 replace(self.data, train_features=[1, 2, 3]),
                 replace(self.data, test_features=np.empty((0, 1)), test_target=[]),
                 replace(self.data, test_target=[0])]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                run_ensemble_experiment(data)
