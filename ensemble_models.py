"""Классификация German Credit ансамблевыми моделями."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


NUMERIC_COLUMNS = [
    "Duration_of_Credit_monthly", "Credit_Amount", "Instalment_per_cent",
    "Duration_in_Current_address", "Age_years", "No_of_Credits_at_this_Bank",
    "No_of_dependents",
]
SELECTED_MODEL = "random_forest_regularized"
MODEL_NAMES = {
    "random_forest": "Random Forest (300 trees, unrestricted depth)",
    "random_forest_regularized": "Random Forest (300 trees, depth=6, leaf=3)",
    "extra_trees": "Extra Trees (300 trees, leaf=2)",
    "gradient_boosting_depth2": "Gradient Boosting (150 trees, depth=2, learning_rate=0.05)",
    "gradient_boosting_depth3": "Gradient Boosting (150 trees, depth=3, learning_rate=0.05)",
    "hist_boosting_leaf7": "HistGradientBoosting (150 iterations, 7 leaves, L2=1)",
    "hist_boosting_leaf15": "HistGradientBoosting (150 iterations, 15 leaves, L2=1)",
}


def _make_estimator(name, seed):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if name == "random_forest_regularized":
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                      random_state=seed, n_jobs=1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name in ("gradient_boosting_depth2", "gradient_boosting_depth3"):
        return GradientBoostingClassifier(n_estimators=150, learning_rate=.05,
                                          max_depth=2 if name.endswith("2") else 3,
                                          random_state=seed)
    if name in ("hist_boosting_leaf7", "hist_boosting_leaf15"):
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=.1,
            max_leaf_nodes=7 if name.endswith("7") else 15, l2_regularization=1.,
            early_stopping=False, random_state=seed)
    raise ValueError(f"Неизвестная модель: {name}")


def _make_preprocessor(features, scale_numeric=False):
    numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if scale_numeric:
        from sklearn.preprocessing import StandardScaler
        numeric_imputer = Pipeline([("imputer", numeric_imputer), ("scaler", StandardScaler())])
    if not isinstance(features, pd.DataFrame):
        # Контракт также принимает уже подготовленные числовые массивы.
        return numeric_imputer
    if set(NUMERIC_COLUMNS).issubset(features.columns):
        numeric = list(NUMERIC_COLUMNS)
    else:
        numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [name for name in features.columns if name not in numeric]
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_imputer, numeric),
        ("categorical", category_pipeline, categorical),
    ], sparse_threshold=0)


def _make_pipeline(features, model_name, seed):
    return Pipeline([
        ("preprocess", _make_preprocessor(features)),
        ("model", _make_estimator(model_name, seed)),
    ])


def _binary_target(values, expected_length, name, require_both=False):
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != expected_length or len(values) == 0:
        raise ValueError(f"{name}: нужна одна метка для каждой строки")
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name}: допустимы только метки 0 и 1")
    if require_both and len(np.unique(values)) != 2:
        raise ValueError("Для обучения нужны оба класса: 0 и 1")
    return values.astype(int)


def _metrics(target, probabilities, predictions):
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)) if len(np.unique(target)) == 2 else float("nan"),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучить выбранный ансамбль на train и оценить его на test.

    Конфигурация фиксирована после CV-экспериментов. Предобработка обучается
    внутри Pipeline только на train. test_target используется лишь в метриках.
    """
    from threadpoolctl import threadpool_limits

    train = data.train_features
    test = data.test_features
    if not isinstance(train, pd.DataFrame):
        train = np.asarray(train, dtype=float)
        test = np.asarray(test, dtype=float)
    if getattr(train, "ndim", None) != 2 or getattr(test, "ndim", None) != 2:
        raise ValueError("Признаки должны быть двумерными таблицами или массивами")
    if train.shape[1] == 0 or train.shape[1] != test.shape[1] or len(test) == 0:
        raise ValueError("Нужны непустые test и признаки с одинаковым числом столбцов")
    if isinstance(train, pd.DataFrame):
        if not isinstance(test, pd.DataFrame) or set(train.columns) != set(test.columns):
            raise ValueError("Train и test должны содержать одинаковые имена столбцов")
    target = _binary_target(data.train_target, len(train), "train_target", require_both=True)
    pipeline = _make_pipeline(train, SELECTED_MODEL, data.random_seed)
    with threadpool_limits(limits=1):
        pipeline.fit(train, target)
        probabilities = pipeline.predict_proba(test)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    test_target = _binary_target(data.test_target, len(test), "test_target")
    return EnsembleResult(
        model_name=MODEL_NAMES[SELECTED_MODEL],
        probabilities=probabilities,
        predictions=predictions,
        **_metrics(test_target, probabilities, predictions),
    )


def _cv_experiments(features, target, seed):
    """Сравнение конфигураций только внутри train; Pipeline заново fit на каждом фолде."""
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {
        "roc_auc": "roc_auc", "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    results = {}
    for key, name in MODEL_NAMES.items():
        scores = cross_validate(_make_pipeline(features, key, seed), features, target,
                                scoring=scoring, cv=cv, n_jobs=1, error_score="raise")
        results[key] = {
            "model_name": name,
            "parameters": _make_estimator(key, seed).get_params(deep=False),
            "metrics": {metric: float(np.mean(scores["test_" + metric])) for metric in scoring},
            "roc_auc_std": float(np.std(scores["test_roc_auc"], ddof=0)),
            "fold_metrics": {metric: [float(v) for v in scores["test_" + metric]] for metric in scoring},
        }
        print(f"CV: {key}, ROC-AUC={results[key]['metrics']['roc_auc']:.6f}", flush=True)
    return results


def _save_figures(output_dir, experiments, target, probabilities, predictions, baseline_probabilities):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

    names = list(experiments)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, [experiments[name]["metrics"]["roc_auc"] for name in names],
            xerr=[experiments[name]["roc_auc_std"] for name in names],
            color=["#d57739" if name == SELECTED_MODEL else "#4472c4" for name in names],
            capsize=4)
    ax.set(xlim=(0, 1), xlabel="Mean ROC-AUC (5-fold CV on train)",
           title="Candidate ensembles; error bars = fold standard deviation")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, values in [("Selected Random Forest", probabilities),
                          ("Logistic baseline", baseline_probabilities)]:
        fpr, tpr, _ = roc_curve(target, values)
        auc = roc_auc_score(target, values)
        axes[0].plot(fpr, tpr, label=f"{label}: AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out test ROC")
    axes[0].legend(loc="lower right")
    ConfusionMatrixDisplay.from_predictions(target, predictions, ax=axes[1],
        display_labels=[0, 1], colorbar=False, cmap="Blues")
    axes[1].set_title("Selected model, threshold = 0.5")
    fig.tight_layout()
    fig.savefig(output_dir / "test_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_report(path, results):
    experiments = results["experiments"]
    selected = results["selection"]["selected_model"]
    chosen = experiments[selected]
    own = results["test"]["selected_model"]
    baseline = results["test"]["logistic_baseline"]
    train = results["train"]["selected_model"]
    lines = [
        "# Лабораторная работа №4. Ансамбли", "",
        "## Данные и подготовка", "",
        "Использован german.csv из задания: 1000 строк, 20 признаков, "
        "цель — первый столбец Creditability. Метки: 700 единиц и 300 нулей. "
        "Пропусков в исходной таблице нет. Положительный класс всех метрик — 1.", "",
        "Внешний split: train_test_split(test_size=0.2, random_state=42, stratify=y). "
        "Обучающая часть содержит 800 строк (560 единиц, 240 нулей); тестовая — "
        "200 строк (140 единиц, 60 нулей). Это же правило разбиения использовалось "
        "в предыдущей лабораторной. Индекс и целевой столбец не включаются в признаки.", "",
        "Количественные признаки: " + ", ".join(NUMERIC_COLUMNS) + ". "
        "Остальные 13 столбцов рассматриваются как категориальные коды и "
        "преобразуются OneHotEncoder(handle_unknown='ignore', sparse_output=False). "
        f"Число признаков после преобразования на полном train: {results['preprocessing']['encoded_features']}. "
        "Новые признаки сверх one-hot не добавлялись, отбор признаков не проводился.", "",
        "Масштабирование для деревьев не применяется: разбиения основаны на порогах "
        "отдельных признаков и не требуют одинаковых единиц измерения. "
        "В Pipeline предусмотрено заполнение числовых пропусков медианой и "
        "категориальных — наиболее частым значением. Эти статистики и словари "
        "категорий обучаются только на train; при CV — заново внутри каждого фолда. "
        "Для логистической модели сравнения числовые признаки дополнительно "
        "стандартизируются внутри её Pipeline.", "",
        "## Гипотезы и эксперименты", "",
        "До итоговой тестовой оценки проверены семь конфигураций. "
        "Критерий выбора — максимальный средний ROC-AUC пятичастной "
        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) "
        "только на внешней обучающей части. В каждом фолде 640 обучающих и "
        "160 валидационных строк. Все конфигурации используют одинаковые фолды. "
        "test_target не участвует в fit, подготовке признаков, CV или выборе порога.", "",
        "1. **Random Forest:** 300 деревьев, bootstrap=True, max_features='sqrt', "
        "без ограничения глубины. Гипотеза: усреднение разных деревьев уменьшит "
        "разброс прогноза по сравнению с одиночным деревом.",
        "2. **Random Forest с ограничениями:** те же 300 деревьев, max_depth=6, "
        "min_samples_leaf=3. Гипотеза: ограничение сложности улучшит качество "
        "на малой выборке и уменьшит подгонку отдельных наблюдений.",
        "3. **Extra Trees:** 300 деревьев, min_samples_leaf=2; bootstrap=False "
        "по умолчанию. Гипотеза: случайный выбор порогов даст дополнительное "
        "разнообразие деревьев; возможен и рост смещения.",
        "4. **Gradient Boosting, глубина 2:** 150 деревьев, learning_rate=0.05. "
        "Гипотеза: последовательное исправление ошибок неглубокими деревьями "
        "учтёт нелинейности при ограниченной сложности.",
        "5. **Gradient Boosting, глубина 3:** остальные параметры те же. "
        "Проверяется, полезны ли более сложные взаимодействия признаков.",
        "6. **HistGradientBoosting, 7 листьев:** 150 итераций, learning_rate=0.1, "
        "max_leaf_nodes=7, l2_regularization=1, early_stopping=False. "
        "Проверяется более сильное ограничение сложности бустинга.",
        "7. **HistGradientBoosting, 15 листьев:** прочие настройки те же. "
        "Проверяется влияние большей сложности деревьев. "
        "Для обоих вариантов min_samples_leaf=20 по умолчанию.", "",
        "Общие параметры случайности — seed=42; вычисления ограничены одним "
        "потоком и n_jobs=1. MLP и другие нейросети в решении не используются: "
        "по условию достаточно ансамбля.", "",
        "## Метрики", "",
        "В таблице — средние метрики на пяти валидационных фолдах внешнего train. "
        "Стандартное отклонение ROC-AUC вычислено с ddof=0; это разброс фолдов, "
        "а не доверительный интервал. ROC-AUC считается по вероятностям класса 1, "
        "остальные метрики — по классам при пороге 0.5. F1 усредняется по фолдам, "
        "а не вычисляется из уже усреднённых Precision и Recall.", "",
        "| Конфигурация | ROC-AUC | Std AUC | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, record in experiments.items():
        m = record["metrics"]
        lines.append(f"| {key} | {m['roc_auc']:.6f} | {record['roc_auc_std']:.6f} | "
                     f"{m['accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |")
    lines += [
        "", "![Кросс-валидация ансамблей](results/cv_comparison.png)", "",
        "После выбора конфигурация зафиксирована в SELECTED_MODEL. "
        "Модель обучена на всех 800 обучающих строках и применена к отложенным "
        "200 строкам. Отдельная логистическая модель сравнения обучена на тех же "
        "800 строках: LogisticRegression(C=1.0, max_iter=2000, random_state=42), "
        "числовые признаки стандартизированы. Эта модель не входит в кандидаты "
        "ансамбля и не влияет на его выбор.", "",
        "| Оценка | ROC-AUC | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in [("Выбранный лес, train", train), ("Выбранный лес, test", own),
                         ("Логистическая модель, test", baseline)]:
        lines.append("| " + name + " | " + " | ".join(f"{values[k]:.6f}" for k in
                     ["roc_auc", "accuracy", "precision", "recall", "f1"]) + " |")
    lines += [
        "", "![ROC и матрица ошибок](results/test_diagnostics.png)", "",
        "При отсутствии предсказанных единиц Precision и F1 определяются как 0 "
        "(zero_division=0). Если в переданном test только один класс, ROC-AUC "
        "не определён и возвращается NaN; в основном эксперименте оба класса есть.", "",
        "## Выбранная модель", "",
        f"Выбран **{chosen['model_name']}**. Его средний CV ROC-AUC — "
        f"**{chosen['metrics']['roc_auc']:.6f}**, максимальный среди семи "
        f"заранее заданных конфигураций. Test ROC-AUC после выбора — **{own['roc_auc']:.6f}**.", "",
        "Случайный лес обучает деревья на bootstrap-выборках обучающих строк. "
        "В каждом узле рассматривается случайное подмножество признаков "
        "(max_features='sqrt'), чтобы уменьшать сходство деревьев. "
        "Вероятность класса 1 — среднее вероятностей деревьев; внутри листа "
        "вероятность основана на долях классов обучающих объектов этого листа. "
        "Использованы 300 деревьев, глубина не более 6 и минимум 3 объекта в листе; "
        "порог класса 1 — p >= 0.5. Число деревьев ограничивает время CPU, "
        "ограничения глубины и размера листа регулируют сложность.", "",
        "## Выводы и ограничения", "",
        "Ограничение глубины и минимального размера листа увеличило средний "
        "CV ROC-AUC леса с 0.773586 до 0.778199. Extra Trees дал 0.765179. "
        "У классического бустинга увеличение глубины с 2 до 3 повысило AUC "
        "с 0.764025 до 0.769568, но не превзошло выбранный лес. "
        "У гистограммного бустинга увеличение числа листьев с 7 до 15 "
        "снизило AUC с 0.764211 до 0.756808. Это наблюдения для данной "
        "схемы CV, а не универсальные свойства алгоритмов.", "",
        f"На test лес имеет Δ ROC-AUC = {own['roc_auc'] - baseline['roc_auc']:+.6f}, "
        f"Δ Accuracy = {own['accuracy'] - baseline['accuracy']:+.6f}, "
        f"Δ F1 = {own['f1'] - baseline['f1']:+.6f} относительно логистической модели. "
        "Высокий Recall леса достигается при большем числе положительных "
        "предсказаний, что снижает Precision. Порог 0.5 не подбирался по test. "
        "Модель, всегда выбирающая класс 1, имеет test Accuracy=0.7, поэтому "
        "одной Accuracy для оценки недостаточно.", "",
        "Тестовый ROC-AUC выше среднего CV; на небольшой выборке качество "
        "зависит от состава разбиения. Разница CV AUC двух лесов мала по "
        "сравнению с межфолдовым разбросом. Статистическая значимость "
        "различий не проверялась. Отдельная оценка калибровки вероятностей, "
        "бизнес-стоимости ошибок и справедливости по подгруппам не проводилась. "
        "Выводы ограничены этим учебным набором и фиксированным split.", "",
        "## Воспроизводимость", "",
        "Весь код находится в ensemble_models.py. Контрактная функция "
        "run_ensemble_experiment использует зафиксированную конфигурацию, "
        "не выполняет подбор по test и принимает как DataFrame, так и готовые "
        "числовые массивы. Её вероятности, предсказания и метрики соответствуют "
        "строке выбранной модели на test. main() повторяет CV, проверяет "
        "выбор, обучает финальную модель и формирует отчёт, JSON и графики.", "",
        "random_seed=42; внешний split 80/20 со стратификацией; внутренние "
        "5 фолдов перемешиваются с тем же seed. Передача иного random_seed "
        "в контракт меняет случайность модели, но не зафиксированные параметры.", "",
        "Версии: " + ", ".join(f"{k} {v}" for k, v in results["environment"].items()) + ".",
        "Зависимости закреплены в requirements.txt; рекомендуется Python 3.12 "
        "(не ниже 3.11). Возможны небольшие платформенные различия "
        "арифметики с плавающей точкой.", "",
        "```bash", "python -m pip install -r requirements.txt",
        "python -m unittest discover -s tests -v", "python ensemble_models.py", "```", "",
        "На Windows команды можно выполнить через Python окружения: "
        "`.\\.venv\\Scripts\\python.exe` вместо `python`; "
        "окружение создаётся командой `python -m venv .venv`.", "",
        "experiment_results.json содержит параметры, средние и пофолдовые "
        "CV-метрики, критерий выбора, итоговые train/test-метрики и версии среды. "
        "Метрики сохраняются без округления. Исходные README, german.csv и "
        "grader_contracts сохранены без изменений. Тесты проверяют формат "
        "результата, вероятности, метрики, пропуски, неизвестные категории, "
        "воспроизводимость и отсутствие влияния test_target на прогноз. "
        "Скрытые тесты преподавателя не предоставлены.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Повторить исследования без использования test в выборе модели."""
    import json
    from pathlib import Path
    import platform
    import matplotlib
    import scipy
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_limits

    seed = 42
    root = Path(__file__).resolve().parent
    frame = pd.read_csv(root / "german.csv", sep=";")
    x_train, x_test, y_train, y_test = train_test_split(
        frame.iloc[:, 1:], frame.iloc[:, 0], test_size=.2, random_state=seed,
        stratify=frame.iloc[:, 0],
    )
    with threadpool_limits(limits=1):
        experiments = _cv_experiments(x_train, y_train, seed)
        best = max(experiments, key=lambda key: experiments[key]["metrics"]["roc_auc"])
        if best != SELECTED_MODEL:
            raise RuntimeError("Результат CV отличается от зафиксированной модели; проверьте версии зависимостей")
        result = run_ensemble_experiment(EnsembleInput(x_train, y_train, x_test, y_test, seed))
        # Отдельный fit той же конфигурации для оценки разрыва train/test.
        selected_pipeline = _make_pipeline(x_train, SELECTED_MODEL, seed).fit(x_train, y_train)
        p_train = selected_pipeline.predict_proba(x_train)[:, 1]
        baseline = Pipeline([
            ("preprocess", _make_preprocessor(x_train, scale_numeric=True)),
            ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]).fit(x_train, y_train)
        p_baseline = baseline.predict_proba(x_test)[:, 1]
        baseline_predictions = baseline.predict(x_test)
        results = {
            "dataset": "german.csv", "random_seed": seed,
            "split": {"train_rows": len(x_train), "test_rows": len(x_test),
                      "test_size": .2, "stratified": True},
            "preprocessing": {"numeric_columns": NUMERIC_COLUMNS,
                "categorical_columns": [c for c in x_train.columns if c not in NUMERIC_COLUMNS],
                "encoded_features": int(selected_pipeline.named_steps['preprocess'].transform(x_train).shape[1])},
            "selection": {"criterion": "maximum mean CV ROC-AUC", "n_splits": 5,
                          "shuffle": True, "random_seed": seed, "selected_model": best},
            "experiments": experiments,
            "train": {"selected_model": _metrics(y_train, p_train, (p_train >= .5).astype(int))},
            "test": {
                "selected_model": {key: float(getattr(result, key)) for key in
                    ["roc_auc", "accuracy", "precision", "recall", "f1"]},
                "logistic_baseline": _metrics(y_test, p_baseline, baseline_predictions),
            },
            "environment": {"Python": platform.python_version(), "NumPy": np.__version__,
                "Pandas": pd.__version__, "SciPy": scipy.__version__,
                "scikit-learn": sklearn.__version__, "Matplotlib": matplotlib.__version__},
        }
        output_dir = root / "results"
        output_dir.mkdir(exist_ok=True)
        _save_figures(output_dir, experiments, y_test, result.probabilities, result.predictions, p_baseline)
    _write_report(root / "REPORT.md", results)
    (root / "experiment_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Выбрана модель:", result.model_name)
    print(json.dumps(results["test"], indent=2))
    print("Сохранены REPORT.md, experiment_results.json и графики в results/")


if __name__ == "__main__":
    main()
