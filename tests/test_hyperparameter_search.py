"""
Tests for parrot.hyperparameter_search: the search space, the space-filling
initial design, TPE sampling, successive halving, and training configurations
stage by stage (on the validation split or on cross-validation folds).
"""

import math

import numpy as np
import pytest
import torch

from parrot import hyperparameter_search as hs
from parrot import tpe_sampler
from parrot import train_network
from parrot.architectures import HyperParameter, build_network
from parrot.tasks import Task

from tests.helpers import FakeTrainer, make_loaders, make_search

TASK = Task('sequence', 1)


# ..........................................................................................
# Stages and the search space

@pytest.mark.parametrize('max_epochs, min_epochs, expected', [(30, 3, [3, 10, 30]), (100, 11, [11, 33, 100]),
                                                              (27, 1, [1, 3, 9, 27]), (5, 1, [1, 5]), (1, 1, [1]),
                                                              (30, 30, [30]), (50, 6, [6, 17, 50]),
                                                              (14, 2, [2, 5, 14]), (500, 56, [56, 167, 500]),
                                                              (9, 2, [2, 9]), (10, 4, [4, 10]), (10, 9, [9, 10])])
def test_stage_epochs(max_epochs, min_epochs, expected):
    assert hs.stage_epochs(max_epochs, min_epochs) == expected


@pytest.mark.parametrize('max_epochs', range(9, 301))
def test_default_stages_are_three_from_nine_epochs(max_epochs):
    # The default first stage is a ninth of the epochs, so there are three stages, each about three times longer
    search = make_search(n_trials=3, max_epochs=max_epochs)
    assert len(search.stages) == 3 and search.stages[-1] == max_epochs
    assert search.stages[0] == max(1, round(max_epochs / 9))


@pytest.mark.parametrize('min_epochs, reduction_factor', [(0, 3), (31, 3), (3, 1)])
def test_stage_epochs_rejects_invalid_settings(min_epochs, reduction_factor):
    with pytest.raises(ValueError):
        hs.stage_epochs(30, min_epochs, reduction_factor)


def test_search_space_dimensions():
    space = hs.SearchSpace('cnn', fixed={'channels': 32})
    names = [hyperparameter.name for hyperparameter in space.dimensions]
    assert names == ['learning_rate', 'num_blocks', 'kernel_size', 'dropout']  # learning rate first, channels fixed
    assert not space.is_numeric(space.dimensions[2])                          # kernel_size: a set of choices

    # Models fitted in one step have no learning rate
    assert [hyperparameter.name for hyperparameter in hs.SearchSpace('knn').dimensions] == [
        'num_neighbors', 'neighbor_weights', 'features']


@pytest.mark.parametrize('architecture, task, input_size, unused', [
    ('composition', Task('sequence', 1), 20, ['window_radius']),
    ('composition', Task('residues', 1), 20, []),
    ('unet2d', Task('matrix', 1, width=4), 20, ['pair_features']),
    ('unet2d', Task('matrix', 1), 20, []),
    ('svm', Task('sequence', 1), 20, []),
    ('svm', Task('sequence', 1), 1280, ['features']),          # e.g. ESM-2 embeddings: no dipeptide features
])
def test_hyperparameters_without_effect_are_not_searched(architecture, task, input_size, unused):
    space = hs.SearchSpace(architecture, task=task, input_size=input_size)
    assert list(space.unused) == unused
    assert not {hyperparameter.name for hyperparameter in space.dimensions} & set(unused)


def test_hyperparameters_a_configuration_does_not_use_take_their_default():
    space = hs.SearchSpace('svm')
    gamma = [index for index, hyperparameter in enumerate(space.dimensions) if hyperparameter.name == 'relative_gamma']
    linear = space.from_point([0.5, space.dimensions[1].search_choices.index('linear'), 0.9, 0])
    rbf = space.from_point([0.5, space.dimensions[1].search_choices.index('rbf'), 0.9, 0])
    assert gamma and linear['relative_gamma'] == 1.0 and rbf['relative_gamma'] != 1.0


def test_unet2d_search_range_stays_a_sensible_size():
    # The largest network the search can build (it once reached 1.5 billion parameters)
    space = hs.SearchSpace('unet2d')
    largest = {hyperparameter.name: hyperparameter.search_range[1] for hyperparameter in space.dimensions
               if space.is_numeric(hyperparameter)}
    largest.update(kernel_size=max(next(hyperparameter.search_choices for hyperparameter in space.dimensions
                                        if hyperparameter.name == 'kernel_size')),
                   pair_features='outer', upsampling='transposed')
    with torch.device('meta'):
        network = build_network('unet2d', 20, Task('matrix', 1), space.architecture_values(largest))
    assert sum(parameter.numel() for parameter in network.parameters()) < 50e6


@pytest.mark.parametrize('hyperparameter', [
    HyperParameter('rate', float, 0.001, 'h', search_range=(1e-5, 1e-2), log_scale=True),
    HyperParameter('width', int, 10, 'h', search_range=(5, 50)),
    HyperParameter('width', int, 64, 'h', search_range=(8, 512), log_scale=True),
    HyperParameter('dropout', float, 0.0, 'h', search_range=(0.0, 0.5)),
])
def test_unit_coordinates_round_trip(hyperparameter):
    low, high = hyperparameter.search_range
    values = range(low, high + 1) if hyperparameter.value_type is int else np.linspace(low, high, 25)
    for value in values:
        unit = hs.SearchSpace._to_unit(hyperparameter, value)
        assert 0 <= unit <= 1
        back = hs.SearchSpace._from_unit(hyperparameter, unit)
        assert back == value if hyperparameter.value_type is int else math.isclose(back, value, rel_tol=1e-9, abs_tol=1e-12)


def test_whole_numbers_are_equally_likely():
    """Uniform positions give every whole number in the range the same share (including the end points)"""
    hyperparameter = HyperParameter('layers', int, 2, 'h', search_range=(1, 5))
    counts = np.bincount([hs.SearchSpace._from_unit(hyperparameter, u) for u in np.linspace(0, 1, 10_001)])[1:]
    assert counts.min() / counts.max() > 0.99


def test_log_scale_whole_numbers_cover_the_range():
    hyperparameter = HyperParameter('width', int, 64, 'h', search_range=(8, 512), log_scale=True)
    values = [hs.SearchSpace._from_unit(hyperparameter, u) for u in np.linspace(0, 1, 1001)]
    assert min(values) == 8 and max(values) == 512
    assert np.median(values) == pytest.approx(64, rel=0.1)  # the geometric middle of 8 and 512


def test_default_values_and_points():
    space = hs.SearchSpace('transformer')
    defaults = space.default_values()
    assert defaults['learning_rate'] == pytest.approx(0.001)
    assert defaults['num_heads'] == 4 and defaults['positional_encoding'] == 'rotary'
    assert space.from_point(space.to_point(defaults)) == defaults


def test_initial_design_fills_the_space():
    space = hs.SearchSpace('unet')
    rng = np.random.default_rng(0)
    points = np.array(space.initial_design(12, rng))

    for dimension, hyperparameter in enumerate(space.dimensions):
        if space.is_numeric(hyperparameter):
            # Latin hypercube: exactly one point in each twelfth of every range
            assert sorted(np.floor(points[:, dimension] * 12).astype(int)) == list(range(12))
        else:
            counts = np.bincount(points[:, dimension].astype(int), minlength=len(hyperparameter.search_choices))
            assert counts.max() - counts.min() <= 1


def test_architecture_values_combine_searched_fixed_and_default():
    space = hs.SearchSpace('cnn', fixed={'pooling': 'max'})
    values = space.architecture_values({'learning_rate': 0.01, 'channels': 24, 'num_blocks': 2, 'kernel_size': 7,
                                        'dropout': 0.1})
    assert values == {'channels': 24, 'num_blocks': 2, 'kernel_size': 7, 'dropout': 0.1, 'pooling': 'max',
                      'global_context': True, 'attention_pooling': True, 'loglength_feedforward': True}
    assert hs.SearchSpace.learning_rate({'learning_rate': 0.01}) == 0.01
    assert hs.SearchSpace.learning_rate({}) == train_network.LEARNING_RATE.default


# ..........................................................................................
# TPE

def test_tpe_suggests_configurations_near_the_good_ones():
    """With losses that grow with distance from a target, TPE's best candidates are near it"""
    space = hs.SearchSpace('cnn')
    rng = np.random.default_rng(1)
    target = np.array([0.8, 0.2, 0.6, 3.0, 0.1])  # kernel_size choice index 3 (9)

    def loss(point: np.ndarray) -> float:
        return float(np.sum((point[[0, 1, 2, 4]] - target[[0, 1, 2, 4]]) ** 2) + (point[3] != target[3]))

    points = np.array([space.random_point(rng) for _ in range(60)])
    losses = np.array([loss(point) for point in points])
    sampler = hs.TPESampler(space, rng)
    suggested = sampler.suggest(points, losses)[:10]

    assert np.mean([loss(point) for point in suggested]) < np.quantile(losses, 0.25)
    assert np.mean(suggested[:, 3] == 3) > 0.5


def test_tpe_needs_enough_results():
    """Both densities need at least one more configuration than there are hyperparameters"""
    sampler = hs.TPESampler(hs.SearchSpace('brnn'), np.random.default_rng(0))  # 3 dimensions
    assert not sampler.model_can_be_fitted(7)
    assert sampler.model_can_be_fitted(8)
    assert sampler.n_good(8) == 4 and sampler.n_good(100) == 15


def test_tpe_treats_failed_configurations_as_bad():
    space = hs.SearchSpace('brnn')
    rng = np.random.default_rng(2)
    points = np.array([space.random_point(rng) for _ in range(20)])
    losses = np.linspace(1, 2, 20)
    losses[:5] = math.inf
    candidates = hs.TPESampler(space, rng).suggest(points, losses)
    assert np.all(np.isfinite(candidates)) and candidates.shape == (tpe_sampler.CANDIDATES_PER_SUGGESTION, 3)


# ..........................................................................................
# Successive halving (with a stand-in for training)

class PromotionCheckingTrainer(FakeTrainer):
    """Records, for every promotion, the trial's rank among the trials that had finished its stage at the time"""

    def train_to(self, trial, epochs):
        if trial.stage_losses:
            stage = len(trial.stage_losses) - 1
            finished = sorted((t for t in self.search.trials if len(t.stage_losses) > stage),
                              key=lambda t: t.stage_losses[stage])
            self.promotions.append((finished.index(trial), len(finished)))
        return super().train_to(trial, epochs)


def test_successive_halving_promotes_only_the_best_third_at_each_step():
    space = hs.SearchSpace('brnn')
    trainer = PromotionCheckingTrainer(space)
    trainer.promotions = []
    search = make_search(n_trials=30, max_epochs=27, min_epochs=3, trainer=trainer)
    trainer.search = search
    best = search.run()

    assert search.stages == [3, 9, 27]
    assert len(search.trials) == 30
    # Every promotion was of a trial in the best third of those that had finished its stage at the time
    assert trainer.promotions and all(rank < finished // 3 for rank, finished in trainer.promotions)
    # Many trials never get past the first stage
    assert sum(len(trial.stage_losses) == 1 for trial in search.trials) >= 10
    assert best.status == hs.COMPLETED and best.epochs == 27
    assert best.stage_losses[-1] == min(t.stage_losses[-1] for t in search.trials if t.status == hs.COMPLETED)
    assert all(t.status in (hs.COMPLETED, hs.STOPPED) for t in search.trials)


def test_search_starts_with_the_defaults_then_a_space_filling_design():
    search = make_search(n_trials=20)
    search.run()
    origins = [trial.origin for trial in search.trials]
    assert origins[0] == 'default' and search.trials[0].values['learning_rate'] == pytest.approx(0.001)
    assert origins[1:9] == ['initial'] * 8
    assert set(origins[9:]) <= {'model', 'random'}
    assert len({tuple(trial.values.values()) for trial in search.trials}) == 20  # no configuration tried twice


def test_search_finds_a_good_learning_rate():
    search = make_search(n_trials=40, seed=3)
    best = search.run()
    assert abs(math.log10(best.values['learning_rate'] / 0.003)) < 0.3


def test_search_is_reproducible_with_the_same_seed():
    first, second = make_search(seed=5), make_search(seed=5)
    first.run()
    second.run()
    assert [t.values for t in first.trials] == [t.values for t in second.trials]


def test_failed_configurations_are_recorded_and_skipped():
    space = hs.SearchSpace('brnn')
    trainer = FakeTrainer(space, fail_if=lambda values: values['num_layers'] > 3)
    search = make_search(trainer=trainer)
    best = search.run()

    failed = [trial for trial in search.trials if trial.status == hs.FAILED]
    assert failed and all(trial.stage_losses == [math.inf] for trial in failed)
    assert best.values['num_layers'] <= 3


def test_every_configuration_failing_is_an_error():
    space = hs.SearchSpace('brnn')
    with pytest.raises(RuntimeError, match='Every configuration failed') as raised:
        make_search(trainer=FakeTrainer(space, fail_if=lambda values: True), n_trials=5).run()

    # The first configuration's own error is kept, so a bug is not mistaken for unsuitable configurations
    assert isinstance(raised.value.__cause__, ValueError)
    assert 'does not suit the data' in str(raised.value.__cause__)


class DivergingTrainer(FakeTrainer):
    """Every configuration diverges: no epoch has a finite validation loss."""

    def train_to(self, trial, epochs):
        super().train_to(trial, epochs)
        return math.inf


@pytest.mark.parametrize('min_epochs', [10, 1])
def test_every_configuration_diverging_is_an_error(min_epochs):
    # With one stage, the diverged trials used to be marked completed, and the search returned one of them
    space = hs.SearchSpace('brnn')
    search = make_search(trainer=DivergingTrainer(space), n_trials=5, max_epochs=10, min_epochs=min_epochs)
    with pytest.raises(RuntimeError, match='Every configuration failed'):
        search.run()
    assert all(trial.status == hs.FAILED and trial.failure_reason == hs.DIVERGED for trial in search.trials)


class LaterStageFailureTrainer(FakeTrainer):
    """Trial 1 scores best in the first stage, then fails in the second (as running out of GPU memory would)."""

    def train_to(self, trial, epochs):
        if trial.number == 1 and epochs > self.first_stage_epochs:
            raise ValueError('pretend this configuration ran out of memory at its second stage')
        super().train_to(trial, epochs)
        return float(trial.number)


def test_a_stopped_configuration_is_trained_again_when_the_promoted_one_fails():
    # Trials 2 and 3 were stopped with good scores; the search used to say every configuration had failed
    space = hs.SearchSpace('brnn')
    trainer = LaterStageFailureTrainer(space)
    search = make_search(trainer=trainer, n_trials=3, max_epochs=9, min_epochs=1)
    trainer.first_stage_epochs = search.stages[0]
    best = search.run()
    assert best.number == 2 and best.status == hs.COMPLETED and best.epochs == 9
    assert search.trials[0].status == hs.FAILED and search.trials[2].status == hs.STOPPED


class FirstPromotionFailsTrainer(hs.TrialTrainer):
    """Trains for real, but the first configuration to reach its second stage fails there."""

    first_stage_epochs = 1
    failed = False

    def train_to(self, trial, epochs):
        if epochs > self.first_stage_epochs and not self.failed:
            self.failed = True
            raise ValueError('pretend this configuration ran out of memory at its second stage')
        return super().train_to(trial, epochs)


def test_a_stopped_configuration_trained_again_is_trained_for_every_epoch():
    # Its training state was dropped when it was stopped, but its epochs were not reset, so it was trained again
    # for only the epochs after the stage it had reached (8 of 9 here), though recorded as trained for all 9
    space = hs.SearchSpace('brnn')
    torch.manual_seed(0)
    trainer = FirstPromotionFailsTrainer(space, make_loaders(), 20, TASK, 'cpu')
    search = make_search(trainer=trainer, n_trials=3, max_epochs=9, min_epochs=1)
    trainer.first_stage_epochs = search.stages[0]
    best = search.run()

    _, train_losses, val_losses = trainer.trained_network(best)
    assert trainer.failed and best.status == hs.COMPLETED
    assert best.epochs == len(train_losses) == len(val_losses) == 9


def test_too_few_trials_still_train_one_configuration_fully():
    search = make_search(n_trials=2, max_epochs=27, min_epochs=3)
    best = search.run()
    assert best.epochs == 27 and best.status == hs.COMPLETED


def test_search_stops_when_every_configuration_has_been_tried():
    # random_forest with the number of trees fixed: 20 leaf sizes x 2 feature sets = 40 configurations
    space = hs.SearchSpace('random_forest', fixed={'num_trees': 10})

    class OneStepTrainer(FakeTrainer):
        def train_to(self, trial, epochs):
            trial.folds = [hs.FoldProgress()]
            return trial.values['min_samples_leaf'] + (trial.values['features'] == 'dipeptides')

    search = hs.HyperparameterSearch(space, OneStepTrainer(space), 100, 10, rng=np.random.default_rng(0), silent=True)
    best = search.run()
    assert search.stages == [1]                       # fitted in one step: a single stage
    assert len(search.trials) == 40
    assert best.values == {'min_samples_leaf': 1, 'features': 'composition'}


def test_nothing_to_search_trains_one_network():
    space = hs.SearchSpace('brnn')
    space.dimensions = []
    search = hs.HyperparameterSearch(space, FakeTrainer(space), 30, 9, rng=np.random.default_rng(0), silent=True)
    search.trainer.train_to = lambda trial, epochs: (setattr(trial, 'folds', [hs.FoldProgress()]), 1.0)[1]
    search.run()
    assert len(search.trials) == 1


def test_trials_are_written_as_a_table(tmp_path):
    search = make_search(n_trials=6, max_epochs=9, min_epochs=1)
    path = tmp_path / 'trials.tsv'
    search.on_progress = lambda progress: progress.write_trials(str(path))
    search.run()

    lines = path.read_text().splitlines()
    assert lines[0] == '# parrot-optimize trials for the brnn architecture'
    header = lines[1].split('\t')
    assert header == ['trial', 'origin', 'status', 'epochs', 'val_loss_1_epochs', 'val_loss_3_epochs',
                      'val_loss_9_epochs', 'learning_rate', 'hidden_size', 'num_layers']
    assert len(lines) == 2 + 6
    assert {line.split('\t')[2] for line in lines[2:]} <= {'completed', 'stopped'}


# ..........................................................................................
# Training configurations for real

@pytest.mark.parametrize('gradient_options', [{}, {'gradient_clip_norm': 0.1, 'check_finite_gradients': True}])
def test_training_resumes_exactly_where_it_stopped(gradient_options):
    """Training to 2 epochs and then on to 4 gives exactly the same result as training to 4 in one go"""
    space = hs.SearchSpace('brnn')
    values = space.default_values()
    results = []
    for stops in ([2, 4], [4]):
        torch.manual_seed(0)
        trainer = hs.TrialTrainer(space, make_loaders(), 20, TASK, 'cpu', **gradient_options)
        trial = hs.Trial(1, values, 'default')
        for epochs in stops:
            loss = trainer.train_to(trial, epochs)
        network, train_losses, val_losses = trainer.trained_network(trial)
        results.append((loss, train_losses, val_losses, network.state_dict()))

    (loss_a, train_a, val_a, state_a), (loss_b, train_b, val_b, state_b) = results
    assert loss_a == loss_b and train_a == train_b and val_a == val_b and len(val_a) == 4
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)


def test_trained_network_holds_the_best_epoch():
    space = hs.SearchSpace('cnn')
    torch.manual_seed(1)
    loaders = make_loaders()
    trainer = hs.TrialTrainer(space, loaders, 20, TASK, 'cpu')
    trial = hs.Trial(1, {**space.default_values(), 'learning_rate': 0.01}, 'default')
    loss = trainer.train_to(trial, 5)

    network, _, val_losses = trainer.trained_network(trial)
    assert loss == min(val_losses)
    assert train_network.evaluate_loss(network, loaders[0][1], TASK, 'cpu') == pytest.approx(loss, rel=1e-5)


def test_cross_validation_scores_the_average_over_folds():
    space = hs.SearchSpace('composition')
    trainer = hs.TrialTrainer(space, make_loaders(n_folds=3), 20, TASK, 'cpu')
    trial = hs.Trial(1, space.default_values(), 'default')
    loss = trainer.train_to(trial, 2)

    assert len(trial.folds) == 3
    assert loss == pytest.approx(np.mean([fold.best_loss for fold in trial.folds]))
    with pytest.raises(ValueError, match='validation split'):
        trainer.trained_network(trial)


def test_models_fitted_in_one_step_are_fitted_once():
    space = hs.SearchSpace('linear')
    trainer = hs.TrialTrainer(space, make_loaders(), 20, TASK, 'cpu')
    trial = hs.Trial(1, space.default_values(), 'default')
    loss = trainer.train_to(trial, 1)

    network, train_losses, val_losses = trainer.trained_network(trial)
    assert network.name == 'linear' and math.isfinite(loss)
    # The fit's losses are kept, so parrot-optimize can report them as parrot-train does
    assert len(train_losses) == 1 and val_losses == [loss]


def test_unsuitable_configuration_fails_without_stopping_the_search():
    """k-nearest neighbours with more neighbours than training sequences cannot be fitted"""
    space = hs.SearchSpace('knn', fixed={'features': 'composition', 'neighbor_weights': 'uniform'})
    trainer = hs.TrialTrainer(space, make_loaders(n_samples=40), 20, TASK, 'cpu')
    search = hs.HyperparameterSearch(space, trainer, 12, 1, rng=np.random.default_rng(0), silent=True)
    best = search.run()

    assert any(trial.status == hs.FAILED for trial in search.trials)  # some num_neighbors > 30
    assert best.values['num_neighbors'] <= 30
