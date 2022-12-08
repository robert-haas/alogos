import os
import random
import sqlite3

import pytest
import shared

import alogos as al


def test_database_on_and_location(tmpdir):
    # No database
    # - implicit (default)
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", max_generations=1
    )
    assert ea.database is None
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    # - explicit
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=1,
        database_on=False,
    )
    assert ea.database is None
    ea.run()
    assert ea.state.best_individual.fitness >= 0

    # SQLite database in RAM
    # - implicit (default if database_on=True)
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=1,
        database_on=True,
    )
    assert ea.database is not None
    shared.check_ea_database(ea)
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    shared.check_ea_database(ea)
    # - explicit
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=1,
        database_on=True,
        database_location=":memory:",
    )
    assert ea.database is not None
    shared.check_ea_database(ea)
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    shared.check_ea_database(ea)

    # SQLite database on disk
    filepath = os.path.join(tmpdir.strpath, "test.db")
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=1,
        database_on=True,
        database_location=filepath,
    )
    assert ea.database is not None
    shared.check_ea_database(ea)
    ea.run()
    assert ea.state.best_individual.fitness >= 0
    assert os.path.isfile(filepath)
    assert os.path.getsize(filepath) > 0
    shared.check_ea_database(ea)


def test_database_export_sql_creates_valid_sql_file(tmpdir):
    # See if the SQLite3 database file is created and can be read
    def check_database_file(filepath):
        assert os.path.isfile(filepath)
        connection = sqlite3.connect(filepath)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM full_search")
        data = cursor.fetchall()
        assert len(data) > 20

    filepath = os.path.join(tmpdir.strpath, "dummy1.db")
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        database_on=True,
        database_location=filepath,
        max_generations=1,
    )
    ea.run()
    check_database_file(filepath)


@pytest.mark.xfail  # TODO: failure modes not clear at the moment
def test_database_export_sql_errors(tmpdir):
    def func():
        filepath = os.path.join(tmpdir.strpath, "dummy1.db")
        ea = al.EvolutionaryAlgorithm(
            shared.GRAMMAR_FLOAT,
            shared.OBJ_FUN_FLOAT,
            "min",
            database_on=True,
            database_location=filepath,
        )
        ea.database.export_sql(filepath)

    message = "TODO"
    shared.emits_exception(
        function=func, error_type=al.exceptions.DatabaseError, expected_message=message
    )


def test_database_generation_range_errors():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        lambda s: float(s),
        "min",
        database_on=True,
        max_generations=2,
    )
    ea.run()

    method_names = [
        "num_individuals",
        "num_genotypes",
        "num_phenotypes",
        "num_fitnesses",
        "individuals",
        "individuals_with_min_fitness",
        "individuals_with_max_fitness",
        "individuals_with_low_fitness",
        "individuals_with_high_fitness",
        "genotypes",
        "genotypes_with_min_fitness",
        "genotypes_with_max_fitness",
        "phenotypes",
        "phenotypes_with_min_fitness",
        "phenotypes_with_max_fitness",
        "details",
        "details_with_min_fitness",
        "details_with_max_fitness",
        "fitnesses",
        "fitness_min",
        "fitness_max",
    ]
    for method_name in method_names:
        method = getattr(ea.database, method_name)

        for gen_range in (3.14,):
            with pytest.raises(TypeError):
                method(generation_range=gen_range)
        for gen_range in (
            "x",
            ("x", 5),
            (5, "y"),
            (3.14, 5),
            (5, 3.14),
            (1,),
            (1, 2, 3),
            (5.0, 7.0),
        ):
            with pytest.raises(ValueError):
                method(generation_range=gen_range)


def test_database_fitness_min_and_fitness_max_errors():
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT, shared.OBJ_FUN_FLOAT, "min", database_on=True
    )
    ea.step()

    # Expected errors
    methods = [
        ea.database.fitness_min_after_num_evals,
        ea.database.fitness_max_after_num_evals,
    ]
    for fun in methods:
        value = fun(1)
        assert isinstance(value, float)
        assert value == value  # not NaN
        with pytest.raises(TypeError):
            fun("nonsense")
        with pytest.raises(ValueError):
            fun(0)
        with pytest.raises(ValueError):
            fun(-1)


@pytest.mark.skip  # TODO: flaky, fails sometimes
@pytest.mark.parametrize(
    "params",
    [
        (dict()),
        (dict(phe_to_fit_cache_size=3)),
        (dict(gen_to_phe_cache_size=3)),
        (dict(parent_selection_operator="rank_proportional")),
        (dict(parent_selection_operator="tournament")),
        (dict(parent_selection_operator="truncation")),
        (dict(parent_selection_operator="uniform")),
        (dict(crossover_operator=None)),
        (dict(mutation_operator=None)),
        (dict(survivor_selection_operator="rank_proportional")),
        (dict(survivor_selection_operator="tournament")),
        (dict(survivor_selection_operator="truncation")),
        (dict(survivor_selection_operator="uniform")),
    ],
)
def test_database_genotype_to_phenotype_and_phenotype_to_fitness_evaluation_order(
    params,
):
    bnf = """
    <tuple> ::= <number>
    <number> ::= <sign><digit>.<digit> | nan | inf | -inf
    <sign> ::= +|-
    <digit> ::= 0|1|2|3|4|5|6|7|8|9
    """
    grammar = al.Grammar(bnf_text=bnf)

    max_generations = 15

    for cache_on in (True, False):
        for db_on in (True, False):
            for system in ("cfggp", "cfggpst", "dsge", "ge", "pige", "whge"):
                # 1) Algorithm with caching and db

                # Objective function
                tracked = []

                def obj_fun(string):
                    phe = string  # noqa: B023
                    fit = float(string)  # noqa: B023
                    tracked.append((phe, fit))  # noqa: B023
                    return fit  # noqa: B023

                # Algorithm
                ea = al.EvolutionaryAlgorithm(
                    grammar,
                    obj_fun,
                    "min",
                    system=system,
                    max_generations=max_generations,
                    population_size=random.choice(range(13, 23)),
                    offspring_size=random.choice(range(13, 23)),
                    database_on=True,
                    gen_to_phe_cache_lookup_on=cache_on,
                    phe_to_fit_cache_lookup_on=cache_on,
                    phe_to_fit_database_lookup_on=db_on,
                    **params,
                )
                ea.run()

                # 1) Genotype-to-phenotype evaluations
                # - Evaluations queried from database by dedicated method
                db_eval_unique = ea.database.gen_to_phe_evaluations()
                num_db_eval_unique = ea.database.num_gen_to_phe_evaluations()
                assert len(db_eval_unique) == num_db_eval_unique

                # - Evaluations extracted from dataframe
                df = ea.database.to_dataframe()
                if (
                    "mutation_operator" not in params
                    or params["mutation_operator"] is not None
                ):
                    # Crossover individuals are evaluated only if they are not changed by mutation
                    df = df[df["label"] != "crossover"]
                df_gen_phe_values = df[["genotype", "phenotype"]].values.tolist()
                df_eval = [(gen, phe) for gen, phe in df_gen_phe_values]
                df_eval_unique = shared.filter_list_unique(df_eval)
                num_df_eval = len(df_eval)
                num_df_eval_unique = len(df_eval_unique)
                assert num_df_eval > num_df_eval_unique
                assert df_eval != df_eval_unique

                # - Evaluation number stored in state (tracks actual objective function calls)
                num_state_eval = ea.state.num_gen_to_phe_evaluations

                # Consistency between results retrieved with different methods
                # - Database and dataframe: always same
                assert num_db_eval_unique == num_df_eval_unique
                assert db_eval_unique == df_eval_unique
                # - Database and state: depends on cache
                if not cache_on or ea.parameters.gen_to_phe_cache_size < 5:
                    # If cache is off or very limited, some genotypes are evaluated more than once
                    assert num_state_eval > num_db_eval_unique
                else:
                    # If cache size is unrestricted, each genotype will be evaluated exactly once
                    assert num_state_eval == num_db_eval_unique

                # 2) Phenotype-to-fitness evaluations
                # - Evaluations tracked directly in objective function calls
                tracked_eval = [
                    (phe, fit if fit == fit else float("inf"))  # NaN to +Inf as in EA
                    for phe, fit in tracked
                ]
                tracked_eval_unique = shared.filter_list_unique(tracked_eval)
                num_tracked_eval = len(tracked_eval)
                num_tracked_eval_unique = len(tracked_eval_unique)
                if cache_on:
                    if db_on:
                        # If db is on, each phenotype is evaluated exactly once
                        assert num_tracked_eval == num_tracked_eval_unique
                        assert tracked_eval == tracked_eval_unique
                    elif ea.parameters.phe_to_fit_cache_size < 5:
                        # If db is off and cache lookup is restricted, some phenotypes are
                        # evaluated more than once
                        assert num_tracked_eval > num_tracked_eval_unique
                        assert tracked_eval != tracked_eval_unique
                elif db_on:
                    # If db is on, each phenotype is evaluated exactly once
                    assert num_tracked_eval == num_tracked_eval_unique
                    assert tracked_eval == tracked_eval_unique
                else:
                    # If db and cache are off, some phenotypes are evaluated more than once
                    assert num_tracked_eval > num_tracked_eval_unique
                    assert tracked_eval != tracked_eval_unique

                # - Evaluations queried from database by dedicated method
                db_eval_unique = ea.database.phe_to_fit_evaluations()
                num_db_eval_unique = ea.database.num_phe_to_fit_evaluations()
                assert len(db_eval_unique) == num_db_eval_unique

                # - Evaluations extracted from dataframe
                df = ea.database.to_dataframe()
                if (
                    "mutation_operator" not in params
                    or params["mutation_operator"] is not None
                ):
                    # Crossover individuals are evaluated only if they are not changed by mutation
                    df = df[df["label"] != "crossover"]
                df_phe_fit_values = df[["phenotype", "fitness"]].values.tolist()
                df_eval = [(phe, fit) for phe, fit in df_phe_fit_values]
                df_eval_unique = shared.filter_list_unique(df_eval)
                num_df_eval = len(df_eval)
                num_df_eval_unique = len(df_eval_unique)
                assert num_df_eval > num_df_eval_unique
                assert df_eval != df_eval_unique

                # - Evaluation number stored in state (tracks actual objective function calls)
                num_state_eval = ea.state.num_phe_to_fit_evaluations

                # Consistency between results retrieved with different methods
                assert num_tracked_eval == num_state_eval
                assert (
                    num_tracked_eval_unique == num_db_eval_unique == num_df_eval_unique
                )
                if "phe_to_fit_cache_size" in params:
                    if db_on:
                        assert (
                            num_tracked_eval == num_tracked_eval_unique
                        )  # due to full lookup
                        assert (
                            tracked_eval
                            == tracked_eval_unique
                            == db_eval_unique
                            == df_eval_unique
                        )
                    else:
                        assert (
                            num_tracked_eval > num_tracked_eval_unique
                        )  # due to partial lookup
                        assert (
                            tracked_eval
                            != tracked_eval_unique
                            == db_eval_unique
                            == df_eval_unique
                        )
                elif cache_on or db_on:
                    assert (
                        num_tracked_eval == num_tracked_eval_unique
                    )  # due to full lookup
                    assert (
                        tracked_eval
                        == tracked_eval_unique
                        == db_eval_unique
                        == df_eval_unique
                    )
                else:
                    assert (
                        num_tracked_eval > num_tracked_eval_unique
                    )  # due to no lookup
                    assert (
                        tracked_eval
                        != tracked_eval_unique
                        == db_eval_unique
                        == df_eval_unique
                    )

                # Consistency with fitness_min_after_num_evals and fitness_max_after_num_evals
                num_evaluations = ea.database.num_phe_to_fit_evaluations()
                tracked_fitnesses = [fit for phe, fit in tracked_eval_unique]
                db_fitnesses = [fit for phe, fit in db_eval_unique]
                for n in range(1, num_evaluations, 7):
                    fit_min = ea.database.fitness_min_after_num_evals(n)
                    fit_max = ea.database.fitness_max_after_num_evals(n)
                    phe_fit_n = ea.database.phe_to_fit_evaluations(num_evaluations=n)
                    phe_fit_det_n = ea.database.phe_to_fit_evaluations(
                        num_evaluations=n, with_details=True
                    )
                    fit_min2 = min(fit for phe, fit in phe_fit_n)
                    fit_min3 = min(fit for phe, fit, det in phe_fit_det_n)
                    fit_max2 = max(fit for phe, fit in phe_fit_n)
                    fit_max3 = max(fit for phe, fit, det in phe_fit_det_n)
                    assert (
                        fit_min
                        == fit_min2
                        == fit_min3
                        == min(tracked_fitnesses[:n])
                        == min(db_fitnesses[:n])
                    )
                    assert (
                        fit_max
                        == fit_max2
                        == fit_max3
                        == max(tracked_fitnesses[:n])
                        == max(db_fitnesses[:n])
                    )


def test_plot_genealogy(tmpdir):
    ea = al.EvolutionaryAlgorithm(
        shared.GRAMMAR_FLOAT,
        shared.OBJ_FUN_FLOAT,
        "min",
        max_generations=3,
        database_on=True,
    )
    ea.run()
    for backend in (None, "d3", "vis", "three"):
        if backend:
            fig = ea.database.plot_genealogy(backend=backend)
        else:
            fig = ea.database.plot_genealogy()

        text1 = fig.to_html()
        assert len(text1) > 0

        filepath = os.path.join(tmpdir.strpath, str(backend) + ".html")
        fig.export_html(filepath)
        with open(filepath) as f:
            text2 = f.read()
        assert text1[:100] == text2[:100]
        assert text1[-100:] == text2[-100:]
