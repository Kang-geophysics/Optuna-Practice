# %%
import optuna
# %%
def objective(trial):
    x = trial.suggest_float("x", -15, 30)
    y = trial.suggest_float("y", -15, 30)

    c0 = (x - 5) ** 2 + y**2 - 25
    c1 = -((x - 8) ** 2) - (y + 3) ** 2 + 7.7

    trial.set_user_attr("constraint", (c0, c1))

    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2

    return v0, v1


def constraints(trial):
    return trial.user_attrs["constraint"]

tmp = 3

sampler = optuna.samplers.NSGAIIISampler(constraints_func=constraints)
study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
study.optimize(objective, n_trials=2000)

fig = optuna.visualization.plot_pareto_front(study, include_dominated_trials=False)
fig.show()