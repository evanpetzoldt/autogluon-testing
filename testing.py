
from autogluon.tabular import TabularDataset, TabularPredictor
from utils import doublelift
from autogluon.core.metrics import make_scorer
from sklearn.metrics import mean_tweedie_deviance
import pandas as pd

ag_tweedie_scorer = make_scorer(name='mean_tweedie_deviance', score_func=mean_tweedie_deviance, greater_is_better=False, power=1.5)

all_data = TabularDataset('data/ret_hitter.csv')
all_data['scored_points'] = all_data['fan_pts'] > 0
starter_data = all_data[all_data['at_bats'] >= 3]
train_data = starter_data[starter_data['season'] < 2022]
test_data = starter_data[starter_data['season'] == 2022]

label = 'fan_pts'
label2 = 'scored_points'
model_vars = [
    'season', 'day_of_week', 'double_header', 'park_id', 'day_night', 'temperature', 'sky', 'precipitation', 'wind_speed',
    'player_games_played', 'player_lineup_position', 'player_team_id', 'opposing_team_id', 'player_home_away', 'team_games_played',
    'hit_percentage_season', 'walk_percentage_season', 'strikeout_percentage_season',
    'obp_5', 'obp_10', 'obp_15', 'obp_season',
    'run_percentage_season', 'rbi_percentage_season', 'steal_percentage_season',
    'fanpts_percentage_5', 'fanpts_percentage_10', 'fanpts_percentage_15', 'fanpts_percentage_season',
]

predictor_regression = TabularPredictor(label=label, problem_type='regression', path='models/ag-regression').fit(train_data[model_vars + [label]], time_limit=60*60)
predictor_binary = TabularPredictor(label=label2, problem_type='binary', path='models/ag-binary', eval_metric='roc_auc').fit(train_data[model_vars + [label2]], time_limit=60*60)
predictor_severity = TabularPredictor(label=label, problem_type='regression', path='models/ag-severity').fit(train_data[train_data['scored_points']][model_vars + [label]], time_limit=60*60)
predictor_tweedie = TabularPredictor(label=label, problem_type='regression', path='models/ag-tweedie', eval_metric=ag_tweedie_scorer).fit(train_data[model_vars + [label]], time_limit=60*60)
predictor_quantile = TabularPredictor(label=label, problem_type='quantile', path='models/ag-quantile', quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]).fit(train_data[train_data['scored_points']][model_vars + [label]], time_limit=60*60, calibrate=True)

pred_reg = predictor_regression.predict(test_data[model_vars])
pred_bin = predictor_binary.predict_proba(test_data[model_vars]).iloc[:, 1]
pred_sev = predictor_severity.predict(test_data[model_vars])
pred_twd = predictor_tweedie.predict(test_data[model_vars])
pred_qtl = predictor_quantile.predict(test_data[model_vars])

predictor_severity.leaderboard(train_data[model_vars + [label]], silent=True)
predictor_severity.feature_importance(test_data)

df_comb = pd.DataFrame({'pred_reg': pred_reg, 'scored_points_pct': pred_bin, 'pred_sev': pred_sev, 'pred_twd': pred_twd, 'fan_pts': test_data['fan_pts']})
df_comb['pred_freqsev'] = df_comb['scored_points_pct'] * df_comb['pred_sev']

chart = doublelift(df_comb, 'pred_reg', 'pred_freqsev', 'fan_pts', rescale=True)
chart.show()

val_tweedie_reg = mean_tweedie_deviance(df_comb['fan_pts'], df_comb['pred_reg'], power=1.5)
val_tweedie_freqsev = mean_tweedie_deviance(df_comb['fan_pts'], df_comb['pred_freqsev'], power=1.5)
val_tweedie_twd = mean_tweedie_deviance(df_comb['fan_pts'], df_comb['pred_twd'], power=1.5)


