import pandas as pd
import numpy as np
import QuantLib
import json
import os
from tqdm import tqdm
from datetime import datetime as dt
from datetime import time
from dateutil.relativedelta import relativedelta
from py_lib.data.snp_constituents import get_snp500_constituents
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns

ls_portfolios = [
    'rp_ml', 'rp_ewma', 'rp_ldw',
    'ivp_ml', 'ivp_ewma', 'ivp_ldw',
    'hrp_ml', 'hrp_ewma', 'hrp_ldw'
]
lo_portfolios = [
    'lo_rp_ml', 'lo_rp_ewma', 'lo_rp_ldw',
    'lo_ivp_ml', 'lo_ivp_ewma', 'lo_ivp_ldw',
    'lo_hrp_ml', 'lo_hrp_ewma', 'lo_hrp_ldw',
    'equal_weight'
]
ewma_ls_portfolios = [
    'rp_ewma',
    'ivp_ewma',
    'hrp_ewma',
]
ewma_lo_portfolios = [
    'lo_rp_ewma',
    'lo_ivp_ewma',
    'lo_hrp_ewma',
]
ml_ls_portfolios = [
    'rp_ml',
    'ivp_ml',
    'hrp_ml',
]
ml_lo_portfolios = [
    'lo_rp_ml',
    'lo_ivp_ml',
    'lo_hrp_ml',
]
ldw_ls_portfolios = [
    'rp_ldw',
    'ivp_ldw',
    'hrp_ldw',
]
ldw_lo_portfolios = [
    'lo_rp_ldw',
    'lo_ivp_ldw',
    'lo_hrp_ldw',
]

rp_lo_portfolios = [
    'lo_rp_ml', 'lo_rp_ewma', 'lo_rp_ldw',
]
ivp_lo_portfolios = [
    'lo_ivp_ml', 'lo_ivp_ewma', 'lo_ivp_ldw',
]
hrp_lo_portfolios = [
    'lo_hrp_ml', 'lo_hrp_ewma', 'lo_hrp_ldw',
]

return_metrics = [
    'sharpe', 'adj_sharpe', 'total_return', 'total_long_return',
    'total_short_return', 'beta', 'sortino'
]

lo_return_metrics = [
    'total_return', 'beta', 'vol'
]
risk_metrics = [
    'vol',
    'parametrical_var_mean', 'parametrical_var_std',
    'net_expo_mean', 'net_expo_std',
    'abs_net_expo_mean', 'abs_net_expo_std',
    'gross_expo_mean', 'gross_expo_std',
]

concentration_metrics = [
    'long_risk_contribution_mean',
    'long_money_contribution_mean', 'short_risk_contribution_mean',
    'short_money_contribution_mean', 'risk_conc_at_1pct_mean',
    'money_conc_at_1pct_mean', 'risk_conc_at_5pct_mean',
    'money_conc_at_5pct_mean', 'risk_conc_at_10pct_mean',
    'money_conc_at_10pct_mean', 'risk_conc_at_25pct_mean',
    'money_conc_at_25pct_mean', 'long_risk_contribution_std',
    'long_money_contribution_std', 'short_risk_contribution_std',
    'short_money_contribution_std', 'risk_conc_at_1pct_std',
    'money_conc_at_1pct_std', 'risk_conc_at_5pct_std',
    'money_conc_at_5pct_std', 'risk_conc_at_10pct_std',
    'money_conc_at_10pct_std', 'risk_conc_at_25pct_std',
    'money_conc_at_25pct_std'
]

all_metrics = ['sharpe', 'adj_sharpe', 'total_return', 'total_long_return',
               'total_short_return', 'beta', 'sortino', 'risk_free', 'vol',
               'parametrical_var_mean', 'net_expo_mean', 'abs_net_expo_mean',
               'gross_expo_mean', 'long_risk_contribution_mean',
               'long_money_contribution_mean', 'short_risk_contribution_mean',
               'short_money_contribution_mean', 'risk_conc_at_1pct_mean',
               'money_conc_at_1pct_mean', 'risk_conc_at_5pct_mean',
               'money_conc_at_5pct_mean', 'risk_conc_at_10pct_mean',
               'money_conc_at_10pct_mean', 'risk_conc_at_25pct_mean',
               'money_conc_at_25pct_mean', 'parametrical_var_std', 'net_expo_std',
               'abs_net_expo_std', 'gross_expo_std', 'long_risk_contribution_std',
               'long_money_contribution_std', 'short_risk_contribution_std',
               'short_money_contribution_std', 'risk_conc_at_1pct_std',
               'money_conc_at_1pct_std', 'risk_conc_at_5pct_std',
               'money_conc_at_5pct_std', 'risk_conc_at_10pct_std',
               'money_conc_at_10pct_std', 'risk_conc_at_25pct_std',
               'money_conc_at_25pct_std']


def format_correlation_table_for_latex(df_pivot, alpha=0.05, label="tab:corr_mse",
                                       caption="Correlation between metrics and RMSE"):
    # Separate correlation and p-value parts
    corr_df = df_pivot.loc[:, "corr"]
    pval_df = df_pivot.loc[:, "pval_adj"]

    # Create string-formatted DataFrame
    formatted = corr_df.copy()
    for col in corr_df.columns:
        corr = corr_df[col]
        pval = pval_df[col]
        formatted[col] = [
            f"{c:.2f}*" if p < alpha else f"{c:.2f}"
            for c, p in zip(corr, pval)
        ]

    # Optionally add a "Significant" count per row
    significance_count = (pval_df < alpha).sum(axis=1)
    formatted["# Significant"] = significance_count

    # Export to LaTeX
    latex_code = formatted.to_latex(
        index=True,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * (len(formatted.columns)),
        multicolumn=True,
        multicolumn_format='c'
    )
    return latex_code

def grouped_boxplots(df, groups=['ml', 'ewma', 'ldw'], levels=['rp', 'ivp', 'hrp'],
                     show=False, export_path=None, title="Boxplots"):
    import matplotlib.pyplot as plt

    # Build list of columns in desired order
    ordered_cols = [(s, m) for s in groups for m in levels]
    if ('weight', 'equal') in df.columns:
        ordered_cols.append(('weight', 'equal'))  # Add benchmark
    elif ('equal', 'weight') in df.columns:
            ordered_cols.append(('equal', 'weight'))

    # Reindex DataFrame with desired order
    plot_df = df[ordered_cols]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create boxplot
    plot_df.boxplot(ax=ax)

    # Set x-axis labels
    if ('weight', 'equal') in df.columns or ('equal', 'weight') in df.columns:
        labels = [f"{s}\n{m}" for s, m in ordered_cols[:-1]] + ['equal weight']
    else:
        labels = [f"{s}\n{m}" for s, m in ordered_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Optional: Add visual group separators
    ax.axvline(3.5, color='gray', linestyle='--')
    ax.axvline(6.5, color='gray', linestyle='--')
    if ('weight', 'equal') in df.columns or ('equal', 'weight') in df.columns:
        ax.axvline(9.5, color='gray', linestyle='--')

    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    if export_path is not None:
        plt.savefig(export_path, format='png')


def grouped_latex_table(
    summary_df,
    benchmark_col=('weight', 'equal'),
    benchmark_name=('equal weight'),
    float_fmt="{:.4f}",
    caption="Summary statistics for strategy and benchmark returns",
    label="tab:return_summary",
    title=None,
    legend=None,
    placement="ht",
    export_path=None,
    landscape=True,
):
    pass
    # # Ensure benchmark is at the end
    # cols = list(summary_df.columns)
    # if benchmark_col in cols:
    #     cols.remove(benchmark_col)
    #     cols.append(benchmark_col)
    # summary_df = summary_df[cols]
    #
    # # Prepare column headers
    # header_groups = [col[0] if isinstance(col, tuple) else col for col in summary_df.columns]
    # header_methods = [col[1] if isinstance(col, tuple) else "" for col in summary_df.columns]
    #
    # # Build LaTeX lines
    # lines = []
    # # if title:
    # #     lines.append(r"\vspace{1em}")
    # #     lines.append(r"\noindent\textbf{" + title + r"}")
    # #     lines.append("")
    #
    # if landscape:
    #     lines.append(r"\begin{landscape}")
    #     lines.append(r"\begin{table}[" + placement + "]")
    # else:
    #     lines.append(r"\begin{table}[" + placement + "]")
    # lines.append(r"\centering")
    # lines.append(r"\renewcommand{\arraystretch}{1.2}")
    # lines.append(r"\begin{tabular}{|l|" + "ccc|" * 3 + "c|}")
    # lines.append(r"\toprule")
    #
    # # Top header (group names)
    # group_header = ["\\textbf{" + str(g).upper() + "}" for g in header_groups[:-1]]
    # grouped = [group_header[i] if i % 3 == 0 else "" for i in range(len(group_header))]
    # grouped.append(f"\\textbf{benchmark_name}")
    # lines.append(" & " + " & ".join(grouped) + r" \\")
    #
    # # Second header (methods)
    # lines.append(r"\textbf{Stat} & " + " & ".join(header_methods) + r" \\")
    # lines.append(r"\midrule")
    #
    # # Table body
    # for idx, row in summary_df.iterrows():
    #     row_label = str(idx).capitalize()
    #     row_data = [float_fmt.format(row[col]) for col in summary_df.columns]
    #     lines.append(f"{row_label} & " + " & ".join(row_data) + r" \\")
    #
    # lines.append(r"\bottomrule")
    # lines.append(r"\end{tabular}")
    # lines.append(rf"\caption{{{caption}}}")
    # lines.append(rf"\label{{{label}}}")
    # if landscape:
    #     lines.append(r"\end{table}")
    #     lines.append(r"\end{landscape}")
    # else:
    #     lines.append(r"\end{table}")
    #
    # if legend:
    #     lines.append("")
    #     lines.append(r"\vspace{0.5em}")
    #     lines.append(r"\noindent\small{" + legend + r"}")
    #
    # if export_path is not None:
    #     with open(export_path, "w") as f:
    #         f.write("\n".join(lines))
    #
    # return "\n".join(lines)


if __name__ == '__main__':
    plt.rcdefaults()

    year_scores_path = 'py_lib/data/year_scores_df.pkl'
    month_scores_path = 'py_lib/data/month_scores_df.pkl'
    week_scores_path = 'py_lib/data/week_scores_df.pkl'
    cvmx_eval_path = 'py_lib/data/cvmx_eval_df.pkl'
    available_tikers_path = 'py_lib/data/available_tikers.pkl'

    portfolio_renames = {
        'equal_weight': 'Equal Weight',
        'hrp_ewma': 'Long and Short Hierarchical RP using exponentially weighted covariance',
        'hrp_ldw': 'Long and Short Hierarchical RP using Ledoit-Wolf shrunk covariance',
        'hrp_ml': 'Long and Short Hierarchical RP using maximum likelihood covariance',
        'ivp_ewma': 'Long and Short Inverse Variance using exponentially weighted covariance',
        'ivp_ldw': 'Long and Short Inverse Variance using Ledoit-Wolf shrunk covariance',
        'ivp_ml': 'Long and Short Inverse Variance using maximum likelihood covariance',
        'rp_ewma': 'Long and Short Risk Parity using exponentially weighted covariance',
        'rp_ldw': 'Long and Short Risk Parity using Ledoit-Wolf shrunk covariance',
        'rp_ml': 'Long and Short Risk Parity using maximum likelihood covariance',
        'lo_hrp_ewma': 'Hierarchical RP using exponentially weighted covariance',
        'lo_hrp_ldw': 'Hierarchical RP using Ledoit-Wolf shrunk covariance',
        'lo_hrp_ml': 'Hierarchical RP using maximum likelihood covariance',
        'lo_ivp_ewma': 'Inverse Variance using exponentially weighted covariance',
        'lo_ivp_ldw': 'Inverse Variance using Ledoit-Wolf shrunk covariance',
        'lo_ivp_ml': 'Inverse Variance using maximum likelihood covariance',
        'lo_rp_ewma': 'Risk Parity using exponentially weighted covariance',
        'lo_rp_ldw': 'Risk Parity using Ledoit-Wolf shrunk covariance',
        'lo_rp_ml': 'Risk Parity using maximum likelihood covariance',
    }

    year_scores_df = pd.read_pickle(year_scores_path)
    month_scores_df = pd.read_pickle(month_scores_path)
    week_scores_df = pd.read_pickle(week_scores_path)
    cvmx_eval_df = pd.read_pickle(cvmx_eval_path)
    available_tikers = pd.read_pickle(available_tikers_path)
    available_tikers.index = pd.to_datetime(available_tikers.index)

    week_scores_df = week_scores_df.rename(columns=portfolio_renames, level=1)
    month_scores_df = month_scores_df.rename(columns=portfolio_renames, level=1)
    year_scores_df = year_scores_df.rename(columns=portfolio_renames, level=1)

    cvmx_eval_pivot = cvmx_eval_df.reset_index()[['level_0', 'level_1', 'mse', 'medae']].pivot_table(index='level_0',
                                                                                                     columns='level_1')
    cvmx_eval_pivot.index = pd.to_datetime(cvmx_eval_pivot.index)
    cvmx_eval_mse = cvmx_eval_pivot['mse']
    cvmx_eval_medae = cvmx_eval_pivot['medae']

    cvmx_eval_mse = cvmx_eval_mse.astype(float)
    cvmx_eval_rmse = np.sqrt(cvmx_eval_mse)


    def fisher_ci(r, n, alpha=0.05):
        """Compute 95% CI for Pearson correlation using Fisher transformation."""
        if np.abs(r) == 1 or n <= 3:
            return r, r, r  # Cannot compute CI reliably
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = 1.96  # for 95% CI
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        return r, np.tanh(z_lower), np.tanh(z_upper)

    cov_mx_groups = {
        'ewma_cov_matrix': list(map(portfolio_renames.get, ewma_lo_portfolios)),
        'ledoit_wolf_cov_matrix': list(map(portfolio_renames.get, ldw_lo_portfolios)),
        'ml_cov_matrix': list(map(portfolio_renames.get, ml_lo_portfolios)),
    }

    correl_mses = {}
    correl_medaes = {}
    correl_mses_ci = {}
    correl_medaes_ci = {}
    for metric in return_metrics + ['vol']:
        metric_correl_mses = []
        metric_correl_medaes = []
        metric_correl_mses_ci = []
        metric_correl_medaes_ci = []
        for cov_mx_g, ps in cov_mx_groups.items():
            ls_df = week_scores_df[metric][ps].copy()
            if 'return' in metric:
                ls_df = ls_df - 1
            ls_df['cov_mx_mse'] = cvmx_eval_rmse[cov_mx_g].to_list()
            ls_df['cov_mx_medae'] = cvmx_eval_medae[cov_mx_g].to_list()

            corrs_mse = []
            ci_mse = []
            corrs_medae = []
            ci_medae = []

            correl_mse = ls_df.corr()['cov_mx_mse'][ps]
            correl_medae = ls_df.corr()['cov_mx_medae'][ps]
            for p in ps:
                r_mse = ls_df[p].corr(ls_df['cov_mx_mse'])
                r_medae = ls_df[p].corr(ls_df['cov_mx_medae'])
                n = ls_df[[p, 'cov_mx_mse']].dropna().shape[0]

                corr_mse, lower_mse, upper_mse = fisher_ci(r_mse, n)
                corr_medae, lower_medae, upper_medae = fisher_ci(r_medae, n)

                corrs_mse.append(pd.Series(corr_mse, name=p))
                ci_mse.append(pd.Series({'lower': lower_mse, 'upper': upper_mse}, name=p))

                corrs_medae.append(pd.Series(corr_medae, name=p))
                ci_medae.append(pd.Series({'lower': lower_medae, 'upper': upper_medae}, name=p))

            metric_correl_mses.append(correl_mse)
            metric_correl_medaes.append(correl_medae)
            # metric_correl_mses.append(pd.concat(corrs_mse, axis=1))
            metric_correl_mses_ci.append(pd.concat(ci_mse, axis=1).T)

            # metric_correl_medaes.append(pd.concat(corrs_medae, axis=1))
            metric_correl_medaes_ci.append(pd.concat(ci_medae, axis=1).T)

        correl_mses[metric] = pd.concat(metric_correl_mses)
        correl_medaes[metric] = pd.concat(metric_correl_medaes)
        correl_mses_ci[metric] = pd.concat(metric_correl_mses_ci)
        correl_medaes_ci[metric] = pd.concat(metric_correl_medaes_ci)

        # correl_mses[metric] = pd.concat(metric_correl_mses)
        # correl_medaes[metric] = pd.concat(metric_correl_medaes)

    correl_mse_df = pd.concat(correl_mses, axis=1)
    correl_mses_ci_df = pd.concat(correl_mses_ci, axis=1)
    correl_medae_df = pd.concat(correl_medaes, axis=1)
    correl_medaes_ci_df = pd.concat(correl_medaes_ci, axis=1)

    ls_cum_results = (np.exp(week_scores_df['total_return'].sort_index().cumsum()) - 1)[list(map(portfolio_renames.get, ls_portfolios))]
    lo_cum_results = (np.exp(week_scores_df['total_return'].sort_index().cumsum()) - 1)[list(map(portfolio_renames.get, lo_portfolios))]

    lo_cum_results_month = (np.exp(month_scores_df['total_return'].sort_index().cumsum()) - 1)[list(map(portfolio_renames.get, lo_portfolios))]
    ls_cum_results_month = (np.exp(month_scores_df['total_return'].sort_index().cumsum()) - 1)[list(map(portfolio_renames.get, ls_portfolios))]

    tables = []

    ####################################################################################################################
    #### Returns
    ####################################################################################################################

    lo_monthly_return_distribution = month_scores_df['total_return'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_return_distribution = month_scores_df['total_return'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_return_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_return_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_return_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_return_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_return_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_return_distribution.columns])
    tables.append(
        grouped_latex_table(
            lo_monthly_return_distribution,
            caption='Long-Only portfolios monthly returns distribution',
            label='lo_return_summary',
            title='Long-Only Monthly Return',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_return_distribution,
            caption='Long and short portfolios monthly returns distribution',
            label='ls_return_summary',
            title='Long and Short Monthly Return',
        )
    )

    boxplot_df = month_scores_df['total_return'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_total_return_boxplots.png',
                     title="Monthly Returns Distribution - Long Only Portfolios - Grouped by covariance matrix estimation method")
    # boxplot_df = month_scores_df['total_return'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_total_return_boxplots.png',
    #                  title="Monthly Returns Distribution - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    ####################################################################################################################
    #### Vol
    ####################################################################################################################

    lo_monthly_vol_distribution = month_scores_df['vol'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_vol_distribution = month_scores_df['vol'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_vol_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_vol_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_vol_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_vol_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_vol_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_vol_distribution.columns])

    tables.append(
        grouped_latex_table(
            lo_monthly_vol_distribution,
            caption='Long-Only portfolios monthly volatility distribution',
            label='lo_vol_summary',
            title='Long-Only Monthly Vol',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_vol_distribution,
            caption='Long and short portfolios monthly volatility distribution',
            label='ls_vol_summary',
            title='Long and Short Monthly Vol',
        )
    )

    boxplot_df = month_scores_df['vol'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_vol_boxplots.png',
                     title="Monthly Return Volatility Distribution - Long Only Portfolios - Grouped by covariance matrix estimation method")
    # boxplot_df = month_scores_df['vol'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_vol_boxplots.png',
    #                  title="Monthly Return Volatility Distribution - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    ####################################################################################################################
    #### Adjusted Sharpe
    ####################################################################################################################

    lo_monthly_adj_sharpe_distribution = month_scores_df['adj_sharpe'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_adj_sharpe_distribution = month_scores_df['adj_sharpe'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_adj_sharpe_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_adj_sharpe_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_adj_sharpe_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_adj_sharpe_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_adj_sharpe_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_adj_sharpe_distribution.columns])

    tables.append(
        grouped_latex_table(
            lo_monthly_adj_sharpe_distribution,
            caption='Long-Only portfolios monthly adjusted sharpe distribution',
            label='lo_adj_sharpe_summary',
            title='Long-Only Monthly Adjusted Sharpe',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_adj_sharpe_distribution,
            caption='Long and short portfolios monthly adjusted sharpe distribution',
            label='ls_adj_sharpe_summary',
            title='Long and Short Monthly Adjusted Sharpe',
        )
    )

    boxplot_df = month_scores_df['adj_sharpe'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_adjusted_sharpe_boxplots.png',
                     title="Monthly Adjusted Sharpe Distribution - Long Only Portfolios - Grouped by covariance matrix estimation method")
    boxplot_df = month_scores_df['adj_sharpe'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_adjusted_sharpe_boxplots.png',
    #                  title="Monthly Adjusted Sharpe Distribution - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    ####################################################################################################################
    #### Long / Short returns
    ####################################################################################################################

    ls_monthly_long_return_distribution = month_scores_df['total_long_return'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_short_return_distribution = month_scores_df['total_short_return'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_long_return_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_long_return_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_long_return_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_short_return_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_short_return_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_short_return_distribution.columns])

    tables.append(
        grouped_latex_table(
            ls_monthly_long_return_distribution,
            caption='Long and short portfolios monthly long returns distribution',
            label='ls_monthly_long_return_summary',
            title='Long and Short Monthly Long Returns',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_short_return_distribution,
            caption='Long and short portfolios monthly short returns distribution',
            label='ls_monthly_short_return_summary',
            title='Long and Short Monthly Short Returns',
        )
    )

    # boxplot_df = month_scores_df['total_long_return'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_long_return_boxplots.png',
    #                  title="Monthly Long Return Distribution - Long and Short Portfolios - Grouped by covariance matrix estimation method")
    # boxplot_df = month_scores_df['total_short_return'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_short_return_boxplots.png',
    #                  title="Monthly Short Return Distribution - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    ####################################################################################################################
    #### Sharpe
    ####################################################################################################################

    lo_monthly_sharpe_distribution = month_scores_df['sharpe'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_sharpe_distribution = month_scores_df['sharpe'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_sharpe_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_sharpe_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_sharpe_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_sharpe_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_sharpe_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_sharpe_distribution.columns])

    tables.append(
        grouped_latex_table(
            lo_monthly_sharpe_distribution,
            caption='Long-Only portfolios monthly sharpe distribution',
            label='lo_sharpe_summary',
            title='Long-Only Monthly Sharpe',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_sharpe_distribution,
            caption='Long and short portfolios monthly sharpe distribution',
            label='ls_sharpe_summary',
            title='Long and Short Monthly Sharpe',
        )
    )

    boxplot_df = month_scores_df['sharpe'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_sharpe_boxplots.png',
                     title="Monthly Sharpe Distribution - Long Only Portfolios - Grouped by covariance matrix estimation method")
    # boxplot_df = month_scores_df['sharpe'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_sharpe_boxplots.png',
    #                  title="Monthly Sharpe Distribution - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    ####################################################################################################################
    #### Beta
    ####################################################################################################################

    lo_monthly_beta_distribution = month_scores_df['beta'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_beta_distribution = month_scores_df['beta'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_beta_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_beta_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_beta_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_beta_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_beta_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_beta_distribution.columns])

    tables.append(
        grouped_latex_table(
            lo_monthly_beta_distribution,
            caption='Long-Only portfolios monthly beta distribution',
            label='lo_beta_summary',
            title='Long-Only Monthly Beta',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_beta_distribution,
            caption='Long and short portfolios monthly beta distribution',
            label='ls_beta_summary',
            title='Long and Short Monthly Beta',
        )
    )

    boxplot_df = month_scores_df['beta'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_beta_boxplots.png',
                     title="Monthly Market Beta Distribution - Long Only Portfolios - Grouped by covariance matrix estimation method")
    # boxplot_df = month_scores_df['beta'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_beta_boxplots.png',
    #                  title="Monthly Market Beta Distribution - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    tables = []

    ####################################################################################################################
    #### Risk Concentration - 10%
    ####################################################################################################################

    lo_monthly_money_conc_at_10pct_mean_distribution = month_scores_df['risk_conc_at_10pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])

    lo_monthly_money_conc_at_10pct_mean_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_money_conc_at_10pct_mean_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_money_conc_at_10pct_mean_distribution.columns.str.removeprefix('lo_')])


    tables.append(
        grouped_latex_table(
            lo_monthly_money_conc_at_10pct_mean_distribution,
            caption='Long-Only portfolios top 10% assets risk distribution',
            label='lo_risk_conc_at_10pct_mean_summary',
            # title='Long-Only Monthly Beta',
        )
    )

    boxplot_df = month_scores_df['risk_conc_at_10pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_risk_conc_at_10pct_mean_boxplots.png',
                     title="Top 10% Assets Risk - Long Only Portfolios - Grouped by covariance matrix estimation method")


    ####################################################################################################################
    #### Risk Concentration - 25%
    ####################################################################################################################

    lo_monthly_money_conc_at_25pct_mean_distribution = month_scores_df['risk_conc_at_25pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_money_conc_at_25pct_mean_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_money_conc_at_25pct_mean_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_money_conc_at_25pct_mean_distribution.columns.str.removeprefix('lo_')])


    tables.append(
        grouped_latex_table(
            lo_monthly_money_conc_at_25pct_mean_distribution,
            caption='Long-Only portfolios top 25% assets risk distribution',
            label='lo_risk_conc_at_25pct_mean_summary',
            # title='Long-Only Monthly Beta',
        )
    )

    boxplot_df = month_scores_df['risk_conc_at_25pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_risk_conc_at_25pct_mean_boxplots.png',
                     title="Top 25% Assets Risk - Long Only Portfolios - Grouped by covariance matrix estimation method")



    ####################################################################################################################
    #### Money Concentration - 10%
    ####################################################################################################################

    lo_monthly_money_conc_at_10pct_mean_distribution = month_scores_df['money_conc_at_10pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_money_conc_at_10pct_mean_distribution = month_scores_df['money_conc_at_10pct_mean'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_money_conc_at_10pct_mean_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_money_conc_at_10pct_mean_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_money_conc_at_10pct_mean_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_money_conc_at_10pct_mean_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_money_conc_at_10pct_mean_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_money_conc_at_10pct_mean_distribution.columns])

    tables.append(
        grouped_latex_table(
            lo_monthly_money_conc_at_10pct_mean_distribution,
            caption='Long-Only portfolios top 10% assets weight distribution',
            label='lo_money_conc_at_10pct_mean_summary',
            # title='Long-Only Monthly Beta',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_money_conc_at_10pct_mean_distribution,
            caption='Long and Short portfolios top 10% assets weight distribution',
            label='ls_money_conc_at_10pct_mean_summary',
            # title='Long and Short Monthly Beta',
        )
    )

    boxplot_df = month_scores_df['money_conc_at_10pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_money_conc_at_10pct_mean_boxplots.png',
                     title="Top 10% Assets Weight - Long Only Portfolios - Grouped by covariance matrix estimation method")
    # boxplot_df = month_scores_df['money_conc_at_10pct_mean'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_money_conc_at_10pct_mean_boxplots.png',
    #                  title="Top 10% Assets Weight - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    ####################################################################################################################
    #### Money Concentration - 25%
    ####################################################################################################################

    lo_monthly_money_conc_at_25pct_mean_distribution = month_scores_df['money_conc_at_25pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    ls_monthly_money_conc_at_25pct_mean_distribution = month_scores_df['money_conc_at_25pct_mean'][list(map(portfolio_renames.get, ls_portfolios))].agg([
        'mean',
        'min',
        lambda x: x.quantile(0.10),
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        'max',
    ])
    lo_monthly_money_conc_at_25pct_mean_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    lo_monthly_money_conc_at_25pct_mean_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in lo_monthly_money_conc_at_25pct_mean_distribution.columns.str.removeprefix('lo_')])
    ls_monthly_money_conc_at_25pct_mean_distribution.index = ['mean', 'min', 'q10', 'q25', 'q50', 'q75', 'q90', 'max']
    ls_monthly_money_conc_at_25pct_mean_distribution.columns = pd.MultiIndex.from_tuples(
        [col.split('_', 1) for col in ls_monthly_money_conc_at_25pct_mean_distribution.columns])

    tables.append(
        grouped_latex_table(
            lo_monthly_money_conc_at_25pct_mean_distribution,
            caption='Long-Only portfolios top 25% assets weight distribution',
            label='lo_money_conc_at_25pct_mean_summary',
            # title='Long-Only Monthly Beta',
        )
    )
    tables.append(
        grouped_latex_table(
            ls_monthly_money_conc_at_25pct_mean_distribution,
            caption='Long and Short portfolios top 25% assets weight distribution',
            label='ls_money_conc_at_25pct_mean_summary',
            # title='Long and Short Monthly Beta',
        )
    )

    boxplot_df = month_scores_df['money_conc_at_25pct_mean'][list(map(portfolio_renames.get, lo_portfolios))].copy()
    boxplot_df.columns = pd.MultiIndex.from_tuples([(' '.join(col.split(' ')[:2]), ' '.join(col.split(' ')[3:-1])) for col in boxplot_df.columns])
    grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
                     groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
                     levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
                     show=False,
                     export_path=fr'out/images/lo_money_conc_at_25pct_mean_boxplots.png',
                     title="Top 25% Assets Weight - Long Only Portfolios - Grouped by covariance matrix estimation method")
    # boxplot_df = month_scores_df['money_conc_at_25pct_mean'][list(map(portfolio_renames.get, ls_portfolios))].copy()
    # boxplot_df.columns = pd.MultiIndex.from_tuples(
    #     [col.split('_', 1) for col in boxplot_df.columns])
    # grouped_boxplots(boxplot_df.reorder_levels([1, 0], axis=1),
    #                  groups=['maximum likelihood', 'exponentially weighted', 'Ledoit-Wolf shrunk'],
    #                  levels=['Risk Parity', 'Inverse Variance', 'Hierarchical RP'],
    #                  show=False,
    #                  export_path=fr'out/images/ls_money_conc_at_25pct_mean_boxplots.png',
    #                  title="Top 25% Assets Weight - Long and Short Portfolios - Grouped by covariance matrix estimation method")

    ####################################################################################################################
    #### Available assets per period
    ####################################################################################################################

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    pd.DataFrame(available_tikers, columns=['Available Assets']).plot(ax=ax)
    ax.set_title("Available Assets per week")
    ax.set_ylabel("Amount")
    ax.set_xlabel("Date")
    ax.grid(True)
    plt.tight_layout()

    # Export to file
    plt.savefig('out/images/available_tickers.png', format='png')
    plt.close()

    ####################################################################################################################
    #### Cumulative Returns
    ####################################################################################################################
    weekly_returns = pd.concat([np.exp(week_scores_df['total_return'][list(map(portfolio_renames.get, lo_portfolios))]),
                                week_scores_df.reorder_levels([1, 0], axis=1)['Equal Weight']['risk_free']], axis=1)

    name_map = {
        'lo_rp_ml': 'True Risk-Parity using makimum-likelihood estimated matrix',
        'lo_rp_ewma': 'True Risk-Parity using EWMA estimated matrix',
        'lo_rp_ldw': 'True Risk-Parity using Ledoit-Wolf estimated matrix',
        'lo_ivp_ml': 'Inverse Volatility using makimum-likelihood estimated matrix',
        'lo_ivp_ewma': 'Inverse Volatility using EWMA estimated matrix',
        'lo_ivp_ldw': 'Inverse Volatility using Ledoit-Wolf estimated matrix',
        'lo_hrp_ml': 'Hierarquical Risk Parity using makimum-likelihood estimated matrix',
        'lo_hrp_ewma': 'Hierarquical Risk Parity using EWMA estimated matrix',
        'lo_hrp_ldw': 'Hierarquical Risk Parity using Ledoit-Wolf estimated matrix',
        'equal_weight': 'Equal Weight',
        'risk_free': 'Bank Prime Loan Rate',
    }

    return_tables = []
    for p in weekly_returns.columns:
        # Ensure datetime index
        returns_df = weekly_returns[[p]].copy()
        returns_df.index = pd.to_datetime(returns_df.index)

        # Convert weekly returns to monthly compounded returns
        monthly_returns = returns_df.resample('ME').prod() - 1
        monthly_returns['month'] = monthly_returns.index.strftime('%b')
        monthly_returns['year'] = monthly_returns.index.strftime('%Y')

        monthly_returns_pivot = monthly_returns.pivot_table(index='year', columns='month', values=p)

        # Ensure 12 months and order
        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns_pivot = monthly_returns_pivot.reindex(all_months, axis=1)

        # Calculate full-year return
        yearly_returns = monthly_returns_pivot.apply(lambda x: (x + 1).prod() - 1, axis=1)
        monthly_returns_pivot['Year'] = yearly_returns
        monthly_returns_pivot.loc['Mean'] = monthly_returns_pivot.mean()


        float_fmt = "{:.2%}"
        return_tables.append(monthly_returns_pivot.map(lambda x: float_fmt.format(x)).to_latex(
            index=True,
            caption=name_map.get(p) if p == 'risk_free' else f"{name_map.get(p)} Portfolio Monthly and Yearly Returns",
            label=f"tab:{p}_monthly_returns",
            column_format="l|" + "c" * (monthly_returns_pivot.shape[1] - 1) + '|c',
            escape=True,
            bold_rows=True,
        ))


    print(('\n' + '%' * 50 + '\n').join([x.replace('\\end{table}\n',
                                                   '\\end{table}\n\\end{landscape}\n'
                                         ).replace('\\begin{table}\n',
                                                   '\\begin{landscape}\n\\begin{table}\n'
                                         ).replace('\n\\textbf{Mean}',
                                                   '\n\\midrule\n\\textbf{Mean}')
                                         for x in return_tables]))

    cumulative = (weekly_returns).cumprod()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative.plot(ax=ax)
    ax.set_title("Cumulative Return Comparison")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.grid(True)
    plt.tight_layout()

    # Export to file
    plt.savefig('out/images/cumulative_returns_all2.png', format='png')
    plt.close()

    g_names = {
        0: 'Risk Parity',
        1: 'Inverse Variance',
        2: 'Hierarchical Risk Parity',
    }


    for i, group in enumerate([
        list(map(portfolio_renames.get, rp_lo_portfolios)),
        list(map(portfolio_renames.get, ivp_lo_portfolios)),
        list(map(portfolio_renames.get, hrp_lo_portfolios)),
        ]):
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        cumulative[group].plot(ax=ax)
        ax.set_title(f"Cumulative Return Comparison - {g_names[i]} Estimated Covariance Matrix")
        ax.set_ylabel("Cumulative Return")
        ax.set_xlabel("Date")
        ax.grid(True)
        plt.tight_layout()

        # Export to file
        plt.savefig(fr'out/images/{g_names[i]}_cumulative_returns.png', format='png')
        plt.close()

    weekly_returns_corr = weekly_returns.copy()
    weekly_returns_corr.columns = weekly_returns_corr.columns.str.removesuffix(' covariance').str.replace('using', '|')
    corr = (weekly_returns_corr.drop('risk_free', axis=1) - 1).corr()

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.9)

    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0.95, vmax=1.0,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'}
    )

    plt.title("Weekly Return Correlation Matrix")
    plt.tight_layout()
    plt.savefig(fr'out/images/weekly_returns_correlation.png', format='png')
    plt.close()


    ####################################################################################################################
    #### Covar Matrix Mean Squared Error
    ####################################################################################################################
    cvmx_eval_rmse.columns.name = 'method'
    cvmx_eval_rmse.index.name = 'date'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cvmx_eval_rmse.plot(ax=ax)
    ax.set_title("Covariance Matrix Root Mean Squared Error")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Date")
    ax.grid(True)
    plt.tight_layout()

    # Export to file
    plt.savefig('out/images/cvmx_eval_rmse.png', format='png')
    plt.close()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cvmx_eval_rmse.drop('ewma_cov_matrix', axis=1).plot(ax=ax)
    ax.set_title("Covariance Matrix Root Mean Squared Error - Just Ledoit-Wolf and Maximum Likelihood")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Date")
    ax.grid(True)
    plt.tight_layout()

    # Export to file
    plt.savefig('out/images/cvmx_eval_rmse_no_ewma.png', format='png')
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Primary axis for main_cols
    cvmx_eval_rmse.drop('ewma_cov_matrix', axis=1).rename(columns={
        'ledoit_wolf_cov_matrix': 'Ledoit-Wolf shrunk',
        'ml_cov_matrix': 'Maximum likelihood',
    }).plot(ax=ax1)
    ax1.set_ylabel("RMSE")
    ax1.set_title("Covariance Matrix Root Mean Squared Error")

    # Secondary axis for standout variable
    ax2 = ax1.twinx()
    cvmx_eval_rmse['ewma_cov_matrix'].plot(ax=ax2, color='green', label='Exponentially weighted [rhs]')
    ax2.set_ylabel(f"Exponentially weighted (Secondary Axis)")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    # Export to file
    plt.savefig('out/images/cvmx_eval_rmse_2y_axis.png', format='png')
    plt.close()


####################################################################################################################
    #### Covar Matrix Median Absolute Error
    ####################################################################################################################

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Primary axis for main_cols
    cvmx_eval_medae.drop('ewma_cov_matrix', axis=1).plot(ax=ax1)
    ax1.set_ylabel("MedAE")
    ax1.set_title("Covariance Matrix Median Absolute Error")

    # Secondary axis for standout variable
    ax2 = ax1.twinx()
    cvmx_eval_medae['ewma_cov_matrix'].plot(ax=ax2, color='green', label='ewma_cov_matrix [rhs]')
    ax2.set_ylabel(f"ewma_cov_matrix (Secondary Axis)")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    # Export to file
    plt.savefig('out/images/cvmx_eval_medae_2y_axis.png', format='png')
    plt.close()

    ####################################################################################################################
    #### Correlation between Covar Matrix Median Absolute Deviation and Returns
    ####################################################################################################################

    correl_medae_df['sharpe']
    correl_medaes_ci_df['sharpe']

    ####################################################################################################################
    #### Correlation between Covar Matrix Median Absolute Deviation and Vol
    ####################################################################################################################

    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    # Predefine structures
    correl_mses = {}
    correl_medaes = {}
    correl_mses_ci = {}
    correl_medaes_ci = {}
    correl_mses_pvals_adj = {}
    correl_medaes_pvals_adj = {}

    # Collect all p-values first for multiple testing correction
    pvals_mse = []
    pvals_medae = []
    correlation_records_mse = []
    correlation_records_medae = []

    # First loop: collect all correlation values and p-values
    for metric in lo_return_metrics:

        for cov_mx_g, ps in cov_mx_groups.items():
            ls_df = week_scores_df[metric][ps].copy()
            if 'return' in metric:
                ls_df = ls_df - 1

            ls_df['cov_mx_mse'] = cvmx_eval_rmse[cov_mx_g].to_list()
            ls_df['cov_mx_medae'] = cvmx_eval_medae[cov_mx_g].to_list()

            for p in ps:
                x = ls_df[p]
                y_mse = ls_df['cov_mx_mse']
                y_medae = ls_df['cov_mx_medae']

                r_mse, pval_mse = pearsonr(x, y_mse, alternative='less' if metric == 'total_return' else 'greater')
                r_medae, pval_medae = pearsonr(x, y_medae, alternative='less' if metric == 'total_return' else 'greater')

                pvals_mse.append(pval_mse)
                pvals_medae.append(pval_medae)

                correlation_records_mse.append((metric, cov_mx_g, p, r_mse))
                correlation_records_medae.append((metric, cov_mx_g, p, r_medae))

    # Apply FDR correction
    _, pvals_mse_adj, _, _ = multipletests(pvals_mse, alpha=0.05, method='fdr_by')
    _, pvals_medae_adj, _, _ = multipletests(pvals_medae, alpha=0.05, method='fdr_by')

    # Organize results into DataFrames
    df_mse = pd.DataFrame(correlation_records_mse, columns=["metric", "cov_group", "portfolio", "corr"])
    df_mse["pval_adj"] = pvals_mse_adj

    df_medae = pd.DataFrame(correlation_records_medae, columns=["metric", "cov_group", "portfolio", "corr"])
    df_medae["pval_adj"] = pvals_medae_adj

    df_mse_pivot = df_mse.pivot_table(index="metric", columns=["cov_group", "portfolio"], values=["corr", "pval_adj"])
    df_medae_pivot = df_medae.pivot_table(index="metric", columns=["cov_group", "portfolio"], values=["corr", "pval_adj"])

    df_mse_pivot.columns.names = ["type", "cov_group", "portfolio"]
    df_medae_pivot.columns.names = ["type", "cov_group", "portfolio"]

    float_fmt = "{:.3f}"
    latex_code_rmse_corr = df_mse_pivot['corr'].map(lambda x: float_fmt.format(x)).to_latex(
        index=True,
        escape=True,
        caption="Correlation between metrics and Covariance Matrix RMSE",
        label="tab:corr_rmse",
        column_format="l|ccc|ccc|ccc",
        multicolumn=True,
        multicolumn_format='c'
    )

    latex_code_rmse_pvalue = df_mse_pivot['pval_adj'].map(lambda x: float_fmt.format(x)).to_latex(
        index=True,
        escape=True,
        caption="Correlation $p$-values between metrics and Covariance Matrix RMSE",
        label="tab:corr_rmse",
        column_format="l|ccc|ccc|ccc",
        multicolumn=True,
        multicolumn_format='c'
    )

    ####################################################################################################################
    #### Correlation between Covar Matrix Median Absolute Deviation and Adjusted Sharpe
    ####################################################################################################################

    ####################################################################################################################
    #### Correlation between Covar Matrix Median Absolute Deviation and Betas
    ####################################################################################################################

    ####################################################################################################################
    #### Correlation between Portfolios
    ####################################################################################################################


    print('done')