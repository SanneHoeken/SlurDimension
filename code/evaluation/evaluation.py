import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency, pearsonr


def print_latex(results_df, average_results):

    if average_results:
        print('Experiment with multiple files encountered: printing average result values')
        mean_df = results_df.groupby('experiment', as_index=False).mean()
        results = mean_df.to_dict('records')
            
    else:
        results = results_df.to_dict('records')

    print("""

    \\begin{table*}[h]
    \\centering
    \\begin{tabular}{@{}c|c|ccccccc@{}}
    \\toprule
    \\multirow{3}{*}{[variable]} & Correlation & \\multicolumn{7}{c}{Classification report} \\\\ \\cmidrule(l){2-9} 
    & \\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Pearson's r\\\\ (* = sig.)\\end{tabular}} & \\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Acc.\\\\ (* = sig.)\\end{tabular}} 
    & \\multicolumn{3}{c}{Hateful} & \\multicolumn{3}{c}{Neutral} \\\\ \\cmidrule(l){4-9} 
    &  &  & Prec. & Recall & F1 & Prec. & Recall & F1 \\\\ \\midrule""")

    for result in results:
        
        experiment_name = str(result['experiment'])
        corr = str(round(result['correlation_r'], 3)) if correlation else '-'
        corr_sig = '*' if correlation and result['correlation_p'] < 0.05 else ''

        acc = str(round(result['accuracy'], 2))
        acc_sig = '*' if result['chi-square_p'] < 0.05 else ''

        per_class = [result['hateful_precision'], result['hateful_recall'], result['hateful_f1'], 
                    result['neutral_precision'], result['neutral_recall'], result['neutral_f1']]
        per_class_string = ' & '.join([str(round(x, 2)) for x in per_class])
        
        row = experiment_name + ' & ' + corr + corr_sig + ' & ' + acc + acc_sig + ' & ' + per_class_string + '\\\\'
        
        print(row)

    # END LATEX TABLE
    print("""\\bottomrule
    \\end{tabular}
    \\caption{[caption]}
    \\label{[lable]}
    \\end{table*}
    """)
    

def evaluate(experiment2files, gold_file, output_file, level, correlation, print_as_latex):
    
    results = []
    for experiment, files in experiment2files.items():
        for projections_file in files:
            df_projections = pd.read_csv(projections_file)
            df_projections['projection'] = df_projections['projection'].astype(float)
            
            df_gold = pd.read_csv(gold_file)
            df_gold['gold'] = df_gold['gold'].astype(int)

            if level == 'type' or correlation: # correlation is always on type level
                mean_projections = df_projections.groupby('LU')['projection'].mean()
                mean_gold = df_gold.groupby('LU')['gold'].mean()
                gold = [mean_gold[key] for key in mean_projections.keys()]
                projections = mean_projections.values

                if correlation:
                    corr, corr_p = pearsonr(gold, projections)
                else:
                    corr = None
                    corr_p = None
            
            if level == 'token':
                merged_df = pd.merge(df_projections, df_gold)
                projections = merged_df['projection']
                gold = merged_df['gold']

            # CLASSIFICATION PERFORMANCE
            projections_categorical = ['neutral' if x <= 0 else 'hateful' for x in projections]
            gold_categorical = ['neutral' if x == 0 else 'hateful' for x in gold]
            cm = confusion_matrix(gold_categorical, projections_categorical)
            tn, fp, fn, tp = cm.ravel()
            obs = [[tn, fp], [fn, tp]]
            _, acc_p, _, _ = chi2_contingency(obs)
            cr = classification_report(gold_categorical, projections_categorical, output_dict=True)
            
            result = {'experiment': experiment, 
                    'correlation_r': corr, 
                    'correlation_p': corr_p,
                    'accuracy': cr['accuracy'],
                    'chi-square_p': acc_p,
                    'hateful_precision': cr['hateful']['precision'],
                    'hateful_recall': cr['hateful']['recall'],
                    'hateful_f1': cr['hateful']['f1-score'],
                    'hateful_support': cr['hateful']['support'],
                    'neutral_precision': cr['neutral']['precision'],
                    'neutral_recall': cr['neutral']['recall'],
                    'neutral_f1': cr['neutral']['f1-score'],
                    'neutral_support': cr['neutral']['support']}
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file)
    
    if print_as_latex:
        average_results = any([len(x) > 1 for x in experiment2files.values()])
        print_latex(results_df, average_results)


if __name__ == '__main__':
    
    # FILES
    experiment2files = {
                        'persons': [f'../../output/projections/1-15slurs_cohyponym/1-15slurs_cohyponym_run{i}-hatexplain_persons-projections.csv' for i in range(10)],
                        'random nouns': [f'../../output/projections/1-15slurs_cohyponym/1-15slurs_cohyponym_run{i}-hatexplain_randomnouns-projections.csv' for i in range(10)]
                        }
    gold_file = '../../data/hatexplain/hatexplain_nouns_rationales.csv'
    output_file = '../../output/evaluations/1-15slurs_cohyponym/randomnouns-performance.csv'

    # SETTINGS
    level = 'token'
    correlation = True
    print_as_latex = True

    evaluate(experiment2files, gold_file, output_file, level, correlation, print_as_latex)