dt_conf = {
    'file_path': './data/paint_quality_assurance_data.csv',
    'file_type': 'csv',
    'target_feature': 'Dust Defect',
    'feature_list': ['Platform', 'Primer Color', 'Topcoat Color'],
    'transform_conf': [
        {
            'method_name': 'drop_redundent_columns',
            'params': {}
        },
        {
            'method_name': 'drop_null_columns',
            'params': {
                'null_percent_threshold': 0.2
            }
        },
        {
            'method_name': 'drop_unique_value_columns',
            'params': {}
        },
        {
            'method_name': 'data_imputation',
            'params': {}
        },
        {
            'method_name': 'feature_scaling',
            'params': {}
        },
        {
            'method_name': 'feature_transformer',
            'params': {}
        }
    ],
    'remove_outlier': True,
    'contamination_factor': 0.01,
    'save_path': './data/transform_pipeline.joblib'
}

fs_conf = {
    'file_path': './data/transformed_data_v1.csv',
    'file_type': 'csv',
    'target_feature': 'target_label',
    'run_parallel': True,

    'drop_low_variance_features': True,
    'variance_thresh': 0.3,

    'drop_high_corr_features': True,
    'corr_threshold': 0.8,
    'corr_method': 'pearson',

    'drop_multicolliner_features': True,

    'feature_select_conf': [
        {
            'select_method': 'anova_f_value_selection',
            'params': {
                'num_feat': 15
            }
        },

        {
            'select_method': 'mutual_info_classif_selection',
            'params': {
                'num_feat': 15
            }
        },

        {
            'select_method': 'logit_selection',
            'params': {
                'fit_method': None
            }
        },

        {
            'select_method': 'permutation_impt_selection',
            'params': {
                'model_list': ['all'],
                'num_feat': 15
            }
        },

        {
            'select_method': 'recursive_feature_elimination',
            'params': {
                'model_list': ['all'],
                'num_feat': 15,
                'step_value': None
            }
        },

        {
            'select_method': 'model_based_importance',
            'params': {
                'model_list': ['all'],
                'num_feat': 15,
            }
        },

        {
            'select_method': 'regularization_selection',
            'params': {
                'model_list': ['all'],
                'num_feat': 15,
            }
        },

        {
            'select_method': 'boruta_selection',
            'params': {
                'model_list': ['random_forest', 'lightgbm'],
            }
        },

        {
            'select_method': 'sequencial_forward_selection',
            'params': {
                'model_list': ['random_forest', 'lightgbm'],
                'num_feat': 15,
                'scoring_metric': None
            }
        }
    ],
    'test_size': 0.2,
    'save_path': './data/feature_selected_data_v1.joblib'
}
