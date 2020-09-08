import datetime

from dsflow.inference.processing.abstracts import Scoringfactory

from {{cookiecutter.project_name}}.train.outils import typecolumns


class Process_{{cookiecutter.project_name}}(Scoringfactory):

    @classmethod
    def score(self, loaded_model, df_to_score, extra_args=None):
        """
            from train phase we saved model as :
           dict_model = dict()
           dict_model['missing_values'] = dict_NA
           dict_model['model_features'] = list(rf_var_select)
           dict_model['model_rf'] = rf_final
        """
        df_to_score = df_to_score[['id_operation', 'lib_transaction', 'credit', 'debit']]
        df_to_score = typecolumns(df_to_score)

        # predict categorie
        predtect_categorie = loaded_model["model_catego"].predict(df_to_score)

        # add predict categorie to df
        df_to_score["lib_categorie"] = predtect_categorie

        # predict under category
        pred_sous_categorie = loaded_model["model_sous_catego"].predict(df_to_score)

        # add predict under category to df
        df_to_score["lib_sous_categorie"] = pred_sous_categorie
        scored_df = df_to_score.drop(['credit_o_n'], axis='columns')
        return scored_df

    @classmethod
    def get_schema(cls):
        schema = {"fields": [{"metadata": {}, "name": "id_operation", "nullable": True, "type": "integer"},
                             {"metadata": {}, "name": "lib_transaction", "nullable": True, "type": "string"},
                             {"metadata": {}, "name": "credit", "nullable": True, "type": "float"},
                             {"metadata": {}, "name": "debit", "nullable": True, "type": "float"},
                             {"metadata": {}, "name": "lib_categorie", "nullable": True, "type": "string"},
                             {"metadata": {}, "name": "lib_sous_categorie", "nullable": True, "type": "string"}
                             ]}

        return schema

    @classmethod
    def get_score_name(cls):
        return "{{cookiecutter.project_name}}"

    @classmethod
    def get_example(self):
        # example for score client with api
        return [{"credit": "", "debit": -13.99, "id_operation": 199,
                 "lib_transaction": "PRLV SEPA SPB DEB-SPB-876-20181228-000026657097 ASSURANCE MOBILE SPB BOUYGUES TELECOM",
                 "lib_categorie": "UNKNOWN"}]

    @classmethod
    def get_score_date(cls):
        return datetime.date.today().strftime("%d-%m-%Y")
