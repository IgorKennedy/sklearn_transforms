from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

    
rm_columns = DropColumns(
    columns=["NOME" , "MATRICULA" , "FALTAS" , "INGLES" , "H_AULA_PRES" , "TAREFAS_ONLINE"]  )

print(rm_columns)




rm_columns.fit(X=df_data_1)


df_data_2 = pd.DataFrame.from_records(
    data=rm_columns.transform(
        X=df_data_1
    ),
)





si = SimpleImputer(
    missing_values=np.nan,  
    strategy='constant',  
    fill_value=0,  
    verbose=0,
    copy=True
)




si.fit(X=df_data_2)


df_data_3 = pd.DataFrame.from_records(
    data=si.transform(
        X=df_data_2
    ),  
    columns=df_data_2.columns  # as colunas originais devem ser conservadas nessa transformação
)




features = [
    'REPROVACOES_DE', 'REPROVACOES_EM', "REPROVACOES_MF", "REPROVACOES_GO",
    "NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO" ,
]


target = ["PERFIL"]


X = df_data_3[features]
y = df_data_3[target]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=337)




dtc_model = DecisionTreeClassifier(max_depth=3 , min_samples_split=2 , min_samples_leaf=2)



dtc_model.fit(
    X_train,
    y_train
)



y_pred = dtc_model.predict(X_test)



from sklearn.metrics import accuracy_score


print("Acurácia: {}%".format(100*round(accuracy_score(y_test, y_pred), 2))) 
