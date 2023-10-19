import pandas as pd 
import joblib 
import json
import pickle
import pathlib


categorical_columns = ['MS.SubClass',
 'MS.Zoning',
 'Land.Contour',
 'Lot.Config',
 'Neighborhood',
 'Bldg.Type',
 'House.Style',
 'Roof.Style',
 'Mas.Vnr.Type',
 'Foundation',
 'Bsmt.Qual',
 'Bsmt.Cond',
 'Bsmt.Exposure',
 'BsmtFin.Type.1',
 'BsmtFin.Type.2',
 'Central.Air',
 'Garage.Type',
 'Garage.Finish',
 'Sale.Type',
 'Sale.Condition',
 'Condition',
 'Exterior']

ordinal_columns = ['Lot.Shape',
 'Land.Slope',
 'Overall.Qual',
 'Overall.Cond',
 'Exter.Qual',
 'Exter.Cond',
 'Heating.QC',
 'Electrical',
 'Kitchen.Qual',
 'Functional',
 'Paved.Drive',
 'Fence']



DATA_DIR = pathlib.Path.cwd().parent / 'data'
print(DATA_DIR)


clean_data_path = DATA_DIR / 'processed' / 'ames_clean.pkl'


with open(clean_data_path, 'rb') as file:
    data = pickle.load(file)

# read a json file and transform it into a dataframe

# with open('testee.json') as json_file:
#     data = json.load(json_file)

# data = pd.DataFrame(data, index=[0])

# print(data)

model_data = data.copy()

for col in ordinal_columns:
    codes, _ = pd.factorize(data[col], sort=True)
    model_data[col] = codes


model_data = pd.get_dummies(model_data, drop_first=True)

for cat in categorical_columns:
    dummies = []
    for col in model_data.columns:
        if col.startswith(cat + "_"):
            dummies.append(f'"{col}"')
    dummies_str = ', '.join(dummies)
    print(f'From column "{cat}" we made {dummies_str}\n')

modelo_carregado = joblib.load("regression_model.joblib")


print(model_data)
# model_data = model_data.drop('SalePrice')
prediction = modelo_carregado.predict(model_data.to_numpy().reshape(1, -1))


print(prediction)