from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)
from fastapi.responses import FileResponse
import csv
from pydantic import BaseModel
from typing import List
from enum import Enum
from typing import Optional
import uvicorn
import joblib
import os
import pandas as pd
from contextlib import asynccontextmanager
from fastapi.encoders import jsonable_encoder

MODEL_PATH = 'models/'
DATA_PATH = 'data/'


def pydantic_model_to_df(pydantic_model):
    return pd.DataFrame([jsonable_encoder(pydantic_model)])


class Fuel(str, Enum):
    petrol = 'Petrol'
    diesel = 'Diesel'
    cng = 'CNG'
    lpg = 'LPG'


class SellerType(str, Enum):
    dealer = 'Dealer'
    individual = 'Individual'
    trustmark_dealer = 'Trustmark Dealer'


class Transmission(str, Enum):
    manual = 'Manual'
    automatic = 'Automatic'


class Owner(str, Enum):
    first = 'First Owner'
    second = 'Second Owner'
    third = 'Third Owner'
    fourth = 'Fourth & Above Owner'
    test_driven = 'Test Drive Car'


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: Fuel
    seller_type: SellerType
    transmission: Transmission
    owner: Owner
    mileage: Optional[str] = None
    engine: Optional[str] = None
    max_power: Optional[str] = None
    torque: Optional[str] = None
    seats: Optional[str] = None


class Items(BaseModel):
    objects: List[Item]


class ItemResponse(Item):
    prediction: float


class ItemsResponse(Items):
    predictions: List[float]


def ridge_ext_regressor(x: pd.DataFrame) -> List[float]:
    with open(os.path.join(MODEL_PATH, 'preprocess_pl.pkl'), 'rb') as preprocess_file:
        preprocess_pl = joblib.load(preprocess_file.name)
    with open(os.path.join(MODEL_PATH, 'gs_ridge_pl.pkl'), 'rb') as model_file:
        model = joblib.load(model_file.name)
    prediction = model.predict(preprocess_pl.transform(x)).tolist()
    return prediction


ml_models = {}


@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["ridge_ext_regressor"] = ridge_ext_regressor
    yield
    ml_models.clear()


app = FastAPI(lifespan=ml_lifespan_manager)


@app.get("/")
async def root():
    return {
        "Name": "Car price prediction",
        "description": "This is a car price prediction model based "
                       "on the model trained on the dataset with Ridge regressor",
    }


@app.post('/predict_item')
async def predict_item(item: Item) -> float:
    return ml_models["ridge_ext_regressor"](pydantic_model_to_df(item))[0]


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        file_path = os.path.join(DATA_PATH, 'loaded_' + file.filename)
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    with open(file_path, 'r') as csvf:
        csvReader = csv.DictReader(csvf)
        items = [Item.model_validate(row).model_dump() for row in csvReader]
    df_items = pd.DataFrame(items).replace({'': None})
    df_items['predictions'] = ml_models["ridge_ext_regressor"](df_items)
    file_path = os.path.join(DATA_PATH, 'data_with_predictions.csv')
    df_items.to_csv(file_path, index=False)
    headers = {'Content-Disposition': f'attachment; filename="data_with_predictions.csv"'}
    return FileResponse(file_path, headers=headers, media_type="text/csv")


# to start
# uvicorn main:app --reload --port 8000
# or in collab
# !uvicorn main:app & npx localtunnel --port 8000 --subdomain fastapi & wget -q -O - https://loca.lt/mytunnelpassword
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
