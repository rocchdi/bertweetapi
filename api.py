import uvicorn
from fastapi import FastAPI
from fastapi import Request,Query
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from transformers import TFRobertaModel



#api de prediction
api = FastAPI(
    title='API Prediction',
    description="API de prediction des tweet de Air Paradise",
    version="1.0.1",
    openapi_tags=[
    {
        'name': 'Home',
        'description': 'Welcome message function'
    },
    {
        'name': 'prediction',
        'description': 'fonction qui permet de soumettre un tweet à un modele de prediction et obtenir le sentiment '
    }
]
    )




#load the model
model = tf.keras.models.load_model('bertweet_model.h5', custom_objects={'TFRobertaModel':TFRobertaModel})

MAX_LEN=128

def tokenize_data(data,tokenizer,max_len=MAX_LEN):
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)







##########################
#routes definition
##########################
@api.get('/', name='Home', tags=['Home'])
async def get_index():
    """Returns a welcome message
    """
    return {"message": "Je suis prêt à analyser votre tweet "}




# ========== Prédition pour un tweet ==========
@api.get("/prediction", name='prediction avec un modele avance BERT', tags=['prediction'])
def get_predict(request:Request, twt:str = Query(None)):
    """le  user lance  une prediction 
    """
 
    received = [twt]

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    pred_input_ids, pred_attention_masks = tokenize_data(received, tokenizer, MAX_LEN)

    sentiment = model.predict([pred_input_ids,pred_attention_masks], batch_size=1)[0]

    #print('received=',received)
    #print(pred_input_ids, pred_attention_masks)
    #print('sentiment=',sentiment)


    if (np.argmax(sentiment) == 0):
        return {'prediction': "Negative"}
    else:
        return {'prediction': "Positive"}
