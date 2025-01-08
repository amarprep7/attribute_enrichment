
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Hash import SHA256
from base64 import b64encode, b64decode 

import openai
import ssl
import httpx
from typing import Dict
import time
import os
import io
import requests
import re
import ast
import datetime
import pytz
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import json
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import concurrent

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--part_num', type=str, required=True)
parser.add_argument('--run_id', type=str, required=True)
args = parser.parse_args()

part_num=int(args.part_num)
run_id=str(args.run_id)
# 0 1200
# 1200 2400
# 2400 3600
# 3600 4800
def get_part_limit(part_num):
    pt_batch_size=1200
    start= (part_num-1)*pt_batch_size
    end=part_num*pt_batch_size
    print(start, end)
    return start, end
    
ptnumber_start,ptnumber_end= get_part_limit(part_num)
# ptnumber_start,ptnumber_end= get_part_limit(2)
# ptnumber_start,ptnumber_end= get_part_limit(3)
# ptnumber_start,ptnumber_end= get_part_limit(4)
print(f'running for part_num-{part_num}, ptnumber {ptnumber_start}  - {ptnumber_end}')

client = bigquery.Client()
# import sys
# sys.exit()

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

    
#run_id = '8'
image = 'no'
attribute_rules = 'Selected'
NUM_THREADS = 240
version = 'ae_result_v3_v2'
market = 'mx'

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###




input_table = "attribute_extraction_llm.ae_input_table_v2"
catalog_table = "intl_catalog.tenant_mx_catalog"
target_location = "attribute_extraction_llm.ae_result_v3"
finetune_location = "attribute_extraction_llm.ae_finetune_data_v3"
track_location = "attribute_extraction_llm.ae_track_v1"


model_name = 'gpt-4o-mini'
model_version = '1'

required_columns = ['brand', 'clothing_size', 'clothing_size_group', 'color', 'gender', 'maximum_recommended_age', 'minimum_recommended_age', 
 'size', 'tire_diameter', 'tire_size', 'tire_width']

config = {
    'model_name' : model_name,
    'endpoint' : 'https://openai',
    'api_version' : '2024-02-01',
    'temperature' : 0
}

import warnings
warnings.filterwarnings('ignore')

if model_name == 'gpt-4o-mini':
    key_path = 'private_key_4o_mini.pem'
else:
    key_path = 'private_key_4o.pem'


with open(key_path, 'rb') as f:
    private_key = f.read()

bucket_name = 'gs://bucket'
parquet_path = f'AE/GenAI/GPT_4o/poc_{model_name}_{attribute_rules}_{run_id}'
blob_path = 'AE/GenAI'

PT_attribute_path = f"{bucket_name}/{blob_path}/ae_attribute_spec_{market}.parquet"

df_attr = pd.read_parquet(PT_attribute_path, engine='pyarrow')

df_attr['product_type'] = df_attr['product_type'].str.lower().str.strip()
df_attr['taxonomy_key'] = df_attr['taxonomy_key'].str.lower().str.strip()


def create_product_info(row):
    return """ Product Title:{},\n 
            Product short Desciption:{},\n
            Product Long Description : {}""".format(row['product_title'], row['product_short_desc'], row['product_long_desc'])

def get_attr_prompt(df,pt_name,attribute_rules,attribute_list=None):
    if attribute_rules == 'Selected':
        filtered_df = df_attr[(df_attr['product_type'] == pt_name.lower().strip()) & (df_attr['taxonomy_key'].isin(attribute_list))]
        filtered_df['output_example'] = filtered_df['attribute'] + ': ""'
        df['output_example'] = ','.join(filtered_df['output_example'].astype(str))
        df['attribute_prompt'] = ','.join(filtered_df['attribute_prompt_en'].astype(str))
    elif attribute_rules == 'Required':
        filtered_df = df_attr[(df_attr['product_type'] == pt_name.lower().strip()) & (df_attr['requirement_level'] == attribute_rules)]
        filtered_df['output_example'] = filtered_df['attribute'] + ': ""'
        df['output_example'] = ','.join(filtered_df['output_example'].astype(str))
        df['attribute_prompt'] = ','.join(filtered_df['attribute_prompt_en'].astype(str))
        attribute_list = filtered_df['taxonomy_key'].astype(str).tolist()
    else:
        filtered_df = df_attr[(df_attr['product_type'] == pt_name.lower().strip())]
        filtered_df['output_example'] = filtered_df['attribute'] + ': ""'
        df['output_example'] = ','.join(filtered_df['output_example'].astype(str))
        df['attribute_prompt'] = ','.join(filtered_df['attribute_prompt_en'].astype(str))
        attribute_list = filtered_df['taxonomy_key'].astype(str).tolist()
    return df,attribute_list

def create_prompt(row):
    base_prompt = f'''You are an experienced Category Manager leading the {row['product_type']} department within
    Walmart and you are able to do attribute extraction easily from the product description.'''

    image_prompt = ''
    if (pd.notna(row['main_image']) or image != 'no'):
        if row['product_type'] is not None:
            if 'tire' in row['product_type'].lower():
                image_prompt = ''' For any attribute that are not there in product description, try to do attribute extraction from image.
                Analyze the provided image of a tire and extract the alphanumeric characters present on the tire or objects related to tire.
                Interpret these extracted characters according to standard tire code notations.
                For example, if you see "P225/65R15 91H", then it's a Passenger tire (P), 225mm wide, 65% aspect ratio, Radial construction (R),
                15-inch wheel diameter, with a load index of 91 and a speed rating of H.
                '''
            else:
                image_prompt = ''' For any attribute that are not there in product description, try to do attribute extraction from image.'''
        else:
            image_prompt = ''' For any attribute that are not there in product description, try to do attribute extraction from image.'''
            
    attributes_prompt = f''' Given the list of attributes {row['attribute']} and the product description
    {row['product_info']}, extract the attributes in JSON format. Your answer must always be in a 
    valid and parseable JSON format. Please carefully and strictly follow the extraction rule for 
    attributes present in the Rules section below: {row['attribute_prompt']}'''

    locale_prompt = ''
    if row['locale'] == 'es':
        locale_prompt = ''' Strictly return the response in Spanish (except if the example attribute values
        mentioned are in English)'''

    image_instruction = ''' Return response in JSON OBJECT like response example below. Strictly 
    don\'t return values that are not there in product description.response example: '''
    if (pd.notna(row['main_image']) or image != 'no'):
        image_instruction = ''' Return response in JSON OBJECT like 
        response example below. Strictly return only values either from product description or image.response example: '''
        
    output_example = row['output_example']

    full_prompt = f"{base_prompt}{image_prompt}{attributes_prompt}{locale_prompt}{image_instruction}{output_example}"

    return full_prompt

def get_ist_timestamp():
    """Returns the current timestamp in IST as a datetime object."""
    utc = pytz.utc
    now = datetime.datetime.now(utc)
    return now

def subtract_time_differences(time1, time2):
        time_difference = time2 - time1
        hours = time_difference.total_seconds() / 3600
        return hours

def create_openai_api_body(prompt,main_image):
    if (image == 'no' or (main_image is None or '' or not main_image)):
        body = json.dumps({
                    "model": config['model_name'],
                    "task": "chat/completions",
                    "model-params": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            },
                        ],
                        "temperature": config['temperature']
                    },
                })
        return body
    else:
        body = json.dumps({
                "model": config['model_name'],
                "task": "chat/completions",
                "model-params": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                               {"type": "image_url", "image_url": {"url": main_image,"detail": "high"}}, 
                            ],
                        },
                    ],
                    "temperature": config['temperature']
                },
            })
        return body

def create_openai_header():
    global url
    global headers
    url = "https://v1/openai"
    keyVersion='1'
    if model_name == 'gpt-4o-mini':
        consumer_id = "123"
    else:
        consumer_id = "1234"
    epoch_time = int(time.time()) * 1000
    data = consumer_id + '\n' + str(epoch_time) + '\n' + keyVersion + '\n'
    signature = {}
    rsakey = RSA.importKey(private_key) 
    signer = PKCS1_v1_5.new(rsakey) 
    digest = SHA256.new()
    digest.update(data.encode('utf-8')) 
    sign = signer.sign(digest) 
    signature['epoch_time'] = epoch_time
    signature['sign'] = b64encode(sign)

    headers = {"WM_CONSUMER.ID":consumer_id, 
               "WM_CONSUMER.INTIMESTAMP": str(signature['epoch_time']), 
               "WM_SEC.KEY_VERSION": str(keyVersion), 
               "WM_SEC.AUTH_SIGNATURE": signature['sign'].decode(), 
               "WM_SVC.NAME": "WMTLLMGATEWAY", 
               "WM_SVC.ENV": "stage",
              "Content-Type": "application/json"}
    return headers

def make_api_call(prompt,main_image):
    body = create_openai_api_body(prompt,main_image)
    headers = create_openai_header()
    response = requests.post(url, data=body, headers=headers, verify=False)
    if response.status_code != 200:
        response = requests.post(url, data=body, headers=headers, verify=False)
    #print(response.text)
    return response.text,response.status_code

def process_output(model_response):
    try:
        model_response = json.loads(model_response)
        content_string = model_response['choices'][0]['message']['content']
        content_string = content_string.replace('```json\n', '').replace('\n```', '')
        return content_string
    except:
        None

def get_attributes(df,attribute_list):
    #attribute_list = df['attribute'][0].split(",")

    for attr in attribute_list:
        try:
            df[attr] = df['response'].apply(lambda x: json.loads(x).get(attr))
        except:
            df[attr] = ""
        
    return df

def finetune_columns(finetune_df):
    columns_to_drop = ["prompt", "tenant", "locale", "main_image", "image","batch","pt_number", "output_example", "response_all", "response"]
    finetune_df = finetune_df.drop(columns=columns_to_drop)

    for column in required_columns:
        if column not in finetune_df.columns:
            finetune_df[column] = None

    finetune_df["update_ts"] = start_timestamp
    return finetune_df

def clean_and_transform_data(tmp_df):
    columns_to_drop = ["prompt", "product_info", "tenant", "locale", "attribute", "product_title", 
                       "product_short_desc", "product_long_desc", "main_image",  "attribute_prompt",
                       "output_example", "response_all", "response",]
    tmp_df = tmp_df.drop(columns=columns_to_drop)
    
    attribute_columns = [col for col in tmp_df.columns if col not in ['item_id', 'wpid', 'gtin', 
                                                                      'product_type', 'model_nm', 
                                                                      'model_version', 'run_id', 'run_ts',"response_code","batch","pt_number"]]
    rows = []
    for index, row in tmp_df.iterrows():
        for attribute_column in attribute_columns:
            value = row[attribute_column]
            rows.append([row['run_id'], row['run_ts'], 
                         row['item_id'], row['wpid'], row['gtin'], row['product_type'],
                         row['model_nm'], row['model_version'],row["response_code"],row["batch"],row["pt_number"],
                         attribute_column, value])

    new_df = pd.DataFrame(rows, columns=['run_id','run_ts','item_id', 'wpid','gtin','product_type', 
                                         'model_nm','model_version',"response_code","batch","pt_number",'taxonomy_key', 'value'])
    new_df['value'] = new_df['value'].replace(["N/A", "", "None"], np.nan)

    new_df['update_ts'] = start_timestamp

    return new_df

def ae_function(tmp_df,attribute_list,thread_num):
    tmp_df[['response_all', 'response_code']] = tmp_df.apply(
    lambda row: pd.Series(make_api_call(row["prompt"], row["main_image"])), axis=1)
    #print(tmp_df["response_all"])
    tmp_df["response"] = tmp_df["response_all"].apply(process_output)
    tmp_df = get_attributes(tmp_df,attribute_list)
    new_df = clean_and_transform_data(tmp_df)
    return new_df

def concurrency_creator(df, func,attribute_list, NUM_THREADS):
    df_split = np.array_split(df, NUM_THREADS)
    output_df_dict = dict()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executors:
        futures = {
            executors.submit(func, thread_df,attribute_list, thread_num=i): i
            for i, thread_df in enumerate(df_split) if len(thread_df) > 0
        }
        # Process completed tasks as they finish
        for future in concurrent.futures.as_completed(futures):
            batch_id = futures[future]
            output_df_dict[batch_id] = future.result()
    out_df = pd.concat(output_df_dict[i] for i in output_df_dict.keys())
    return out_df

def input_data_processing(df):
    try:
        df['model_nm'] = model_name
        df['model_version'] = model_version
        current_batch = df['batch'][0]
        current_pt = df['pt_number'][0]
        run_ts = df['run_ts'][0]
        locale = df['locale'][0]
        pt_name = df['product_type'][0]
        #print(pt_name)
        if pd.isnull(df['attribute'][0]):
            attribute_list = None
        else:
            attribute_list = df['attribute'][0].split(",")
        df['product_info'] = df.apply(create_product_info, axis=1)
        df,attribute_list = get_attr_prompt(df,pt_name,attribute_rules,attribute_list)
        df['prompt'] = df.apply(create_prompt, axis=1)
        predicted_df = concurrency_creator(df, ae_function,attribute_list, NUM_THREADS=NUM_THREADS)
        predicted_df['value'] = predicted_df['value'].astype(str)
        predicted_df.to_parquet(f"{bucket_name}/{parquet_path}/{version}/pt_{current_pt}_batch_{current_batch}.parquet", index=False)
        return predicted_df
    except Exception as e:
        print(f"Error : {str(e)}, Failed for {pt_name} product_type")
        data = [(run_id, "Failed",None, None, start_timestamp, subtract_time_differences(start_timestamp,get_ist_timestamp()), f"Error at batch {current_batch}, pt {current_pt}: {str(e)}, Failed for {pt_name} product_type")]
        track_df = create_track_df(data)
        client.load_table_from_dataframe(track_df, track_location).result()

def input_data_processing1(df):
    df['model_nm'] = model_name
    df['model_version'] = model_version
    current_batch = df['batch'][0]
    current_pt = df['pt_number'][0]
    run_ts = df['run_ts'][0]
    locale = df['locale'][0]
    pt_name = df['product_type'][0]
    #print(pt_name)
    if pd.isnull(df['attribute'][0]):
        attribute_list = None
    else:
        attribute_list = df['attribute'][0].split(",")
    df['product_info'] = df.apply(create_product_info, axis=1)
    df,attribute_list = get_attr_prompt(df,pt_name,attribute_rules,attribute_list)
    df['prompt'] = df.apply(create_prompt, axis=1)
    predicted_df = concurrency_creator(df, ae_function,attribute_list, NUM_THREADS=NUM_THREADS)
    predicted_df['value'] = predicted_df['value'].astype(str)
    predicted_df.to_parquet(
            f"{bucket_name}/{parquet_path}/{version}/pt_{current_pt}_batch_{current_batch}.parquet", index=False
        )
    #client.load_table_from_dataframe(predicted_df, target_location).result()
    return predicted_df

query_batch = f"""
select pt_number,max(batch) as batch,count(*) as count,
from {input_table} where run_id = '{run_id}'
group by pt_number
"""


print('**** Data Loaded***')

batch_df = client.query(query_batch).result().to_dataframe()

batch_df=batch_df[(batch_df['pt_number']>=ptnumber_start)&(batch_df['pt_number']<ptnumber_end)]
# ptnumber_start,ptnumber_end
print(batch_df['pt_number'].describe())
batch_df = batch_df.sort_values(by='pt_number')
batch_df = batch_df.reset_index(drop=True)

print('**** batch_df Sample***')
print(batch_df.head())

print(batch_df['pt_number'].min())
print(batch_df['pt_number'].max())
# import sys
# sys.exit()

batch_df_len = len(batch_df)

item_count = batch_df['count'].sum()
print(f'item_count-{item_count}')

schema = {
    'run_id': 'string',
    'Status': 'string',
    'pt_count': 'string',
    'item_count': 'string',
    'update_ts': 'datetime64[ns, UTC]',
    'complete_ts': 'float',
    'comments': 'string'
}

def create_track_df(data):
    return pd.DataFrame(data, columns=list(schema.keys())).astype(schema)

start_timestamp = get_ist_timestamp()
print(f"Current timestamp in IST: {start_timestamp}")

data = [(run_id, "Started",batch_df_len, item_count, start_timestamp, None, f"Started for {batch_df_len} product_type")]
track_df = create_track_df(data)
client.load_table_from_dataframe(track_df, track_location).result()




for  i, row in batch_df.iterrows():
    pt_num, batch_count =row['pt_number'], row['batch']
    # import sys
    # sys.exit()
    for j in range(batch_count):
        query = f"""select * from {input_table} where run_id = '{run_id}' and batch = {j+1} and pt_number = {pt_num}"""
        df = client.query(query).result().to_dataframe()
        print(f" pt_number - {pt_num} , batch={j+1}  , {df.shape} ")
        df = input_data_processing(df)

end_timestamp = get_ist_timestamp()
print(f"Current timestamp in IST: {end_timestamp}")

run_time = subtract_time_differences(start_timestamp,end_timestamp)

data = [(run_id, "Completed",batch_df_len, item_count, end_timestamp, run_time, f"Completed for {batch_df_len} product_type")]
track_df = create_track_df(data)
client.load_table_from_dataframe(track_df, track_location).result()

df.head()

job_config = bigquery.LoadJobConfig()
job_config.source_format = bigquery.SourceFormat.PARQUET
job_config.autodetect = True

uri = f"{bucket_name}/{parquet_path}/{version}/*.parquet"

job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET)
load_job = client.load_table_from_uri(
           uri, f"{target_location}", job_config=job_config
       )
load_job.result()







