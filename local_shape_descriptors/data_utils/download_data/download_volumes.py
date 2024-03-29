"""Full tutorial to be devised after sign-up to aws is complete!!
Tutorial to include:
1. AWS Signup process
2. Parsing keys for each volume type. Potentially port some code from Arlo's lsd_nm_experiments.

"""
import boto3
import bson
import numpy as np
import os
import pandas as pd
import zarr
from cloudvolume import CloudVolume

# set bucket credentials
access_key = 'put_your_key'
secret_key = 'put_your_key'
bucket = 'open-neurodata'

session = boto3.session.Session(
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

client = session.client('s3')

# connect to client
# client = session.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

# list data
print(client.list_objects(Bucket=bucket, Prefix="funke"))

# download directory structure file - this shows exactly how the s3 data is stored
if not os.path.exists("./structure.md"):
    client.download_file(
        Bucket=bucket,
        Key="funke/structure.md",
        Filename="structure.md")

# visualize the directory structure to fetch selectively/custom use
with open('./structure.md', 'r') as f:
    text = f.read()
    html = markdown.markdown(text)

with open('structure.html', 'w') as f:
    f.write(html)


# function to download all files nested in a bucket path
def download_directory(
        bucket_name,
        path,
        access_key,
        secret_key):
    resource = boto3.resource(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key)

    bucket = resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=path):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))

        key = obj.key

        print(f'Downloading {key}')
        bucket.download_file(key, key)


if __name__ == '__main__':
    # download example hemi/zebrafinch/fib25 training data
    # download_directory(
    #     bucket,
    #     'funke/hemi/testing/ground_truth/data.zarr',
    #     access_key,
    #     secret_key)
    pass
