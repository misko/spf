#!/usr/bin/env python3

import argparse
import logging
import tempfile
import boto3  # REQUIRED! - Details here: https://pypi.org/project/boto3/
from botocore.exceptions import ClientError
from botocore.config import Config
from dotenv import load_dotenv  # Project Must install Python Package:  python-dotenv
import os
import sys

from spf.scripts.run_filters_on_data import generate_configs_to_run
from spf.utils import get_md5_of_file

os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required" 
os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"

# Return a boto3 client object for B2 service
def get_b2_client(endpoint, key_id, application_key):
        b2_client = boto3.client(service_name='s3',
                                 endpoint_url=endpoint,                # Backblaze endpoint
                                 aws_access_key_id=key_id,              # Backblaze keyID
                                 aws_secret_access_key=application_key, # Backblaze applicationKey
                                 config=Config(
                                        s3={
                                            'checksum_mode': 'disabled',
                                        }
                                    ))
        return b2_client


# Return a boto3 resource object for B2 service
def get_b2_resource(endpoint, key_id, application_key):
    b2 = boto3.resource(service_name='s3',
                        endpoint_url=endpoint,                # Backblaze endpoint
                        aws_access_key_id=key_id,              # Backblaze keyID
                        aws_secret_access_key=application_key, # Backblaze applicationKey
                        config = Config(
                            signature_version='s3v4',
                            s3={
                                            'checksum_mode': 'disabled',
                                        }
                    ))
    return b2


# Upload specified file into the specified bucket
def upload_file(bucket, directory, file, b2, b2path=None):
    file_path = directory + '/' + file
    remote_path = b2path
    if remote_path is None:
        remote_path = file
    try:
        response = b2.Bucket(bucket).upload_file(file_path, remote_path)
    except ClientError as ce:
        print('error', ce)

    return response

def download_folder(b2_client, bucket, prefix, local_dir):
    os.makedirs(local_dir)
    resp = b2_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in resp.get('Contents', []):
        filename = obj['Key']
        local_filename=f'{local_dir}/{os.path.basename(filename)}'
        print(f"Downloading {filename} -> {local_filename}")
        b2_client.download_file(bucket, filename, local_filename)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        required=False,
    )

    return parser

def main():
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = get_parser()
    args = parser.parse_args()

    load_dotenv()
    
    endpoint_rw = os.getenv("ENDPOINT_URL_YOUR_BUCKET")
    key_id_rw = os.getenv("KEY_ID_YOUR_ACCOUNT")
    application_key_rw = os.getenv("APPLICATION_KEY_YOUR_ACCOUNT")

    b2_client = get_b2_client(endpoint_rw, key_id_rw, application_key_rw)

    bucket = 'projectspf'
    
    #first need to download the model and get checksums

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname='/tmp/x'

        inference_model_prefix='md2/cache/inference_models'+'/paired_checkpoints_windowedbeamformer_nosig_3p5_randx_wd0p02_gains_vehicle_0p2dropout_noroverbounce'

        local_model_folder=os.path.join(tmpdirname,'model')
        #download_folder(b2_client,bucket,inference_model_prefix,local_model_folder)

        model_config_fn=f'{local_model_folder}/config.yml'
        model_checkpoint_fn=f'{local_model_folder}/best.pth'
        config_checksum = get_md5_of_file(model_config_fn)
        checkpoint_checksum = get_md5_of_file(model_checkpoint_fn)

        print(config_checksum,checkpoint_checksum)

        #download inference cache /mnt/md2$ ls cache/inference/dec18_mission1_rover1.zarr/3.500/dc0661eb09c048996e81545363ff8e33/d1655af080f3721a7e4852221955950e.npz 
        prefix = 'md2/cache/nosig_data'

        # Use the *custom* client for listing and downloading:
        resp = b2_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get('Contents', []):
            filename = obj['Key']
            if '.yaml' in filename :
                remote_zarr_fn=filename.replace('.yaml','.zarr')
                base_zarr_fn=os.path.join(tmpdirname,os.path.basename(remote_zarr_fn))
                print(remote_zarr_fn,filename)

                # download the nosig data
                download_folder(b2_client,bucket,remote_zarr_fn,base_zarr_fn)
                #breakpoint()
                jobs=generate_configs_to_run(
                        yaml_config_fn=args.config,
                        work_dir=tmpdirname,
                        dataset_fns=[base_zarr_fn],
                        seed=args.seed,
                    )
                breakpoint()
                a=1

    # if args.debug:

    #     results = list(
    #         tqdm.tqdm(
    #             map(run_jobs_with_one_dataset, jobs),
    #             total=len(jobs),
    #         )
    #     )
    # else:
    #     torch.set_num_threads(1)
    #     with Pool(args.parallel) as pool:  # cpu_count())  # cpu_count() // 4)
    #         results = list(
    #             tqdm.tqdm(
    #                 pool.imap_unordered(run_jobs_with_one_dataset, jobs),
    #                 total=len(jobs),
    #             )
    #         )
    #         #print(f"Downloading {filename} -> {base_filename}")
    #         # Use that same custom client
    #         #b2_client.download_file(bucket, filename, base_filename)
    #         break

        

# Optional (not strictly required)
if __name__ == '__main__':
    main()