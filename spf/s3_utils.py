import os
import tempfile
import time
from contextlib import contextmanager

import boto3  # REQUIRED! - Details here: https://pypi.org/project/boto3/
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv  # Project Must install Python Package:  python-dotenv
from filelock import FileLock

os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"

import io
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv


def get_b2_client():
    """
    Return a boto3 client object configured for Backblaze B2.
    You can adjust or load ENV variables as needed.
    """
    load_dotenv()
    endpoint = os.getenv("B2_ENDPOINT")
    key_id = os.getenv("B2_KEY_ID")
    application_key = os.getenv("B2_APP_KEY")

    return boto3.client(
        service_name="s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=application_key,
        config=Config(s3={"checksum_mode": "disabled"}),
    )


class B2ReadIO:
    """
    A simple context manager / file-like object for reading
    from a B2 object using only boto3.
    """

    def __init__(self, client, bucket, key):
        self.client = client
        self.bucket = bucket
        self.key = key
        self.response = self.client.get_object(Bucket=self.bucket, Key=self.key)
        self.body = self.response["Body"]

    def read(self, size=-1):
        """
        Read `size` bytes, or -1 to read all.
        """
        return self.body.read(size)

    def close(self):
        """
        Close the underlying body.
        """
        if self.body:
            self.body.close()
            self.body = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class B2WriteIO:
    """
    A simple context manager / file-like object for writing
    to a B2 object using only boto3.

    For simplicity, all data is stored in-memory and uploaded
    once the context closes.
    """

    def __init__(self, client, bucket, key):
        self.client = client
        self.bucket = bucket
        self.key = key
        self.buffer = io.BytesIO()

    def write(self, data):
        # You can write bytes (or in text-mode, you’ll want to encode first)
        return self.buffer.write(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        # No-op for now
        pass

    def close(self):
        """
        When closing, upload the entire buffer to B2.
        """
        if self.buffer is not None:
            self.buffer.seek(0)
            self.client.upload_fileobj(self.buffer, self.bucket, self.key)
            self.buffer.close()
            self.buffer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


b2_cache_folder = None


def b2_get_or_set_cache():
    global b2_cache_folder
    if b2_cache_folder is None:
        b2_cache_folder = tempfile.TemporaryDirectory()
    return b2_cache_folder


def b2_reset_cache():
    global b2_cache_folder
    old_b2_cache_folder = b2_cache_folder
    b2_cache_folder = tempfile.TemporaryDirectory()
    return old_b2_cache_folder


def b2_download_folder_cache(fn):
    b2_client = get_b2_client()
    bucket, b2_path = b2path_to_bucket_and_path(fn)
    resp = b2_client.list_objects_v2(Bucket=bucket, Prefix=b2_path)
    for obj in resp.get("Contents", []):
        b2_file_to_local_with_cache(f'b2://{bucket}/{obj["Key"]}')
    return os.path.join(b2_cache_folder.name, bucket, b2_path)


def b2_file_to_local_with_cache(fn, *args, **kwargs):
    global b2_cache_folder
    if "b2:" == fn[:3]:
        b2_client = get_b2_client()
        if b2_cache_folder is None:
            b2_cache_folder = tempfile.TemporaryDirectory()
        tmpdirname = b2_cache_folder.name
        local_fn = f"{tmpdirname}/{fn[5:]}"
        os.makedirs(os.path.dirname(local_fn), exist_ok=True)

        # Create a lock file *specific* to this B2 file to avoid collisions with other files
        # For instance, local_fn + ".lock"
        lock_file = local_fn + ".lock"
        lock = FileLock(lock_file)

        # Acquire the lock before checking/downloading
        with lock:
            # Double-check after acquiring the lock in case another process just downloaded it
            if not os.path.exists(local_fn):
                print("DOWNLOADING", fn, local_fn)
                bucket, b2_path = b2path_to_bucket_and_path(fn)
                b2_client.download_file(bucket, b2_path, local_fn)

        return local_fn
    return fn


@contextmanager
def b2_file_as_local(fn, *args, **kwargs):
    if "b2:" == fn[:3]:
        b2_client = get_b2_client()
        with tempfile.TemporaryDirectory() as tmpdirname:
            local_fn = f"{tmpdirname}/{fn.split('/')[-1]}"
            bucket, b2_path = b2path_to_bucket_and_path(fn)
            b2_client.download_file(
                bucket,
                b2_path,
                local_fn,
            )
            with open(local_fn, *args, **kwargs) as f:
                yield f
    else:
        with open(fn, *args, **kwargs) as f:
            yield f


def spf_open(fn, mode="r", **kwargs):
    """
    A custom open function that:
      - If `fn` starts with 'b2:', uses a Backblaze B2 client (via boto3).
      - Otherwise, uses the built-in `open` for local files.

    Supported modes (basic):
      - 'r' or 'rb' for read
      - 'w' or 'wb' for write
    """
    # If it's not a B2 path, just fallback to normal open
    if not fn.startswith("b2:"):
        return open(fn, mode, **kwargs)

    bucket, key_name = b2path_to_bucket_and_path(fn)
    # Get a B2 client
    b2_client = get_b2_client()

    # Decide read or write
    # (Check if 'r' is in mode for reading, or 'w' is in mode for writing)
    # You can refine this logic as needed.
    if "r" in mode:
        return B2ReadIO(b2_client, bucket, key_name)
    elif "w" in mode:
        return B2WriteIO(b2_client, bucket, key_name)
    else:
        raise ValueError(
            f"Mode '{mode}' is not supported for B2 paths. Use 'r', 'rb', 'w', or 'wb'."
        )


def spf_exists(fn):
    """
    Check if the given path exists. Works for:
      - Local paths (e.g., "/tmp/test.txt" or "relative/path.txt")
      - B2 paths (e.g., "b2:mybucket/some/object.txt")

    Returns True if the path exists, False otherwise.
    """
    # Fallback to local file existence if not a b2 path
    if not fn.startswith("b2://"):
        return os.path.exists(fn)

    # Otherwise, parse the B2 path: "b2:bucketName/key"
    path = fn[5:]  # remove 'b2://'
    parts = path.split("/", 1)
    if len(parts) == 1:
        # Edge case: "b2:mybucket" (no key)
        bucket_name = parts[0]
        key_name = ""
    else:
        bucket_name, key_name = parts

    if not bucket_name:
        # If there's no bucket name, there's no valid key
        return False

    # Make sure there's a key to check; an empty key might be treated differently
    if not key_name:
        # For some cases, users might want to check if a bucket itself exists,
        # but here we'll assume an empty key always returns False.
        return False

    # Create a B2 client
    client = get_b2_client()

    # Attempt to HEAD the object. If it's missing, B2 will raise a 404 error.
    try:
        client.head_object(Bucket=bucket_name, Key=key_name)
        return True
    except ClientError as e:
        # Check for "404" or "NoSuchKey" error codes
        error_code = e.response["Error"]["Code"]
        if error_code in ("NoSuchKey", "404"):
            return False
        # Other errors might indicate other issues (e.g., permissions)
        # Reraise or handle as needed:
        raise


def b2path_to_bucket_and_path(b2path):
    components = b2path.replace("b2://", "").split("/")
    return components[0], "/".join(components[1:])


# Return a boto3 client object for B2 service
def get_b2_client():
    load_dotenv()
    endpoint = os.getenv("B2_ENDPOINT")
    key_id = os.getenv("B2_KEY_ID")
    application_key = os.getenv("B2_APP_KEY")
    b2_client = boto3.client(
        service_name="s3",
        endpoint_url=endpoint,  # Backblaze endpoint
        aws_access_key_id=key_id,  # Backblaze keyID
        aws_secret_access_key=application_key,  # Backblaze applicationKey
        config=Config(
            s3={
                "checksum_mode": "disabled",
            }
        ),
    )
    return b2_client


# Return a boto3 resource object for B2 service
def get_b2_resource():
    load_dotenv()
    endpoint = os.getenv("B2_ENDPOINT")
    key_id = os.getenv("B2_KEY_ID")
    application_key = os.getenv("B2_APP_KEY")
    b2 = boto3.resource(
        service_name="s3",
        endpoint_url=endpoint,  # Backblaze endpoint
        aws_access_key_id=key_id,  # Backblaze keyID
        aws_secret_access_key=application_key,  # Backblaze applicationKey
        config=Config(
            signature_version="s3v4",
            s3={
                "checksum_mode": "disabled",
            },
        ),
    )
    return b2


# Upload specified file into the specified bucket
def b2_upload_file(bucket, directory, file, b2, b2path=None):
    file_path = directory + "/" + file
    remote_path = b2path
    if remote_path is None:
        remote_path = file
    try:
        response = b2.Bucket(bucket).upload_file(file_path, remote_path)
    except ClientError as ce:
        print("error", ce)

    return response


def b2_download_folder(b2_client, bucket, prefix, local_dir):
    os.makedirs(local_dir)
    resp = b2_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in resp.get("Contents", []):
        filename = obj["Key"]
        local_filename = f"{local_dir}/{os.path.basename(filename)}"
        print(f"b2_download_folder Downloading {filename} -> {local_filename}")
        b2_client.download_file(bucket, filename, local_filename)
