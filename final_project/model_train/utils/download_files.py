import boto3
import os

# fetch credentials from env variables
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

# setup a AWS S3 client/resource
s3 = boto3.resource(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# point the resource at the existing bucket
bucket = s3.Bucket("anyoneai-datasets")

# download the training dataset
with open("tt0078841.tar", "wb") as data:
    bucket.download_fileobj("shot-type/keyframes/tt0078841.tar", data)

for file in bucket.objects.filter(Prefix="shot-type/movie-shot-trailers"):
    print(file)
