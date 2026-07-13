import boto3
from boto3.dynamodb.conditions import Attr

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table("spf_filter_results")

response = table.scan(
    FilterExpression=Attr("full_name").begins_with("march1")
)

items = response["Items"]
print(items)
a=1
