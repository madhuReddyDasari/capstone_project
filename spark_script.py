import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName('Sentiment_Analysis').getOrCreate()
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

from configurations import s3_bucket_path, deployment_type
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def display_heading(data):
	print("\n\n " +"-"*(len(data)+5))
	print("|  "+data+ "   |")
	print(" "+"-"*(len(data)+5)+"\n\n")


display_heading("Read datasets and store in dataframes")

if(deployment_type =="dev"):
	members_df = spark.read.json("memberships.json")
	persons_df = spark.read.json("persons.json")
	organizations_df = spark.read.json("organizations.json")
else:
	members_df = spark.read.json(s3_bucket_path+"memberships.json")
	persons_df = spark.read.json(s3_bucket_path+"persons.json")
	organizations_df = spark.read.json(s3_bucket_path+"organizations.json")




display_heading("join all the datasets into 1 resultant Dataset")
persons_df1 = persons_df.withColumnRenamed('id','person_id')
organizations_df_new = organizations_df.withColumnRenamed('id', 'organization_id')
result = members_df.join(persons_df1, on="person_id", how ="inner").join(organizations_df_new, on="organization_id", how="inner")





display_heading("Function to remove redundant columns")
def remove_redundant_columns(df):
	parsed_cols = set()
	all_new_cols = []
	for i in result.columns:
		all_new_cols.append(i+'_duplicate') if i in parsed_cols else all_new_cols.append(i) 
		parsed_cols.add(i)
	distinct_result = result.toDF(*all_new_cols).select([col for col in all_new_cols if not col.endswith('_duplicate')])
	return distinct_result

organised_result = remove_redundant_columns(result).drop(*['organization_id','person_id'])


display_heading("Dropping Array Columns")
array_col_type_list = [(k,v) for k,v in organised_result.dtypes if 'array' in str(v.lower())]
array_cols = [k for k,v in array_col_type_list]
new_result = organised_result.drop(*[col for col in array_cols if col!='contact_details'])



display_heading("Splitting into tables")
w = Window().orderBy(lit('A'))
contact_details = new_result.select('contact_details').distinct().withColumn("contact_details_id", row_number().over(w))
main_table = new_result.join(contact_details, on='contact_details', how = 'inner').drop('contact_details')
contact_details = contact_details.withColumn('type',contact_details.contact_details.type).withColumn('value', contact_details.contact_details.value)


display_heading("Partitioning Dataframe")
if (deployment_type=="dev"):
	main_table.write.partitionBy('gender').parquet('capstone_output')
else:
	main_table.write.partitionBy('gender').parquet(s3_bucket_path+'capstone_output')
