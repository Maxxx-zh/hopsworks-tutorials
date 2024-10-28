import pandas as pd

def compute_ranking_dataset(trans_fg, articles_fg, customers_fg):
    trans_df = trans_fg.select(["article_id", "customer_id", "month_sin", "month_cos"]).read()
    articles_df = articles_fg.select_except(['article_description', 'embeddings', 'image_url']).read()
    customers_df = customers_fg.select(["customer_id", "age"]).read()
    
    df = trans_df.merge(articles_df, on="article_id", how="left")
    df = df.merge(customers_df, on="customer_id", how="left")
    
    # Define the features used in the query
    query_features = ["customer_id", "age", "month_sin", "month_cos", "article_id"]
    
    df = df[query_features]
    
    # Copy the positive pairs for ranking
    positive_pairs = df.copy()
    
    # Define the number of negative pairs to generate
    n_neg = len(positive_pairs) * 10

    # Initialize the negative_pairs DataFrame
    negative_pairs = pd.DataFrame()

    # Generate random article_id for negative_pairs that are not in positive_pairs
    negative_pairs['article_id'] = positive_pairs["article_id"].drop_duplicates().sample(n_neg, replace=True, random_state=2)

    # Add customer_id to negative_pairs
    negative_pairs["customer_id"] = positive_pairs["customer_id"].sample(n_neg, replace=True, random_state=3).to_numpy()

    # Add other features to negative_pairs
    negative_pairs[["age", "month_sin", "month_cos"]] = positive_pairs[["age", "month_sin", "month_cos"]].sample(n_neg, replace=True, random_state=4).to_numpy()

    # Add labels to positive and negative pairs
    positive_pairs["label"] = 1
    negative_pairs["label"] = 0

    # Concatenate positive and negative pairs
    ranking_df = pd.concat([positive_pairs, negative_pairs[positive_pairs.columns]], ignore_index=True)
    
    # Keep unique article_id from item features
    item_df = articles_fg.read()
    item_df.drop_duplicates(subset="article_id", inplace=True)
    
    # Keep only the necessary columns from item features
    item_df = item_df[["article_id", "product_type_name", "product_group_name", "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name", 
                       "perceived_colour_master_name", "department_name", "index_name", "index_group_name", "section_name", "garment_group_name"]]
    
    # Merge with item features
    ranking_df = ranking_df.merge(item_df, on="article_id")
    
    return ranking_df



# import polars as pl
# import numpy as np

# def compute_ranking_dataset(trans_fg, articles_fg, customers_fg):
#     # Define the features used in the query
#     query_features = ["customer_id", "age", "month_sin", "month_cos", "article_id"]
    
#     # Perform the necessary joins to create the feature set
#     fg_query = trans_fg.select(["month_sin", "month_cos"]).join(
#         articles_fg.select_except(['article_description', 'embeddings', 'image_url']), 
#         on=["article_id"],
#     ).join(customers_fg.select(["customer_id", "age"]))
#     df = fg_query.read()

#     # Convert pandas DataFrame to polars DataFrame
#     df = pl.from_pandas(df[query_features])
    
#     # Copy the positive pairs for ranking
#     positive_pairs = df.clone()
    
#     # Define the number of negative pairs to generate
#     n_neg = len(positive_pairs) * 10
    
#     # Generate random article_id for negative_pairs that are not in positive_pairs
#     unique_articles = df["article_id"].unique()
#     negative_article_ids = np.random.choice(unique_articles, n_neg, replace=True)
    
#     # Generate random customer_id for negative_pairs
#     negative_customer_ids = np.random.choice(df["customer_id"], n_neg, replace=True)
    
#     # Generate random age, month_sin, month_cos for negative_pairs
#     random_indices = np.random.randint(0, len(df), n_neg)
#     negative_age = df["age"].take(random_indices)
#     negative_month_sin = df["month_sin"].take(random_indices)
#     negative_month_cos = df["month_cos"].take(random_indices)
    
#     # Create negative_pairs DataFrame
#     negative_pairs = pl.DataFrame({
#         "article_id": negative_article_ids,
#         "customer_id": negative_customer_ids,
#         "age": negative_age,
#         "month_sin": negative_month_sin,
#         "month_cos": negative_month_cos,
#     })
    
#     # Add labels to positive and negative pairs
#     positive_pairs = positive_pairs.with_columns(pl.lit(1).alias("label"))
#     negative_pairs = negative_pairs.with_columns(pl.lit(0).alias("label"))
    
#     # Concatenate positive and negative pairs
#     ranking_df = pl.concat([positive_pairs, negative_pairs], how="vertical")
    
#     # Keep unique article_id from item features
#     item_df = articles_fg.read()

#     # Convert pandas DataFrame to polars DataFrame
#     item_df = pl.from_pandas(item_df)
    
#     # Keep only the necessary columns from item features and drop duplicates
#     item_columns = [
#         "article_id", "product_type_name", "product_group_name", "graphical_appearance_name", 
#         "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name", 
#         "department_name", "index_name", "index_group_name", "section_name", "garment_group_name"
#     ]
#     item_df = item_df.select(item_columns).unique(subset="article_id")
    
#     # Merge with item features
#     ranking_df = ranking_df.join(item_df, on="article_id")
    
#     # Convert back to pandas DataFrame for compatibility
#     return ranking_df.to_pandas()